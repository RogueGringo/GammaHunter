# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Experiment harness: param-parity gate + RunMetricRecord JSONL logging.

Torch is OPTIONAL. Without torch, arms use pure-Python stubs that still emit
the ADR-001 schema (placeholder trajectories) so schema tests pass.

With torch present, the feed-forward arm attaches a real ``FeedForward``
module: eval rows get real CE loss / accuracy. FF trajectories are empty by
design (``drift_trajectory=[]``, ``terminal_drift=None``,
``perturbation_delta=None``, ``tokens_decoded=None``). The geometric arm may
also attach a real ``GeometricRecurrent`` (weight-tied Phi + tau) with real
drift telemetry; Loop / CoT remain stubs. No science OPEN claims.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, TextIO, Union

from reachability_gen.adr_invariants import (
    EPS_SIGMA,
    PARAM_TOL,
    REQUIRED_METRIC_KEYS,
    SCIENCE_OPEN_DEFAULT,
    assert_f_block_matches_implementation,
    validate_metric_record,
)
from reachability_gen.arms import (
    ChainOfThoughtArm,
    EuclideanLoopArm,
    FeedForwardArm,
    GeometricRecurrentArm,
)
from reachability_gen.flops import F_block
from reachability_gen.param_match import ParamMatchResult, check_param_match
from reachability_gen.schema import RunMetricRecord

try:
    import torch  # type: ignore

    HAS_TORCH = True
except ImportError:  # pragma: no cover - exercised in CI without torch
    torch = None  # type: ignore
    HAS_TORCH = False

ArmLike = Any
ForwardFn = Callable[..., Any]

# Back-compat re-export
RUN_METRIC_REQUIRED_KEYS = REQUIRED_METRIC_KEYS


def _arm_cycles_or_depth(arm: ArmLike) -> int:
    """T for Geo/Loop, L for FF/CoT (depth / layers — not K_used)."""
    if hasattr(arm, "T"):
        return int(arm.T)
    if hasattr(arm, "L"):
        return int(arm.L)
    return 0


def _arm_tokens_decoded(arm: ArmLike) -> Optional[int]:
    """K_used for CoT; None for FF / Geo / Loop."""
    # Prefer explicit CoT marker over bare K_used on other arms.
    if isinstance(arm, ChainOfThoughtArm) or (
        hasattr(arm, "K_used") and hasattr(arm, "K_cap")
    ):
        return int(arm.K_used)
    return None


def _arm_d_model(arm: ArmLike) -> int:
    if hasattr(arm, "d"):
        return int(arm.d)
    cfg = getattr(arm, "config", None)
    if cfg is not None and hasattr(cfg, "d"):
        return int(cfg.d)
    return 0


def _is_ff_arm(arm: ArmLike) -> bool:
    return isinstance(arm, FeedForwardArm) or (
        hasattr(arm, "name") and str(getattr(arm, "name", "")).startswith("ff-")
    )


def _is_geo_arm(arm: ArmLike) -> bool:
    return isinstance(arm, GeometricRecurrentArm) or (
        hasattr(arm, "name") and str(getattr(arm, "name", "")).startswith("geo-")
    )


def _run_forward(
    arm: ArmLike,
    batch: Mapping[str, Any],
    *,
    return_trajectory: bool,
) -> tuple[list[Any], list[Any]]:
    """Call arm.forward if present; else forward_stub. Always returns (logits, traj).

    Logits may be a flat list of scalars (stubs) or a list of ``[logit0, logit1]``
    pairs from the real torch FeedForward head.
    """
    if hasattr(arm, "forward") and callable(arm.forward):
        out = arm.forward(batch, return_trajectory=return_trajectory)
        if isinstance(out, tuple) and len(out) == 2:
            logits, traj = out
            traj_list: list[Any] = [] if traj is None else list(traj)
            return list(logits), traj_list
        if not return_trajectory and isinstance(out, dict):
            # Legacy stub dict path
            drifts = out.get("drift_trajectory")
            if drifts is None:
                drifts = out.get("z_drift_series") or []
            return list(out["logits"]), list(drifts)
        if return_trajectory:
            raise TypeError(
                f"{arm!r}.forward(return_trajectory=True) must return (logits, trajectory)"
            )
        raise TypeError(f"unexpected forward return type: {type(out)}")

    stub = arm.forward_stub(batch)
    logits = list(stub["logits"])
    traj = []
    if return_trajectory:
        traj = [{"t": 0, "arm": stub.get("arm"), "placeholder": True}]
    return logits, traj


def _logits_to_ce_and_acc(
    logits: list[Any],
    y: int,
) -> tuple[float, float, int]:
    """Compute CE loss + accuracy from stub scalars or real ``[logit0, logit1]``.

    Returns ``(loss, accuracy, yhat)``.
    """
    if not logits:
        return 0.0, 0.0, 0
    first = logits[0]
    # Real FF: nested [logit_class0, logit_class1]
    if isinstance(first, (list, tuple)) and len(first) == 2:
        import math

        l0, l1 = float(first[0]), float(first[1])
        m = max(l0, l1)
        e0 = math.exp(l0 - m)
        e1 = math.exp(l1 - m)
        z = e0 + e1
        p_y = (e1 / z) if y == 1 else (e0 / z)
        loss = -math.log(max(p_y, 1e-12))
        yhat = 1 if l1 > l0 else 0
        acc = 1.0 if yhat == y else 0.0
        return float(loss), float(acc), int(yhat)
    # Stub scalar logit
    logit0 = float(first)
    yhat = 1 if logit0 > 0.0 else 0
    acc = 1.0 if yhat == y else 0.0
    # Placeholder stub loss remains 0.0 unless caller overrides
    return 0.0, float(acc), int(yhat)


def measure_phase_diagnostics(
    trajectory: Sequence[Any],
    *,
    eps_sigma: float = EPS_SIGMA,
) -> dict[str, Any]:
    """Phase / perturbation diagnostics (MEASURE plumbing only).

    When ``trajectory`` holds real latent tensors / arrays with a ``.shape``,
    computes ``δ_t = mean_batch ||z_{t+1}-z_t||_2``. Otherwise emits placeholder
    zeros (stub arms). Does **not** stamp science OPEN.
    """
    drifts: list[float] = []
    # Real tensor trajectory?
    real = False
    if trajectory and HAS_TORCH:
        first = trajectory[0]
        if hasattr(first, "detach") or hasattr(first, "shape"):
            try:
                from reachability_gen.models.geometric import drift_from_trajectory

                # Ensure torch tensors
                tensors = []
                for z in trajectory:
                    if hasattr(z, "detach"):
                        tensors.append(z)
                    else:
                        tensors.append(torch.as_tensor(z))
                drifts = drift_from_trajectory(tensors)
                real = True
            except Exception:
                real = False
    if not real:
        for _ in range(max(len(trajectory) - 1, 0)):
            # Placeholder: no real latent norms yet. δ_t = ||z_{t+1}-z_t||_2
            drifts.append(0.0)
    terminal_drift: Optional[float] = drifts[-1] if drifts else None
    perturbation_delta: Optional[float] = float(eps_sigma) if eps_sigma is not None else None
    return {
        "drift_trajectory": drifts,
        "terminal_drift": terminal_drift,
        "perturbation_delta": perturbation_delta,
        "eps_sigma": eps_sigma,
        "science_open": SCIENCE_OPEN_DEFAULT,
    }


class ExperimentHarness:
    """Eval harness that asserts param parity and logs RunMetricRecord rows."""

    def __init__(
        self,
        arms: Sequence[ArmLike],
        *,
        metrics_path: Union[str, Path],
        param_tolerance: float = PARAM_TOL,
        assert_parity: bool = True,
        exclude_cot_from_parity: bool = True,
        eps_sigma: float = EPS_SIGMA,
        return_trajectory: bool = True,
        run_id: Optional[str] = None,
        epoch: int = 0,
    ) -> None:
        self.arms = list(arms)
        self.metrics_path = Path(metrics_path)
        self.param_tolerance = param_tolerance
        self.eps_sigma = eps_sigma
        self.return_trajectory = return_trajectory
        self.run_id = run_id or f"stub-{uuid.uuid4().hex[:12]}"
        self.epoch = int(epoch)
        self._step = 0
        self._fh: Optional[TextIO] = None
        self._n_logged = 0

        # ADR runtime assertions
        assert_f_block_matches_implementation(F_block)

        exclude = ("cot",) if exclude_cot_from_parity else ()
        self.parity: ParamMatchResult = check_param_match(
            self.arms,
            tolerance=param_tolerance,
            exclude_names=exclude,
        )
        if assert_parity and not self.parity.passed:
            raise AssertionError(
                f"param parity failed (±{param_tolerance:.0%}): {self.parity.notes}"
            )
        if self.parity.science_open:
            raise AssertionError("param match must never self-stamp science_open")

    def open(self) -> "ExperimentHarness":
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.metrics_path.open("w", encoding="utf-8")
        self._n_logged = 0
        return self

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "ExperimentHarness":
        return self.open()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @property
    def n_logged(self) -> int:
        return self._n_logged

    def log_eval_step(self, record: Mapping[str, Any]) -> RunMetricRecord:
        """Validate and append one RunMetricRecord as a JSONL line."""
        data = dict(record)
        validate_metric_record(data)
        if self._fh is None:
            raise RuntimeError("harness not opened; use as context manager or call open()")
        self._fh.write(json.dumps(data, sort_keys=True) + "\n")
        self._fh.flush()
        self._n_logged += 1
        return data  # type: ignore[return-value]

    def eval_example(
        self,
        example: Mapping[str, Any],
        arm: ArmLike,
        *,
        context_len: Optional[int] = None,
        loss: Optional[float] = None,
        accuracy: Optional[float] = None,
    ) -> RunMetricRecord:
        """Run one forward (real FF when attached) and log a RunMetricRecord."""
        batch = {
            "batch_size": 1,
            "encoding": [example.get("encoding", "")],
            "y": [example.get("y", 0)],
            "s": [example.get("s", 0)],
            "t": [example.get("t", 0)],
        }
        t0 = time.perf_counter()
        logits, trajectory = _run_forward(
            arm, batch, return_trajectory=self.return_trajectory
        )
        _latency_ms = (time.perf_counter() - t0) * 1000.0  # measured; not logged
        del _latency_ms

        y = int(example.get("y", 0))
        ce_loss, ce_acc, yhat = _logits_to_ce_and_acc(logits, y)
        del yhat

        m = context_len
        if m is None:
            enc = example.get("encoding") or ""
            m = max(len(str(enc).split()), 1)

        report = arm.inference_flops(int(m))
        is_ff = _is_ff_arm(arm)
        is_geo = _is_geo_arm(arm)
        real_ff = is_ff and bool(getattr(arm, "has_real_model", False))
        real_geo = is_geo and bool(getattr(arm, "has_real_model", False))
        real_torch = real_ff or real_geo

        # FF trajectories empty by design (ADR note); skip phase stub fields.
        if is_ff:
            drift_trajectory: list[float] = []
            terminal_drift: Optional[float] = None
            perturbation_delta: Optional[float] = None
            tokens_decoded: Optional[int] = None
        else:
            diag = measure_phase_diagnostics(trajectory, eps_sigma=self.eps_sigma)
            drift_trajectory = list(diag["drift_trajectory"])
            terminal_drift = diag["terminal_drift"]
            perturbation_delta = diag["perturbation_delta"]
            tokens_decoded = _arm_tokens_decoded(arm)

        step = self._step
        self._step += 1

        if loss is not None:
            out_loss = float(loss)
        elif real_torch:
            out_loss = float(ce_loss)
        else:
            out_loss = 0.0  # stub arms

        if accuracy is not None:
            out_acc = float(accuracy)
        elif real_torch:
            out_acc = float(ce_acc)
        else:
            out_acc = float(ce_acc)  # stub scalar sign → 0/1

        record: dict[str, Any] = {
            # --- required user TypedDict keys ---
            "run_id": self.run_id,
            "seed": int(example.get("seed", 0)),
            "arm": str(getattr(arm, "name", type(arm).__name__)),
            "step": int(step),
            "epoch": int(self.epoch),
            "param_count": int(arm.param_count()),
            "d_model": _arm_d_model(arm),
            "seq_len": int(m),
            "cycles_or_depth": _arm_cycles_or_depth(arm),
            "tokens_decoded": tokens_decoded,
            "cumulative_flops": float(report.flops),
            "hop_distance": int(example.get("hop_distance", -1)),
            "is_ood": bool(example.get("is_ood", False)),
            "loss": out_loss,
            "accuracy": out_acc,
            "drift_trajectory": drift_trajectory,
            "terminal_drift": terminal_drift,
            "perturbation_delta": perturbation_delta,
            # --- optional documented example-linkage extras ---
            "split": example.get("split", ""),
            "n": int(example.get("n", 0)),
            "p": float(example.get("p", 0.0)),
            "edge_hash": example.get("edge_hash", ""),
            "s": int(example.get("s", 0)),
            "t": int(example.get("t", 0)),
            "y": y,
        }
        return self.log_eval_step(record)


def load_examples_jsonl(path: Union[str, Path], *, limit: Optional[int] = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def eval_jsonl(
    examples_path: Union[str, Path],
    metrics_path: Union[str, Path],
    arms: Sequence[ArmLike],
    *,
    limit: Optional[int] = None,
    assert_parity: bool = True,
    run_id: Optional[str] = None,
    epoch: int = 0,
) -> int:
    """Eval loop: JSONL examples → arm stubs → metrics JSONL. Returns rows written."""
    examples = load_examples_jsonl(examples_path, limit=limit)
    with ExperimentHarness(
        arms,
        metrics_path=metrics_path,
        assert_parity=assert_parity,
        run_id=run_id,
        epoch=epoch,
    ) as harness:
        for ex in examples:
            for arm in arms:
                harness.eval_example(ex, arm)
    return harness.n_logged


def default_stub_arms(
    *,
    d: int = 64,
    T: int = 4,
    L: int = 1,
    attach_ff_model: Optional[bool] = None,
    attach_geo_model: Optional[bool] = None,
    ff_seed: int = 0,
    geo_seed: int = 0,
) -> list[ArmLike]:
    """Param-matched Geo (no tau) / FF / Loop + CoT.

    When torch is available (and ``attach_ff_model`` is not False), attaches a
    real ``FeedForward`` module to the FF arm so eval rows get real CE
    loss/accuracy. Set ``attach_geo_model=True`` to also attach a real
    ``GeometricRecurrent``. Loop / CoT stay stubs by default.
    """
    geo = GeometricRecurrentArm(T=T, d=d, use_tau=False)
    ff = FeedForwardArm(L=L, d=d)
    loop = EuclideanLoopArm(T=T, d=d)
    cot = ChainOfThoughtArm(K_cap=16, d=d, L=L, K_used=4)
    do_attach_ff = HAS_TORCH if attach_ff_model is None else bool(attach_ff_model)
    if do_attach_ff and HAS_TORCH:
        ff.attach_default_model(seed=ff_seed)
    if attach_geo_model and HAS_TORCH:
        geo.attach_default_model(seed=geo_seed)
    return [geo, ff, loop, cot]


__all__ = [
    "ExperimentHarness",
    "HAS_TORCH",
    "REQUIRED_METRIC_KEYS",
    "RUN_METRIC_REQUIRED_KEYS",
    "default_stub_arms",
    "eval_jsonl",
    "load_examples_jsonl",
    "measure_phase_diagnostics",
    "validate_metric_record",
]
