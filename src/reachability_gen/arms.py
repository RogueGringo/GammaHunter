# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Arm interfaces for the geometric-recurrence testbed (RESEARCH / MEASURE).

CoT remains a Protocol stub (placeholder logits). The feed-forward,
geometric, and Euclidean-loop arms optionally host real torch modules
(``FeedForward`` / ``GeometricRecurrent`` / ``EuclideanLoop``) when torch
is installed; otherwise stub paths.

Encoding remains the locked edge-list scheme in ``encode.py``.
No science OPEN claims.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, runtime_checkable

from reachability_gen.adr_invariants import PARAM_TOL
try:
    import torch  # type: ignore

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    HAS_TORCH = False

from reachability_gen.flops import (
    FlopReport,
    flops_cot,
    flops_euclidean_loop,
    flops_feedforward,
    flops_geometric,
)

# Default non-embedding param match tolerance (±5% per ADR-001).
PARAM_MATCH_TOLERANCE: float = PARAM_TOL


@dataclass(frozen=True)
class SharedArmConfig:
    """Shared width / context / matching knobs across arms.

    Parameters
    ----------
    d :
        Embedding / hidden width.
    m :
        Default context length (tokenized edge-list + query). Overrideable
        per ``inference_flops`` call.
    param_match_tolerance :
        Relative tolerance for non-embedding param matching (default ±5%).
    vocab_size :
        Placeholder vocab size for stub param estimates (not used in FLOPs).
    mlp_expansion :
        Must match FLOP schematic (default 4).
    """

    d: int = 64
    m: int = 32
    param_match_tolerance: float = PARAM_MATCH_TOLERANCE
    vocab_size: int = 128
    mlp_expansion: int = 4


def params_per_block(d: int, *, mlp_expansion: int = 4) -> int:
    """Schematic non-embedding params for one transformer-style block.

    Attention Q,K,V,O: 4 * d * d
    MLP up+down:       2 * d * (mlp_expansion * d)
    (Biases / LayerNorm omitted — same order as FLOP schematic.)
    """
    attn = 4 * d * d
    mlp = 2 * d * (mlp_expansion * d)
    return int(attn + mlp)


def tau_embed_params(d: int) -> int:
    """Params for scalar->d tau/phase linear (weight d + bias d)."""
    return int(2 * d)


@runtime_checkable
class Arm(Protocol):
    """Minimal arm interface for FLOP / param scaffolding."""

    @property
    def name(self) -> str: ...

    def param_count(self) -> int:
        """Non-embedding parameter estimate (schematic)."""
        ...

    def inference_flops(self, context_len: int, **kwargs: Any) -> FlopReport:
        """Return a :class:`FlopReport` for the given context length."""
        ...

    def forward_stub(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Placeholder forward — returns logits stub, no real computation."""
        ...


@dataclass
class FeedForwardArm:
    """Feed-forward transformer control: L unshared layers.

    When ``model`` is a real torch ``FeedForward`` (or torch is available and
    ``attach_default_model()`` has been called), ``forward`` runs the real
    network. Otherwise falls back to the placeholder stub. FF trajectories are
    empty by design (``[]`` / ``None``) — no recurrent latent to drift.
    """

    L: int
    d: int
    config: SharedArmConfig = field(default_factory=SharedArmConfig)
    mlp_expansion: int = 4
    # Optional real torch module; default None → stub path.
    model: Any = field(default=None, repr=False, compare=False)
    vocab: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.L < 0:
            raise ValueError(f"L must be non-negative, got {self.L}")
        if self.d < 1:
            raise ValueError(f"d must be >= 1, got {self.d}")

    @property
    def name(self) -> str:
        return f"ff-L{self.L}-d{self.d}"

    @property
    def has_real_model(self) -> bool:
        return self.model is not None and HAS_TORCH

    def attach_default_model(
        self,
        *,
        vocab_size: Optional[int] = None,
        n_heads: int = 4,
        max_len: int = 256,
        seed: Optional[int] = None,
    ) -> Any:
        """Build and attach a real ``FeedForward`` if torch is available."""
        if not HAS_TORCH:
            return None
        from reachability_gen.models.feedforward import FeedForward
        from reachability_gen.tokenize import build_vocab

        if self.vocab is None:
            self.vocab = build_vocab()
        vs = int(vocab_size) if vocab_size is not None else len(self.vocab)
        if seed is not None:
            torch.manual_seed(int(seed))
        heads = n_heads if self.d % n_heads == 0 else 1
        self.model = FeedForward(
            vocab_size=vs,
            d=self.d,
            L=max(self.L, 1),
            n_heads=heads,
            max_len=max_len,
            mlp_expansion=self.mlp_expansion,
            pad_id=int(self.vocab.pad_id),
        )
        return self.model

    def param_count(self) -> int:
        if self.has_real_model and hasattr(self.model, "non_embedding_param_count"):
            # Prefer schematic L * block for parity with Geo/Loop; real count
            # includes LayerNorm / biases. Keep schematic for ADR parity gate.
            pass
        return self.L * params_per_block(self.d, mlp_expansion=self.mlp_expansion)

    def inference_flops(self, context_len: int, **kwargs: Any) -> FlopReport:
        return flops_feedforward(
            context_len,
            self.d,
            self.L,
            params_estimate=self.param_count(),
        )

    def forward_stub(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        n = _batch_size(batch)
        return {
            "logits": [0.0] * n,
            "arm": self.name,
            "placeholder": True,
            "drift_trajectory": [],  # FF: empty by design
        }

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        return_trajectory: bool = False,
    ) -> tuple[Any, Any] | dict[str, Any]:
        if self.has_real_model:
            return self._forward_real(batch, return_trajectory=return_trajectory)
        return _forward_impl(self, batch, return_trajectory=return_trajectory, steps=0)

    def _forward_real(
        self,
        batch: Mapping[str, Any],
        *,
        return_trajectory: bool,
    ) -> tuple[Any, Any]:
        """Run attached torch FeedForward; trajectory always empty/None."""
        from reachability_gen.tokenize import batch_encode, build_vocab

        if self.vocab is None:
            self.vocab = build_vocab()
        encodings = batch.get("encoding")
        if encodings is None:
            encodings = [""]
        if isinstance(encodings, str):
            encodings = [encodings]
        ids, mask = batch_encode(list(encodings), self.vocab)
        token_ids = torch.tensor(ids, dtype=torch.long)
        attention_mask = torch.tensor(mask, dtype=torch.long)
        self.model.eval()
        with torch.no_grad():
            logits_t, _traj = self.model(
                token_ids, attention_mask, return_trajectory=return_trajectory
            )
        # Harness historically expected a list of per-example scalars for
        # stub arms; for real FF we pass class-1 logit as the scalar score
        # and stash full [B,2] on the model output path via CE in harness.
        # Return raw 2-class logits as nested lists for harness CE.
        logits_list = logits_t.detach().cpu().tolist()  # [[logit0, logit1], ...]
        # Trajectory empty by design for FF.
        traj: list = []
        if return_trajectory:
            return logits_list, traj
        return logits_list, None


@dataclass
class GeometricRecurrentArm:
    """Geometric recurrence treatment: weight-tied Phi reused T times.

    When ``model`` is a real torch ``GeometricRecurrent`` (or torch is available
    and ``attach_default_model()`` has been called), ``forward`` runs the real
    network and can return latent trajectory tensors for drift telemetry.
    Otherwise falls back to the placeholder stub.
    """

    T: int
    d: int
    use_tau: bool = True
    config: SharedArmConfig = field(default_factory=SharedArmConfig)
    mlp_expansion: int = 4
    model: Any = field(default=None, repr=False, compare=False)
    vocab: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.T < 0:
            raise ValueError(f"T must be non-negative, got {self.T}")
        if self.d < 1:
            raise ValueError(f"d must be >= 1, got {self.d}")

    @property
    def name(self) -> str:
        tau = "tau" if self.use_tau else "notau"
        return f"geo-T{self.T}-d{self.d}-{tau}"

    @property
    def has_real_model(self) -> bool:
        return self.model is not None and HAS_TORCH

    def attach_default_model(
        self,
        *,
        vocab_size: Optional[int] = None,
        n_heads: int = 4,
        max_len: int = 256,
        seed: Optional[int] = None,
    ) -> Any:
        """Build and attach a real ``GeometricRecurrent`` if torch is available."""
        if not HAS_TORCH:
            return None
        from reachability_gen.models.geometric import GeometricRecurrent
        from reachability_gen.tokenize import build_vocab

        if self.vocab is None:
            self.vocab = build_vocab()
        vs = int(vocab_size) if vocab_size is not None else len(self.vocab)
        if seed is not None:
            torch.manual_seed(int(seed))
        heads = n_heads if self.d % n_heads == 0 else 1
        self.model = GeometricRecurrent(
            vocab_size=vs,
            d=self.d,
            T=max(self.T, 1),
            n_heads=heads,
            max_len=max_len,
            mlp_expansion=self.mlp_expansion,
            pad_id=int(self.vocab.pad_id),
            use_tau=self.use_tau,
        )
        return self.model

    def param_count(self) -> int:
        # Weight-tied: one block regardless of T (schematic ADR parity).
        p = params_per_block(self.d, mlp_expansion=self.mlp_expansion)
        if self.use_tau:
            p += tau_embed_params(self.d)
        return p

    def inference_flops(self, context_len: int, **kwargs: Any) -> FlopReport:
        T = int(kwargs.get("T", self.T))
        use_tau = bool(kwargs.get("use_tau", self.use_tau))
        return flops_geometric(
            context_len,
            self.d,
            T,
            use_tau=use_tau,
            params_estimate=self.param_count(),
        )

    def forward_stub(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        n = _batch_size(batch)
        return {
            "logits": [0.0] * n,
            "arm": self.name,
            "T": self.T,
            "placeholder": True,
            "drift_trajectory": [],  # filled by real runs; empty in stub
        }

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        return_trajectory: bool = False,
    ) -> tuple[Any, Any] | dict[str, Any]:
        if self.has_real_model:
            return self._forward_real(batch, return_trajectory=return_trajectory)
        return _forward_impl(self, batch, return_trajectory=return_trajectory, steps=self.T)

    def _forward_real(
        self,
        batch: Mapping[str, Any],
        *,
        return_trajectory: bool,
    ) -> tuple[Any, Any]:
        """Run attached torch GeometricRecurrent; trajectory = list of z_t."""
        from reachability_gen.tokenize import batch_encode, build_vocab

        if self.vocab is None:
            self.vocab = build_vocab()
        encodings = batch.get("encoding")
        if encodings is None:
            encodings = [""]
        if isinstance(encodings, str):
            encodings = [encodings]
        ids, mask = batch_encode(list(encodings), self.vocab)
        token_ids = torch.tensor(ids, dtype=torch.long)
        attention_mask = torch.tensor(mask, dtype=torch.long)
        self.model.eval()
        with torch.no_grad():
            logits_t, traj = self.model(
                token_ids, attention_mask, return_trajectory=return_trajectory
            )
        logits_list = logits_t.detach().cpu().tolist()
        if return_trajectory:
            traj_out: list = list(traj) if traj is not None else []
            return logits_list, traj_out
        return logits_list, None


@dataclass
class EuclideanLoopArm:
    """Weight-tied Euclidean loop control — no phase regularizers / tau.

    When ``model`` is a real torch ``EuclideanLoop`` (or torch is available and
    ``attach_default_model()`` has been called), ``forward`` runs the real
    network and can return latent trajectory tensors for drift telemetry.
    Otherwise falls back to the placeholder stub.
    """

    T: int
    d: int
    config: SharedArmConfig = field(default_factory=SharedArmConfig)
    mlp_expansion: int = 4
    model: Any = field(default=None, repr=False, compare=False)
    vocab: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.T < 0:
            raise ValueError(f"T must be non-negative, got {self.T}")
        if self.d < 1:
            raise ValueError(f"d must be >= 1, got {self.d}")

    @property
    def name(self) -> str:
        return f"loop-T{self.T}-d{self.d}"

    @property
    def has_real_model(self) -> bool:
        return self.model is not None and HAS_TORCH

    def attach_default_model(
        self,
        *,
        vocab_size: Optional[int] = None,
        n_heads: int = 4,
        max_len: int = 256,
        seed: Optional[int] = None,
    ) -> Any:
        """Build and attach a real ``EuclideanLoop`` if torch is available."""
        if not HAS_TORCH:
            return None
        from reachability_gen.models.euclidean_loop import EuclideanLoop
        from reachability_gen.tokenize import build_vocab

        if self.vocab is None:
            self.vocab = build_vocab()
        vs = int(vocab_size) if vocab_size is not None else len(self.vocab)
        if seed is not None:
            torch.manual_seed(int(seed))
        heads = n_heads if self.d % n_heads == 0 else 1
        self.model = EuclideanLoop(
            vocab_size=vs,
            d=self.d,
            T=max(self.T, 1),
            n_heads=heads,
            max_len=max_len,
            mlp_expansion=self.mlp_expansion,
            pad_id=int(self.vocab.pad_id),
        )
        return self.model

    def param_count(self) -> int:
        return params_per_block(self.d, mlp_expansion=self.mlp_expansion)

    def inference_flops(self, context_len: int, **kwargs: Any) -> FlopReport:
        T = int(kwargs.get("T", self.T))
        return flops_euclidean_loop(
            context_len,
            self.d,
            T,
            params_estimate=self.param_count(),
        )

    def forward_stub(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        n = _batch_size(batch)
        return {
            "logits": [0.0] * n,
            "arm": self.name,
            "T": self.T,
            "placeholder": True,
            "drift_trajectory": [],
        }

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        return_trajectory: bool = False,
    ) -> tuple[Any, Any] | dict[str, Any]:
        if self.has_real_model:
            return self._forward_real(batch, return_trajectory=return_trajectory)
        return _forward_impl(self, batch, return_trajectory=return_trajectory, steps=self.T)

    def _forward_real(
        self,
        batch: Mapping[str, Any],
        *,
        return_trajectory: bool,
    ) -> tuple[Any, Any]:
        """Run attached torch EuclideanLoop; trajectory = list of z_t."""
        from reachability_gen.tokenize import batch_encode, build_vocab

        if self.vocab is None:
            self.vocab = build_vocab()
        encodings = batch.get("encoding")
        if encodings is None:
            encodings = [""]
        if isinstance(encodings, str):
            encodings = [encodings]
        ids, mask = batch_encode(list(encodings), self.vocab)
        token_ids = torch.tensor(ids, dtype=torch.long)
        attention_mask = torch.tensor(mask, dtype=torch.long)
        self.model.eval()
        with torch.no_grad():
            logits_t, traj = self.model(
                token_ids, attention_mask, return_trajectory=return_trajectory
            )
        logits_list = logits_t.detach().cpu().tolist()
        if return_trajectory:
            traj_out: list = list(traj) if traj is not None else []
            return logits_list, traj_out
        return logits_list, None


@dataclass
class ChainOfThoughtArm:
    """Token-level CoT control: FLOPs debit realized K_used, not cap K."""

    K_cap: int
    d: int
    L: int = 1
    config: SharedArmConfig = field(default_factory=SharedArmConfig)
    mlp_expansion: int = 4
    # Default realized tokens for stub demos; override in inference_flops.
    K_used: int = 0

    def __post_init__(self) -> None:
        if self.K_cap < 0:
            raise ValueError(f"K_cap must be non-negative, got {self.K_cap}")
        if self.d < 1:
            raise ValueError(f"d must be >= 1, got {self.d}")
        if self.L < 1:
            raise ValueError(f"L must be >= 1, got {self.L}")
        if self.K_used < 0:
            raise ValueError(f"K_used must be non-negative, got {self.K_used}")
        if self.K_used > self.K_cap:
            raise ValueError(
                f"K_used={self.K_used} exceeds K_cap={self.K_cap}"
            )

    @property
    def name(self) -> str:
        return f"cot-Kcap{self.K_cap}-d{self.d}-L{self.L}"

    def param_count(self) -> int:
        # CoT may differ from Geo/FF/Loop; still report schematic block params.
        return self.L * params_per_block(self.d, mlp_expansion=self.mlp_expansion)

    def inference_flops(self, context_len: int, **kwargs: Any) -> FlopReport:
        K_used = int(kwargs.get("K_used", self.K_used))
        if K_used > self.K_cap:
            raise ValueError(
                f"K_used={K_used} exceeds K_cap={self.K_cap}"
            )
        return flops_cot(
            context_len,
            self.d,
            K_used,
            L=self.L,
            K_cap=self.K_cap,
            params_estimate=self.param_count(),
        )

    def forward_stub(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        n = _batch_size(batch)
        return {
            "logits": [0.0] * n,
            "arm": self.name,
            "K_cap": self.K_cap,
            "K_used": self.K_used,
            "placeholder": True,
        }

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        return_trajectory: bool = False,
    ) -> tuple[list[float], list[dict[str, Any]]] | dict[str, Any]:
        return _forward_impl(
            self, batch, return_trajectory=return_trajectory, steps=self.K_used
        )


def _batch_size(batch: Mapping[str, Any]) -> int:
    if "batch_size" in batch:
        return int(batch["batch_size"])
    for key in ("y", "s", "t", "encoding"):
        if key in batch and hasattr(batch[key], "__len__"):
            return len(batch[key])  # type: ignore[arg-type]
    return 1


def _stub_trajectory(arm_name: str, steps: int) -> list[dict[str, Any]]:
    """Placeholder latent trajectory for harness schema (no real dynamics)."""
    return [{"t": i, "arm": arm_name, "z_norm": 0.0, "placeholder": True} for i in range(max(steps, 0))]


def _forward_impl(
    arm: Any,
    batch: Mapping[str, Any],
    *,
    return_trajectory: bool,
    steps: int,
) -> tuple[list[float], list[dict[str, Any]]] | dict[str, Any]:
    """Shared stub forward used when torch is absent.

    When ``return_trajectory`` is True, returns ``(logits, trajectory)``.
    Otherwise returns the legacy dict from ``forward_stub``.
    """
    base = arm.forward_stub(batch)
    logits = list(base["logits"])
    if return_trajectory:
        traj = _stub_trajectory(arm.name, steps)
        return logits, traj
    return base
