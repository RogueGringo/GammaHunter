# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Fast tests for id_2k rematch param scaling / parity / artifact hygiene."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reachability_gen.run_id_2k_rematch import (
    FF_BASELINE_PARAMS,
    assert_param_parity,
    param_window,
    within_5pct,
)

ROOT = Path(__file__).resolve().parents[1]
REMATCH_PATH = ROOT / "artifacts" / "id_2k_rematch.json"
ID_2K_PATH = ROOT / "data" / "id_2k.jsonl"


def test_param_window_matches_user_bounds():
    lo, hi = param_window(FF_BASELINE_PARAMS)
    assert lo == 115_157
    assert hi == 127_279
    assert within_5pct(FF_BASELINE_PARAMS, FF_BASELINE_PARAMS)
    assert within_5pct(lo, FF_BASELINE_PARAMS)
    assert within_5pct(hi, FF_BASELINE_PARAMS)
    assert not within_5pct(lo - 1, FF_BASELINE_PARAMS)
    assert not within_5pct(hi + 1, FF_BASELINE_PARAMS)


def test_assert_param_parity_hard_fail():
    ok = assert_param_parity(
        ff_count=121_218, geo_count=121_794, loop_count=120_770
    )
    assert ok["within_5pct"] is True
    assert ok["science_open"] is False
    with pytest.raises(AssertionError, match="Geo params"):
        assert_param_parity(
            ff_count=121_218, geo_count=72_258, loop_count=120_770
        )
    with pytest.raises(AssertionError, match="Loop params"):
        assert_param_parity(
            ff_count=121_218, geo_count=121_794, loop_count=50_000
        )


@pytest.mark.skipif(not ID_2K_PATH.exists(), reason="id_2k.jsonl missing")
def test_scale_recurrent_lands_in_window():
    torch = pytest.importorskip("torch")
    del torch
    from reachability_gen.overfit_ff import load_jsonl
    from reachability_gen.run_id_2k_rematch import scale_recurrent_to_window
    from reachability_gen.tokenize import build_vocab
    from reachability_gen.train.ff_trainer import examples_to_batch

    rows = load_jsonl(ID_2K_PATH)
    vocab = build_vocab()
    probe = examples_to_batch(rows[:2], vocab)
    max_len = max(int(probe[0].shape[1]) + 8, 64)
    _, _, _, vocab = examples_to_batch(rows, vocab, max_len=max_len)
    lo, hi = param_window(FF_BASELINE_PARAMS)
    for use_tau in (True, False):
        cfg = scale_recurrent_to_window(
            vocab_size=len(vocab),
            max_len=max_len,
            pad_id=vocab.pad_id,
            use_tau=use_tau,
            ref_count=FF_BASELINE_PARAMS,
        )
        assert lo <= cfg.param_count <= hi
        assert within_5pct(cfg.param_count, FF_BASELINE_PARAMS)


def test_rematch_json_science_open_false_if_present():
    if not REMATCH_PATH.exists():
        pytest.skip("artifacts/id_2k_rematch.json not written yet")
    data = json.loads(REMATCH_PATH.read_text())
    assert data.get("science_open") is False

    def _walk(obj):
        if isinstance(obj, dict):
            if obj.get("science_open") is True:
                return False
            return all(_walk(v) for v in obj.values())
        if isinstance(obj, list):
            return all(_walk(v) for v in obj)
        return True

    assert _walk(data)
    pm = data["param_match"]
    assert pm["within_5pct"] is True
    assert pm["geo_within_5pct"] is True
    assert pm["loop_within_5pct"] is True
    assert pm["science_open"] is False
    for arm in ("ff", "geo", "loop"):
        assert arm in data
        by_hop = data[arm]["val_by_hop"]
        for k in ["-1", "2", "3", "4", "5", "6"]:
            assert k in by_hop, f"missing hop {k} in {arm}"
            assert "acc_mean" in by_hop[k]
            assert "loss_mean" in by_hop[k]
    for arm in ("geo", "loop"):
        h2 = data[arm]["val_by_hop"]["2"]
        assert "mean_drift_trajectory" in h2
        assert "mean_terminal_drift" in h2
        assert "mean_perturbation_delta" in h2
        assert "damp_regime" in h2 or "terminal_drift_regime" in h2
        assert "drift_summary" in data[arm]
