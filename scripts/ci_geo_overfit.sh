#!/usr/bin/env bash
# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

# CI-ish: install torch CPU if needed, run pytest geo (non-slow), then balanced
# geo overfit gate. RESEARCH / MEASURE plumbing — no science OPEN.
# Does NOT run 2000-scale geo train.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

python - <<'PY' || pip install -q 'torch' --index-url https://download.pytorch.org/whl/cpu
import torch
print("torch", torch.__version__)
PY

pip install -q -e ".[dev]" 2>/dev/null || pip install -q -e .

echo "== pytest (schema + geo non-slow) =="
pytest -q --ignore=tests/test_ff_torch.py --ignore=tests/test_geo_torch.py
pytest -q tests/test_geo_torch.py -m "not slow"
echo "== balanced geo overfit gate (T=6) =="
python -m reachability_gen.overfit_geo \
  --balanced \
  --examples data/train_tiny.jsonl \
  --steps 100 \
  --d 64 --T 6 --lr 3e-3
echo "CI geo balanced overfit script OK"
