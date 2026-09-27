#!/usr/bin/env bash
# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

# CI-ish: install torch CPU if needed, run pytest (torch tests conditional),
# then the balanced overfit gate. RESEARCH / MEASURE plumbing — no science OPEN.
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

echo "== pytest =="
pytest -q --ignore=tests/test_ff_torch.py
# Torch tests (including slow overfit) when torch is importable:
pytest -q tests/test_ff_torch.py -m "not slow"
echo "== balanced overfit gate =="
python -m reachability_gen.overfit_ff \
  --balanced \
  --examples data/train_tiny.jsonl \
  --steps 100 \
  --d 64 --L 2 --lr 3e-3
echo "CI FF balanced overfit script OK"
