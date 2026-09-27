# Usage

All commands run from the repository root. Instance files (`data/*.jsonl`)
are not versioned. The generators are seeded; the three fixed-set generators
also write a generation report to `artifacts/`.

## Install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

Python 3.10 or later. A CPU build of PyTorch is sufficient for every command
below. On Windows, run Python in UTF-8 mode (`python -X utf8 ...` or
`PYTHONUTF8=1`): some help and log text uses non-ASCII symbols that the
default console code page cannot print.

## Tests

```bash
pytest -q -m "not slow"     # fast suite; tests that need local data skip without it
pytest -q                   # full suite, including longer training checks
```

## Instance generation

| Command | Output |
|---|---|
| `python -m reachability_gen.generate --all-splits --n-per-cell 4` | Base train / val / test / size-OOD splits in `data/` |
| `python -m reachability_gen.gen_id_2k` | Fixed 2,000-instance in-distribution set, `data/id_2k.jsonl` |
| `python -m reachability_gen.gen_ood_hops` | Extended-step evaluation set, `data/ood_hops.jsonl` |
| `python -m reachability_gen.gen_covariate_matched_ood` | Length-matched extended-step set, `data/covariate_matched_ood.jsonl` |

Each fixed set's generator accepts `--verify-only <path>` to check an existing
file against its specification.

## Sanity gates

| Command | Purpose |
|---|---|
| `python -m reachability_gen.overfit_ff --balanced` | Baseline arm must fit a small balanced batch |
| `python -m reachability_gen.overfit_geo --balanced` | Recurrent arm must fit the same batch, with finite per-step telemetry |
| `bash scripts/ci_ff_overfit.sh`, `bash scripts/ci_geo_overfit.sh` | The same gates as CI scripts |

## Early baseline run

```bash
python -m reachability_gen.train_ff_id
```

Baseline-only training on generated base-split examples
(`data/train_tiny.jsonl`); writes `artifacts/ff_id_train_summary.json` (see
[LIMITATIONS.md](LIMITATIONS.md), item 7).

## Comparison runs

Runs on `data/id_2k.jsonl`. Each writes a result file to `artifacts/`.

| Command | Result file | Scope |
|---|---|---|
| `python -m reachability_gen.run_id_2k_comparison --skip-gen` | `artifacts/id_2k_comparison.json` | Baseline and one recurrent arm, not parameter-matched |
| `python -m reachability_gen.run_id_2k_rematch` | `artifacts/id_2k_rematch.json` | Three arms under the parity gate |
| `python -m reachability_gen.run_id_2k_rematch_fixed30` | `artifacts/id_2k_rematch_fixed30.json` | Three arms, fixed budget; saves best checkpoints |
| `python -m reachability_gen.run_id_2k_rematch_bound30` | `artifacts/id_2k_rematch_bound30.json` | Three arms, fixed budget; saves best checkpoints |

Without `--skip-gen`, the first command regenerates `data/id_2k.jsonl` and
overwrites `artifacts/id_2k_generation_report.json`; the other runners never
regenerate the dataset. The runs are successive protocol revisions, and the
module docstrings record what each revision fixes. Only the two fixed-budget
runs save checkpoints, so only they can be audited. The runners size their
context from the first rows of the dataset and warn once if any input is
truncated (see [LIMITATIONS.md](LIMITATIONS.md)). New runners should size
inputs with `reachability_gen.tokenize.required_max_len` and pass
`on_overflow="error"`.

## Extended-step evaluation

```bash
python -m reachability_gen.run_ood_gate2               # extended-step set
python -m reachability_gen.run_covariate_matched_ood   # length-matched set
```

Inference only, on the saved best checkpoints of the latest comparison run.
They write `artifacts/id_2k_rematch_bound30_gate2_ood.json` and
`artifacts/id_2k_rematch_bound30_gate2_matched_ood.json`.

## Checkpoint audit

```bash
python -m reachability_gen.audit_checkpoints
```

Re-scores every saved best checkpoint on the exact validation split and exits
with status 1 if any re-score differs from the recorded accuracy. Writes
`artifacts/id_2k_checkpoint_audit.json`, containing per-step accuracy (all
instances and answerable instances), telemetry, breakdown flags, dataset
composition, input-truncation accounting and, for each extended-step set
present in `data/`, evaluation-set coverage.

## Utilities

| Command | Purpose |
|---|---|
| `python -m reachability_gen.flops_demo` | Print the schematic compute table for one context length |
| `python -m reachability_gen.eval_demo` | Write sample metric records in the ADR-001 schema |

## Output conventions

- Result files never set `science_open` to true; no component marks a result
  as an established finding.
- Per-instance metric records follow the `RunMetricRecord` schema in ADR-001
  §7.
