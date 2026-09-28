# Usage

All commands run from the repository root. Instance files (`data/*.jsonl`)
are not versioned. The generators are seeded; the fixed-set generators also
write a generation report to `artifacts/`.

## Install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

Python 3.10 or later. A CPU build of PyTorch is sufficient for every command
below and is the reference for the committed results. For an NVIDIA GPU,
install a CUDA build instead, in a separate environment (the CUDA 13.0 build
includes kernels for compute capability 7.5 through 12.0):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

On Windows, run Python in UTF-8 mode (`python -X utf8 ...` or
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
| `python -m reachability_gen.gen_id_disjoint` | Graph-disjoint, label-paired 2,000-instance in-distribution set, `data/id_disjoint_2k.jsonl` (`--n-total` / `--n-val` scale it; `--spec extended` builds the long-path evaluation set) |
| `python -m reachability_gen.gen_crossed` | Crossed 20,000-instance in-distribution set, `data/id_crossed_20k.jsonl`: two reachable queries and the two queries crossing them per graph, so every endpoint appears once with each label (`--spec extended` builds the long-path version) |
| `python -m reachability_gen.gen_ood_hops` | Extended-step evaluation set, `data/ood_hops.jsonl` |
| `python -m reachability_gen.gen_covariate_matched_ood` | Length-matched extended-step set, `data/covariate_matched_ood.jsonl` |

Each fixed set's generator accepts `--verify-only <path>` to check an existing
file against its specification. The crossed long-path set:

```bash
python -m reachability_gen.gen_crossed --spec extended --n-total 2000 --n-val 2000 --seed 191000 --out data/extended_crossed_2k.jsonl --report artifacts/extended_crossed_2k_generation_report.json
```

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

## Multi-seed run on the graph-disjoint set

```bash
python -m reachability_gen.run_disjoint_rematch            # seeds 0 1 2, 30 epochs
python -m reachability_gen.run_disjoint_rematch --seeds 0 --epochs 2   # smoke check
python -m reachability_gen.run_disjoint_rematch --device cuda           # on a GPU
```

`--device cuda` trains on the GPU from the same initial weights and data
order as on the CPU; GPU numerics differ slightly, so the device is recorded
in the result file.

The 20,000-instance version (10,000 graphs, independent seed) and its run:

```bash
python -m reachability_gen.gen_id_disjoint --n-total 20000 --n-val 4000 --seed 170000 --out data/id_disjoint_20k.jsonl --report artifacts/id_disjoint_20k_generation_report.json
python -m reachability_gen.run_disjoint_rematch --data data/id_disjoint_20k.jsonl --device cuda --out artifacts/id_disjoint_20k_rematch.json --ckpt-dir artifacts/id_disjoint_20k_rematch
```

Per-example telemetry is computed on the first 400 validation rows, so it is
comparable across dataset sizes.

Two options change how the arms learn, not what they are compared on:
`--readout query` feeds the classifier the final states at the query's two
nodes instead of the mean over all tokens, and `--curriculum` trains on graphs
whose reachable pair is at most 2 hops first, adding one hop per stage.
Validation always covers every hop and also reports accuracy grouped by each
graph's hop, where chance is exactly 0.5.

Trains the three arms of the latest comparison protocol on
`data/id_disjoint_2k.jsonl` for each seed. Inputs are sized so that nothing is
truncated, validation is logged every epoch with breakdown diagnostics, and
both the best and the final checkpoint are saved to
`artifacts/id_disjoint_rematch/` and re-scored before the result file
`artifacts/id_disjoint_rematch.json` is written (exit status 1 if a re-score
differs).

## Calibration: message passing, trained short, tested long

```bash
python -m reachability_gen.gen_id_disjoint --spec extended --n-total 2000 --n-val 2000 --seed 190000 --out data/extended_disjoint_2k.jsonl --report artifacts/extended_disjoint_2k_generation_report.json
python -m reachability_gen.run_mp_calibration --device cuda
```

The first command builds an evaluation-only set with the same construction as
the graph-disjoint sets but longer paths (8–16 hops) on larger graphs (24–48
nodes). The second trains two parameter-matched message-passing arms on
`data/id_disjoint_20k.jsonl` with 6 steps: an unlooped one with 6 distinct
layers and a looped one that reuses a single step. It then evaluates both on
the long-path set: the unlooped arm at its fixed depth, the looped arm at 6,
16, 32 and 48 steps. It is the harness's positive control for a pattern
reported in the literature (how its results vary by seed is described in
[LIMITATIONS.md](LIMITATIONS.md), item 9), and it writes
`artifacts/mp_calibration.json` with self-audited best and final checkpoints
in `artifacts/mp_calibration/`.

```bash
python -m reachability_gen.untrained_control --device cuda
```

The untrained-model control rebuilds, for each seed and arm, the exact
weights its training started from and scores them with the calibration's own
evaluation, on the same sets and step counts: an accuracy reached before any
training cannot be attributed to learning. It writes
`artifacts/mp_calibration_untrained_control.json`, including the learned
margin (trained minus untrained accuracy) for both saved checkpoints when the
calibration's result file is present.

## Stability ring

```bash
python -m reachability_gen.run_stability --device cuda
```

Trains nine parameter-matched looped arms on `data/id_crossed_20k.jsonl`
(5 seeds, 30 epochs, one optimiser setting for all). The arms are the
standard looped step, the recurrence under study in the message-passing frame
(with and without per-step vectors), both trained with 6 steps or with a
random step count per batch, and a ported arm that infers the graph from the
edge list, with default and with its original initialisation. Each arm is
scored untrained and at its best and final checkpoints (saved and re-scored)
on the validation split and on `data/extended_crossed_2k.jsonl` at 6 to 192
steps, with the relative change of the query target's state per step; a seed
counts as stable when its long-path accuracy is at least 0.99 at every step
count from 16 to 192. It writes `artifacts/stability_ring.json` and
checkpoints in `artifacts/stability_ring/`. Seeds can run as separate
processes and be merged:

```bash
python -m reachability_gen.run_stability --device cuda --seeds 0 --out artifacts/stability_ring_seed0.json
python -m reachability_gen.run_stability --merge artifacts/stability_ring_seed0.json artifacts/stability_ring_seed1.json
```

## Take-off study

```bash
python -m reachability_gen.run_takeoff --device cuda
```

Measures how reliably three parameter-matched looped arms start learning on
`data/id_crossed_20k.jsonl`: the standard step, the recurrence under study,
and an anchored variant that combines features of both research lines (nodes
the source has not reached keep an exactly zero state, the source is
re-injected at every step, the answer is read from the target's state, and no
node identities are embedded). Each arm starts cold, after one epoch on the
paired set (`data/id_disjoint_20k.jsonl`), or on a hop curriculum, for 20
seeds and 5 epochs. A run counts as taking off when its validation accuracy
reaches 0.99; rates come with Wilson 95% intervals and two-sided Fisher exact
tests, and each final checkpoint is re-scored and evaluated on
`data/extended_crossed_2k.jsonl` at 16, 48 and 192 steps. It writes
`artifacts/takeoff_study.json`; the checkpoints (`artifacts/takeoff/`) are
not versioned, and their SHA-256 hashes are recorded instead. Seeds can run
in parts and be merged with `--merge`, as for the stability ring.

## LLM reference points

```bash
pip install -e ".[llm]"                                  # adds transformers, accelerate
python -m reachability_gen.run_llm_reference --device cuda
python -m reachability_gen.run_llm_cot --device cuda
```

Both runners load open language models only from the local Hugging Face cache
(`local_files_only`; nothing is downloaded) and run inference only. The
direct protocol asks for an immediate answer after four solved examples and
scores whichever of " Yes" and " No" the model rates higher at the first token
where they differ (also giving a threshold-free AUROC), on seeded samples of
the crossed validation and long-path sets, the paired validation split, and
the crossed questions with the edge list withheld (a model that reads the
graph cannot beat chance there). The step-by-step protocol puts each question
to instruction-tuned models through their chat template, asks them to reason
and end with "Answer: Yes" or "Answer: No", decodes greedily and records every
generation in `artifacts/llm_cot_generations.jsonl`. They write
`artifacts/llm_reference.json` and `artifacts/llm_cot.json`.

```bash
python -m reachability_gen.run_llm_edge_probe --device cuda
```

The edge-lookup probe asks the same models whether single edges are listed
(two listed edges, one reversed edge and one absent pair per graph, on the
same graphs), with the same scoring, to separate reading the edge list from
searching it. It writes `artifacts/llm_edge_probe.json`.

```bash
python -m reachability_gen.llm_cot_paths
```

The path audit reads the recorded step-by-step generations and checks every
written path (the last chain of three or more nodes joined by arrows in a
generation) against the edges shown: each step is a listed edge, a listed
edge followed backwards, or a pair that is not listed. Results are grouped by
label and answer, per model and set, in `artifacts/llm_cot_path_audit.json`.

```bash
python -m reachability_gen.llm_cue_conflict --enriched --device cuda
```

The cue-conflict test reads the paired-split margins stored in
`artifacts/llm_reference.json` and, with `--enriched` (optionally followed by
model names; by default the reference file's models), also scores every
paired graph in which the target-reach cue conflicts with the label and as
many in which it agrees. It reports how often each model ranks the reachable
question higher in each group (wins, ties and losses, Wilson intervals on
graphs without ties, and a two-sided Fisher exact test between the groups)
and writes `artifacts/llm_cue_conflict.json`.

```bash
python -m reachability_gen.run_llm_reference --device cuda --int8 --models MODEL --out EIGHT_BIT.json
python -m reachability_gen.llm_int8_fidelity --model MODEL --pair SIXTEEN_BIT.json EIGHT_BIT.json
```

With `--int8` (accepted by every runner above), each linear layer's weights
are stored in 8 bits with one scale per output channel and expanded to 16
bits at each call, so a model about twice as large fits the same GPU memory.
The fidelity gate compares one model's 16-bit and 8-bit result files question
by question (correlation and Yes/No agreement of the margins, and the AUROC
shift, against limits fixed in advance), writes
`artifacts/llm_int8_fidelity.json` and exits with status 1 if any limit is
missed. 8-bit results are used only for a model family that passed the gate
at a smaller size.

## Reader pipeline

```bash
python -m reachability_gen.run_reader --device cuda
```

Asks whether the graph the anchored arm needs can come from the edge-list
text. A reader that never sees the question and gives every node token the
same embedding turns the edge list into an explicit 0/1 adjacency, which the
anchored arm then searches. Three regimes: the reader trained on the true
edges, with a solver trained first on true graphs and then frozen; the reader
trained from the reachability answers alone, with the same frozen solver; and
reader and solver trained together from the answers alone. Each run is
scored in three layers: the reader (edge precision, recall and F1,
exact-graph rate, reversed edges, the topology loss against the true edges,
and agreement of the reachability closure with the true graph's), the
pipeline (accuracy and AUROC on the crossed validation split and on the
long-path set at 16, 48 and 192 steps), and the attribution of every wrong
answer to the reader or the solver. Controls: the untrained reader, and the
solver on the true graph. The pass criteria are fixed in the runner. It
writes `artifacts/reader_pipeline.json`; the checkpoints
(`artifacts/reader_pipeline/`) are not versioned, and their SHA-256 hashes are
recorded instead. Regimes can run as separate processes
(`--regimes ... --out PART.json`) and be merged with `--merge`.

## External benchmarks

```bash
pip install -e ".[benchmarks]"                          # adds pyarrow
python -m reachability_gen.benchmarks.nlgraph --download
```

Fetches the NLGraph dataset (`tasksource/nlgraph` on Hugging Face, about
1.4 MB) into the git-ignored `data/benchmarks/`, converts its connectivity
questions to the locked encoding (each undirected edge in both directions),
checks every label against its graph, and writes
`artifacts/nlgraph_connectivity_audit.json`: rules that need no path search
(direct edge, isolated endpoint, graph density), the hop profile of reachable
pairs, and graph reuse between its train and test splits. Benchmark files are
never committed.

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

## Reach-cue audit

```bash
python -m reachability_gen.reach_cues
```

On the validation rows of each graph-disjoint set (by default the paired and
the crossed sets), scores rules that read only one endpoint's reach: how many
nodes the source reaches, how many nodes reach the target, and how far those
sets extend, within 1 hop, within 6 hops and without limit. It also scores the
distance between the two endpoints with edge direction ignored. Each rule's
threshold is fitted on the audited rows themselves, so every score is a
ceiling for that rule. Writes `artifacts/reach_cue_audit.json`.

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
