# Overview

## Purpose

GammaHunter measures how architectures that iterate a shared computation
compare with fixed-depth architectures on problems that require a known
number of sequential reasoning steps. It is built to produce comparisons that
are controlled for model size and compute, stratified by the number of
required steps, and checked for common measurement failures.

## Task

Each instance is a directed graph, serialized as a token sequence, together
with a query pair (s, t). The label is whether t is reachable from s. The
shortest-path length is computed exactly, so every reachable instance carries
the number of reasoning steps it requires and evaluation can be stratified by
it.
Instances are generated from fixed seeds, with class balance and hard
negatives (both endpoints connected to the graph, yet mutually unreachable).
Crossed sets go further: each graph contributes two reachable queries and the
two queries that cross them, so every endpoint appears once with each label
and only the relation between the two endpoints decides the answer.

## Components

| Component | Role |
|---|---|
| Generation | Reproducible instance sets for in-distribution and extended-step evaluation, each with a generation report |
| Encoding and tokenization | One locked serialization shared by every arm, with a guard that reports or refuses inputs that would be truncated |
| Model arms | A fixed-depth baseline and weight-tied recurrent variants behind a common interface; in the gated comparison runs, a parity gate holds recurrent arms within ±5% of the baseline's parameter count |
| Message-passing arms | Arms whose state lives on the graph's nodes, in a looped (shared step) and an unlooped (distinct layers) variant matched within ±5% on parameters; used to calibrate the harness against known results. Looped variants also carry the recurrence under study, and a ported arm infers the graph from the edge list |
| Compute accounting | Schematic inference cost per arm under the frozen specification (ADR-001) |
| Runners | Training and evaluation with set epoch budgets, per-instance metric logging in a common schema and, in the fixed-budget runs, best-checkpoint saving |
| Telemetry | Per-step measurements of recurrent state: drift between steps, state norms, token-level coherence and propagated perturbation gain |
| Audit | Re-scores saved checkpoints against recorded results; reports dataset composition, evaluation coverage and breakdown flags |

## Measurement principles

- **Fixed protocol.** Encoding, compute formulas, parity tolerance, step
  strata and the logging schema are frozen in ADR-001 and enforced at runtime.
- **Matched comparison.** The gated runs compare arms at matched parameter
  counts, with schematic compute logged alongside per-instance accuracy.
- **Stratified evaluation.** Results are reported per required-step count,
  not only in aggregate.
- **Traceable results.** A reported table should be reproducible from a saved
  checkpoint; the audit checks this.
- **Fail-closed reporting.** No component marks a result as an established
  finding; interpretation is a separate, human decision.

## Integrity controls

| Check | What it detects |
|---|---|
| Checkpoint re-score | Reported results that do not correspond to the saved model |
| Input-length guard | Instances whose query would be cut off by the model's context size |
| Dataset composition | Graph reuse across splits, measured against a query-blind baseline |
| Endpoint-cue baseline | Negatives decidable from one endpoint without any path search |
| Reach-cue ceilings | How far rules that read only one endpoint's reach (nodes the source reaches, nodes that reach the target) get without a path between the endpoints, and how far the endpoints' distance with edge direction ignored gets |
| Untrained control | Accuracy an arm already reaches at the weights its training starts from, which cannot be attributed to learning |
| Evaluation coverage | Tokens or graph sizes present in evaluation data but absent from training data |
| Breakdown flags | Representation collapse, input-independent outputs, abrupt training regressions, and final-epoch results well below the best checkpoint |

## Extending

A new architecture is added by implementing the arm interface in
`src/reachability_gen/arms.py`. The parity gate, compute accounting and
diagnostics operate on generic model properties (parameter counts, logits,
per-step states), so they can be reused. The comparison runners and the audit
currently enumerate the three sequence arms, and each needs a corresponding
entry; the message-passing arms read the parsed graph rather than tokens and
have their own calibration runner.
