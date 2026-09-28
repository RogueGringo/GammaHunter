# Changelog

## Unreleased

- Pull-request template recording evidence paths, test plan and
  `science_open` status for each change.
- Graph-disjoint, label-paired in-distribution set whose negatives cannot be
  decided from one endpoint.
- Multi-seed fixed-budget runner with per-epoch diagnostics, best and final
  checkpoints, and a self-audit of every saved checkpoint.
- Endpoint-rule baseline in the checkpoint audit.
- Optional CUDA device for the graph-disjoint runner; the CPU remains the
  default and the reference for committed results.
- Dataset size is a generator parameter; a 20,000-instance graph-disjoint set
  and its three-seed GPU run.
- Optional query readout (the final states at the query's two nodes) and a
  hop curriculum for the graph-disjoint runner, with validation accuracy
  stratified by each graph's hop.
- NLGraph connectivity adapter and integrity audit (external benchmark,
  downloaded on demand, never committed).
- Evaluation-only long-path set (8–16 hops on 24–48-node graphs); the dataset
  generator now takes a set specification.
- Looped and unlooped message-passing arms, parameter-matched, and a
  calibration runner that trains on short paths and tests on long ones.
- Reach-cue audit: ceilings for rules that read only one endpoint's reach,
  on the graph-disjoint sets.
- Untrained-model control for the calibration: each arm scored at the
  weights its training starts from, with the learned margin.
- Crossed reachability sets: two reachable queries and the two queries
  crossing them per graph, so no rule on a single endpoint beats 0.5; a
  20,000-instance in-distribution set and a long-path set. The reach-cue
  audit also scores the endpoints' distance with edge direction ignored.
- Stability ring: nine parameter-matched looped arms (standard step, the
  recurrence under study, per-step vectors, random step counts in training,
  and a ported graph-inference arm with default and original initialisation),
  scored untrained and at best and final checkpoints up to 192 steps; seeds
  run in parallel and merge. Five-seed results recorded (limitations item
  12).
- Anchored message-passing arm: bias-free steps, an RMS cap that leaves zero
  in place (with a finite gradient there), the source re-injected every step
  and a readout of the target's state and norm, so unreached nodes stay at
  exactly zero at any step count.
- Take-off study: take-off rates of three looped arms under three training
  starts over 20 seeds, with Wilson intervals, Fisher exact tests (Holm
  adjusted, for take-off and for holding long-path answers) and the final
  checkpoints' reach up to 192 steps. Results recorded (limitations item 13).
- LLM reference points: open models from the local cache, inference only, on
  the crossed, long-path and paired sets with a withheld-graph control; an
  immediate-answer protocol (logit scoring, AUROC) and a step-by-step protocol
  (greedy decoding, every generation recorded). Optional `llm` extra.
- Cue-conflict test: whether the LLMs' paired-split signal follows one-endpoint
  reach cues, on every cue-conflicting paired graph and as many agreeing ones.
- Related-work page mapping the measurements to published terms and papers.
- Edge-lookup probe: whether the reference LLMs read single edges (listed,
  reversed, absent) of the same graphs, separating reading from search.
- 8-bit weight storage for the reference LLMs (one scale per output channel,
  expanded in place at each call) and a fidelity gate comparing a model's
  16-bit and 8-bit margins question by question against limits fixed in
  advance; Qwen2.5-7B-Instruct added to every reference protocol through it.
  Results recorded (limitations item 14).
- Path audit of the step-by-step generations: whether each written path uses
  listed edges, follows a listed edge backwards or steps along a pair that is
  not listed, grouped by label and answer.
- Reader pipeline: a question-blind reader with no node identities turns the
  edge list into an explicit 0/1 adjacency for the anchored arm; supervised
  and answers-only regimes over 10 seeds, scored on the reader (edges,
  topology loss, closure), the pipeline (accuracy and AUROC to 192 steps) and
  the attribution of each error, with untrained and true-graph controls and a
  gradient probe. Edge evidence computed in log space. Results recorded
  (limitations item 15).
- Answers-only reader variants (soft graph in training, a prior on edge
  density, both) and language models as the reader, feeding the reader
  study's frozen solvers.

## 0.1.0 (2026-09-27)

Initial private release.

- Instance generation for in-distribution and extended-step strata,
  including a length-matched extended-step set, with seeded generation
  reports.
- Matched model arms, parameter-parity gate and compute accounting under
  ADR-001.
- Training and evaluation runners with per-instance metric logging.
- Dynamics telemetry for recurrent arms.
- Checkpoint audit, dataset-composition and coverage review, breakdown flags
  and input-length guard.
- Proprietary license and per-file notices.
