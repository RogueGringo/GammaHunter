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
