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
