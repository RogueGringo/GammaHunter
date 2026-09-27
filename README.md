# GammaHunter

**Proprietary and confidential.** Copyright (c) 2026 B.Jones. All rights
reserved. Use requires a separate signed written agreement; see
[LICENSE](LICENSE).

GammaHunter is a controlled measurement harness for iterative computation in
sequence models. It compares models that re-apply shared computation over
several steps with fixed-depth models of matched size, on a synthetic
multi-step reasoning task in which the number of reasoning steps each instance
requires is known exactly.

## Capabilities

- **Instance generation.** Labeled directed-graph reachability problems with
  exact shortest-path step counts, class balancing, hard negatives, and
  separate in-distribution and extended-step strata.
- **Matched model arms.** A fixed-depth transformer baseline and weight-tied
  recurrent variants behind a common interface, held to a parameter-parity
  gate so that model size is matched.
- **Compute accounting.** Per-arm inference cost under a frozen specification.
- **Dynamics telemetry.** Per-step state measurements for recurrent arms:
  drift between steps, state norms, representation coherence and response to
  small input perturbations.
- **Integrity controls.** Checkpoint re-scoring against recorded results,
  input-length guards, dataset-composition and coverage checks, and automatic
  flags for degenerate training outcomes.

## Why it matters

Claims about iterative reasoning in neural networks are easy to overstate.
Differences in size or compute, overlap between training and evaluation data,
or silently truncated inputs can all look like architectural effects.
GammaHunter freezes its measurement specification in advance (ADR-001) and
provides an audit that re-checks saved evidence, so that such confounds can be
surfaced before results are interpreted. The parity gate, compute accounting
and diagnostics are architecture-agnostic; adding a model family also
requires entries in the runners and the audit.

## Status

Research-stage measurement infrastructure (v0.1.0). Result files in
`artifacts/` are single-seed engineering measurements, not claims of
capability. Known constraints on interpreting them are listed in
[docs/LIMITATIONS.md](docs/LIMITATIONS.md).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pytest -q -m "not slow"
```

Full command reference: [docs/USAGE.md](docs/USAGE.md).

## Repository layout

| Path | Contents |
|---|---|
| `src/reachability_gen/` | Generation, tokenization, model arms, training and evaluation runners, audit |
| `tests/` | Unit, integration and artifact-contract tests |
| `artifacts/` | Result files and saved checkpoints |
| `docs/` | Overview, usage, limitations and the measurement specification |
| `scripts/` | CI helpers for model sanity gates |

## Documentation

- [Overview](docs/OVERVIEW.md): components and measurement principles
- [Usage](docs/USAGE.md): commands and outputs
- [Limitations](docs/LIMITATIONS.md): current constraints on interpreting results
- [ADR-001](docs/ADR-001-metrics-and-compute.md): frozen metrics and compute specification
- [Changelog](CHANGELOG.md)

## License

Proprietary. No rights are granted without a separate signed written
agreement with the copyright holder. See [LICENSE](LICENSE). Enquiries:
B.Jones@jtech.ai.
