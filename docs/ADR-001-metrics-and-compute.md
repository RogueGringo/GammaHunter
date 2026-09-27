# ADR-001: Metrics and compute accounting

**Status:** Accepted for scaffold  
**Date:** 2026-09-26  
**Mode:** RESEARCH / MEASURE plumbing — **not** science OPEN  
**Immutability:** Values below are runtime assertions via
`reachability_gen.adr_invariants`. Changing them requires a new ADR revision.

---

## Context

The geometric-recurrence reachability testbed needs a frozen contract for
(1) input encoding, (2) schematic FLOP formulas, (3) parameter-parity gates,
(4) hop stratification for ID vs OOD eval, and (5) the per-example logging
schema written by the train/eval harness. Without this freeze, FLOP tables and
JSONL metrics drift across arms and cannot support fair MEASURE runs.

## Decision

### 1. Encoding (locked)

Canonical **sorted edge-list + QUERY**, as implemented in
`src/reachability_gen/encode.py`:

```text
N <n> EDGES <u>,<v> ... QUERY <s> <t>
```

No scratchpad / CoT tokens in Geo / FF / Loop encodings. Arms must not alter
this string scheme.

### 2. Block FLOPs

With assumptions documented in `flops.py` (attention + MLP expansion 4,
MAC = 2 FLOPs, no bias / LayerNorm / softmax FLOPs):

\[
F_{\mathrm{block}}(m,d) = 24\, m\, d^{2} + 4\, m^{2}\, d
\]

Runtime checksum: `adr_invariants.f_block_checksum` must equal `flops.F_block`.

### 3. CoT KV-cache decode schematic

\[
\mathrm{FLOPs}_{\mathrm{CoT}} =
  F_{\mathrm{prefill}}(m)
  + \sum_{k=1}^{K_{\mathrm{used}}} F_{\mathrm{decode}}(m+k)
\]

FLOPs **must** use realized `K_used`, never the cap `K_cap`. Cap is metadata
only.

### 4. Parameter parity

Non-embedding params among **Geo / FF / Loop** must match within
**±5%** (`PARAM_TOL = 0.05`). CoT may differ but must still report FLOPs.
`science_open` on match results is always `False` — matching is hygiene, not
an experimental claim.

### 5. Hop stratification

Shortest-path hop distance \(K\) on the directed graph:

| Regime | Hop distance \(K\) |
|--------|--------------------|
| ID     | \(K \in [2, 6]\)   |
| OOD    | \(K \in \{8, 12, 16\}\) |

`is_ood = (K > K_train_max)` with `K_train_max = 6`.

For **unreachable** pairs (`y = 0`): write `hop_distance = -1` in JSONL
(sentinel). Consumers may treat `-1` or JSON `null` as “undefined hop”; the
generator uses `-1` for type stability. Unreachable rows have `is_ood = false`
(hop-OOD is only defined for reachable \(K \ge 0\)).

Positive (`y = 1`) sampling prefers stratified hop buckets toward the ID/OOD
ranges above when the pool permits; otherwise falls back to any reachable pair.

### 6. Perturbation diagnostic

Default noise scale for `perturbation_delta`: `EPS_SIGMA = 1e-4`.

### 7. `RunMetricRecord` fields

TypedDict in `reachability_gen.schema.RunMetricRecord`. **Required keys** =
user contract exactly (also `REQUIRED_METRIC_KEYS` /
`validate_metric_record()` in `adr_invariants.py`):

| Field | Type | Notes |
|-------|------|-------|
| `run_id` | str | run identifier |
| `seed` | int | |
| `arm` | str | |
| `step` | int | global step within the run |
| `epoch` | int | |
| `param_count` | int | non-embedding estimate (was `params`) |
| `d_model` | int | hidden / embedding width |
| `seq_len` | int | context length `m` |
| `cycles_or_depth` | int | T loops (Geo/Loop) / L layers (FF/CoT) |
| `tokens_decoded` | int \| null | `K_used` for CoT; `null` for FF/Geo/Loop |
| `cumulative_flops` | float | schematic FLOPs (was `flops`) |
| `hop_distance` | int | `-1` if unreachable |
| `is_ood` | bool | hop OOD vs `K_TRAIN_MAX` |
| `loss` | float | real CE for torch FF when attached; stub `0.0` otherwise |
| `accuracy` | float | real 0/1 for torch FF; stub sign(logit) otherwise |
| `drift_trajectory` | list[float] | δ_t = ||z_{t+1}-z_t||_2 (was `z_drift_series`); **empty `[]` for FF by design** |
| `terminal_drift` | float \| null | last δ_t, or null if empty; **`null` for FF by design** |
| `perturbation_delta` | float \| null | uses `EPS_SIGMA` for recurrent stubs; **`null` for FF by design** |

#### Field migration (scaffold → user TypedDict)

| Current (legacy) | Target | Type | Notes |
|------------------|--------|------|-------|
| `z_drift_series` | `drift_trajectory` | `List[float]` | δ_t = ||z_{t+1}-z_t||_2 |
| `T_or_L_or_K` | `cycles_or_depth` | `int` | T loops / L FF; CoT uses L here |
| *(add)* | `tokens_decoded` | `Optional[int]` | `K_used` for CoT; None for FF/Geo/Loop |
| `flops` | `cumulative_flops` | `float` | |
| *(add)* | `run_id` | `str` | |
| `seed` | `seed` | `int` | keep |
| *(add)* | `epoch` | `int` | |
| *(add)* | `step` | `int` | |
| *(add)* | `loss` | `float` | |
| *(add)* | `accuracy` | `float` | |
| `params` | `param_count` | `int` | rename |
| *(add)* | `d_model` | `int` | |
| *(add)* | `seq_len` | `int` | |
| *(add)* | `terminal_drift` | `Optional[float]` | |

Emitters **must not** write legacy names (`z_drift_series`, `T_or_L_or_K`,
`params`, `flops`). `validate_metric_record` rejects those keys.

#### Optional documented extras (example linkage / hygiene)

Not required by `REQUIRED_METRIC_KEYS`. Harness may still emit them for
row↔example joins:

| Field | Type | Notes |
|-------|------|-------|
| `split` | str | |
| `n` | int | |
| `p` | float | |
| `edge_hash` | str | |
| `s`, `t` | int | |
| `y` | int | 0/1 gold label |
| `science_open` | bool | if present, **must** be false |

Removed from required logging (no longer part of the user TypedDict):
`latency_ms`, `yhat`, `correct`, `trajectory`.

### 8. `science_open` never self-stamped

Harnesses, param matchers, and FLOP demos **must not** set `science_open=True`.
OPEN is a human / ADR gate outside this scaffold. Status of this ADR:
**Accepted for scaffold; MEASURE plumbing — not OPEN.**

## Consequences

- `adr_invariants.py` is the single import surface for tolerances and formulas.
- Eval JSONL writers validate required `RunMetricRecord` keys via
  `adr_invariants.validate_metric_record` / `REQUIRED_METRIC_KEYS`.
- Changing `F_block`, `PARAM_TOL`, hop bounds, or encoding requires ADR-002+.

## Feed-forward trajectories (empty by design)

The feed-forward control arm has **no recurrent latent**. Harness rows for FF
therefore always set:

- `drift_trajectory = []`
- `terminal_drift = null`
- `perturbation_delta = null`
- `tokens_decoded = null`

When torch is available, FF still emits **real** CE `loss` / `accuracy` from an
attached `FeedForward` module (`models/feedforward.py`). Loop / CoT remain
stubs; Geo may attach a real recurrent module (see below). This is MEASURE
plumbing — **not** science OPEN.


## Geometric recurrent trajectories (real when attached)

When a real torch `GeometricRecurrent` is attached (weight-tied `Phi` × `T`
cycles with optional cycle embedding `τ_t`):

- `drift_trajectory` = `[δ_0, …]` with `δ_t = mean_batch ||z_{t+1}-z_t||_2`
  (length `T-1` or `T`; from `forward(..., return_trajectory=True)`)
- `terminal_drift` = last `δ_t`
- `perturbation_delta` optional (uses `EPS_SIGMA`)
- Overfit gate (MEASURE plumbing): same balanced 16+16 batch as FF, plus
  finite / non-zero drift check. Fixed `T=6` (max ID hop). **Not** science OPEN.

Stub Geo (no torch / not attached) still emits placeholder zero drifts via
`measure_phase_diagnostics`.

## Non-goals


No accuracy claims, no Lyapunov / drift plots as pass stamps, no training
loops that declare science OPEN.
