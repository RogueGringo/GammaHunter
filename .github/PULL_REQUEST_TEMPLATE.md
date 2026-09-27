## Mode

<!-- MEASURE | FREEZE | … -->

-

## Cycle name

-

## Substrate + prohibited defaults

- Substrate:
- Prohibited defaults / fail-closed bands:

## Prereg success criteria

-

## Evidence paths (artifacts)

- `artifacts/`

## science_open

- [x] **false** (default — leave checked unless a human gate explicitly flips)
- [ ] true — **human gate only**; cite ADR / seal + artifact paths below

If true, cite:

-

## STOP / OPEN-candidate / RESIDUE

<!-- Pick one primary class; cite artifacts. Do not invent OPEN. -->

- [ ] STOP
- [ ] OPEN-candidate (engineering / not science OPEN)
- [ ] RESIDUE

Notes:

-

## Test plan

```bash
pytest -m "not slow"
```

- [ ] `pytest -m "not slow"` passes
- [ ] Artifact / report keys consistent with run scripts (if touched)
- [ ] No harness self-stamps `science_open=true`
