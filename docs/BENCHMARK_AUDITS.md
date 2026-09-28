# Benchmark audits

The integrity controls applied to GammaHunter's own datasets are applied to
public benchmarks before those are used for comparison. Findings describe each
benchmark as packaged at the stated source; they are not claims about any
model evaluated on it.

## NLGraph: connectivity task

Source: `tasksource/nlgraph` on Hugging Face (Wang et al. 2023, "Can Language
Models Solve Graph Problems in Natural Language?", arXiv 2305.10037). The
dataset page states no licence; files are downloaded on demand and not
redistributed. Reproduce with `python -m reachability_gen.benchmarks.nlgraph
--download`; results are in `artifacts/nlgraph_connectivity_audit.json`.

| Measure (test split) | Value |
|---|---|
| Questions (distinct graphs) | 371 (188) |
| Share answered "yes" | 0.542 |
| Rule: "yes" iff the two nodes share an edge | 0.833 (easy 0.964, medium 0.842, hard 0.756) |
| Rule: "no" iff either node has no edge | 0.642 |
| Rule: graph-density threshold, ignoring the question | 0.464 |
| Reachable pairs by path length | 1 hop: 139, 2: 49, 3: 11, 4: 2 |
| Test graphs that also appear in the train split | 188 of 188 |

About two thirds of the reachable pairs are joined by a single edge, so a rule
that only checks for that edge scores 0.833 without searching any path.
Accuracy on this task mostly reflects one-step lookups and is best compared
with that rule rather than with 0.5. Every test graph also appears in the
train split, with different questions; that matters for models trained on the
split, less for the prompting setting the benchmark was designed for.
