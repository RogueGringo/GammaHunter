# Relation to published work

This page places GammaHunter's measurements in the vocabulary of published
work on algorithmic reasoning, length generalization and shortcut learning.
It paraphrases each paper's stated findings; it adds no claim beyond the
result files cited, and every GammaHunter result keeps the scope stated in
[LIMITATIONS.md](LIMITATIONS.md).

## Measurements and the terms the field uses for them

| GammaHunter measurement | Established term | Published work (arXiv) | Where recorded |
|---|---|---|---|
| Rules that read one endpoint's reach score up to 0.78–0.86 on paired sets | Shortcut learning; degree-based heuristics for graph connectivity | Transformers trained on connectivity learn a node-degree heuristic when training graphs exceed their capacity (2510.19753); shortcuts as unintended decision rules (2004.07780) | Limitations item 10 |
| Removing one-endpoint cues exposed a direction-blind cue, removed in turn | Shortcuts come in multiples; mitigating one can amplify another | 2212.04825 | Limitations items 10–11 |
| Crossed sets: every endpoint appears once with each label | Contrast sets, minimal pairs | Evaluating local decision boundaries with contrast sets (2004.02709) | Limitations items 10–11 |
| Untrained control; a ported arm correct before training | Compiled versus learned parameters | Neural compilation writes an algorithm into weights; learned models can match accuracy yet differ in mechanism (2505.18623) | Limitations items 12–13 |
| Standard looped arm extrapolates in some seeds only | Length generalization that is not robust across seeds | Length generalization depends strongly on initialisation and data order (2402.09371) | Limitations item 9 |
| Answers degrade as the step count grows far past training | Overthinking in recurrent-depth models | Extra recurrence unlocks deeper reasoning but excessive recurrence degrades it (2604.07822); recall of the input limits overthinking in recurrent networks (2202.05826) | Limitations items 12–13 |
| Anchored variant: source re-injected every step, unreached nodes exactly zero, no node identities | Input recall; algorithmic alignment of message passing with breadth-first search | 2202.05826; 1905.13211; neural execution of graph algorithms (1910.10593) | Limitations item 13 |
| Trained from question-answer pairs only | Learning without intermediate supervision | 2306.13411 | All runners |
| Sequence arms and open LLMs at chance on cue-free sets | Transformers struggle to learn search, increasingly with graph size | 2412.04703; LLMs on graph problems in natural language (2305.10037) | Limitations items 8 and 14 |

## Direction

Read together, these results point to a division of labour that the
literature describes from both sides:

* **Processor.** A looped, algorithmically aligned processor that recalls its
  input at every step learned reachability reliably from question-answer
  pairs alone and kept its answers far beyond its training depth and graph
  size (limitations item 13). In our measurements it showed neither the
  overthinking nor the seed-to-seed fragility that the papers above describe
  for looped and length-generalizing transformers on their tasks.
* **Reader.** The step no model here performs reliably is reading the graph:
  sequence arms trained from scratch stayed at chance even on the paired
  sets, and open LLMs of up to 3.8 billion parameters stayed at chance on the
  crossed sets, in line with the search and heuristics findings above.

The next measurement follows from that split: whether a model that sees the
edge list only as tokens can learn to supply the processor's input, with the
anchored processor behind it, trained end to end from question-answer pairs,
and evaluated on the crossed sets with the same controls (untrained model,
cue ceilings, several seeds, re-scored checkpoints). Its outcome would say
whether depth- and size-robust search can be obtained from text, not only
from given structure.
