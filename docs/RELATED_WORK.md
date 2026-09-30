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
| Sequence arms and open LLMs (to 7.6 billion parameters) at chance on cue-free sets | Transformers struggle to learn search, increasingly with graph size | 2412.04703; LLMs on graph problems in natural language (2305.10037) | Limitations items 8 and 14 |
| The same LLMs read single edges of those graphs almost perfectly (AUROC up to 0.99, from 3 to 7.6 billion parameters alike) | Compositionality gap: sub-questions answered, their composition not; in the cited work the gap did not narrow as models grew | 2210.03350 | Limitations item 14 |
| A question-blind reader trained on the edges supplies the graph from text; the anchored arm then keeps every answer up to 192 steps | Encode–process–decode around an algorithmic processor; here the encoder must emit the structure itself | Neural algorithmic reasoning (2105.02761); TransNAR (2406.09308) gives its processor the graph in structured form and lets a transformer read the text alongside it | Limitations item 15 |
| Trained from the answers alone, the reader ends with an empty or a complete graph; no gradient reaches edges into nodes the source has not reached | Learning a discrete graph from a downstream objective | Interaction graphs inferred as the latent code of a variational autoencoder (1802.04687); graph structure and network learned by bilevel optimisation (1903.11960); straight-through estimation (1308.3432) | Limitations item 15 |
| Trained from the answers alone with an unbiased score-function estimator (sampled graphs, leave-one-out baselines), the reader's graph turned deterministic, empty or complete, within an epoch at the study's rate, and stayed nearly empty at a tenth of it | Score-function (REINFORCE) gradients for discrete latent structure; variance reduction with several samples per input and leave-one-out baselines | REINFORCE (Williams, Machine Learning 8, 1992; not on arXiv); Monte Carlo gradient estimation surveyed (1906.10652); leave-one-out baselines from several samples per input (Kool, van Hoof and Welling, ICLR 2019 workshop; OpenReview r1lgTGL5DE); REINFORCE-style optimisation revisited for language models (2402.14740) | Limitations item 18 |
| Reading the graph from sentences worded unlike any in training: a word reader trained from scratch read only the held-out wordings that keep the training word order; frozen features of a small language model read more of them (edge F1 up to 0.91) but at most 2.3% of the graphs exactly; language models listing successors read the sentences worse than the edge list, and most often read backwards the sentence that names the target first | Sensitivity of graph reasoning to how the graph is written as text; probing frozen representations, with controls for what the probe itself learns | How a graph is encoded as text changes LLM graph-reasoning accuracy (2310.04560); linear probes (1610.01644); control tasks for probes (1909.03368) | Limitations item 19 |
| Trained on 51 wordings instead of five, the reader on frozen language-model features read held-out sentences whose constructions occur in training almost completely and a construction absent from training only partly; no reader read more than 3.1% of the graphs exactly | Lexical versus structural generalisation (novel combinations of a familiar primitive and a familiar structure, against novel combinations of familiar syntactic structures); broadening the training data | COGS separates the two and finds structural generalisation harder (2010.05465); systematic generalisation beyond small differences between training and test (SCAN, 1711.00350); training data broadened by recombining fragments (1904.09545) | Limitations item 20 |

## Direction

Read together, these results point to a division of labour that the
literature describes from both sides:

* **Processor.** A looped, algorithmically aligned processor that recalls its
  input at every step learned reachability reliably from question-answer
  pairs alone and kept its answers far beyond its training depth and graph
  size (limitations item 13). In our measurements it showed neither the
  overthinking nor the seed-to-seed fragility that the papers above describe
  for looped and length-generalizing transformers on their tasks.
* **Text.** Every model that reads the graph as text stayed at chance:
  sequence arms trained from scratch even on the paired sets, and open LLMs
  of up to 7.6 billion parameters on the crossed sets, in line with the
  search and heuristics findings above. An edge-lookup probe separates
  reading from search: the LLMs of 3 to 7.6 billion parameters read single
  edges of the same graphs almost perfectly (AUROC 0.89–0.99), so their
  failure is in the search, and where their answers carry a signal, it
  follows a one-endpoint cue.

A reader placed in front of the anchored processor tests this division of
labour directly (limitations item 15). It emits an explicit graph without
seeing the question, so it cannot write the answer into the graph, and it
embeds no node identities. Trained on the edges, it read every graph exactly,
including edge lists longer than any in training, and the pipeline kept every
answer up to 192 steps: on this task, depth- and size-robust search was
obtained from text once the reading was supervised. Trained from the answers
alone, the same pipeline stayed at chance. The exact zeros that keep the
processor's answers stable at any depth also leave the answers loss with no
gradient on edges into the part of the graph the source does not reach.

Two follow-up questions were then measured (limitations item 16). A soft
graph during training, a prior on edge density, or both, did not teach the
reader from answers alone: with the density held near the truth and the
gradient alive, the answers still did not single out which edges exist. And
the language models' edge reading did not carry over from checking one edge to
listing a node's edges: asked for every node's successors, they read at most
1% of the graphs exactly (edge F1 0.53–0.77), and the exact processor searching
their graphs scored at most 0.58 (their own step-by-step answers: 0.48–0.53). Among
the routes measured, only the reader trained on edges supplied graphs exact
enough for the processor's search.

The exactness frontier (limitations item 17) says why the bar is so high and
what answers cannot supply. Errors compound along a path: over paths of 8–16
edges, missing 1–2% of the edges already costs 6–11 points of accuracy,
exactly as a single-path model predicts, so a reader must be nearly exact.
About a quarter of the listed edges are implied by other paths and cannot be
revealed by any answer. And even with every answer given and the zero-gradient
kink removed, answers alone did not teach the reader the closure here.

Two further studies (limitations items 18–19) measured the routes still
open. An unbiased score-function estimator, which never differentiates
through the solver, did not teach the reader from answers either: at the rate
of the earlier studies the reader's graph turned deterministic within an
epoch, which leaves the estimator no gradient, and at a tenth of it the
reader stayed nearly empty. And no reader read graphs worded unlike its
training sentences exactly enough for the processor: words learned from
scratch carried only the wordings that keep the training word order, frozen
features of a small language model carried more but read at most 2.3% of the
graphs exactly, and language models read the sentences worse than they had
read the edge list. On this task the processor searches exactly whatever
graph it is given, and the reading came close to exact only where it was
supervised on the same wording. Hybrid designs such as TransNAR, which hand
the processor the graph in structured form, leave out the step these studies
found hardest.

Training on 51 wordings instead of five (limitations item 20) narrowed that
gap without closing it: the reader on a small language model's frozen
features then read new words almost completely in constructions it had seen,
but one construction it had not seen left only 0.4–3.1% of the graphs read
exactly, a pattern consistent with, though not shown to be, a failure of
structural rather than lexical generalisation in the terms of COGS.
