# Limitations

Current constraints on interpreting the results in `artifacts/`. Figures come
from `artifacts/id_2k_checkpoint_audit.json` unless noted otherwise.

1. **Single seed, small strata.** Each comparison run on `id_2k` uses one
   training seed. Validation strata hold 40 instances per step count, about
   ±13 points at 95% confidence near 75% accuracy.

2. **The in-distribution set reuses graphs and carries a shortcut.** Its
   2,000 instances are drawn from 20 distinct graphs, and every validation
   graph also appears in training. A baseline that ignores the query and
   predicts each graph's majority training label scores 0.7275. In addition,
   131 of its 200 validation negatives can be decided from one endpoint (the
   source has no outgoing edge or the target no incoming edge), and a rule
   using only that cue scores 0.8275. Accuracy on this set therefore mixes
   reachability with per-graph memorization and a one-step cue. The
   graph-disjoint set (`id_disjoint_2k`) removes both: each graph appears once,
   with one reachable and one unreachable query, and no negative carries the
   endpoint cue, so both baselines score 0.5 by construction. Cues that read
   one endpoint at a time remain (item 10).

3. **Input truncation in the comparison runs.** The comparison runners size
   their context from the first rows of the dataset. 50 of the 2,000
   instances exceed that size and lost their query tokens during training and
   evaluation (8 of 400 validation instances). The input-length guard now
   reports this.

4. **Reported tables versus saved checkpoints.** The fixed-budget comparison
   result files report best accuracy from the best checkpoint but per-step
   tables from the final epoch. Use the audit, which re-scores each saved
   checkpoint, when comparing arms.

5. **Training stability of recurrent arms.** Some recurrent configurations
   degrade late in training or converge to representations that no longer
   depend on the input. The audit's breakdown flags identify these cases.

6. **Extended-step set design.** Two held-out sets exist.
   - In the first, all 240 negatives come from one 32-node graph, while 236 of
     240 positives come from larger graphs and contain tokens that never occur
     in the training data.
   - The second, length-matched set uses only tokens seen in training and
     draws each positive from its own graph. Its 240 negatives come from 3
     graphs with only two sizes (14 and 18 nodes), while 157 of 240 positives
     use other sizes, so graph size still partly separates the classes. Most
     of its graph sizes do not occur among the training graphs.

   Because the in-distribution models were trained on 20 graphs (item 2),
   results on either set also reflect generalization to unseen graphs, not
   step count alone.

7. **Earlier runs outside the audit.** `id_2k_comparison.json` compares the
   baseline with a recurrent arm that is not parameter-matched (about 0.60 of
   the baseline's parameter count, as recorded in that file). The runs behind
   it and `id_2k_rematch.json` saved no checkpoints, so the audit cannot
   re-score them. The parameter count in
   `ff_id_train_summary.json` implies a 64-token context, shorter than most of
   its inputs, so most inputs lost their query and its per-step results are
   not interpretable.

8. **No learning on the graph-disjoint sets.** Trained with the same arms and
   settings (three seeds, 30 epochs) on `id_disjoint_2k` (1,600 training
   instances, CPU) and on the ten-times-larger `id_disjoint_20k` (16,000
   training instances, GPU), no arm exceeded chance on validation or fit its
   own training set. Best validation accuracy was at most 0.5225 on the small
   set and at most 0.5025 on the large one (4,000 validation instances), and
   training accuracy never passed 0.52 (`artifacts/id_disjoint_rematch.json`,
   `artifacts/id_disjoint_20k_rematch.json`). The recurrent arm with cycle
   embeddings showed full token collapse from the first epoch in every seed.
   Accuracy on `id_2k` should therefore not be read as learned reachability.
   A query readout with a hop curriculum (one seed,
   `artifacts/id_disjoint_20k_learnability.json`) let the fixed-depth baseline
   begin fitting its training set (training accuracy 0.59) with at most a
   faint signal on 2-hop questions (0.51–0.54, where one standard error is
   0.018), which the cues in item 10 could also produce; the recurrent arms
   stayed at chance. Other architectures and much
   longer training remain untested.

9. **Calibration results depend on the seed.** In the message-passing
   calibration (`artifacts/mp_calibration.json`, three seeds, GPU), the looped
   arm reached 1.000 on the `id_disjoint_20k` validation split in every seed,
   but running it for more steps than it was trained with reached the longer
   paths only in some seeds. At the best checkpoint (epoch 1), two seeds
   scored at least 0.997 on the long-path set with 16, 32 and 48 steps; the
   third scored 0.500 at every step count above 6, on the validation split as
   well. At the final checkpoint (epoch 30), one of the two also fell to 0.500
   at 32 and 48 steps. With 6 steps, fewer than any path in the long-path set
   needs, the looped arm scored 0.500 in every seed. The unlooped arm reached
   1.000 in two seeds; in the third it peaked at 0.915 and ended at 0.720, and
   it scored 0.7355 on the long-path set although every path there is longer
   than its depth (see item 10). At the weights their training starts from,
   both arms score exactly 0.500 on both sets at every step count
   (`artifacts/mp_calibration_untrained_control.json`), so their results
   above 0.5 were learned, not built in. These arms receive each graph as explicit structure,
   whereas the sequence arms must recover it from tokens, so the calibration
   is a reference point for the harness, not a like-for-like comparison with
   the sequence arms.

10. **Cues from one endpoint's reach.** Pairing each graph's reachable and
    unreachable query defeats rules that ignore the query and rules that flag
    an endpoint without edges, but not every cue. In these sparse random
    graphs, the sources of unreachable queries tend to reach fewer nodes and
    their targets tend to have fewer ancestors; neither feature needs a path
    between the two endpoints. Single-threshold rules on one such feature,
    fitted on the set itself, score up to 0.78 on the graph-disjoint
    validation splits and up to 0.86 on the long-path set (0.79 when limited
    to 6 hops from the endpoint); endpoint degrees alone score up to 0.62
    (`artifacts/reach_cue_audit.json`). Accuracy between 0.5 and these levels
    does not by itself show path search; results at 1.000, such as those of
    the message-passing arms on the validation splits, exceed them. The
    crossed sets (`id_crossed_20k`, `extended_crossed_2k`) remove these cues
    by construction: each graph contributes two reachable queries and the two
    queries that cross them, so every source and every target appears once
    with each label, and every one-endpoint rule scores exactly 0.5. A rule on
    the distance between the two endpoints with edge direction ignored scores
    0.51 on them (0.53–0.54 on the paired sets).

11. **Crossed-set construction.** In a crossed graph, each reachable query has
    exactly one path, and graph sizes (16–24 nodes, and 40–48 for the
    long-path set) and densities differ from the paired sets, so results on
    the two families are not directly comparable.

12. **Learning and stability on the crossed sets.** In the stability ring
    (`artifacts/stability_ring.json`: nine parameter-matched looped arms,
    five seeds, 30 epochs, one optimiser setting), every arm scored 0.5
    before training except the ported arm with its original initialisation,
    which scored 1.000. The message-passing arms fitted the crossed
    validation split in 13 of 35 runs (0 to 4 of 5 seeds per arm), each
    within its first two epochs; the other runs stayed at 0.5 for all 30
    epochs, whereas on the paired sets every looped seed fitted it in its
    first epoch (item 9). Three message-passing runs kept their long-path
    answers at every step count from 16 to 192 at the final checkpoint, all
    trained with random step counts (one each: the standard step with
    [6, 10] and with [6, 18], and the recurrence under study with [6, 10]);
    none trained with a fixed 6 steps did. With a fixed 6 steps, the
    recurrence under study fitted the split in 4 of 5 seeds (the standard
    step in 2), but its state stopped changing without carrying the answer
    at 32 steps and beyond, and one of the four fell back to 0.5 at epoch 28;
    with per-step vectors, the one seed that fitted the split lost the answer
    beyond 6 steps. The ported arm with default initialisation fitted the
    split in every seed within one epoch and scored 0.63–0.84 on the
    long-path set at every step count: it answered every unreachable pair
    correctly but missed reachable ones, because it gates edges through
    node-identity embeddings and left 5–18% of the edges involving node
    identities never seen in training gated off. With its original
    initialisation it was correct at every step count in every seed, as it
    was before training. The range [6, 10] was added after a 3-epoch pilot
    (seed 0) in which the planned [6, 18] did not start learning.

13. **Take-off study.** Over 20 seeds per cell
    (`artifacts/takeoff_study.json`: three parameter-matched looped arms,
    three training starts, 5 epochs), the anchored variant fitted the crossed
    validation split in all 60 runs (58 in the first epoch) and scored 1.000
    on the long-path set at 16, 48 and 192 steps in every run, and 1.000 on
    the validation split at 192 steps; untrained, it scored 0.40–0.50. The
    standard step fitted the split in 41 of 60 runs and the recurrence under
    study in 51; 2 and 5 of their runs held the long-path answers at all three
    step counts. For holding those answers, the anchored variant differed from
    both other arms under every start (two-sided Fisher exact test,
    Holm-adjusted over 15 comparisons, p ≤ 2.6×10⁻⁷). For fitting the split,
    it differed significantly after adjustment only from the standard step
    with the curriculum start (p = 0.018; p = 0.12 against a cold start). The
    recurrence under study did not differ from the standard step, and neither
    a first epoch on the paired set nor the hop curriculum changed any arm's
    rate significantly. Seeds shared with the stability ring reproduce its
    first epochs exactly. Like every message-passing arm, the anchored
    variant is given the graph's edges, and its design keeps unreached nodes
    at exactly zero, the structure breadth-first search relies on. Its result
    therefore shows that this structure is learned reliably at this size, not
    that it emerges from token input.

14. **LLM reference points.** Open language models from the local cache
    answered the same questions, inference only: eight models of 0.49 to 3.82
    billion parameters in 16-bit precision, and Qwen2.5-7B-Instruct (7.6
    billion) with its linear-layer weights stored in 8 bits, because it does
    not fit this machine's GPU memory in 16 bits. The 8-bit weights were used
    only after a check with limits fixed in advance: Qwen2.5-3B-Instruct,
    scored both ways on the same questions, gave margins correlated at
    r ≥ 0.995, the same answer on at least 95.8% of the questions and AUROC
    within 0.004 on every set (`artifacts/llm_int8_fidelity.json`).
    - *Immediate answer* after four solved examples
      (`artifacts/llm_reference.json`, `artifacts/llm_reference_7b.json`,
      1,000 questions per set): all nine models scored at chance on the
      crossed validation split (AUROC 0.48–0.50) and on the long-path set
      (0.49–0.52), as on the same questions with the edge list withheld
      (0.49–0.50). On the paired validation split, which carries one-endpoint
      cues, AUROC was 0.51–0.62, highest for the 7B model (0.62) and
      Falcon3-3B-Instruct (0.61).
    - *Cue conflict.* Scored on every paired graph in which the target-reach
      cue conflicts with the label and on as many in which it agrees (548 of
      each; `artifacts/llm_cue_conflict.json`,
      `artifacts/llm_cue_conflict_7b.json`), these two models ranked the
      reachable question higher in 65% and 62% of agreeing graphs but in 41%
      and 50% of conflicting ones (two-sided Fisher exact tests on graphs
      without ties, p ≈ 5×10⁻¹⁶ and p ≈ 9×10⁻⁵): their signal follows the
      one-endpoint cue, not a path. Qwen2.5-1.5B showed the same pattern more
      weakly (56% against 46%, p = 0.003) and the two Qwen2.5-3B models the
      reverse (48% against 57%, p ≤ 0.005), which a model that searches would
      not show either.
    - *Edge lookup.* Asked only whether single edges are listed
      (`artifacts/llm_edge_probe.json`, `artifacts/llm_edge_probe_7b.json`,
      1,000 questions per set: two listed edges, one reversed edge and one
      absent pair per graph), the models of 3 to 7.6 billion parameters
      scored AUROC 0.89–0.99 on both sets (the 7B model 0.99 on both) and
      rejected reversed edges in 67–87% of cases, while scoring at chance on
      reachability over the same graphs: they read the edge list but do not
      search it. Models of 1.7 billion parameters or fewer scored 0.62–0.81,
      with strong biases towards one answer.
    - *Step by step* (`artifacts/llm_cot.json`, `artifacts/llm_cot_7b.json`:
      six instruction-tuned models including the 7B model, greedy decoding,
      400 questions per set with edges and 100 without, every generation
      recorded in the matching `*_generations.jsonl`): they reached 0.45–0.55
      among parsed answers on the crossed validation split (the 7B model
      0.54) and 0.48–0.52 on the long-path set (0.52). On the long-path set,
      3–52% of the answers were cut off at the 768-token budget before a
      final answer (18% for the 7B model). Without the edges, one model
      declined to answer 97% of the questions and the others stayed at
      chance. The written paths locate the errors
      (`artifacts/llm_cot_path_audit.json`; a written path is the last chain
      of three or more nodes joined by arrows in a generation). Of the Yes
      answers on unreachable pairs, 828 of 1,537 contain a written path; 688
      of these claim to run from the source to the target, which the listed
      edges do not allow: 441 include a listed edge followed backwards and
      656 a pair that is not listed at all (a path can do both). The 7B
      model, which rejected reversed edges in 82–85% of single-edge
      questions, followed an edge backwards in 87 of the 151 paths behind its
      wrong Yes answers. Behind correct Yes answers, only 148 of 828 written
      paths use listed edges alone, so most correct answers are not backed
      by a valid path either.

    These models read the edges as text, whereas the message-passing arms are
    given the graph as structure, so the comparison is not like-for-like: the
    reference points show where standard open models of these sizes stand on
    the same questions, not that one architecture outperforms another.

15. **Reader pipeline.** A reader that sees only the edge list, never the
    question, and gives every node token the same embedding supplied the
    anchored arm with an explicit 0/1 graph
    (`artifacts/reader_pipeline.json`: three regimes, 10 seeds each, 5 epochs
    on `id_crossed_20k`; pass criteria fixed in the runner).
    - *Trained on the true edges*, with the solver trained first on true
      graphs and then frozen, the reader passed in all 10 seeds, each from
      its first epoch. It read every validation and long-path graph exactly,
      although 788 of the 2,000 long-path edge lists are longer than any in
      training (up to 116 edges, against at most 69), with a topology loss of
      0.0016 per node pair on the validation split and 0.0011 on the
      long-path set (0.29–0.85 for the untrained reader), and
      the pipeline scored accuracy and AUROC 1.000 on the validation split at
      6 steps and on the long-path set at 16, 48 and 192 steps.
    - *Trained from the answers alone*, whether the solver was frozen or
      trained together with the reader, the pipeline stayed at chance in
      every seed (AUROC 0.50 at every step count). Each reader ended with an
      empty graph (3 of 10 seeds with the frozen solver, 9 of 10 trained
      together) or with nearly every pair connected (7 and 1; edge F1 at most
      0.16). With the frozen solver, all 10,000 wrong long-path answers came
      from the reader's graph. The exact-graph criterion could not have been
      met by any reader trained from answers alone: 28% of the validation
      graphs' listed edges are implied by other paths, so no answer depends
      on them, and only 5 of the 1,000 validation graphs have none
      (`artifacts/reader_identifiability.json`). These readers also missed the
      target that answers can reveal, the true reachability closure
      (agreement 0.41 and 0.71 on the long-path graphs). The gradient probe
      shows the mechanism. On an
      empty graph the answers loss has exactly zero gradient on every edge:
      an edge into a node the solver has not reached meets the solver's
      bias-free update at an input of exactly zero, where its ReLU has zero
      slope, so the loss cannot propose an edge into a part of the graph the
      source does not yet reach. With the frozen solver, the first gradient
      pushed almost uniformly towards removing or towards adding edges,
      depending on how many edges the untrained reader emitted (0.3 to 208
      per graph, against 32.5 true), and in 8 of the 9 seeds with any
      gradient the reader reached, within the first epoch, the state that
      push pointed to.
    - *Disclosures.* The first full run was stopped during its second seed:
      the supervised reader of seed 0 had fallen into a float32 underflow of its
      edge evidence, where every gradient is exactly zero (logs kept as
      `artifacts/reader_pipeline_underflow_*_run.log`); the evidence is now
      computed in log space, and all runs above are from that version. The
      reader's learning rate (1e-2, against the solver's 1e-3) and the
      straight-through carrier of the answers-only regimes were set after
      pilots on 2,000 rows. Reader runs on the GPU are not bit-reproducible,
      because the edge evidence is summed with atomic operations; every final
      checkpoint was re-scored and matched its record.

    These results show that the anchored arm keeps its answers when the graph
    is read from text by a reader trained on the edges, not that the graph can
    be learned from the answers alone. Other readers, relaxations of the hard
    graph, priors on edge density and solvers without an exactly-zero kink
    remain untested.

16. **Reader follow-ups.** The two questions raised by item 15 were run as
    separate studies, each fixed in code before its runs.
    - *Answers-only variants* (`artifacts/reader_variants.json`; the frozen
      solver of item 15, 10 seeds each): during training the solver received
      the reader's soft graph P(edge) instead of the hard one, a prior pulled
      the density of the graph it received towards the training graphs'
      pooled edge density (0.084), or both; evaluation used the hard graph.
      With the answers-only regime of item 15 this forms a 2×2 design, and no
      seed in it passed: every pipeline stayed at chance (AUROC 0.50). Both
      soft variants stayed in the state they reached in the first epoch, a
      nearly complete graph (8 of 10 seeds without the prior, 1 with it) or
      an empty one; with the prior, a soft graph whose every pair sits near
      the target density satisfies the prior and still thresholds to an
      empty graph. With the hard graph, the prior did what it was built to
      do: 7 of 10 readers ended with 27–39 edges per graph against 32.5
      true, and in 7 the gradient on the graph never vanished. But edge F1
      stayed between 0.08 and 0.26 with no upward trend over the five epochs
      (a random graph of that size scores about 0.08), and accuracy between
      0.50 and 0.51. Removing the zero-gradient trap and holding the density
      near the truth did not let the answers single out which edges exist
      (agreement with the true closure on the long-path graphs: 0.36–0.71).
      The first runs of the two prior variants clamped the density, which
      gave the prior no gradient at an empty graph; they were stopped and
      rerun with the fix (logs kept as
      `artifacts/reader_variants_clamped_*_run.log`).
    - *Language models as the reader* (`artifacts/llm_reader.json`:
      Qwen2.5-3B-Instruct, Phi-3.5-mini-instruct and Falcon3-3B-Instruct,
      greedy decoding, the 100 crossed-validation and 100 long-path graphs of
      the step-by-step samples, 6,396 replies per model, all recorded): shown
      a graph's edge list, each model listed every node's successors without
      ever seeing the question. No model read more than 1% of the graphs
      exactly. Edge F1 was 0.53–0.77: Qwen2.5-3B missed about half the edges
      (recall 0.45–0.49), while Phi-3.5 and Falcon3 listed too many
      (precision 0.50–0.68), 14–31% of the extra ones being listed edges
      read backwards. The frozen solvers, which answer every
      question correctly on the true graphs, then scored 0.51–0.58 on the
      crossed questions (Phi-3.5 0.58, Wilson 95% 0.53–0.63; Falcon3 0.55)
      and 0.48–0.51 on the long-path questions; all but a handful of the
      wrong answers trace to the graph read. On the same crossed questions
      these models, reasoning step by step, had scored 0.48–0.53. They
      verify a single listed edge at AUROC 0.89–0.99 (item 14), so listing a
      node's edges is a different and, here, much harder task than checking
      one. The run stopped after two models when the third could not be
      loaded while the second was still held in GPU memory; the third ran on
      its own and the two parts were merged (same protocol and sample). An
      earlier attempt was stopped on a misread of allocator warnings; its
      replies were identical (log kept as
      `artifacts/llm_reader_stopped_run.log`).

    These results do not show that answers can never teach a reader, or that
    no language model can read the graph exactly: other readers, training
    signals, prompts and larger models remain untested.

17. **The exactness frontier.** Three measurements, fixed in code before they
    ran, on how exact a reader's graph must be and why answers alone did not
    teach it.
    - *What answers can reveal* (`artifacts/reader_identifiability.json`): an
      edge u→v is implied when v stays reachable from u without it, so no
      reachability answer depends on it. Of the listed edges, 28–29% are
      implied in the crossed training and validation graphs and 23% in the
      long-path graphs; only 24 of 4,000 training, 5 of 1,000 validation and
      1 of 500 long-path graphs have none. From answers alone the attainable
      target is therefore the true reachability closure, not the edge list;
      items 15 and 16 are read against it.
    - *How exact the graph must be* (`artifacts/reader_noise.json`; the ten
      frozen solvers of item 15, the full crossed validation split and
      long-path set, five random corruptions per rate): on the true graphs
      the pipeline scored 1.000. Deleting each edge with probability p
      lowered accuracy exactly as the single-path structure predicts,
      1/2 + 1/2 · mean (1 − p)^h: on the long-path set (paths of 8–16 edges)
      0.971, 0.942, 0.895, 0.771 and 0.647 at p = 0.005, 0.01, 0.02, 0.05 and
      0.1 (expected 0.971, 0.943, 0.893, 0.773 and 0.648). Reversed edges cost
      more (0.943 at 0.5% of edges, 0.900 at 1%), and inserted edges cost more
      at 48 and 192 steps than at 16, since more steps traverse longer false
      paths (1% extra edges: 0.946 at 16 steps, 0.928 at 48 and 192). A reader
      with edge F1 0.99 thus leaves 0.87–0.90 on the long paths and
      0.95–0.96 on the crossed validation paths; one with F1 0.95 leaves
      0.60–0.67 on the long paths. The language-model readers of item 16
      (F1 0.53–0.77) sit where the curves are at or near chance. Random
      readings with each model's recall, extra edges and share of reversals
      scored within 0.03 of the model's own pipeline in four of six
      model–set pairs; on the crossed questions Phi-3.5 and Falcon3 scored
      0.05 and 0.03 above their matched noise, differences of the size of the
      sampling error for 400 questions. Their reading errors cost about what
      random errors at the same rates cost.
    - *Why answers alone did not teach the reader*
      (`artifacts/reader_answers_density.json`; frozen solver, hard graph,
      density prior, 10 seeds per cell, scored on the true closure): the
      reader learned from four answers per graph or from every source's
      answer for every target, while the solver's message and update ReLUs
      kept PyTorch's slope 0 at an input of exactly 0 or took slope 1 there,
      which changes only the gradient. No seed of the 2×2 passed, and
      accuracy stayed at 0.50–0.51 throughout. Dense answers raised edge F1
      slightly (0.20 to 0.25) without moving accuracy, and closure agreement
      on the long-path graphs stayed at 0.64–0.69. Slope 1 at zero let the
      gradient reach edges into unreached nodes, as intended, but turned the
      first gradient almost wholly towards adding edges (net push to add
      0.999, against 0.62 with slope 0), and the readers over-connected: with
      dense answers every seed ended with most pairs connected (recall 1.000;
      262–316 edges per graph against 32.5). Neither the sparsity of the
      answers nor the zero-gradient kink, alone or together, accounts for the
      failure; what does (for example, learning a discrete graph through
      straight-through gradients at all) these studies do not isolate.

18. **Measurement-only status.** No result in this repository is presented as
    an established finding.
