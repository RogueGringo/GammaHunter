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
    (`artifacts/reach_cue_audit.json`).
    - *Both endpoints, paired sets.* Rules that combine both endpoints' reach,
      still without a path between them, come much closer. The pigeonhole
      rule is fitted on nothing: reachable iff |desc(s) ∪ {s}| +
      |anc(t) ∪ {t}| > n, when the two sets must share a node. It is sound, so
      it never fires on an unreachable pair. It scores 0.9878 and 0.9925 on
      the two graph-disjoint validation splits (firing on 1,951 of the 2,000
      reachable pairs of the larger) and 0.9715 on the long-path set. A
      two-threshold rule on the two reach fractions, fitted on the training
      split, scores 0.9788 on the larger split and 0.9750 on the smaller with
      the endpoints left out of the counts, and 0.9750 and 0.9800 with them
      counted. A gradient-boosted learner (`artifacts/cue_ceiling.json`,
      five learner seeds) was trained on the training split and scored on the
      larger validation split. With 13 endpoint features it scores
      0.9885–0.9892 (0.9900 on the smaller split), and 0.9995 in grouped
      cross-validation on the long-path set; with the target's side alone,
      0.7990–0.8033. With 42 richer features, added after a first comparison
      with an independent review's figures, it scores 0.9935–0.9942 (target
      side alone 0.8020–0.8067). Refitted on shuffled training labels, the
      same learner scores 0.35–0.64 on the larger split. On the paired sets,
      then, accuracy up to about 0.994 on the validation splits and 0.9995
      on the long-path set does not by itself show path search. Results at
      1.000 on the validation splits, such as those of the message-passing
      arms, exceed the best such rule measured by 0.58 points on the larger
      split (0.9942) and 0.75 on the smaller (the pigeonhole rule's 0.9925).
      The long-path results of item 9 (at least 0.997) do not exceed it.
    - *Both endpoints, crossed sets.* The crossed sets (`id_crossed_20k`,
      `extended_crossed_2k`) remove the one-endpoint cues by construction.
      Each graph contributes two reachable queries and the two queries that
      cross them, so every source and every target appears once with each
      label, and every one-endpoint rule scores exactly 0.5. The same
      construction lets the pigeonhole rule fire on at most one of each
      graph's two reachable queries, and it fired on none. A rule on the
      distance between the two endpoints with edge direction ignored scores
      0.51 (0.53–0.54 on the paired sets). The learner, trained on the crossed
      training split and scored on its validation split, gives:
      - with the 13 endpoint features, 0.5068–0.5152, just above its
        range on shuffled labels (0.4925–0.5030; 0.4905–0.5100 over all
        feature sets);
      - with three direction-blind pair features added, 0.5285–0.5373;
      - with the richer features, 0.5188–0.6098;
      - with the richer features and the pair features, 0.5780–0.6070
        (0.5275–0.675 by graph hop).

      In grouped cross-validation on the long-path set it scores
      0.5035–0.5105 (shuffled labels: 0.4945–0.5120). Crossed-set accuracies
      up to about 0.61 on the validation split, and up to about 0.51 on the
      long-path set, therefore do not by themselves show search. Items 14,
      16, 17, 19 and 20 cite this ceiling where crossed-set accuracies fall
      between chance and about 0.61; results at 0.50 (items 12, 13, 15, 18)
      are at chance and are not annotated.
    - *The review's figures, one by one* (recorded as claims in both result
      files).
      - Reproduced:
        - pigeonhole 0.9878, firing on 1,951 of 2,000 reachable pairs with
          precision 1.0, and 0.9925;
        - the two-threshold rule's 0.980 on the smaller split, with the
          endpoints counted;
        - the learner's 0.9995 in grouped cross-validation on the paired
          long-path set;
        - about 0.50–0.52 in grouped cross-validation on the crossed
          long-path set (0.5105 with the original features).
      - Not reproduced:
        - the two-threshold rule's 0.976 on the larger split (0.9788 or
          0.9750);
        - the learner's 0.996 on the larger paired split (0.9885–0.9892;
          0.9935–0.9942 with the later features);
        - 0.934 with the target's side alone (0.7990–0.8067);
        - 0.54 with per-endpoint features on the crossed validation split
          (0.5068–0.5152; 0.5188–0.6098 with the later features);
        - 0.60–0.61 in four of five learner seeds and 0.54 in one, and
          0.57–0.67 at every hop, with pair features added. No feature set
          here reproduces that pattern: 0.5285–0.5373 with the original
          features, and 0.5780–0.6070 with the later ones, whose lowest
          hop is 0.5275–0.5675.

      The review's exact features were not available. Its interactions are
      read here as those the trees learn, and every feature set includes the
      graph size. The learner's seed only matters with more than 10,000
      training rows, where it draws the held-out split for early stopping, so
      the smaller sets give the same score for every seed.

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
    at exactly zero. In exact arithmetic that zero pattern computes
    reachability within the step budget except for weights under which a
    newly reached node's update is exactly zero (every pre-activation at or
    below zero); in float32 it did so on these sets
    (`artifacts/anchored_untrained_control.json`, five random
    initialisations at the take-off width). A frozen random core,
    read by the fixed rule "reachable iff some element of the target's state
    is non-zero", scored 1.000 on the validation split at 6 steps and on the
    long-path set at 16, 48 and 192 steps in every initialisation. It scored
    0.900 and 0.800 at 5 and 4 steps, where only reachable pairs beyond the
    step budget were missed. Read by the state's norm instead, it scored 0.90
    on the long-path set at 16 steps: for every target 16 hops from the
    source the float32 norm underflowed to 0, although its elements did not.
    In a random core the non-zero states shrink by a factor of 37–370 per
    hop (at 6 steps, median norm at the target 5–9 × 10⁻² two hops from the
    source, 0.3–1.1 × 10⁻⁹ at six). A readout head trained on the frozen
    core (the take-off protocol, 5 epochs) reached only 0.70 on the
    validation split at 6 steps, and 0.50 on the long-path set at 16 and 48
    steps (1.000 on both sets at 192 steps), in every seed: it did not tell
    the small states
    from zero. In the take-off study's trained checkpoints (cold
    start, seeds 0–4), every reached target sits at the norm cap (12.2) at
    every hop and step count measured. Both readings score 1.000 within the
    step budget (6 steps on the validation split; 16, 48 and 192 on the
    long-path set) and, like the random core, 0.800 and 0.900 at 4 and 5
    steps, where targets beyond the budget are not yet reached. The search
    itself, which nodes are non-zero, is therefore fixed by the
    architecture, and training does two things: it learns the readout, and
    it keeps the reached states at full size. This departs from the
    independent review's proposed wording ("training learns only the
    zero/non-zero readout"), which the trained-head control does not
    support. The take-off result shows how reliably training achieves this
    at this size, not that search is learned or that it emerges from token
    input.

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
      0.54) and 0.48–0.52 on the long-path set (0.52): within what a
      learned rule that needs no search reaches on the crossed validation
      split (up to 0.61), and at or just above what it reaches on the
      long-path set (0.50–0.51; item 10). On the long-path set,
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
    anchored arm with an explicit 0/1 graph. It is also given each token's
    slot within its edge (source, separator, target) as an input embedding,
    so the edge list reaches it already parsed into source and target
    positions; the natural-language readers of items 19–20 receive no slots.
    Results are in `artifacts/reader_pipeline.json` (three regimes, 10 seeds
    each, 5 epochs on `id_crossed_20k`; pass criteria fixed in the runner).
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
      and 0.48–0.51 on the long-path questions, within what a learned rule
      that needs no search reaches on those sets (up to 0.61 and 0.51;
      item 10); all but a handful of the
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
      sampling error for 400 questions, and their 0.58 and 0.55 are within
      what a learned rule needing no search reaches there (up to 0.61;
      item 10). Their reading errors cost about what random errors at the
      same rates cost.
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

18. **An unbiased estimator for answers-only reading.** Fixed in code before
    it ran (`artifacts/reader_estimator.json`,
    `artifacts/reader_estimator_lr1e-3.json`; the frozen solvers of item 15,
    hard graph, density prior, 10 seeds per arm, scored on the true closure).
    Items 15–17 trained the reader from answers through straight-through
    gradients, which are biased. Here the same readers learned through a
    score-function (REINFORCE) estimator instead: for each graph, four graphs
    were sampled edge by edge from the reader's probabilities, the frozen
    solver's answers scored each sample (the log-likelihood of the correct
    answers), and each sample's baseline was the mean score of the other
    three. No gradient passed through the solver, so its zero-gradient kink
    played no part. No seed passed, and accuracy stayed at 0.50 throughout.
    - *At the reader rate of every earlier study (1e-2)* each reader's graph
      became deterministic within its first epoch and stayed so: empty in 17
      of the 20 readers (9 of 10 with four answers per graph, 8 of 10 with
      every source's answer for every target), complete in the other 3. Once
      every sample is the same graph, each sample's score equals the others'
      mean and the estimator's gradient is exactly zero; the density prior's
      gradient also vanishes when every probability is 0 or 1 in float32.
      From then on, weight decay kept shrinking the readers' scores, but
      within five epochs not far enough to change a single edge.
    - *At a tenth of that rate (1e-3)*, fixed in advance so that a failure
      could not be put down to the step size (every answer, 10 seeds), the
      readers stayed stochastic, but the graph they asserted stayed nearly
      empty: 0.3–3.4 edges per graph against 32.5 true. Of those edges 52–63%
      were true, against 8.5% for a random graph of that size and 12–35% for
      the same readers untrained (the architecture alone already favours
      true edges). Edge F1 was 0.01–0.10, and closure agreement
      on the long-path graphs 0.759–0.763, against 0.758 for an empty graph.
      The topology loss fell in every seed (0.226–0.231 per pair after the
      first epoch, 0.211–0.216 after the fifth); whether that reflects the
      answers, or only the density prior and weight decay recalibrating the
      untrained readers' bias, these runs do not separate.

    By the reading fixed in advance, failure at both rates places the
    obstacle in credit assignment from answers alone rather than in the bias
    of straight-through gradients: one score per sampled graph must
    apportion credit among some 380 candidate edges. The estimator stalled in
    two distinct ways, so this reading holds at this budget only; more
    samples per graph, an entropy bonus or longer training remain untested.

    The study's first launch was stopped when the dense arm at 1e-2 raised a
    CUDA illegal-memory-access error in its first epoch. Its cause is unknown
    (an overflow in the sampling code was tested and ruled out); the code was
    hardened (log evidence capped, finiteness checks) and the relaunch, with
    synchronous kernel launches in the dense arms, raised none. The first
    launch's logs were overwritten by the relaunch. The two dense arms saved
    their checkpoints under the same names: each run re-scored its own
    checkpoint right after saving, and every re-score matched, but only the
    later file of each seed was kept.

19. **Reading natural language.** Three readers, fixed in code before they
    ran, read each graph from sentences instead of an edge list, without the
    slot input that parses the edge list for item 15's reader
    (`nl_render`): one sentence per edge in a seeded shuffled order, worded
    by one of five training templates or one of four held-out templates
    whose wording never appears in training (in each set, one template names
    the target first), with a quarter as many distractor sentences that name
    a single node. The question never appears. Every reader feeds the frozen
    solvers of item 15 and is scored on the held-out wording; the training
    wording of the same validation graphs is the in-distribution control.
    - *A word reader trained from scratch* (`artifacts/nl_reader_words.json`:
      the reader of item 15 over words, node numbers sharing one
      identity-free kind and unseen words mapped to a single unknown word;
      supervised on the true edges; 10 seeds). In distribution it read
      almost exactly: edge F1 0.999–1.000, exact graphs 0.93–1.00, pipeline
      accuracy 0.995–1.000. On the held-out wording no seed passed. Almost
      every edge it listed exists (precision 1.000), but it found at most
      half of them (recall 0–0.50) and read no graph exactly; the pipeline
      scored 0.50–0.55 on the validation questions and 0.50 on the long
      paths. A post-hoc breakdown from the saved readers
      (`artifacts/nl_templates_words.json`, which reproduces each reader's
      recorded recall exactly) shows which half. The two held-out templates
      that keep the training templates' "{u} … to {v}." order were read
      almost completely (recall ≥ 0.99) by 4 and 3 of the seeds. The other
      two, one naming the target first and one a longer clause, were read by
      none.
    - *A reader on frozen language-model features*
      (`artifacts/nl_reader_lm.json`: the hidden states at layer 12 of 24
      of Qwen2.5-0.5B from the local cache, frozen, with each node number
      marked at its last token, and the same reader on top; 10 seeds). In
      distribution 9 of the 10 seeds read almost exactly (edge F1 1.000,
      exact graphs 0.97–1.00). The tenth diverged in its last epoch (mean
      training loss 100, after 0.001) and ended listing 88 edges per graph
      against 32.5, with precision 0.15. On the held-out wording the
      features carried more across: for the nine, edge F1 0.68–0.91,
      precision 0.85–1.00, recall 0.55–0.83. Still no seed passed. The best
      read 2.3% of the validation graphs exactly. The nine pipelines scored
      0.55–0.76 on the validation questions, and all ten scored 0.50 on the
      long paths (0.4975–0.5040), where a path runs through 8–16 edges. The
      same breakdown (`artifacts/nl_templates_lm.json`, again reproducing
      the recorded recall exactly) shows where: 7 of the 10 seeds read both
      "{u} … to {v}." templates almost completely, but no seed read more
      than 51% of the longer clause (19–51%) or 85% of the sentences naming
      the target first (3–85%, mean 27%).
    - *Language models listing successors* (`artifacts/llm_reader_nl.json`:
      the three models of item 16, prompted as there but shown the held-out
      rendering instead of the edge list, on the same 400 + 400 questions;
      19,188 replies, all recorded). Every model read the sentences worse
      than it had read the edge list: edge F1 0.41–0.50 on the validation
      graphs (edge list: 0.55–0.77) and 0.31–0.40 on the long-path graphs
      (0.53–0.67). No model read a graph exactly, and the pipeline scored
      0.51–0.52 on the validation questions and 0.495–0.500 on the long
      paths. The same breakdown of the recorded replies
      (`artifacts/nl_templates_llm.json`, reproducing each model's recorded
      recall exactly) places much of the loss in the template that names the
      target first: the models found 13–32% of its edges, against 35–86% for
      the other templates, and listed 27–63% of them backwards, against
      6–31%.

    The per-seed held-out figures of the word reader describe one run: rerun
    with the same code path and seeds (item 21), it did not reproduce them,
    seed by seed or in their mean (held-out edge F1 0.371 against 0.463).

    Only the trained readers came close to exact, and only in the wording
    they were trained on. On the held-out wording no reader read the graphs
    exactly enough for the search, and every pipeline stayed at chance on
    the long paths. On the validation questions, accuracies up to about 0.61
    are within what a learned rule that needs no search reaches (item 10).
    A small language model's frozen features carried more of
    the reading to new wording than words learned from scratch, while the
    language models read the sentences worse than the edge list. The
    sentence that names the target first was the hardest for all three.
    These results cover five training and four held-out templates, one
    small encoder at one layer and three models of 3–4 billion parameters;
    other renderings, encoders, prompts and larger models remain untested.

    Training on sentences at the reader rate of the earlier studies (1e-2)
    was unstable: in 12 of the 20 runs the mean training loss of some epoch
    rose above 10, against about 0.002 at the end of most runs, and one run
    (the language-model reader above) did not recover. The first words run
    ran out of GPU memory while scoring the long-path set, rendered as
    sentences of up to 1,127 words. Scoring batches were then sized by
    rendering length, which leaves the graphs read, accuracy and edge
    metrics unchanged (tested). A second run was stopped when four
    concurrent runs had filled the machine's memory; the third ran to
    completion. Both earlier logs are kept
    (`artifacts/nl_reader_words_oom_run.log`,
    `artifacts/nl_reader_words_stopped_run.log`).

20. **Wording diversity.** Fixed in code before it ran (commit ba7a5b5): the
    two trained readers of item 19, trained on 51 wordings instead of five
    (`artifacts/nl_reader_words_diverse.json`,
    `artifacts/nl_reader_lm_diverse.json`; otherwise as in item 19, without
    slot input, 10 seeds each, except that the word reader's vocabulary grows
    with the wordings).
    The wordings are the five training templates and 46 more; a quarter name
    the target first, and "by" introduces the source in some and the target in
    another. None uses an open-class word of any held-out or novel template.
    Two sets of held-out templates were scored. Item 19's four held-out
    templates carry the pass criteria, unchanged; the new wordings contain the
    constructions of three of them, with other words. Four novel templates are
    built from constructions the wordings lack (an imperative, a conditional,
    a locative inversion and an "endpoint of" phrase); they were scored with
    the same criteria as a stress test, reported separately. No seed of either
    reader passed either.
    - *Frozen language-model features.* In distribution every seed read
      closely (edge F1 0.995–0.999, exact graphs 0.73–0.95). On the held-out
      templates edge F1 was 0.90–0.93 in all ten seeds (item 19: 0.68–0.91 in
      the nine seeds that did not diverge), with precision 0.965–0.998 and
      recall 0.83–0.87, and the pipeline scored 0.73–0.80 on the validation
      questions (item 19: 0.55–0.76 in those nine). Exact graphs stayed at
      0.4–3.1%, and the long paths at 0.51–0.54 (item 19: 0.50), at or just
      above the 0.50–0.51 that a learned rule needing no search reaches on
      that set in grouped cross-validation (item 10). The template
      audit (`artifacts/nl_templates_lm_diverse.json`, reproducing each
      reader's recorded recall exactly) shows where. The three held-out
      templates whose constructions occur in the training wordings were now
      read almost completely (mean recall 1.000, 0.996 and 0.985; item 19:
      0.91, 0.91 and 0.27). The fourth, "Starting at {u}, one step takes you
      to {v}.", a fronted clause whose construction the wordings lack, stayed
      at 37–50% in every seed, no better than in item 19 (19–51%); about a
      quarter of each graph's edges are written this way, so few graphs were
      read exactly. This is consistent with a failure of structural rather
      than lexical generalisation in the terms of COGS (Kim and Linzen, 2020),
      which distinguishes generalisation to novel combinations of a familiar
      primitive and a familiar structure (lexical) from generalisation to
      novel combinations of familiar syntactic structures (structural). The
      design does not establish it: the held-out words are new rather than
      familiar primitives, and in this template "at" introduces the source,
      whereas in the training wordings it occurs only in "{u} points at {v}.",
      where it introduces the target (the novel conditional, which also uses
      "at" for the source, was read at 0–42%). On the novel templates, recall
      over all four was 34–56% per seed (item 19's reader: 11–28%), unevenly:
      81–98% for the locative inversion, 52–95% for the imperative, 0–42% for
      the conditional and at most 0.51% for the "endpoint of" phrase. Accuracy
      on the novel wording stayed at 0.51–0.56 on the validation questions and
      0.50 on the long paths.
    - *Words from scratch.* In distribution eight seeds read their wording
      closely (edge F1 0.993–1.000, exact graphs 0.68–0.99). Two seeds ended
      their last epoch in a jump of the mean training loss (to 799 and 1,408)
      and read their own wording only partly (exact graphs 0.013); they are
      also the word reader's two best seeds on the held-out templates (edge F1
      0.80 and 0.78), so their held-out scores come with a broken
      in-distribution reading and are not gains. Without those two the word
      reader did not gain on the held-out templates: mean edge F1 0.42
      (0.01–0.63) against 0.46 in item 19, precision 0.29–1.00 against 1.00,
      and no graph read exactly; the pipeline scored 0.50–0.54 on the
      validation questions (0.50–0.61 with the two) and 0.49–0.50 on the long
      paths. On the novel templates it read almost nothing (edge F1 at most
      0.12 without the two, 0.25 with them). The template audit
      (`artifacts/nl_templates_words_diverse.json`) shows the reading moving
      between templates rather than growing (counts over all ten seeds, then
      without the two damaged ones): "A link runs from {u} to {v}." was read
      at recall 0.99 or more by 6 seeds, 4 without them (item 19: 3), and
      the target-first template, whose construction the new wordings
      contain, by 2, 1 without them (item 19: none; mean recall 0.40, or
      0.26), while "{u} leads to {v}." fell from mean recall 0.68 to 0.38, or
      0.23 (seeds at 0.99 or more: 4 to 2, or 1). The word reader also read
      21,725 edges that the graphs do not have (item 19: 5), 69% of them in
      seeds 1, 5 and 6, the three whose last epoch ended in a loss jump.

    Diverse wording let the language-model reader read new words almost
    completely in constructions it had seen. Constructions it had not seen it
    read partly and unevenly, never completely, and the one such construction
    among the four held-out templates left only 0.4–3.1% of the graphs read
    exactly and the long paths near chance. The word reader, which has no
    knowledge of the new words, read more of some templates and less of
    others, and, without its two damaged seeds, not more overall. These
    results cover one grammar of 51
    wordings, one held-out and one novel set of four templates each, and one
    small encoder at one layer; other grammars, larger or fine-tuned encoders
    and generated paraphrases remain untested.

    Training was again unstable at the reader rate of the earlier studies
    (1e-2): in 10 of the 20 runs the mean training loss of some epoch rose
    above 10, and four runs ended their last epoch on such a rise (word
    reader seeds 1, 5 and 6, to 799, 1,408 and 199; language-model reader
    seed 3, to 37, which supplies that reader's lowest held-out scores above
    except precision). Wrong answers attributed to the solver, whose read
    graph had the true answer to the question, occurred once on the long
    paths at 16 steps (language-model reader, seed 7; none at 192 steps),
    once on the novel validation questions (seed 8), and 22 times on the
    validation questions for the word reader (24 counting the question
    sample, which repeats validation questions), all in seeds 1, 5 and 6.
    The pass criterion admits up to ten misread graphs among the 1,000
    validation graphs; for graphs drawn as those are, a
    passing reader's misread rate is then below 1.7% (one-sided 95%, exact
    binomial), and below 0.3% if it reads all 1,000 exactly. The item-19
    template audits were rerun with the novel templates after the grammar
    and the novel templates were written, but before the commit that fixed
    them; their earlier numbers were unchanged, and a rerun after the commit
    reproduced them. From this study on, the language-model features are
    held in CPU memory between batches, which leaves the reads unchanged: the
    reruns reproduce item 19's recorded recall exactly. On the crossed
    validation questions a learned rule that needs no search reaches up to
    0.61 (item 10); the word reader's 0.50–0.61 and the 0.51–0.56 on the
    novel wording are within it. On the long-path set such a rule reaches
    0.50–0.51; every long-path figure above is within 0.05 of chance.

21. **Selection normalizers.** Fixed in code before it ran (commit b7baef4),
    with predictions, decision rules, controls and the departures from its
    queued specification stated in the runner. The word reader of item 19
    was retrained exactly as there, with its attention normalized by
    softmax (the original code path), scalable softmax (softmax of
    s · ln n · z, s learned per layer and head from 1), entmax-1.5 or
    sparsemax (`artifacts/selection_reader_<arm>.json`, 10 seeds each;
    verdicts in `artifacts/selection_reader.json`).
    - *Operators* (`artifacts/selection_conformance.json`,
      `artifacts/selection_stage_a.json`). Every reference normalizer passed
      the conformance battery, a deliberately broken sparsemax failed it
      (simplex, certificate, gradient), and float32 outputs matched float64
      within 7.7 × 10⁻⁷ on the CPU and the GPU, on rows of up to 2,048
      positions. Every prediction fixed for stage A held. To keep half the
      weight against n distractors at logit 0, one position needs exactly
      ln n with softmax but less than ½ with sparsemax and less than √2 with
      entmax-1.5 at every n (largest error 4.4 × 10⁻¹⁰). Against 100,000
      N(0, 1) distractors the median margin was 12.02 for softmax (predicted
      12.01), 4.64 for sparsemax (mean-field prediction 4.58), 5.02 for
      entmax-1.5 and 4.63 for scalable softmax. At 1,000 and 10,000
      distractors sparsemax lay 0.29 and 0.31 from its prediction, inside the
      tolerance of 0.5. Positions appended far below the threshold left
      entmax-1.5 and sparsemax unchanged and diluted softmax; with scalable
      softmax they raised the relevant position's weight (0.52 to 0.80 with
      1,000 appended), because they raise ln n. Of the figures supplied with
      the specification, 12.0 for softmax was reproduced (draws 12.016–12.020)
      and 4.5 for sparsemax was reproduced by the rule fixed in advance
      (draws 4.42–5.25), although the median of the draws was 4.64.
    - *Dilution and length (H1, H2).* Both were moot by their fixed rule,
      which applies them only if softmax loses at least 0.01 of mean edge F1.
      From 0.25 to 4 distractor sentences per edge (16 times as many) in the
      training wording, softmax's mean edge F1 fell by 0.0096 (0.998 to
      0.988): it fell in 9 of 10 seeds (by 0.0003 to 0.043) and rose in seed
      4, whose training had degraded (see below); without that seed the mean
      fall would have been 0.012 and the rule would have applied. From the
      validation graphs to the long-path graphs it fell by 0.0001. The sparse
      arms' mean edge F1 rose slightly under dilution, by 0.001 for
      sparsemax and by 0.003 for entmax-1.5, most of it from its stalled seed
      4 (0.0006 without it). Moot means the comparison was not made; it is
      not evidence that dilution does or does not limit the reading.
    - *Held-out wording (H3).* The rule fixed in advance returned
      "inconclusive" for all three normalizers: it could neither show a gain
      of 0.05 in held-out edge F1 over softmax nor exclude one. Mean edge F1
      was 0.43 for entmax-1.5, 0.35 for sparsemax and 0.31 for scalable
      softmax against 0.37 for softmax (differences +0.058, −0.022 and
      −0.058; one-sided p for a gain 0.30, 0.58 and 0.71; a gain of 0.05 or
      more rejected at p 0.53, 0.25 and 0.16, so for no arm). As stated in
      advance, "inconclusive" is the expected outcome at this spread of seeds
      and supports neither side. No run read a held-out graph exactly or met
      item 19's pass criteria; pipeline accuracy stayed at 0.50–0.55 on the
      validation questions, within the no-search ceiling of item 10, and at
      0.498–0.504 on the long paths. The solvers scored 1.000 on the true
      graphs in every run.
    - *Start of training (H4).* Not supported: the median per-batch loss of
      the first epoch was 0.146 for entmax-1.5 and 0.150 for sparsemax
      against 0.133 for softmax (p 0.22 and 0.15).
    - *Replication of item 19.* Not replicated by the criterion fixed in
      advance. In the training wording the softmax arm's mean edge F1 was
      within 0.002 of item 19's (p 0.79), although its seed 4 read only 30%
      of the validation graphs exactly (item 19's seed 4: 99%; the other
      seeds 98–100%). On the held-out wording its mean edge F1 was 0.091
      lower (0.371 against 0.463; the criterion allows 0.05; p 0.42), so the
      level of item 19's held-out reading was not reproduced either. Its
      held-out precision fell below 1.000 in three seeds (0.033 in seed 0,
      0.78 in seed 8, 0.98 in seed 6; item 19: 1.000 in every seed), and the
      same seed read the held-out wording very differently in the two runs
      (seed 0: 0.007 against 0.666; seed 3: 0.657 against 0.000).
      The code path, seeds, data, torch version and GPU were the same. The
      epoch-1 mean losses agreed within 0.06 in seeds 6–9 and differed by up
      to 13.4 in the others (seed 3: 4.62 against 18.06), so the runs parted
      within the first epoch. This is consistent with the GPU's
      non-reproducible summation (item 15) amplified by the unstable reader
      rate, which these runs did not isolate. Item 19's held-out scores
      were therefore not reproduced, seed by seed or in their mean; no
      verdict above uses them.
    - *Not fixed in advance (descriptive only, no test).* Edge F1 sits near
      its ceiling in the training wording; exact graphs, a stricter measure,
      moved with dilution. From 0.25 to 4 distractor sentences per edge,
      softmax read fewer validation graphs exactly in 9 of 10 seeds (mean 0.92
      to 0.81), whereas the other arms read as many or more in 26 of their 27
      seeds that read their wording (entmax-1.5 0.87 to 0.90, sparsemax 0.82
      to 0.89, scalable softmax 0.88 to 0.90). Whether this holds would need a
      study fixed on that measure. The sparse arms' attention was sparse:
      after training, a position of the first layer gave weight to 8 keys on
      average with sparsemax (2.8% of them) and 14 with entmax-1.5 (5.1%;
      18.7% before training) at 0.25 distractor sentences per edge, and to 24
      and 38 at 4. The number of keys attended grew with the distractors
      while their share fell. Scalable softmax learned scales from −0.29 to
      2.52. Its first layer gave exactly zero weight to more than 1% of
      positions in 5 of its other 9 seeds (up to 25% on the validation graphs
      and 27% on the long-path graphs; traces in a sixth) and to up to 73% in
      seed 0, whose reading collapsed; for a dense normalizer such zeros can
      only come from float32 underflow.

    Training was again unstable at the reader rate of item 19 (1e-2): the
    mean training loss of some epoch rose above 10 in 5 of 10 softmax runs
    (item 19: 7 of 10) and in 8, 8 and 9 runs of the scalable-softmax,
    entmax-1.5 and sparsemax arms. Four runs ended their last epoch on such a
    rise (entmax-1.5 seed 2, sparsemax seeds 4 and 9, scalable softmax seed
    0); two of them read nothing in distribution (sparsemax seed 9, to 9,271;
    scalable softmax seed 0, to 14,601). Entmax-1.5 seed 4 stalled at a loss
    of 0.126–0.128 from its second epoch (in-distribution edge F1 0.39).
    Softmax seed 4 rose to 963.5 in its fourth epoch and ended at 0.110
    (the other softmax seeds at about 0.002), reading 30% of the validation
    graphs exactly; it is the seed that makes H1 moot. All runs are included
    in every verdict, as fixed. Stage A was smoke-tested at
    up to 1,000 distractors during development before its finite-n
    tolerances were written, and the float32 agreement of the reference
    backends was measured before its long cases were added. Checkpoints are
    not kept in the repository; their SHA-256 are recorded, and each was
    re-scored after saving and matched its record. These results cover one
    word reader of two layers, one training rate, distractor sentences that
    name a single node, and renderings of up to 1,764 tokens; other readers,
    rates, forms of distraction and longer contexts remain untested.

22. **Measurement-only status.** No result in this repository is presented as
    an established finding.
