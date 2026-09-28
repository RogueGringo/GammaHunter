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
    the message-passing arms on the validation splits, exceed them.

11. **Measurement-only status.** No result in this repository is presented as
    an established finding.
