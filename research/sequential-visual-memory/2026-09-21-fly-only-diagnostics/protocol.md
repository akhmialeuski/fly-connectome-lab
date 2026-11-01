## P0-P2 preregistration — first diagnostic campaign

Parent #39; execution #40. This is exploratory development, not a new confirmation result. No neural dynamics, connectome weights, or specialized visual models will be changed in this campaign.

### Data availability audit (before scoring)

Read-only annotation/index audit, excluding every filename in existing preprocessing caches:

| Existing cohort | Identities | Unused images | Identities with at least three unused images | Minimum unused per identity |
| --- | ---: | ---: | ---: | ---: |
| Smoke | 20 | 86 | 11 | 0 |
| Main | 100 | 586 | 69 | 0 |

A balanced new three-image-per-identity confirmation test for all original identities is not available. This does not justify reusing the old inspected test as confirmation. Reserve up to three unused images per identity deterministically after exact/perceptual-duplicate screening, record missing coverage, and keep these reserve images out of all diagnostic fitting/scoring. A later confirmatory study requires an explicit cohort decision; results in this first campaign remain exploratory.

### Locked development data and fitting

Use the original smoke training/validation membership (280/60), then the original main training/validation membership (1,400/300) for declared follow-ups. Never load final-test feature rows for classifier fitting/scoring. Input generation follows archived effective configurations and trace identities. Preserve all existing evidence.

Use the same StandardScaler and multinomial logistic classifier at all diagnostic stages, full deterministic PCA where selected, C in {0.01,0.1,1,10}, five stratified training-only folds, seed 0, and tolerance 1e-6 with a maximum of 5,000 iterations. PCA/scaling are refitted inside each fold. Select the smallest C on tied mean CV accuracy. A nonconverged fit is recorded as failed rather than silently weakening tolerance. No-PCA means scaled original features, not an identity matrix materialized in memory.

"Last" means the sixteenth observation. "All" means the ordered concatenation of observations available through sixteen. Pixel and encoded-input probes are transparent diagnostic controls, never replacement recognizers. Neural feature selection uses the archived feature order.

### Ordered smoke candidate budget

1. Pixel last, PCA60.
2. Pixel all, PCA60.
3. Encoded input last, PCA60.
4. Encoded input all, PCA60.
5. Persistent neural last, both feature blocks, PCA60 (reproduction anchor).
6. Reset neural last, both, PCA60.
7. Reset neural all, both, PCA60 (external-history control).
8. Persistent neural all, both, PCA60.
9-12. Persistent neural last, both, PCA20 / PCA120 / PCA240 / no PCA.
13-14. Persistent neural last, spike trace only / voltage only, PCA60.
15-16. Training-label-permuted encoded-all and persistent-neural-last probes, PCA60. Validation labels remain untouched; permutation uses a separately named deterministic seed stream.
17-18. Tiny training-only memorization checks for encoded-all and persistent-neural-last: first five sorted class labels, first four sorted training sample IDs per class (or all available if fewer), PCA60 bounded by fold rank. Report in-sample accuracy explicitly, retain CV diagnostics, and do not label this generalization.

Run candidates serially to avoid timing/CPU interference. Save attempted configuration, status, exceptions, elapsed time, peak RSS, selected C, full CV scores, training/validation top-1/top-5/log loss, prediction probabilities, fitted arrays, and train-feature variance/sparsity/spectrum/PCA explained variance. Resume by skipping completed immutable candidate directories; any changed retry gets a new ID.

### Main-cohort decision

After all smoke attempts and their failures are reported, run the four preregistered stage-localization anchors on main: pixel-all/PCA60, encoded-all/PCA60, persistent-neural-last/both/PCA60, reset-neural-all/both/PCA60. Promote at most one additional neural readout variant only if its smoke validation improvement over the corresponding anchor is at least ten percentage points; choose by mean training CV first among qualifying variants, then fewer PCA dimensions, then candidate order. Treat the smoke selection as exploratory and report the selection rule.

### Integrity and causal limits

Check sample IDs, labels, cache/config/brain hashes, candidate-neuron mapping, and archived-model reproduction. Audit exact source-file/RGB duplicates and 64-bit grayscale difference-hash pairs at Hamming distance <=3 across development splits; perceptual matches are review candidates, not proof of duplication. Exclude reserve candidates that collide with historical samples or previously accepted reserve images under these rules. Do not change the original development split after viewing accuracy; report contamination sensitivity separately if needed.

The same-image repeated-noise, blank-input, float32-versus-float16 and neuron-population probes require new simulations and follow after cached stage-localization results. They are not claimed complete by this cache-only experiment. A diagnostic probe can reveal accessible information but cannot prove its absence under every decoder. Report uncertainty and small validation support; do not manufacture a root cause from one difference.

Next decision: use the first reproducible stage drop to choose the smallest additional measurement in P1/P3. Do not begin a broad dynamical or plasticity sweep based only on total firing rate.
