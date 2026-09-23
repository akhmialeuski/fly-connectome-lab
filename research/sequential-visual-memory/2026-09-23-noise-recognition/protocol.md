# Matched recognition under episode noise and storage precision

Preregistered follow-up to #48 within #39. The prior training-only geometry study found noise-induced response changes larger than between-identity image changes, while float16 rounding error was small. That is not evidence of improved recognition from removing noise.

## Fixed inputs and isolation

Use the original persistent 20-identity smoke configuration, 280 training photographs and 60 validation photographs in their existing order. Freeze their exact sample IDs and split membership before generating traces. Exclude all 60 historical test photographs and all reserve photographs from simulation and scoring in this study. Keep all 16 raster image patches, original encoder and seed, input amplitude, graph and synaptic weights, descending-neuron readout population, observation timing, and readout spike-filter state behavior unchanged. CPU only: four Numba threads and one BLAS thread.

Every condition starts from the same baseline noise-enabled warmed rest state, computed with original configuration seed zero. Change episode noise independently of warmup, encoder, image membership, trajectory, and classifier seeds. Use original sample-specific noise identifiers, not one common stream across images: the common-stream intervention in #48 was a geometry measurement, not the intended recognition protocol.

## Frozen schedule

Generate three immutable native float32 trace attempts in this order:

1. Original episode noise, episode seed 0. Its float16 cast must exactly reproduce all selected archived trace rows before proceeding.
2. Original episode noise, episode seed 1, with the same seed-zero warmed rest state.
3. Episode noise disabled, again from the same baseline warmed rest state. This is one shared deterministic endpoint, not two independent replicates.

Each trace contains exactly 340 ordered train/validation episodes, with all 16 observations saved. No original or transformed input images or encoded input arrays enter the archive. Save generated neural responses, spike summaries, warmed rest state and neuron IDs, parameters, source hashes, elapsed time and RSS. If baseline replay fails, stop and diagnose that discrepancy before interpreting any intervention. Never overwrite failed attempts.

Fit two readouts for each of the three traces, in float32-then-float16 order: six unique fits. Float16 means casting the same newly generated native trace to float16 and back for fitting; float32 means retaining its native values. This isolates storage precision without changing a simulation. Use the last sequential observation, both original feature blocks, training-only scaler and fold-local PCA capped at 60, unchanged C grid [0.01, 0.1, 1, 10], five folds, classifier seed 0, tolerance 1e-6 and the previously preregistered 50000 iteration cap. No label permutations, specialized vision models, neuron training, or adaptive budget changes are part of this study.

A failed or nonconverged readout is retained as failed evidence and receives no valid recognition score; continue other preregistered fits without substituting a solver or seed. The seed-zero float16 model must reproduce the previous full-data neural diagnostic endpoint's exported arrays and predictions before interpreting comparisons. Training/preprocessing never uses validation labels.

## Measurements and decisions

Report training, CV and validation accuracy/log loss, convergence, C selection, prediction probabilities, precision-induced prediction disagreements and maximum probability differences. Preserve every fitted scaler, PCA and linear coefficient. Compare noise-off against each noisy seed separately at each precision, plus the mean of the two paired differences. Report 2000-resample identity-cluster intervals with seed 0 and namespace `noise-recognition-identity-bootstrap`, conditional on the fixed validation cohort and the two declared episode seeds. The shared noise-off endpoint must not be counted twice as independent training. These are exploratory validation comparisons, not new test or population confirmation.

Promote noise-off to a larger recognition study only if, at both precisions, the mean validation improvement over the two noisy seeds is at least 10 percentage points and the paired identity-cluster 95% lower bound is positive. This is a practical screening gate, not a biological significance threshold. Otherwise preserve and report the null/inconclusive outcome and reconsider operating regime or population access. Regardless of this gate, no recognition improvement establishes sequential memory; a later matched persistent/reset comparison is required.

Do not choose a preferred precision based on validation performance or silently revise the threshold. Publish all results and failures in the owning issue, cross-link #39, and preserve source, metadata, models and generated traces in a new sibling study. Verify the archive by independent remote restoration before declaring it complete.
