# Objective

Test whether the fixed MaleCNS trace loses identity information because of the selected readout feature block, PCA rank, or use of only the final observation. This is the next bounded P2 experiment under #39. It follows the negative noise-removal gate in #50; it does not change the brain, encoder, image cohort, spike filter, or model family.

# Frozen inputs

- Use the original persistent 20-identity smoke configuration and the exact 280 train / 60 validation sample IDs frozen in `research/sequential-visual-memory/2026-09-23-noise-recognition/membership.json` (membership SHA-256 `5af10356b3ff565bf515144983758763fb8b346737520c50647885cdf88a92d8`). The historical 60 test images and all reserve images are excluded from fitting and scoring.
- Pass `--train-per-class 14 --subset-seed 0` on every new fit, exactly as in the #50 B0 report. Verify that the resulting training subset contains the same 280 IDs before comparing cases.
- Use the completed noise-enabled episode-seed-zero native float32 trace from #50 as the primary source. Keep its graph, effective sensory mask, warmup, all 16 raster windows, input currents, neuron ordering, episode noise, and observation timing fixed. The existing #50 seed-one trace is reserved only for the declared robustness step.
- Float32 is the primary trace treatment because it retains the measured values before archival rounding. Do not choose a treatment based on validation accuracy. The corresponding original float16 model remains a provenance control, not a candidate selected for this screen.
- Keep the original five stratified training folds, classifier seed zero, C grid `[0.01, 0.1, 1, 10]`, tolerance `1e-6`, and 50,000 maximum logistic iterations. Fit scaling and PCA separately inside each training fold. Retain the same regularized multinomial linear readout. No specialized or pretrained vision model, new trainable encoder, or fly synapse changes.

# Fixed screening schedule

Run the following cases in order on episode seed zero. `B0` is the exact existing #50 float32 model and is reused after verifying its source and classifier hashes; all other IDs are new immutable attempts. `all` concatenates the 16 recorded observation responses in time order; it does not synthesize unrecorded within-window dynamics.

| ID | Feature blocks | Observation history | PCA cap | Purpose |
| --- | --- | --- | ---: | --- |
| B0 | voltage + spike trace | final | 60 | Original reference |
| B1 | voltage only | final | 60 | Remove external spike-filter block |
| B2 | spike trace only | final | 60 | Measure filtered spikes without voltage |
| B3 | voltage + spike trace | final | 20 | Lower-rank control |
| B4 | voltage + spike trace | final | 120 | Higher-rank control |
| B5 | voltage + spike trace | final | 240 | Near-training-rank control; record the actual fold cap |
| B6 | voltage + spike trace | final | none | Scaled, regularized no-PCA control |
| B7 | voltage + spike trace | all 16 | 60 | Explicit observation-history control |

This eight-case schedule is deliberately not the full feature-block × history × PCA Cartesian product. Run one case at a time on the laptop, record wall time and peak RSS, and retain every failure/nonconvergence as its own attempt. A failed case receives no valid recognition score; do not silently increase its iteration budget, change solver, or substitute another case. If the all-history fit exceeds a preregistered 12 GiB peak RSS or 30-minute wall-time budget, stop that case and record a resource failure. Previously completed cases remain valid.

# Analysis and decision

For each valid case report training, fold-held-out CV, and validation top-one accuracy and log loss; chosen C, solver iterations, feature dimensions/rank, variance/sparsity statistics, fitted preprocessing and classifier coefficients, per-image probabilities, wall time, and peak RSS. Independently replay all saved probabilities from exported coefficients. Compare each candidate against B0 on paired validation photographs. Use 2,000 fixed-seed identity-cluster bootstrap draws over the 20 identities (three photographs each), and show all case-wise intervals without interpreting an unadjusted interval as family-wise confirmation. Do not score historical test or reserve photographs.

Only advance a case to the seed-one robustness trace if its seed-zero validation accuracy is at least 10 percentage points above B0 and the paired identity-cluster 95% lower bound is positive. If more than one qualifies, advance the single case with the largest seed-zero paired improvement, breaking ties by lower validation log loss and then the earlier fixed schedule ID. Refit that exact case on the seed-one trace with unchanged settings and compare it with the existing seed-one B0. Seed one shares the same images, so it is a perturbation robustness check, **not** independent confirmation. A main 100-identity study or a memory claim requires a separate frozen protocol and evidence.

Publish the protocol before the first new fit; post every outcome and failure to this issue. Keep working attempts under `FLYSTATE_HOME`, archive the new attempt reports, coefficients, predictions, and metadata in a dated sibling research directory, use Git LFS for large NPZ payloads, and independently verify restoration from a fresh remote clone. Do not duplicate external images, downloaded brain files, or the already archived #50 source trace.

# Interpretation limits

If none passes the gate, document that readout choices in this limited screen did not rescue validation recognition, without claiming the connectome has no decodable signal. If one passes, the result remains exploratory until a new independent confirmation cohort and matched persistent/reset memory evaluation. Within-window voltage averages and spike counts were not saved in the source traces and need a separately preregistered measurement study; do not infer them from final-window values.
