# T27: temporal response and population access

Owning issue: [#54](https://github.com/akhmialeuski/fly-connectome-lab/issues/54). Parent roadmap: [#39](https://github.com/akhmialeuski/fly-connectome-lab/issues/39). This protocol is frozen before the first response simulation. The study is a fixed-connectome physiology diagnostic, not an identity-recognition or sequential-memory result.

## Fixed sources and cohort

Use the exact 20-identity training membership from the archived #50 study (membership SHA-256 `5af10356b3ff565bf515144983758763fb8b346737520c50647885cdf88a92d8`). The eight chosen training photographs, each source JPEG SHA-256, and each aligned RGB array SHA-256 are in `cohort.json`. The deterministic rule is two smallest `stable_int(sample_id)` ranks within each of labels 0, 5, 10, and 15. Only raster windows 0, 7, and 15 are measured. These are separate, rest-state-initialized presentations; no sequential state carries from one window to another. Do not inspect validation, historical test, or reserve photographs for this pilot. Never copy photographs into the study or a run directory.

Reuse the original `configs/celeba-smoke.yaml` settings except the declared stimulus/noise interventions below: episode and encoder seeds zero, 32 × 32 RGB windows from aligned 128 × 128 images, fixed sparse projection into `visual_projection`, original amplitude 0.05, 25 warmup steps, `dt=0.020 s`, original 1.2 Hz/0.22 episode noise, and `sensory_input=false`. Use four Numba threads and one BLAS thread. The exact source files have SHA-256 `cc9bd1ecd00bd703a6fa648bc6ad145c93c7c1ee53debdcc9ce0d1f4305e6aca` (`brain.npz`) and `c29919aa44069a271b1ee978abe05fa9bf6e45e4ba3e436e92b624ef1b5be40c` (`weights.npz`). The released graph has 25,582,938 stored edges; the existing sensory-target mask leaves 25,088,107 effective edges. No edge weight or neuron equation is changed.

## Frozen observation masks

Observe four disjoint superclasses: `visual_projection` (9,201 neurons), `ol_intrinsic` (89,403), `cb_intrinsic` (32,164), and `descending_neuron` (1,314). Observe exactly 1,314 cells from each. The first three masks select the 1,314 lowest `(stable_int(str(bodyId)), bodyId)` ranks; the descending mask includes every member. `population-masks.json` records sorted neuron indices, bodyIds, canonical-JSON SHA-256 for both arrays, source file hashes, and internal/inbound/outbound edge counts in the effective graph. Superclass names describe annotation groups; their order does not assert a feedforward anatomical hierarchy. Equal feature count controls measurement dimension, but does not equalize connectivity or firing rate.

## Stimulus schedule

Use the same warmed rest state and the same per-`sample_id|window` random stream for each paired noise-enabled condition. Replay every condition independently. The encoder's output for an image window is constant over its stimulus steps. Scaling the resulting float32 current is the only input intervention. Blank means no encoder current; it does not mean zero tonic or synaptic activity. The original equation remains a per-step voltage kick, so two times as many steps at the same kick also deliver two times the aggregate kick. All durations below use the unchanged 20 ms step.

| Case | Stimulus | Stimulus steps | Subsequent blank steps | Noise | Aggregate kick relative to C1 |
| --- | --- | ---: | ---: | --- | ---: |
| C0 | Blank | 20 | 10 | On | 0 |
| C1 | Original current | 10 | 10 | On | 1 |
| C2 | Original current | 20 | 10 | On | 2 |
| C3 | Half original current | 20 | 10 | On | 1 |
| C4 | Double original current | 10 | 10 | On | 2 |
| C5 | Original current | 10 | 10 | Off | 1 |
| C6 | Blank | 20 | 10 | Off | 0 |

C0 provides time-matched blank values through step 30 for C1-C4. C6 is the noise-off blank reference for C5. For the noise-enabled cases, the random stream must have the same draws at shared physical steps; do not compare different random-number positions. Recovery is ten blank steps after each nonblank stimulus. Record physical time since stimulus onset and since offset. Preserve all seven attempts, including failed or stopped attempts.

Record the rest state at step 0; stimulus checkpoints 1, 2, 5, 10, and 20 where present; and recovery checkpoints 1, 2, 5, and 10. At each checkpoint retain per-sample/per-population float32 voltage vectors and cumulative integer spike counts, plus total spikes, active-neuron count, episode-noise kick count, and finite-state checks. Derive evoked-minus-time-matched-blank vectors, population mean/variance, silent fractions, near-threshold fractions, and between-image variation without discarding the raw numeric evidence. The threshold for any descriptive `near-threshold` bin must be declared in the analysis code and report; do not tune it to the response. Increased firing alone is not evidence of preserved identity information.

## Parity, resources, and decision

Before interpretation, compare C1 terminal descending-neuron voltage and exponential spike trace with an independent call through the existing `EpisodeBrain.run` path from the same rest state, current, and noise stream. Mismatch invalidates the instrumented attempt. Verify that C0 and C6 have no encoder input and that C5/C6 inject no episode noise. Record each attempt's effective configuration, input/mask/source hashes, seeds, package versions, Numba threads, wall time, peak RSS, and failures. Stop the pilot and record a resource failure after 60 minutes total or at 12 GiB peak RSS; do not silently reduce the fixed sample/condition set.

Only after the pilot is valid, freeze a separate expansion protocol for up to two training images per 20 identities. That later analysis may compare same-identity versus different-identity responses and use a simple training-only held-out diagnostic decoder. It must not choose populations, windows, amplitude, or duration by looking at a validation score. These stage-0 observations cannot establish a causal recognition bottleneck or a memory benefit. Report measured effects and remaining uncertainty in issue #54, archive the completed numeric evidence in this dated sibling study, and verify the archive from a fresh remote clone.
