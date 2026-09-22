## Completed numerical follow-up and valid input controls

Scientific execution commit: `f821470ac263a584eb2a42e05108030d7fbb2630`. The numerical protocol was registered in [comment 5763354124](https://github.com/akhmialeuski/fly-connectome-lab/issues/40#issuecomment-5763354124); full retries were registered before scoring in [comment 5769606778](https://github.com/akhmialeuski/fly-connectome-lab/issues/40#issuecomment-5769606778).

All four original failures reproduce as iteration-limit stops at 5,000. Increasing only the iteration cap converges on the same first failed fold and C in 5,411–10,396 iterations; every final gradient infinity norm is below 1e-6. Independent replay of all eight exported fold parameter sets reproduces objective values within 1.12e-16 and gradient norms within 1.93e-15. These fold models are numerical evidence, not selected recognition models.

All four complete five-fold C-grid retries then converge under the common 50,000-iteration cap, unchanged tolerance 1e-6, original fold-local scaler/PCA60, and seed 0. There is no discarded C or relaxed accuracy criterion.

| Fresh attempt | Training correct | Validation correct | Validation top-1 | Validation log loss | Selected C | Best training CV |
| --- | --- | --- | --- | --- | --- | --- |
| retry50000-smoke-02-pixels-persistent-all-both-pca60-true | 280/280 | 31/60 | 51.67% | 3.552894 | 1.0 | 47.14% |
| retry50000-smoke-04-encoded-persistent-all-both-pca60-true | 280/280 | 31/60 | 51.67% | 2.299052 | 0.01 | 44.29% |
| retry50000-main-01-pixels-persistent-all-both-pca60-true | 1400/1400 | 112/300 | 37.33% | 3.960701 | 0.01 | 36.36% |
| retry50000-main-02-encoded-persistent-all-both-pca60-true | 1400/1400 | 107/300 | 35.67% | 4.312019 | 0.01 | 33.93% |

### Interpretation

- Identity information is measurably available in the original pixels and in the fixed encoded currents. Smoke: both full-history controls score 31/60 (51.67%), versus the frozen persistent-last neural anchor at 4/60 (6.67%) and 5% chance. Main: pixels score 112/300 (37.33%) and encoded currents 107/300 (35.67%), versus neural persistent-last at 1/300 (0.33%), neural reset-all at 2/300 (0.67%), and 1% chance.
- The cached neural PCA/feature/history screening did not pass its preregistered promotion gate. This localizes the practical failure to the measured neural representation/readout portion of the current pipeline, rather than complete absence of information in the input. It does not identify a specific biological layer, prove the encoder optimal, or distinguish noise, population access, timing, quantization, and state/readout filtering as causes.
- The original optimizer failures were numerical failures, not zero recognition. The historical pixel scores used a different tolerance (1e-4); retain them as historical measurements, not as replacements for these matched 1e-6 controls. Numerical stopping can affect CV selection and held-out scores, so report the actual repeated controls.
- Training accuracy alone remains inadequate evidence: every new input control fits training perfectly; the smoke neural model also memorizes training without established useful generalization. No useful recognition or sequential-memory advantage has been established for the fly model.
- Every score above is exploratory validation. No old final-test or reserved-photo predictions were scored. The independent confirmation limitation from the cohort audit remains.

### Next decisions

Proceed with the fixed-cohort learning curve in #42 to address the explicit training-size question. Its 48 unique attempts and memberships are frozen before fitting. Both cohorts previously had only 14 training photos per identity. Then continue the separate same-image/noise-seed, direct float32/float16, and signal-delivery diagnostics required by #39. Do not jump from this readout result to arbitrary equation changes or claim that additional data or constrained plasticity cannot help.

### Evidence and engineering checks

The dated diagnostic study retains 26 full-probe attempts (22 completed models and four original failures), two cohort audits, and four numerical diagnoses with eight exported fold parameter sets. All per-attempt reports were posted to #40. Model coefficients, predictions, failures, provenance, and checksum inventories are being appended to `research/sequential-visual-memory/2026-09-21-fly-only-diagnostics/`; remote verification is reported separately after publication.

The numerical implementation passed 313 tests with two optional skips, static checks, pre-commit, and source/wheel builds; its CI check passed. During the independent verification script, an incorrect two-value unpack of the three-value objective/gradient API caused a verifier-only exception. The local script was corrected to combine coefficient and intercept gradient norms, and all eight fold models then replayed successfully. Both verifier logs are retained. This did not modify or rerun a scientific fit.
