Parent: #39. Prerequisite: the cached diagnostic campaign #40 and its valid input-control retries. This is the explicit follow-up to the question whether 280 training examples are sufficient.

## Question and scope

Does increasing the number of independent training photographs per identity improve held-out recognition in the fixed fly representation, or only in its input controls? The original smoke task has 20 identities with 14 training photographs each (280 total); the main task has 100 identities with the same 14 photographs each (1,400 total). Comparing those tasks does not isolate training-set size.

Keep the fly dynamics, effective connectome, encoder, original sample membership, and noise realizations fixed. No specialized vision model or internal plasticity. Pixel and encoder readouts remain diagnostic controls. Do not borrow old test, validation, or reserved images for training.

## Preregistered smoke design

- Use the original smoke persistent configuration and 20 identities. Freeze the original 60 validation photographs across every comparison. The old test remains excluded.
- Use nested balanced training subsets of 2, 4, 8, and 14 photographs per identity: 40, 80, 160, and 280 total. Subset seeds are 0, 1, 2, 3, and 4. Derive independent per-label permutations from a named seed stream and sorted training sample IDs; prefixes define nested membership. Record selected membership and hashes before fitting. Restore original dataset order for fitting so the 14-photo endpoint exactly matches its existing anchor.
- Separate subset seed from the experiment seed. Do not change the cohort, encoder, trajectory, brain noise, rest state, or CV shuffle seed when changing subset membership. Never choose the best subset seed.
- Probe pixels/all observations, encoded input/all observations, and neural persistent/last observation/both feature blocks. Use the same scaler/PCA60/logistic procedure, fitted inside every training fold, C grid [0.01, 0.1, 1, 10], tolerance 1e-6, maximum 50,000 iterations, CV shuffle seed 0. Requested five folds become min(5, photographs per identity); record this and each fold's actual PCA dimension. This is the learning curve of the fixed fitting procedure, not a claim that representation dimension and fold size stay constant.
- Run each representation at 2, 4, and 8 photos for every subset seed, followed by its unique 14-photo endpoint. The 14-photo membership is identical for every subset seed, so fit it once per representation and explicitly reference that shared endpoint rather than counting five duplicate fits as independent evidence. Total: 48 unique fitted-probe attempts, subject to numerical failures.
- Precompute and save all membership records before scoring. Execute serially on CPU. A failed fit remains a failed attempt; do not remove C values, relax tolerance, or substitute zero accuracy. A new numerical failure is documented before choosing a follow-up.

## Measurements and decision

Preserve training/CV/validation top-1, top-5, log loss, selected C, model coefficients, convergence information, feature statistics, fold dimensions, per-image predictions, and resource usage. Report every seed and aggregate mean/range across the five subset realizations at 2/4/8 photos. The single shared full endpoint has no estimated between-subset variance.

Use paired validation predictions across nested subsets. Report identity-cluster bootstrap sensitivity with 2,000 resamples and a named fixed seed, retaining all three validation photographs of each sampled identity; label the intervals exploratory and conditional on this fixed validation cohort. Distinguish image/identity sampling uncertainty from training-subset variation. No multiplicity-adjusted superiority claim and no new untouched confirmation claim.

A rising input-control curve with a flat near-chance neural curve supports investigating neural signal transmission/noise next; it does not prove that more neural training data can never help. If the neural curve rises, quantify the observed gain before deciding whether a separately preregistered larger-per-identity cohort is worthwhile. CelebA has at most 35 source photographs per identity in the installed annotations; more than 14 training photos requires a new split protocol, not reuse of held-out photographs. Augmented views are not additional independent photographs.

## Implementation and preservation

Add explicit training-subset controls to diagnostics without changing existing defaults. Test nested membership, repeatability, order independence of selection, preservation of validation membership, independent seed streams, invalid support, CLI behavior, and replay of the full endpoint. Keep failed and successful attempts in a new dated sibling study with checksums and Git LFS coefficients, and verify remote restoration. Publish every attempt and aggregate interpretation here, cross-link #7/#14/#39, and document CLI usage in English.

Subsequent same-image/noise-seed and float32/float16 causal diagnostics from P1 remain separate requirements; this learning curve does not replace them. Main-cohort repetition or changed dynamics require a reported decision after this bounded smoke study.
