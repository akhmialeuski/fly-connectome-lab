# Results: frozen-trace readout screen

The [preregistered protocol](protocol.md) for [issue #52](https://github.com/akhmialeuski/fly-connectome-lab/issues/52) was frozen at `f54d671` before fitting B1-B7. All seven new fits and the reused B0 reference completed; no failed or nonconverged fit was excluded. The source was the same native float32 noise-enabled seed-zero trace for all eight cases. It contains 280 training and 60 validation photographs from 20 identities, with 16 recorded sequential responses per photograph. The original historical test and reserve photographs were not newly scored.

| Case | Feature block and readout history | PCA cap | Input features | Selected C | Mean CV | Train top-1 | Validation top-1 | Validation log loss | Final iterations |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | Both, final observation | 60 | 2,628 | 0.01 | 5.71% | 81.07% | 3.33% (2/60) | 3.7166 | 56 |
| B1 | Voltage only, final observation | 60 | 1,314 | 1 | 6.07% | 100.00% | 6.67% (4/60) | 5.9709 | 150 |
| B2 | Spike trace only, final observation | 60 | 1,314 | 0.01 | 4.64% | 75.36% | 10.00% (6/60) | 3.6552 | 43 |
| B3 | Both, final observation | 20 | 2,628 | 0.01 | 5.71% | 28.21% | 1.67% (1/60) | 3.6631 | 48 |
| B4 | Both, final observation | 120 | 2,628 | 0.01 | 4.64% | 100.00% | 3.33% (2/60) | 3.8092 | 54 |
| B5 | Both, final observation | 240 | 2,628 | 10 | 5.36% | 100.00% | 3.33% (2/60) | 4.4711 | 34 |
| B6 | Both, final observation | None | 2,628 | 10 | 5.36% | 100.00% | 5.00% (3/60) | 4.3908 | 29 |
| B7 | Both, all 16 observations | 60 | 42,048 | 10 | 6.07% | 100.00% | 8.33% (5/60) | 6.0017 | 256 |

Uniform 20-way chance is 5%. These are exploratory development scores. Every classifier used the same five stratified training folds, fold-local scaling and PCA, C grid `[0.01, 0.1, 1, 10]`, and 50,000-iteration convergence budget. B5's PCA cap of 240 yielded 240 components in its final full-training fit; each 224-row training fold capped PCA at 223 components. B6 used scaling and regularization without PCA. B7 concatenated the **recorded** 16 responses, not inferred within-window activity.

The paired differences below are `case − B0`, in percentage points on the same 60 validation photographs. Intervals are 95% percentile intervals from 2,000 seed-zero bootstrap resamples over the 20 identities, with three paired photographs per identity. They are exploratory per-case intervals, not family-wise multiplicity-adjusted confirmation.

| Case | Paired top-1 difference | 95% identity-cluster interval |
| --- | ---: | ---: |
| B1 | +3.33 | [−3.33, +10.00] |
| B2 | +6.67 | [+1.67, +13.33] |
| B3 | −1.67 | [−6.67, +3.33] |
| B4 | 0.00 | [0.00, 0.00] |
| B5 | 0.00 | [−5.00, +5.00] |
| B6 | +1.67 | [−3.33, +8.33] |
| B7 | +5.00 | [−1.67, +11.67] |

The preregistered advancement gate required **both** at least +10 percentage points versus B0 and a positive lower identity-cluster interval on seed zero. No case met it. B2's interval lower bound was positive, but its observed gain was only +6.67 points; B7 did not meet either criterion. Thus no candidate is advanced to the seed-one robustness trace or a larger recognition experiment under this protocol. This screen does not identify a unique root cause of poor recognition. In particular, B1, B4, B5, B6, and B7 fit all training photographs but generalized poorly, consistent with overfitting or weak class-stable signal in this readout. B2's small validation lead does not establish spike-trace superiority after screening seven candidates on the same images.

The all-history result does not establish fly-network sequential memory: its linear readout receives an explicit concatenation of past observations, and the external spike filter itself carries state. A separate matched persistent/reset intervention is required for a memory claim. No fly-neuron or synaptic coefficient was trained in this study.

The machine-readable [analysis report](snapshot/analysis/report.json) contains the full C-grid CV scores, train/validation metrics, paired intervals, feature statistics, model metadata, timing, RSS, and gate. All 2,720 saved train/validation probability vectors were independently replayed from their exported scaler, PCA, and linear arrays; the largest absolute discrepancy was `2.22e-15`. Peak RSS among new fits was 0.948 GiB (B7), far below its 12 GiB limit; B7's fit elapsed 22.33 seconds, below its 30-minute limit. All source and new attempt checksum inventories verified. The eight new attempts in `snapshot/` are byte-identical to the original working attempts: 103 files, 32,236,289 bytes, canonical file-digest tree SHA-256 `8e6339ae463dc7c64fd20bd00ac35ae9763baac3e138a8603c5f2479c3068b00`. See [local archive verification](provenance/local-archive-verification.json). B0 and both native source traces are referenced from the prior [#50 archive](../2026-09-23-noise-recognition/README.md) to avoid duplicating them.

A fresh shallow GitHub clone of archive commit `b223f81bf0452ed64385cd24ae234c5db4e1b609` restored all seven Git LFS NPZ coefficient payloads, verified all eight attempt inventories, and independently produced the same 103-file digest tree. See [remote LFS verification](provenance/remote-lfs-verification.json). The fresh checkout remained clean after restoration.
