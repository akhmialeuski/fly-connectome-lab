# T30 result: fit-only input measurement gate passed

The [protocol](protocol.md) and exact source/schedule hashes were pushed at `60817ab` before T30 fitting. A0 ran from clean source `418cd4d` and retained the original T29 200 fit photographs, 80 previously inspected training queries, all 16 raster windows, sparse encoder, five stratified folds, and scaler/PCA(60)/logistic readout family. The only methodological change was to choose C by pooled **fit-only out-of-fold log loss**, rather than by fit-fold accuracy. Both representations selected C=0.01 before scoring any prior query.

| Input | C | Fit-only OOF top-1 | Fit-only OOF top-5 | Fit-only OOF log loss |
| --- | ---: | ---: | ---: | ---: |
| Raster pixels | **0.01** | **88/200 (44.0%)** | 75.5% | **2.50673** |
| Raster pixels | 0.1 | 46.0% | 75.5% | 3.04276 |
| Raster pixels | 1 | 45.5% | 75.5% | 3.70214 |
| Raster pixels | 10 | 46.0% | 74.5% | 4.45483 |
| Sparse encoded current | **0.01** | **92/200 (46.0%)** | 75.5% | **2.37658** |
| Sparse encoded current | 0.1 | 46.0% | 76.0% | 2.83736 |
| Sparse encoded current | 1 | 46.5% | 76.5% | 3.41726 |
| Sparse encoded current | 10 | 47.0% | 76.0% | 4.05955 |

The frozen primary gate required **both** selected controls to reach at least 25% OOF top-1 and OOF log loss strictly below `ln(20) = 2.99573`. **Both passed.** These are fit-cohort cross-validated measurements, not untouched confirmation; C was itself chosen from the four OOF losses, so selected OOF performance is exploratory. The complete 1,600 OOF probability rows and all 40 fold metrics are archived, rather than only favorable candidates.

After both C choices and the gate were fixed, models were refit on all 200 fit photographs. On the *same 80 T29 training queries already inspected*, pixels scored 35/80 (43.75%) top-1 and log loss 2.27142, versus T29's 38/80 and 2.77046; encoded current scored 36/80 (45.0%) and log loss 2.29866, versus T29's 37/80 and 4.15056. The current model's much lower query loss at nearly unchanged accuracy supports the diagnosis that accuracy-only C selection made the T29 current probabilities poorly calibrated. It does not establish C as the sole root cause of low fly-network recognition, and the reused 80 queries cannot be treated as fresh confirmation or override the OOF decision. The T29 negative gate remains valid under its own frozen protocol.

A0 completed in 43.70 seconds of internal wall time with 1,117,290,496 bytes peak RSS, below the 60-minute/12-GiB limits. All source, alignment, cohort, fold, encoder, brain-file, feature-array, coefficient, and attempt-inventory hashes verified. [Independent numeric replay](provenance/numeric-replay.json) used a separate sklearn Pipeline to reproduce all 1,600 OOF probability rows exactly (maximum absolute difference 0), all 40 fold metrics and iterations, all eight candidate summaries, selected C values, and the passing gate. SciPy inference from the exported NPZ coefficients reproduced final fit/query probabilities within `7.3e-16`. An initial **verification-script** assertion failed because it used `PCA.fit()` then `transform()` rather than the source pipeline's `fit_transform()`; the corrected exact-order replay passed, and the immutable experiment attempt was never changed. [Source-copy verification](provenance/source-copy-verification.json) covers 18 files and 56,485,249 original bytes, including two Git LFS models; no images or downloaded graph are archived. [Independent fresh-clone verification](provenance/remote-lfs-verification.json) restored the two LFS arrays at commit `5d607c4` and checked every file hash, the attempt inventory, all eight Parquet row counts, and the passing gate.

The next step is a separately frozen, **matched** fly-response recognition comparison on the same 200/80 training-only roles and all 16 windows, with the same OOF log-loss-selected readout budget and explicit original-noise controls. Only that comparison can begin to localize signal loss relative to the fly network. It must not select masks/windows or neuron equations from the already viewed query scores; no dynamics/plasticity change is justified by this input-only result. Sequential memory remains a separate question requiring a later persistent/reset intervention.
