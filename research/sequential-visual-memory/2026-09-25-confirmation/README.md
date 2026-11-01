# T35: single confirmation on untouched identities

Issue: [#70](https://github.com/akhmialeuski/fly-connectome-lab/issues/70). Protocol and run script: [protocol.md](protocol.md), [run.sh](run.sh), committed in `ce5fd5e` before any confirmation recording. Results and decisions: [results.md](results.md).

## Contents

- `snapshot/record/<persistent|reset|shuffled|shuffled-reset>/`: metadata, reports and checksum inventories of the four recordings of all 400 photographs of `configs/celeba-confirm.yaml`.
- `snapshot/evaluate/`: the single evaluation. `report.json` holds every case's held-out score, Wilson interval, binomial test, selected C and CV accuracies, and the six paired comparisons. `predictions.parquet` holds the prediction for every held-out photograph and case (3,720 rows).
- `snapshot/evaluate-model-export/`: a separate completed replay from clean code commit `faa63fd`. It preserves the fitted scaler, PCA and logistic arrays for all 31 cases in `models/<case>/weights.npz`, with `model.json` metadata and SHA-256 checks. The 31 NPZ files contain 187,011,920 bytes and are tracked by Git LFS. The original evaluation attempt is unchanged.
- `analysis/replay.py` recomputes, from the saved predictions alone, every held-out score, binomial p-value, paired difference and McNemar count. `provenance/numeric-replay.json` records its outcome: 31 cases, 6 comparisons, 0 mismatches.
- `analysis/verify_model_export.py` checks all model files, selected hyperparameters, scores and the byte-identical prediction table. With the registered external data and original recordings, it independently applies the saved arrays and verifies all 3,720 predictions. `provenance/model-export-replay.json` records that full check.

The four recordings' final-state arrays (about 100 MB each) are kept in the working data home. They are not in Git, because every reported number is replayable from the predictions and the arrays regenerate deterministically from the committed code.

## Reproduction

From a checkout of `ce5fd5e` with `FLYSTATE_HOME` set and the CelebA dataset registered:

```bash
research/sequential-visual-memory/2026-09-25-confirmation/run.sh
uv run python research/sequential-visual-memory/2026-09-25-confirmation/analysis/replay.py
uv run python research/sequential-visual-memory/2026-09-25-confirmation/analysis/verify_model_export.py
```

`run.sh` skips recordings that already exist. The evaluation attempt directory must not exist, because attempts are never overwritten.

To replay inference from the saved coefficients after restoring the external data and recordings, run:

```bash
uv run python research/sequential-visual-memory/2026-09-25-confirmation/analysis/verify_model_export.py --home "$FLYSTATE_HOME"
```
