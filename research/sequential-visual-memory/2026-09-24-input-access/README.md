# T29: training-only input access

Owning issue [#58](https://github.com/akhmialeuski/fly-connectome-lab/issues/58); stacked implementation PR [#59](https://github.com/akhmialeuski/fly-connectome-lab/pull/59). Read the [frozen protocol](protocol.md), [results](results.md), [exact cohort](cohort.json), and [exact CV schedule](schedule.json) before interpreting scores. This dated study is a sibling of T27 and T28; its fit/query photographs are drawn only from the already selected original training split and are not new confirmation data.

`snapshot/A0` is the immutable failed prefit protocol-validation attempt from commit `309a3b8`. `snapshot/A1` is the immutable completed attempt from corrective commit `e69ffe7`. Each has its original manifest, environment, config, and SHA-256 inventory. A1 also has one exported scaler/PCA/logistic model per input representation, fit/query predictions, metrics, provenance, and a report. The two `model/weights.npz` files use Git LFS. [Source-copy verification](provenance/source-copy-verification.json) records all 19 exact copied files and hashes. [Numeric replay](provenance/numeric-replay.json) reconstructs all 560 stored predictions from the exported numeric coefficients and recomputed external training-image features. The attempt contains no source JPEG, aligned image, raster window, encoded-current feature matrix, or connectome file.

The original CelebA photographs and downloaded MaleCNS graph are deliberately **not** in Git. Register and validate the same source dataset and download the graph according to the root README, then set `FLYSTATE_HOME` to that prepared local home. Exact source JPEG/aligned-RGB hashes are in `cohort.json`; the original 20-identity split membership is in [the archived #50 membership](../2026-09-23-noise-recognition/membership.json). The original configuration is [`configs/celeba-smoke.yaml`](../../../configs/celeba-smoke.yaml). Source and cache digests must match before a reproduced attempt can run.

To create a *new* attempt rather than modify A0/A1:

```bash
uv sync --frozen
FLYSTATE_HOME=/path/to/flystate-home uv run flystate diagnose input-access configs/celeba-smoke.yaml \
  --output runs/diagnostics/2026-09-24-input-access/reproduction-01 \
  --cohort research/sequential-visual-memory/2026-09-24-input-access/cohort.json \
  --schedule research/sequential-visual-memory/2026-09-24-input-access/schedule.json \
  --membership research/sequential-visual-memory/2026-09-23-noise-recognition/membership.json \
  --json
```

The command emits one JSON object on stdout; errors and logs go to stderr. Keep the historical `snapshot/` files unchanged. The browser viewer discovers this archived study through its generic research catalog when the branch is checked out, with no hard-coded experiment IDs.
