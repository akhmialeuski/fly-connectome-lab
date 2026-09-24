# T30: fit-only input log-loss selection

Owning issue [#60](https://github.com/akhmialeuski/fly-connectome-lab/issues/60); implementation PR [#61](https://github.com/akhmialeuski/fly-connectome-lab/pull/61). The [protocol](protocol.md) and [fixed schedule](schedule.json) were committed before T30 fitting. [Results](results.md) distinguish the primary 200-row fit-only OOF gate from descriptive reuse of the 80 T29 queries, which were already inspected. This is a separate dated sibling of [T29](../2026-09-24-input-access/README.md), not a revision of its negative gate.

`snapshot/A0` is an exact copy of the completed working attempt, including manifest, environment/config, full SHA-256 inventory, 1,600 out-of-fold probability rows, 40 fold metrics, eight candidate summaries, two exported scaler/PCA/logistic coefficient sets, final fit/query predictions, and report. Both `model/weights.npz` files use Git LFS. [Source-copy verification](provenance/source-copy-verification.json) lists every original file and hash. [Independent numeric replay](provenance/numeric-replay.json) refits each frozen fold/C combination through sklearn and reconstructs both final models from their saved arrays; it confirms the gate and every stored probability. [Fresh-clone LFS verification](provenance/remote-lfs-verification.json) at archived commit `5d607c4` restored both models and checked all 18 file hashes, the attempt inventory, all eight Parquet row counts, and the passing gate. The archive contains no original photograph, aligned image, raster window, encoded-current feature matrix, or downloaded connectome file.

The original CelebA dataset and MaleCNS brain must be obtained and validated through the root README, subject to their original use terms. The exact external JPEG and aligned-image digests are in the [T29 cohort](../2026-09-24-input-access/cohort.json); its [schedule](../2026-09-24-input-access/schedule.json) fixes the same five fit folds. The original configuration is [`configs/celeba-smoke.yaml`](../../../configs/celeba-smoke.yaml), and the original train/validation/test member IDs are in [the archived membership](../2026-09-23-noise-recognition/membership.json). Source digest checks prevent accidental reuse of different images, graph files, or encoder output.

To create a *new* attempt rather than change the archived one:

```bash
uv sync --frozen
FLYSTATE_HOME=/path/to/flystate-home uv run flystate diagnose input-loss-selection configs/celeba-smoke.yaml \
  --output runs/diagnostics/2026-09-24-input-loss-selection/reproduction-01 \
  --cohort research/sequential-visual-memory/2026-09-24-input-access/cohort.json \
  --parent-schedule research/sequential-visual-memory/2026-09-24-input-access/schedule.json \
  --membership research/sequential-visual-memory/2026-09-23-noise-recognition/membership.json \
  --baseline research/sequential-visual-memory/2026-09-24-input-access/snapshot/A1 \
  --schedule research/sequential-visual-memory/2026-09-24-input-loss-selection/schedule.json \
  --json
```

The command writes one JSON object to stdout and logs/errors to stderr. Working artifacts remain under `FLYSTATE_HOME`; the archived snapshot is immutable. The browser viewer discovers the new diagnostic from its generic catalog rather than a hard-coded experiment list.
