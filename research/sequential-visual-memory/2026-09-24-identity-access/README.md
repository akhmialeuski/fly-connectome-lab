# Training-only identity access in fixed fly responses

This study follows [issue #56](https://github.com/akhmialeuski/fly-connectome-lab/issues/56) and the completed [T27 signal-delivery pilot](../2026-09-24-temporal-population/results.md). The [protocol](protocol.md), [40-image cohort](cohort.json), and [machine-readable schedule](schedule.json) were frozen at commit `7477e5a` before any new neural response run. The completed [results](results.md) report the negative primary gate and its limits. All images come from the original training membership; the eight T27 pilot images recur, so these data are not independent confirmation. No source photograph or downloaded brain file is stored here.

Use the same `FLYSTATE_HOME` that contains the registered original CelebA source, the MaleCNS files, and the validated preprocessing cache. The [source-acquisition instructions](../2026-09-20-celeba-poc1/README.md) give the exact external input locations and hashes. From the repository root, after `uv sync --frozen`:

```bash
export FLYSTATE_HOME=/path/to/existing/flystate-home
study=research/sequential-visual-memory/2026-09-24-identity-access
for case in N0 N1 OFF; do
  uv run flystate diagnose identity configs/celeba-smoke.yaml \
    --output "runs/diagnostics/2026-09-24-identity-access/$case" \
    --case "$case" \
    --cohort "$study/cohort.json" \
    --masks research/sequential-visual-memory/2026-09-24-temporal-population/population-masks.json \
    --membership research/sequential-visual-memory/2026-09-23-noise-recognition/membership.json \
    --schedule "$study/schedule.json" \
    --json
done
```

Each command requires a fresh output directory. It verifies the frozen source, config, cohort, masks, schedule, and prior attempt order before simulation, then preserves a completed or failed immutable attempt with SHA-256 inventory. N0/N1 use identical image-independent episode-noise streams within each window and save exact per-step noise-index digests; OFF saves zero-noise controls. `responses.npz` and `blanks.npz` contain numeric neural states only and can be opened with `numpy.load(..., allow_pickle=False)`. `sample-windows.json` contains image IDs, labels, raster-window indices, input hashes, and input-current norms, not image pixels. The browser viewer discovers these attempts in Diagnostics from the same `FLYSTATE_HOME` without a run-name registry.

After all three response attempts have completed, create a fresh immutable analysis attempt:

```bash
uv run flystate diagnose identity-analyze configs/celeba-smoke.yaml \
  --source runs/diagnostics/2026-09-24-identity-access \
  --output runs/diagnostics/2026-09-24-identity-access/analysis \
  --cohort "$study/cohort.json" \
  --masks research/sequential-visual-memory/2026-09-24-temporal-population/population-masks.json \
  --membership research/sequential-visual-memory/2026-09-23-noise-recognition/membership.json \
  --schedule "$study/schedule.json" \
  --json
```

The analyzer verifies the three source inventories and exact per-step noise-index matching before computing the frozen primary identity feature. It recomputes the pixel and encoded-current controls from the external training images without storing their arrays. `pairs.parquet` retains every same-/different-identity squared distance for the primary and diagnostic views, `queries.parquet` retains every fixed one-nearest-neighbor decision, `cross-seed.parquet` retains per-image noise sensitivity, and `report.json` contains the complete permutation histograms, stage gate, and limitations. Failed analysis attempts remain in the working ledger and cannot be overwritten.

The completed `snapshot/` is a byte-for-byte copy of N0, N1, OFF, and the analysis: 32 files and 62,971,389 original bytes. The six response/blank NPZ payloads use Git LFS. [Source-copy verification](provenance/source-copy-verification.json) records every file's SHA-256. Restore them after cloning and verify the archive:

```bash
git lfs pull --include='research/sequential-visual-memory/2026-09-24-identity-access/snapshot/**' --exclude=''
uv run python -c 'import hashlib, json, pathlib; p=pathlib.Path("research/sequential-visual-memory/2026-09-24-identity-access"); v=json.loads((p/"provenance/source-copy-verification.json").read_text()); assert all(hashlib.sha256((p/"snapshot"/x["path"]).read_bytes()).hexdigest()==x["sha256"] for x in v["files"]); print(v["file_count"], "files verified")'
```

An [independent analysis replay](provenance/analysis-replay.json) recomputed every primary and input-control distance, all one-nearest-neighbor predictions, the three permutation histograms, and the negative gate using SciPy distances rather than the analyzer's distance loop.
