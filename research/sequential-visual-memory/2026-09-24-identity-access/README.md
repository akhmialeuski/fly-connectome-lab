# Training-only identity access in fixed fly responses

This study follows [issue #56](https://github.com/akhmialeuski/fly-connectome-lab/issues/56) and the completed [T27 signal-delivery pilot](../2026-09-24-temporal-population/results.md). The [protocol](protocol.md), [40-image cohort](cohort.json), and [machine-readable schedule](schedule.json) were frozen at commit `7477e5a` before any new neural response run. All images come from the original training membership; the eight T27 pilot images recur, so these data are not independent confirmation. No source photograph or downloaded brain file is stored here.

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
