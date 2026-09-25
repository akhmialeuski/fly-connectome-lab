# T32: drive amplitude, noise and population access in the spiking MaleCNS model

Issue: [#64](https://github.com/akhmialeuski/fly-connectome-lab/issues/64). Protocol: [protocol.md](protocol.md), frozen with the code in commit `dfc958d` before the formal sweep. Results and decisions: [results.md](results.md).

## Contents

- `snapshot/record/<condition>/`: every recording attempt's manifest, effective configuration, environment, report (mean spikes per window per population, sample order) and `checksums.sha256`. Conditions are `sN-on` and `sN-off` for encoder amplitude multiplier N with and without episode noise, plus the seed-1 replication `s16-on-seed1`.
- `snapshot/decode/<condition>/`: every decode attempt's report (all OOF metrics, selected C, candidates, failures) and `oof-predictions.parquet` with the selected-C out-of-fold probabilities of all 200 fit photographs.
- `analysis/`: the scripts that produced the preliminary physiology, spectral and Kenyon-cell numbers quoted in the issue, and the table generator used for the results comment.

The recordings' `responses.npz` spike-count arrays (about 1.5 GB in total) were deleted after decoding to protect the host drive. Each file's SHA-256 remains in its attempt's `checksums.sha256`. They are exactly regenerable from the committed code: two independent recorder implementations produced bit-identical arrays, and the recorder is deterministic for a fixed configuration, seed and thread count.

## Reproduction

From the repository root at commit `dfc958d`, with `FLYSTATE_HOME` pointing at a data home that contains the brain files and the validated CelebA dataset:

```bash
R=research/sequential-visual-memory
COHORT=(--cohort $R/2026-09-24-input-access/cohort.json
        --parent-schedule $R/2026-09-24-input-access/schedule.json
        --membership $R/2026-09-23-noise-recognition/membership.json)
uv run flystate diagnose drive-record configs/celeba-smoke.yaml \
  --output runs/diagnostics/2026-09-25-drive-sweep/record/s16-on \
  --amplitude-scale 16 --noise --episode-seed 0 --workers 3 "${COHORT[@]}" --json
uv run flystate diagnose drive-decode configs/celeba-smoke.yaml \
  runs/diagnostics/2026-09-25-drive-sweep/record/s16-on \
  --output runs/diagnostics/2026-09-25-drive-sweep/decode/s16-on "${COHORT[@]}" \
  --selection-schedule $R/2026-09-24-input-loss-selection/schedule.json --json
```

Replace `s16-on`, `--amplitude-scale` and `--noise/--no-noise` for the other conditions. No validation, historical test, reserve or previously inspected query photograph is read by either command.
