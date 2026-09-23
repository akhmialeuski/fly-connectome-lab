# Isolated episode-noise recognition — 2026-09-23

This sibling study follows the [preregistered protocol](protocol.md) in
[issue #50](https://github.com/akhmialeuski/fly-connectome-lab/issues/50).
The exact 280 training and 60 validation sample IDs are in [membership.json](membership.json);
the three trace and six fit destinations are frozen in [schedule.json](schedule.json).
The completed comparison and its limits are in [results.md](results.md).
The original 20-identity persistent configuration, real MaleCNS files, and CelebA
source are required under `FLYSTATE_HOME`. Acquisition and fingerprint details are
documented in the preceding study directories. No source photograph, transformed
image, connectome download, or dataset archive is copied into this directory.

The baseline trace must replay the original cached float16 development rows
exactly before any intervention is interpreted. Every noise condition uses the
same noise-enabled warmed state and all 16 image patches. The readout fits use
only the final observation, both original neural blocks, 280 training images,
fold-local PCA capped at 60, and the original regularized logistic model.
Historical test and reserve images are excluded from new simulation and scoring.

From this repository, set `FLYSTATE_HOME` to the original data home and use the
same effective configuration as the original smoke run. Each command creates a
fresh immutable directory. The three trace commands must run in this order:

```bash
flystate diagnose noise-trace "$FLYSTATE_HOME/runs/20260920-071344-celeba-smoke-9406ee/config.yaml" \
  --membership research/sequential-visual-memory/2026-09-23-noise-recognition/membership.json \
  --output runs/diagnostics/2026-09-23-noise-recognition/traces/seed0 \
  --episode-seed 0 --json
flystate diagnose noise-trace "$FLYSTATE_HOME/runs/20260920-071344-celeba-smoke-9406ee/config.yaml" \
  --membership research/sequential-visual-memory/2026-09-23-noise-recognition/membership.json \
  --output runs/diagnostics/2026-09-23-noise-recognition/traces/seed1 \
  --episode-seed 1 --json
flystate diagnose noise-trace "$FLYSTATE_HOME/runs/20260920-071344-celeba-smoke-9406ee/config.yaml" \
  --membership research/sequential-visual-memory/2026-09-23-noise-recognition/membership.json \
  --output runs/diagnostics/2026-09-23-noise-recognition/traces/off \
  --episode-seed 0 --noise-disabled --json
```

For every trace, fit `float32` and then `float16` with `flystate diagnose run`:

```bash
flystate diagnose run "$FLYSTATE_HOME/runs/20260920-071344-celeba-smoke-9406ee/config.yaml" \
  --output runs/diagnostics/2026-09-23-noise-recognition/probes/seed0-float32 \
  --trace-source runs/diagnostics/2026-09-23-noise-recognition/traces/seed0 \
  --trace-precision float32 --train-per-class 14 --max-iterations 50000 --json
```

Repeat the final command with every `case-precision` pair in `schedule.json`,
changing only `--output`, `--trace-source`, and `--trace-precision`.
The analyzer verifies each attempt, requires exact seed-zero float16 coefficient
and prediction replay against the earlier full-data endpoint, and uses 2,000
seeded identity-cluster bootstrap draws for paired exploratory validation
differences:

```bash
flystate diagnose noise-analyze "$FLYSTATE_HOME/runs/20260920-071344-celeba-smoke-9406ee/config.yaml" \
  --schedule research/sequential-visual-memory/2026-09-23-noise-recognition/schedule.json \
  --output runs/diagnostics/2026-09-23-noise-recognition/analysis --json
```

The ten completed immutable attempts are preserved in `snapshot/`, including
all native traces, learned scaler/PCA/classifier arrays, predictions, timing,
environment metadata, and checksums. Git LFS stores all generated NPZ arrays:
run `git lfs pull` before checking the snapshot inventories. Local byte-for-byte
copy and independent numerical-replay records are in `provenance/`. The study's
validation comparison is exploratory and does not establish sequential memory.
All 12 NPZ payloads and 116 snapshot files were also restored from a fresh
GitHub clone and checked independently; see
[remote LFS verification](provenance/remote-lfs-verification.json).
