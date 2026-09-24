# Temporal response and population access pilot

This study follows [issue #54](https://github.com/akhmialeuski/fly-connectome-lab/issues/54). The committed `protocol.md`, `analysis-plan.md`, `cohort.json`, and `population-masks.json` define the inputs before the first real response run. It measures the unchanged fly simulation at selected times within a single image window. It does not train a classifier or claim sequential memory.

Use the same `FLYSTATE_HOME` that contains the validated CelebA source, downloaded MaleCNS brain files, and existing preprocessing cache. The archived [source-acquisition instructions](../2026-09-20-celeba-poc1/README.md) specify how to obtain the external inputs; the user-supplied JPEGs and connectome files are never copied into Git. The [results](results.md) describe the completed measurements and their limits. Run from the repository root after `uv sync --frozen`:

```bash
export FLYSTATE_HOME=/path/to/existing/flystate-home
study=research/sequential-visual-memory/2026-09-24-temporal-population
uv run flystate diagnose temporal configs/celeba-smoke.yaml \
  --output runs/diagnostics/2026-09-24-temporal-population/C0 \
  --case C0 \
  --cohort "$study/cohort.json" \
  --masks "$study/population-masks.json" \
  --membership research/sequential-visual-memory/2026-09-23-noise-recognition/membership.json \
  --json
```

Run C0-C6 in the frozen order, changing both `--case` and the final output directory component for each invocation. Every output directory must be new; a completed attempt cannot be overwritten. The command verifies the source configuration, exact training membership, original and aligned image hashes, graph file hashes, and selected neuron bodyIds before simulation. Failed attempts retain their manifest and checksum inventory. `--json` emits one result object on stdout and logs on stderr.

The first blank attempts omitted three checkpoints needed for ten-step recovery comparisons. See `correction-2026-09-24.md`; run C0R and C6R as new attempts with the same command pattern before calculating recovery contrasts. Keep C0/C6 as recorded, and verify exact equality at all overlapping checkpoints.

Each completed case stores `responses.npz` (numeric arrays readable with `numpy.load(..., allow_pickle=False)`), `sample-windows.json` (sample IDs, labels, window indices), `report.json`, `manifest.json`, effective `config.json`, environment metadata, and `checksums.sha256` under `$FLYSTATE_HOME/runs/diagnostics/2026-09-24-temporal-population/<case>/`. New attempts appear automatically in the viewer's Diagnostics menu when it is pointed at the same data home.

After all nine attempts (C0-C6, C0R, and C6R) complete, run the immutable paired analysis:

```bash
uv run flystate diagnose temporal-analyze configs/celeba-smoke.yaml \
  --source runs/diagnostics/2026-09-24-temporal-population \
  --output runs/diagnostics/2026-09-24-temporal-population/analysis \
  --json
```

The analysis refuses altered or missing case files and verifies the corrected blank overlap and paired noise streams. It writes `per-unit.parquet` with every training-image/window/checkpoint/population contrast and a `report.json` with descriptive summaries and the predeclared advancement gate. It does not train or evaluate an identity classifier.

The completed `snapshot/` contains the nine response attempts and the analysis as byte-identical copies of the working attempts. `provenance/source-copy-verification.json` records SHA-256 for all 69 archived files. Restore Git LFS payloads after cloning, then check the snapshot against the recorded inventory:

```bash
git lfs pull --include='research/sequential-visual-memory/2026-09-24-temporal-population/snapshot/**' --exclude=''
uv run python -c 'import hashlib, json, pathlib; p=pathlib.Path("research/sequential-visual-memory/2026-09-24-temporal-population"); v=json.loads((p/"provenance/source-copy-verification.json").read_text()); assert all(hashlib.sha256((p/"snapshot"/x["path"]).read_bytes()).hexdigest()==x["sha256"] for x in v["files"]); print(v["file_count"], "files verified")'
```

The archive does not contain source photographs, preprocessed image arrays, downloaded brain files, or duplicated #50 neural traces. The frozen external input hashes and exact training IDs are in `cohort.json` and `population-masks.json`.

The [fresh-clone record](provenance/remote-lfs-verification.json) confirms that all nine LFS arrays and all 69 archive files restored from GitHub with their recorded hashes, and that the 4,512-row analysis remained readable.
