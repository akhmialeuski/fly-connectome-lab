# T36: standard runs of the graded MaleCNS model

Issue: [#72](https://github.com/akhmialeuski/fly-connectome-lab/issues/72). Protocol: [protocol.md](protocol.md). Run script: [run.sh](run.sh). Results: [results.md](results.md).

## Contents

- `snapshot/runs/<run_id>/`: the three complete run directories (configuration, manifest, environment, per-observation validation metrics, readout weights, and the test evaluation with predictions, confusion and timings), in the same format as the POC 1 runs.
- `snapshot/runs/reports/`: the two paired comparison reports.
- `snapshot/cache/features/<key>/`: the persistent and reset trace caches (400 episodes × 16 observations × 1,314 descending neurons, float16). The reset-concat run reads the reset cache.
- `provenance/acquisition/t36/`: stdout JSON and logs of every pipeline stage.

## Restore for the viewer

The snapshot mirrors a `FLYSTATE_HOME`. Copy it into a new home, add the brain files and the registered CelebA dataset as described in the POC 1 study README, and start the viewer. The three runs then appear under **Experiments** with the episode inspector.

```bash
archive=research/sequential-visual-memory/2026-09-25-rate-runs
export FLYSTATE_HOME="$HOME/data/flystate-restored-t36"
mkdir "$FLYSTATE_HOME" && cp -a "$archive/snapshot/." "$FLYSTATE_HOME/"
uv run flystate serve
```

Regenerating everything from the committed code runs `run.sh` in a fresh home.
