# T31: matched original-fly descending-state identity access

Owning issue [#62](https://github.com/akhmialeuski/fly-connectome-lab/issues/62); stacked implementation PR [#63](https://github.com/akhmialeuski/fly-connectome-lab/pull/63). Read the [precommitted protocol](protocol.md), [source/analysis schedule](schedule.json), and [results](results.md) before drawing conclusions. T31 follows the passing T30 input-measurement gate and keeps T29's separate negative gate intact. The T29/T30 80 query photographs were already inspected and remain development evidence only.

`snapshot/A0` is a byte-for-byte copy of the completed working attempt. It contains the original manifest, config, environment, SHA-256 inventory, 2,400 neural OOF prediction rows, 60 fold records, 12 candidate summaries, 600 paired input/neural rows, 30,000 identity-bootstrap rows, three exported scaler/PCA/logistic models, descriptive fit/query predictions, and the report. Three learned `model/weights.npz` files use Git LFS. [Source-copy verification](provenance/source-copy-verification.json) lists every archived byte. [Independent numeric replay](provenance/numeric-replay.json) refits all source-trace fold/C combinations, reconstructs all selected-model probabilities from NPZ arrays, and checks the paired bootstrap and gate. [Fresh-clone LFS verification](provenance/remote-lfs-verification.json) at commit `603322f` restored all three models and checked 25 hashes, the attempt inventory, 13 Parquet row counts, and the passing gate. The three large **source neural traces are not duplicated** here; they were already preserved under [the 2026-09-23 noise-recognition study](../2026-09-23-noise-recognition/README.md), and `schedule.json` freezes each source array/manifest/sample hash. No external photograph or downloaded graph file is stored in this study.

For a reproduction, obtain and validate the original CelebA dataset and MaleCNS graph through the root README and restore the archived source-trace attempts to the exact `FLYSTATE_HOME/runs/diagnostics/2026-09-23-noise-recognition/traces/{seed0,seed1,off}` paths. Set `FLYSTATE_HOME` to that prepared local home. The exact original training cohort and image hashes are in [T29](../2026-09-24-input-access/cohort.json); the matching five folds are in its [schedule](../2026-09-24-input-access/schedule.json), and the frozen input control is in [T30](../2026-09-24-input-loss-selection/README.md). Source SHA-256 checks reject any different image, trace, encoder, or graph identity.

To create a *new* attempt without modifying A0:

```bash
uv sync --frozen
FLYSTATE_HOME=/path/to/flystate-home uv run flystate diagnose matched-neural configs/celeba-smoke.yaml \
  --output runs/diagnostics/2026-09-24-matched-neural-access/reproduction-01 \
  --cohort research/sequential-visual-memory/2026-09-24-input-access/cohort.json \
  --parent-schedule research/sequential-visual-memory/2026-09-24-input-access/schedule.json \
  --input-schedule research/sequential-visual-memory/2026-09-24-input-loss-selection/schedule.json \
  --input-attempt research/sequential-visual-memory/2026-09-24-input-loss-selection/snapshot/A0 \
  --membership research/sequential-visual-memory/2026-09-23-noise-recognition/membership.json \
  --schedule research/sequential-visual-memory/2026-09-24-matched-neural-access/schedule.json \
  --json
```

The command emits one JSON object on stdout; errors and logs go to stderr. Working attempts stay under `FLYSTATE_HOME`. The browser viewer discovers this attempt from its generic diagnostics catalog, without experiment-ID code changes.
