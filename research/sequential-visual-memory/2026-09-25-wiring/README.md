# T37: wiring specificity at matched operating points

Issue: [#74](https://github.com/akhmialeuski/fly-connectome-lab/issues/74). The protocol and run script, [protocol.md](protocol.md) and [run.sh](run.sh), were committed in `5e8ee78` before any recording. The Phase A selection [selection.json](selection.json) was committed in `bedd1cf` before Phase B. Results and decisions: [results.md](results.md).

## Contents

- `snapshot/phase-a/`: the 15 development recordings (`record/<graph>-a<alpha>/s0/`), the development evaluation (`evaluate/s0/`), the selection attempt (`select/`) and the command logs (`logs/`). The failed first evaluation attempt, which stopped on dotted case names, survives as `logs/evaluate-s0-failed-dotted-names.*`.
- `snapshot/phase-b/`: the 120 recordings (`record/<graph>/s<seed>/`), ten cohort evaluations (`evaluate/s<seed>/`), the pooled analysis (`analyze/`) and the logs. The failed s8 evaluation, stopped by SciPy's L-BFGS evaluation cap, is preserved in `evaluate-failed/s8-lbfgs-evaluation-cap/` with its log.
- Every attempt keeps its manifest, environment, config, report and `checksums.sha256`. Every evaluation keeps `predictions.parquet`, one held-out prediction per photograph and case, and `models/<case>/weights.npz` with the exact affine readout (`weights`, `bias`, `classes`) and its `model.json`. The `weights.npz` files are tracked by Git LFS.
- `analysis/verify.py` checks every archived attempt against its own inventory. It recomputes every held-out score, the Phase A selection, and every pooled accuracy, contrast, McNemar count and two-level bootstrap interval from the predictions. With `--home` it also replays every prediction from the affine readouts on the recorded states. `provenance/verify-with-states.json` records that full run: 149 attempts, 1,317 files, 32,400 replayed predictions.
- `inventory.sha256` lists every file of this study.

The recorded final states (`responses.npz`, float32, about 47 MB per recording) are not archived. Their SHA-256 values remain in each recording's inventory and report. They regenerate deterministically from the committed code, and the propagation kernel's result does not depend on the thread count.

## Reproduction

From a checkout of `bedd1cf` (recordings) or `e5b812c` (evaluation fix), with `FLYSTATE_HOME` set and the CelebA dataset registered:

```bash
research/sequential-visual-memory/2026-09-25-wiring/run.sh phase-a
research/sequential-visual-memory/2026-09-25-wiring/run.sh phase-b
cd research/sequential-visual-memory/2026-09-25-wiring
uv run python analysis/verify.py                          # archive only
uv run python analysis/verify.py --home "$FLYSTATE_HOME"  # also replay from the recorded states
```

`run.sh` skips completed recordings and evaluations. Attempts are never overwritten.
