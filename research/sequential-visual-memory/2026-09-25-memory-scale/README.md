# T34: memory time scale of the graded MaleCNS network and where the memory lives

Issue: [#66](https://github.com/akhmialeuski/fly-connectome-lab/issues/66). Protocol: [protocol.md](protocol.md), committed with the code in `299c1be` before any recording. The grid ran as the bounded-disk pipeline [run-pipeline.sh](run-pipeline.sh) from `73bb697`. The anchor `l0.25-dsame-persistent` ran from `299c1be` via [run.sh](run.sh). `src/` is identical in both commits. Results and decisions: [results.md](results.md).

## Contents

- `snapshot/record/l<leak>-d<same|1.0>-<persistent|reset>/`: metadata, reports and checksum inventories of the 16 recordings. `d1.0` means the driven neurons had leak 1.0.
- `snapshot/record-superseded/`: three attempts kept for the record but not used. `l0.25-dsame-reset` and `l0.25-d1.0-persistent` ran while uncommitted T35 code was present in the working tree (`git_dirty: true`). `l0.25-dsame-reset-interrupted` was stopped when the host drive filled up. The issue documents both events.
- `snapshot/decode/<setting>/`: last-window decoding of each setting's persistent and reset recordings, plus the input references.
- `snapshot/curve/<setting>/`: the label-free window-recall curve of the persistent final state.
- `snapshot/memory/`: the preregistered persistent-versus-reset test at the Bonferroni level 1 − 0.05/8.

The recordings' state arrays were deleted by the pipeline right after their decode and curve were computed, to keep the host drive from filling. Their SHA-256 values remain in each attempt's `checksums.sha256`, and the anchor showed element-for-element regeneration.

## Reproduction

From a checkout of `73bb697` with `FLYSTATE_HOME` set, `research/sequential-visual-memory/2026-09-25-memory-scale/run-pipeline.sh` reruns the grid, then:

```bash
B=runs/diagnostics/2026-09-25-memory-scale
S="l0.25-dsame l0.25-d1.0 l0.1-dsame l0.1-d1.0 l0.05-dsame l0.05-d1.0 l0.02-dsame l0.02-d1.0"
uv run flystate diagnose rate-memory configs/celeba-smoke.yaml $(for s in $S; do echo $B/decode/$s; done) \
  --output $B/memory $(for s in $S; do echo --setting $s; done) \
  --population central_brain --population descending --json
```
