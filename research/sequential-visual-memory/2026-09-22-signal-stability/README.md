# Training-only signal stability — 2026-09-22

Protocol: [issue #48](https://github.com/akhmialeuski/fly-connectome-lab/issues/48)
and [frozen measurement design](protocol.md). Membership is fixed in
[membership.json](membership.json). All 163 episodes completed with exact native replay. See [results and limitations](results.md).
The complete immutable attempt and execution records are preserved in `snapshot/`.
Run `git lfs pull` before verifying `checksums.sha256`; all generated NPZ arrays
use Git LFS. All eight NPZ payloads were independently fetched from origin and
verified; see [remote verification](provenance/remote-lfs-verification.json).

This study measures native float32 neural responses, float16 storage error,
paired noise variability, and zero-current responses on 40 training photographs.
It fits no classifier, changes no synapses, and estimates no recognition accuracy.

```bash
flystate diagnose stability /path/to/original/config.yaml \
  --membership research/sequential-visual-memory/2026-09-22-signal-stability/membership.json \
  --output runs/diagnostics/2026-09-22-signal-stability/paired-responses --json
```

The original verified trace cache and exact external inputs must already exist
under `FLYSTATE_HOME`; acquisition instructions are in the parent archived studies.
Each output directory must be new. Failure to reproduce the archived float16
rows stops interpretation before the intervention cases. All controls retain the
same baseline warmed rest state. Native responses, spike counts, rest state and
neuron indices are saved as non-pickle NPZ files; no input photographs or encoded
image arrays are saved in the study.
