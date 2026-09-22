# Fixed-cohort training-size study — 2026-09-22

Research issue: [#42](https://github.com/akhmialeuski/fly-connectome-lab/issues/42).
Parent diagnostics: [2026-09-21](../2026-09-21-fly-only-diagnostics/README.md).

All 48 attempts completed. See [results and limitations](results.md). The [protocol](protocol.md)
freezes 48 unique attempts on the original 20-identity smoke cohort: nested
2/4/8/14 training photographs per identity, five independent subset draws, and a
single shared full-data endpoint for each representation. The 60 validation
photographs stay fixed. No test or reserved photographs enter training or scoring.

`preregistration/schedule.json` specifies the ordered commands. Configuration paths
refer to the original effective smoke configuration in the parent's source data
home; change only the local data-home prefix when restoring on another machine.
`preregistration/frozen-memberships.json` fixes every selected sample ID and its
SHA-256 membership hash before fitting. It also records the source membership
file hash and the fixed validation IDs. External input acquisition and exact
source hashes are documented in the parent studies; photographs and input arrays
are excluded from Git.

Working outputs belong under `$FLYSTATE_HOME/runs/diagnostics/2026-09-22-training-size-curve/`.
The `snapshot/` directory preserves all completed attempts and execution records.
Learned NPZ coefficients use Git LFS. Run `git lfs pull` before verification.
Copy the contents of `snapshot/` to that working output directory to inspect
the attempts in the viewer Diagnostics menu. Fetch the exact external inputs
using the parent study acquisition records; no photographs are included.
All 48 NPZ payloads were independently fetched from origin into an initially
empty object store and verified against their SHA-256 digests. See
[remote verification](provenance/remote-lfs-verification.json).

Verify the current inventory after checking out this commit:

```bash
cd research/sequential-visual-memory/2026-09-22-training-size-curve
sha256sum --check checksums.sha256
```

See the repository README for `flystate diagnose run --train-per-class` and
`--subset-seed`. Every new attempt requires a fresh output directory.
