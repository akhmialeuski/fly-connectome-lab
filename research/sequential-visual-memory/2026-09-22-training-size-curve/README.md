# Fixed-cohort training-size study — 2026-09-22

Research issue: [#42](https://github.com/akhmialeuski/fly-connectome-lab/issues/42).
Parent diagnostics: [2026-09-21](../2026-09-21-fly-only-diagnostics/README.md).

This preregistration contains no fitted scientific results yet. The [protocol](protocol.md)
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
Completed and failed attempts will be copied into a `snapshot/` directory with
coefficients in Git LFS. Until that copy and remote verification are complete,
this directory is a protocol archive, not a completed result archive.

Verify the current inventory after checking out this commit:

```bash
cd research/sequential-visual-memory/2026-09-22-training-size-curve
sha256sum --check checksums.sha256
```

See the repository README for `flystate diagnose run --train-per-class` and
`--subset-seed`. Every new attempt requires a fresh output directory.
