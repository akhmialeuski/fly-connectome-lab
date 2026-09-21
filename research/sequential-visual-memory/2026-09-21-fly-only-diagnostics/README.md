# Fly-only recognition diagnostics — 2026-09-21

Execution issue: [#40](https://github.com/akhmialeuski/fly-connectome-lab/issues/40).
Research roadmap: [#39](https://github.com/akhmialeuski/fly-connectome-lab/issues/39).

The [protocol](protocol.md) was posted before new classifier scores were obtained:
[preregistration](https://github.com/akhmialeuski/fly-connectome-lab/issues/40#issuecomment-5759062409).
This study is exploratory and uses the original training/validation cohorts.
It does not reuse the previously inspected test split as a confirmation test.

Source data, cohort selection, existing neural traces, and historical models come
from the [2026-09-20 study](../2026-09-20-celeba-poc1/README.md). Follow that study's
exact external-input restoration instructions. This directory will preserve new
computed evidence separately from the immutable original study. No photographs,
transformed image arrays, or downloaded brain files belong here.

Working attempts are created under
`$FLYSTATE_HOME/runs/diagnostics/2026-09-21-fly-only-diagnostics/` using
`flystate diagnose run`; see the repository README for command options and numeric
model replay. Each completed or failed attempt is immutable and has its own
SHA-256 inventory. The issue records the current execution status; the presence
of a protocol alone does not imply a completed experiment.
