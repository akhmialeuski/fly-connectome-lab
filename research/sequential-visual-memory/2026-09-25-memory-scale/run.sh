#!/usr/bin/env bash
# T34 grid: gain 1.0, network leak x driven-neuron leak x persistent/reset, then last-window decoding.
set -euo pipefail
: "${FLYSTATE_HOME:?Set FLYSTATE_HOME to the data home}"
R=research/sequential-visual-memory
BASE=runs/diagnostics/2026-09-25-memory-scale
COMMON=(--cohort $R/2026-09-24-input-access/cohort.json
        --parent-schedule $R/2026-09-24-input-access/schedule.json
        --membership $R/2026-09-23-noise-recognition/membership.json)
for leak in 0.25 0.1 0.05 0.02; do
  for driven in same 1.0; do
    for state in persistent reset-each-window; do
      name="l${leak}-d${driven}-${state%-each-window}"
      [ -f "$FLYSTATE_HOME/$BASE/record/$name/report.json" ] && continue
      extra=()
      [ "$driven" != same ] && extra=(--driven-leak "$driven")
      uv run flystate diagnose rate-record configs/celeba-smoke.yaml --output "$BASE/record/$name" \
        --gain 1.0 --leak "$leak" "${extra[@]}" --input-scale 20 --steps-per-window 4 "--$state" \
        "${COMMON[@]}" --json 2>/dev/null >/dev/null
      echo "RECORDED $name"
    done
  done
done
