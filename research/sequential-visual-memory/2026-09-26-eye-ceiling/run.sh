#!/usr/bin/env bash
# T39: sample both development cohorts through the modelled compound eye, score every case with
# the standard readout, and apply the frozen decision rule. The 100-identity cohort decides.
set -euo pipefail
: "${FLYSTATE_HOME:?Set FLYSTATE_HOME to the data home}"
BASE=runs/diagnostics/2026-09-26-eye-ceiling
mkdir -p "$FLYSTATE_HOME/$BASE/logs"

step() {
  local name="$1" output="$2"; shift 2
  [ -f "$FLYSTATE_HOME/$output/report.json" ] && return 0
  uv run flystate diagnose "$@" --output "$output" --json >"$FLYSTATE_HOME/$BASE/logs/$name.json" \
    2>"$FLYSTATE_HOME/$BASE/logs/$name.log"
  echo "DONE $name"
}

for cohort in identities100:configs/celeba-persistent.yaml identities20:configs/celeba-smoke.yaml; do
  label="${cohort%%:*}"
  config="${cohort#*:}"
  step "$label-record" "$BASE/$label/record" eye-record "$config"
  step "$label-evaluate" "$BASE/$label/evaluate" eye-evaluate "$config" \
    --recording "$BASE/$label/record"
  step "$label-decide" "$BASE/$label/decide" eye-decide "$config" \
    --evaluation "$BASE/$label/evaluate"
done
