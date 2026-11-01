#!/usr/bin/env bash
# T34 grid as a per-setting pipeline that keeps disk use bounded: record the persistent and
# reset states, decode their last-window states, compute the label-free memory curve, then
# delete the two large state arrays (their SHA-256 stays in each attempt's checksums.sha256).
set -euo pipefail
: "${FLYSTATE_HOME:?Set FLYSTATE_HOME to the data home}"
R=research/sequential-visual-memory
BASE=runs/diagnostics/2026-09-25-memory-scale
COHORT=(--cohort $R/2026-09-24-input-access/cohort.json
        --parent-schedule $R/2026-09-24-input-access/schedule.json
        --membership $R/2026-09-23-noise-recognition/membership.json)
MIN_FREE_KB=1000000
guard() {
  local free
  free=$(df --output=avail -k /mnt/c | tail -1)
  if [ "$free" -lt "$MIN_FREE_KB" ]; then echo "STOP: host drive below 1 GB free"; exit 1; fi
}
for leak in 0.25 0.1 0.05 0.02; do
  for driven in same 1.0; do
    setting="l${leak}-d${driven}"
    extra=()
    [ "$driven" != same ] && extra=(--driven-leak "$driven")
    for state in persistent reset-each-window; do
      name="${setting}-${state%-each-window}"
      [ -f "$FLYSTATE_HOME/$BASE/record/$name/report.json" ] && continue
      guard
      uv run flystate diagnose rate-record configs/celeba-smoke.yaml --output "$BASE/record/$name" \
        --gain 1.0 --leak "$leak" "${extra[@]}" --input-scale 20 --steps-per-window 4 "--$state" \
        "${COHORT[@]}" --json >/dev/null 2>&1
      echo "RECORDED $name"
    done
    if [ ! -d "$FLYSTATE_HOME/$BASE/decode/$setting" ]; then
      guard
      uv run flystate diagnose drive-decode configs/celeba-smoke.yaml \
        "$BASE/record/$setting-persistent" "$BASE/record/$setting-reset" \
        --output "$BASE/decode/$setting" "${COHORT[@]}" \
        --selection-schedule $R/2026-09-24-input-loss-selection/schedule.json \
        --representation last_window_state --json >/dev/null 2>&1
      echo "DECODED $setting"
    fi
    if [ ! -d "$FLYSTATE_HOME/$BASE/curve/$setting" ]; then
      uv run flystate diagnose rate-memory-curve configs/celeba-smoke.yaml \
        "$BASE/record/$setting-persistent" --output "$BASE/curve/$setting" "${COHORT[@]}" \
        --population central_brain --population descending --json >/dev/null 2>&1
      echo "CURVE $setting"
    fi
    rm -f "$FLYSTATE_HOME/$BASE/record/$setting-persistent/responses.npz" \
          "$FLYSTATE_HOME/$BASE/record/$setting-reset/responses.npz"
  done
done
