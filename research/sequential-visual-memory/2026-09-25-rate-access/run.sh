#!/usr/bin/env bash
# Formal T33 grid: 3 gains x 2 leaks x persistent/reset, recorded from the committed worktree.
set -euo pipefail
cd /home/anatolk/Workspace/fly-connectome-lab-rate
export FLYSTATE_HOME=/home/anatolk/data/flystate
R=research/sequential-visual-memory
BASE=runs/diagnostics/2026-09-25-rate-access
COMMON=(--cohort $R/2026-09-24-input-access/cohort.json
        --parent-schedule $R/2026-09-24-input-access/schedule.json
        --membership $R/2026-09-23-noise-recognition/membership.json
        --input-scale 20 --steps-per-window 4)
for gain in 0.5 1.0 1.5; do
  for leak in 1.0 0.25; do
    for state in persistent reset-each-window; do
      name="g${gain}-l${leak}-${state%-each-window}"
      [ -f "$FLYSTATE_HOME/$BASE/record/$name/report.json" ] && continue
      uv run flystate diagnose rate-record configs/celeba-smoke.yaml --output "$BASE/record/$name" \
        --gain "$gain" --leak "$leak" "--$state" "${COMMON[@]}" --json 2>/dev/null \
        | python3 -c "import json,sys;d=json.load(sys.stdin);print('RECORDED','$name',round(d.get('simulation_seconds',0)),d.get('final_state_mean_abs',d),d.get('final_state_saturated_fraction'))"
    done
  done
done
