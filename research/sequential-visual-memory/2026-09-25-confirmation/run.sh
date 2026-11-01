#!/usr/bin/env bash
# T35: record the frozen T34 operating point and its controls once, then evaluate once.
set -euo pipefail
: "${FLYSTATE_HOME:?Set FLYSTATE_HOME to the data home}"
CFG=configs/celeba-confirm.yaml
BASE=runs/diagnostics/2026-09-25-confirmation
MODEL=(--gain 1.0 --leak 0.02 --driven-leak 1.0 --input-scale 20 --steps-per-window 4)
record() {
  local name="$1"; shift
  [ -f "$FLYSTATE_HOME/$BASE/record/$name/report.json" ] && return 0
  uv run flystate diagnose confirm-record "$CFG" --output "$BASE/record/$name" "${MODEL[@]}" "$@" \
    --json >/dev/null 2>&1
  echo "RECORDED $name"
}
record persistent --persistent
record reset --reset-each-window
record shuffled --persistent --shuffle-seed 0
record shuffled-reset --reset-each-window --shuffle-seed 0
populations=()
for p in central_brain descending driven visual_projection_other optic_lobe_sample kenyon vnc; do
  populations+=(--population "$p")
done
uv run flystate diagnose confirm-evaluate "$CFG" --output "$BASE/evaluate" \
  --recording "persistent=$BASE/record/persistent" --recording "reset=$BASE/record/reset" \
  --recording "shuffled=$BASE/record/shuffled" --recording "shuffled-reset=$BASE/record/shuffled-reset" \
  "${populations[@]}" \
  --compare persistent/central_brain:reset/central_brain \
  --compare persistent/central_brain:shuffled/central_brain \
  --compare persistent/descending:reset/descending \
  --compare persistent/descending:shuffled/descending \
  --compare shuffled/central_brain:shuffled-reset/central_brain \
  --compare input/encoded_current_all:persistent/central_brain \
  --json
