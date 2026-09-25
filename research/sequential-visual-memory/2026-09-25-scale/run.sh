#!/usr/bin/env bash
# T38: record the confirmed graded model on the 100-identity development cohort for encoder seeds
# 0 to 4 and once on 100 untouched identities, evaluate every cohort, then apply the frozen rules.
set -euo pipefail
: "${FLYSTATE_HOME:?Set FLYSTATE_HOME to the data home}"
CFG=configs/celeba-persistent.yaml
BASE=runs/diagnostics/2026-09-25-scale
CONFIRMATION_SEED=92
MIN_FREE_BYTES=1000000000
MODEL=(--gain 1.0 --leak 0.02 --driven-leak 1.0 --input-scale 20 --steps-per-window 4
       --batch-size 250 --threads 12)
DELAYS=(--delay 0 --delay 1 --delay 2 --delay 4 --delay 8 --delay 16)
mkdir -p "$FLYSTATE_HOME/$BASE/logs"

guard() {
  local free
  free=$(df --output=avail -B1 /mnt/c | tail -1)
  if [ "$free" -lt "$MIN_FREE_BYTES" ]; then
    echo "STOP: C: has only $free bytes free" >&2
    exit 1
  fi
}

# run <log name> <command> <output> <arguments...>; skipped when its report already exists.
run() {
  local name="$1" command="$2" output="$3"; shift 3
  [ -f "$FLYSTATE_HOME/$output/report.json" ] && return 0
  guard
  uv run flystate diagnose "$command" "$CFG" --output "$output" "$@" --json     >"$FLYSTATE_HOME/$BASE/logs/$name.json" 2>"$FLYSTATE_HOME/$BASE/logs/$name.log"
  echo "DONE $name"
}

cohort() {
  local label="$1" references="$2"; shift 2
  local sets=("$@")
  run "$label-record-persistent" scale-record "$BASE/$label/persistent" "${sets[@]}"     "${MODEL[@]}" "${DELAYS[@]}" --persistent
  run "$label-record-reset" scale-record "$BASE/$label/reset" "${sets[@]}" "${MODEL[@]}"     --reset-each-window
  run "$label-evaluate" scale-evaluate "$BASE/$label/evaluate" "${sets[@]}" "${DELAYS[@]}"     --persistent-recording "$BASE/$label/persistent" --reset-recording "$BASE/$label/reset"     "$references"
}

for seed in 0 1 2 3 4; do
  references=--no-references
  [ "$seed" = 0 ] && references=--references
  cohort "development-encoder$seed" "$references" --set "encoder.seed=$seed"
done
cohort confirmation --references --set "dataset.subset.selection_seed=$CONFIRMATION_SEED" \
  --set name=celeba-scale-confirm
development=()
for seed in 0 1 2 3 4; do
  development+=(--development "encoder$seed=$BASE/development-encoder$seed/evaluate")
done
run analyze scale-analyze "$BASE/analyze" "${development[@]}" \
  --confirmation "$BASE/confirmation/evaluate" "${DELAYS[@]}"
