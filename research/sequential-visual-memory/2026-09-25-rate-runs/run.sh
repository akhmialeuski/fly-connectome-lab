#!/usr/bin/env bash
# T36: standard persistent, reset and reset-concat runs of the frozen graded model, as in POC 1.
set -euo pipefail
: "${FLYSTATE_HOME:?Set FLYSTATE_HOME to the data home}"
CFG=configs/celeba-confirm-rate.yaml
LOGS="$FLYSTATE_HOME/acquisition/t36"
mkdir -p "$LOGS"
stage() {
  local name="$1"; shift
  [ -f "$LOGS/$name.json" ] && return 0
  uv run flystate "$@" --json > "$LOGS/$name.json.part" 2> "$LOGS/$name.log"
  mv "$LOGS/$name.json.part" "$LOGS/$name.json"
  echo "DONE $name"
}
run_dir() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_dir"])' "$LOGS/$1.json"; }
stage trace-persistent trace build "$CFG"
stage trace-reset trace build "$CFG" --set memory.mode=reset
stage train-persistent train "$CFG" --set name=celeba-confirm-rate-persistent
stage train-reset train "$CFG" --set memory.mode=reset --set name=celeba-confirm-rate-reset
stage train-concat train "$CFG" --set memory.mode=reset_concat --set name=celeba-confirm-rate-reset-concat
persistent=$(run_dir train-persistent); reset=$(run_dir train-reset); concat=$(run_dir train-concat)
stage evaluate-persistent evaluate "$persistent"
stage evaluate-reset evaluate "$reset"
stage evaluate-concat evaluate "$concat"
stage compare-reset compare "$persistent" "$reset" --markdown runs/reports/t36-rate-persistent-vs-reset.md
stage compare-concat compare "$persistent" "$concat" --markdown runs/reports/t36-rate-persistent-vs-concat.md
