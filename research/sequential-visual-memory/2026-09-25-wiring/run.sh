#!/usr/bin/env bash
# T37: phase-a selects each graph family's operating point on the development cohort; phase-b
# records ten untouched cohorts at the selected points, evaluates each cohort and pools them.
set -euo pipefail
: "${FLYSTATE_HOME:?Set FLYSTATE_HOME to the data home}"
PHASE="${1:?Usage: run.sh phase-a|phase-b}"
CFG=configs/celeba-confirm.yaml
STUDY=research/sequential-visual-memory/2026-09-25-wiring
BASE=runs/diagnostics/2026-09-25-wiring
THREADS=12
MIN_FREE_BYTES=1000000000
DEV=0
COHORTS=(4 5 6 8 10 11 15 22 24 25)
ALPHAS=(0.25 0.5 0.75 1.0 1.25)
MODEL=(--leak 0.02 --driven-leak 1.0 --input-scale 20 --steps-per-window 4 --threads "$THREADS")

guard() {
  local free
  free=$(df --output=avail -B1 /mnt/c | tail -1)
  if [ "$free" -lt "$MIN_FREE_BYTES" ]; then
    echo "STOP: C: has only $free bytes free" >&2
    exit 1
  fi
}

# record <root> <name> <cohort flags...> -- <graph flags...>
record() {
  local root="$1" name="$2"; shift 2
  guard
  uv run flystate diagnose wiring-record "$CFG" --output "$BASE/$root/record/$name" "$@" \
    "${MODEL[@]}" --json >"$FLYSTATE_HOME/$BASE/$root/logs/$name.json" \
    2>"$FLYSTATE_HOME/$BASE/$root/logs/$name.log"
  echo "RECORDED $root/$name"
}

evaluate() {
  local root="$1" seed="$2"; shift 2
  local output="$BASE/$root/evaluate/s$seed"
  [ -f "$FLYSTATE_HOME/$output/report.json" ] && return 0
  local items=()
  for name in "$@"; do items+=(--recording "$name=$BASE/$root/record/$name/s$seed"); done
  uv run flystate diagnose wiring-evaluate "$CFG" --output "$output" --cohort "$seed" "${items[@]}" \
    --json >"$FLYSTATE_HOME/$BASE/$root/logs/evaluate-s$seed.json" \
    2>"$FLYSTATE_HOME/$BASE/$root/logs/evaluate-s$seed.log"
  echo "EVALUATED $root/s$seed"
}

if [ "$PHASE" = phase-a ]; then
  mkdir -p "$FLYSTATE_HOME/$BASE/phase-a/logs"
  names=()
  candidates=()
  for alpha in "${ALPHAS[@]}"; do
    record phase-a "fly-a$alpha" --cohort "$DEV" --family fly --alpha "$alpha"
    record phase-a "degree0-a$alpha" --cohort "$DEV" --family degree --graph-seed 0 --alpha "$alpha"
    record phase-a "random0-a$alpha" --cohort "$DEV" --family random_target --graph-seed 0 \
      --alpha "$alpha"
    names+=("fly-a$alpha" "degree0-a$alpha" "random0-a$alpha")
    candidates+=(--candidate "fly:$alpha=fly-a$alpha/central_brain"
                 --candidate "degree:$alpha=degree0-a$alpha/central_brain"
                 --candidate "random_target:$alpha=random0-a$alpha/central_brain")
  done
  evaluate phase-a "$DEV" "${names[@]}"
  uv run flystate diagnose wiring-select "$CFG" --output "$BASE/phase-a/select" --cohort "$DEV" \
    --evaluation "$BASE/phase-a/evaluate/s$DEV" "${candidates[@]}" --json
  exit 0
fi

# Phase B reads the operating points committed with the Phase A archive.
selected() {
  python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['families'][sys.argv[2]]['selected_alpha'])" \
    "$STUDY/selection.json" "$1"
}
FLY_ALPHA=$(selected fly)
DEGREE_ALPHA=$(selected degree)
RANDOM_ALPHA=$(selected random_target)
mkdir -p "$FLYSTATE_HOME/$BASE/phase-b/logs"
flags=()
for seed in "${COHORTS[@]}"; do flags+=(--cohort "$seed"); done
record phase-b fly "${flags[@]}" --family fly --alpha "$FLY_ALPHA"
for seed in 0 1 2 3 4; do
  record phase-b "degree$seed" "${flags[@]}" --family degree --graph-seed "$seed" --alpha "$DEGREE_ALPHA"
done
for seed in 0 1; do
  record phase-b "random$seed" "${flags[@]}" --family random_target --graph-seed "$seed" \
    --alpha "$RANDOM_ALPHA"
done
record phase-b fly-g1 "${flags[@]}" --family fly --gain 1.0
record phase-b fly-g1-reset "${flags[@]}" --family fly --gain 1.0 --reset-each-window
record phase-b degree0-g1 "${flags[@]}" --family degree --graph-seed 0 --gain 1.0
record phase-b feedforward-g1 "${flags[@]}" --family feedforward --gain 1.0
names=(fly degree0 degree1 degree2 degree3 degree4 random0 random1 fly-g1 fly-g1-reset degree0-g1
       feedforward-g1)
evaluations=()
for seed in "${COHORTS[@]}"; do
  evaluate phase-b "$seed" "${names[@]}"
  evaluations+=(--evaluation "s$seed=$BASE/phase-b/evaluate/s$seed")
done
contrasts=()
for population in central_brain descending; do
  p="/$population"
  contrasts+=(
    --contrast "W1-degree-$population=fly$p:degree0$p,degree1$p,degree2$p,degree3$p,degree4$p"
    --contrast "W2-random-$population=fly$p:random0$p,random1$p"
    --contrast "M1-memory-$population=fly-g1$p:fly-g1-reset$p"
    --contrast "M2-recurrence-$population=fly-g1$p:feedforward-g1$p"
    --contrast "R-t35-replication-$population=fly-g1$p:degree0-g1$p"
  )
done
uv run flystate diagnose wiring-analyze "$CFG" --output "$BASE/phase-b/analyze" "${evaluations[@]}" \
  "${contrasts[@]}" --json
