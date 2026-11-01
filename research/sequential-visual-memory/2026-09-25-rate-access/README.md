# T33: identity and memory of the MaleCNS graph under graded dynamics

Issue: [#65](https://github.com/akhmialeuski/fly-connectome-lab/issues/65). Protocol and run script: [protocol.md](protocol.md), [run.sh](run.sh), frozen in commit `d51e12c` with the code at `93839fb` before any recording. The memory test used `9497986`, whose only change merges several decode attempts. Results and decisions: [results.md](results.md).

## Contents

- `snapshot/record/g<gain>-l<leak>-<persistent|reset>/`: every recording attempt's manifest, configuration, environment, report (state statistics, sample order) and `checksums.sha256`.
- `snapshot/decode/g<gain>-l<leak>/`: the decode of each setting's persistent and reset recordings together, with all OOF metrics, failures and the selected-C out-of-fold probabilities.
- `snapshot/memory/`: the preregistered persistent-versus-reset test with its identity-cluster bootstrap intervals at the 95% and Bonferroni levels.
- `analysis/summarize.py` prints the tables posted in the issue. `analysis/memory_curve.py` computes the label-free memory curve quoted there. The same method is committed as `flystate diagnose rate-memory-curve` for T34 onwards.

The recordings' `responses.npz` state arrays (about 13 GB) were deleted after decoding because the host drive was full (#39). Their hashes remain in each attempt's `checksums.sha256`. They are exactly regenerable: T34's anchor re-recorded `g1.0-l0.25-persistent` with later code and matched every array element for element.

## Reproduction

`run.sh` records the grid as it was run, with the author's absolute paths. From a checkout of commit `93839fb` with `FLYSTATE_HOME` set, one setting is:

```bash
R=research/sequential-visual-memory
COHORT=(--cohort $R/2026-09-24-input-access/cohort.json
        --parent-schedule $R/2026-09-24-input-access/schedule.json
        --membership $R/2026-09-23-noise-recognition/membership.json)
for state in persistent reset-each-window; do
  uv run flystate diagnose rate-record configs/celeba-smoke.yaml \
    --output runs/diagnostics/2026-09-25-rate-access/record/g1.0-l0.25-${state%-each-window} \
    --gain 1.0 --leak 0.25 --input-scale 20 --steps-per-window 4 --$state "${COHORT[@]}" --json
done
```

Decoding uses `flystate diagnose drive-decode` with both recordings, and the memory test uses `flystate diagnose rate-memory`, with the arguments listed in each attempt's `manifest.json`.
