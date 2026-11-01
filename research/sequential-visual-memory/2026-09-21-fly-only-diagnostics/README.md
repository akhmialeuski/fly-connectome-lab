# Fly-only recognition diagnostics — 2026-09-21

Execution: [#40](https://github.com/akhmialeuski/fly-connectome-lab/issues/40).
Roadmap: [#39](https://github.com/akhmialeuski/fly-connectome-lab/issues/39).
Implementation: [#41](https://github.com/akhmialeuski/fly-connectome-lab/pull/41).

## Scope and current status

This snapshot preserves two completed cohort audits and all 26 smoke/main
probe attempts: 22 fitted diagnostic models and four original optimizer failures.
Four numerical diagnoses additionally preserve eight training-fold parameter sets,
including four unconverged fits explicitly marked as diagnostic-only. Subsequent
causal and learning-curve investigations remain separate studies. Results are exploratory. No historical-test or reserved-image
predictions were scored, and no fly dynamics or synaptic weights changed.

The [protocol](protocol.md) was
[posted before scoring](https://github.com/akhmialeuski/fly-connectome-lab/issues/40#issuecomment-5759062409).
The [numerical diagnosis and retry results](provenance/numerical-summary.md),
[main result table](provenance/main-summary.md),
[smoke result table](snapshot/runs/diagnostics/2026-09-21-fly-only-diagnostics/execution/smoke-summary.md)
and [interpretation](provenance/smoke-conclusions.md) preserve successful and failed
experiments. The same reports are posted to their research issues.

## Findings and limits

- Persistent neural PCA60: 280/280 training correct, 4/60 validation correct;
  chance is 5%. All parameter arrays reproduce the historical final-observation
  model exactly. This is memorization without established useful generalization.
- Main neural anchors: persistent-last scores 1/300 validation correct (0.33%);
  reset-all scores 2/300 (0.67%), against 1% chance. Both reproduce the historical
  model probabilities. Their original input controls fail the 5,000-iteration budget; the separately
  preregistered 50,000-cap retries converge. See the numerical report for scores.
- Pixel-last: 10/60 validation correct; encoded-last: 11/60. Pixel-all and
  encoded-all true-label controls reached the 5,000-iteration optimizer limit at
  tolerance 1e-6. Their missing scores are not zero accuracy.
- More PCA components, no PCA, and concatenated neural history did not achieve
  the preregistered improvement threshold. No extra variant is promoted.
- Training-label permutation and tiny memorization controls reinforce that high
  training accuracy alone is not recognition. Tiny controls have 20 training
  photographs across five identities and no validation score.
- The audits reserve 37 smoke and 224 main photographs after screening; only
  11/20 and 67/100 identities respectively have three reserved photographs.
  Cohort reserves can overlap and must not be added as independent images.
- No exact cross-split duplicates were found in these cohorts. Perceptual review
  candidates are reported separately and are not proven duplicates.
- The existing sensory-mask policy leaves 25,088,107 effective graph edges out
  of 25,582,938 source entries. This documents the runtime, not a causal finding.
- Both existing cohorts have only 14 training photographs per identity. Main's
  1,400 training examples increase the class count, not that per-identity support.
  The source contains at most 35 photographs for any one identity. A controlled
  learning curve remains necessary to assess the data-size hypothesis.

## Evidence and reproducibility

`experiment.json` records the source commit, status, and parent study.
`snapshot/runs/diagnostics/2026-09-21-fly-only-diagnostics/` contains immutable
attempt directories and execution records. Each attempt has its own checksum
inventory, effective configuration, environment, manifest, and source provenance.
Completed probes include training/validation probabilities, metrics, feature
statistics, CV scores, and numeric scaler/PCA/classifier coefficients. Failed
attempts retain their exception and pre-failure evidence; no completed model is
claimed for them. `execution/` includes issue-comment readback verification and
independent replay of all 12,280 stored prediction rows from the 22 models, recorded
in separate smoke, main, and retry reports. Numerical fold coefficient replay
reproduces the saved training objective and gradient norms without scoring validation.

Large model NPZ files and main training-prediction Parquet files use Git LFS.
JSON, YAML, Markdown, and logs remain directly readable; Parquet tables can be
inspected after fetching their payloads. No source photographs, transformed image arrays, input feature
matrices, or downloaded connectome files are stored here.

Source data and historical neural traces come from the
[parent study](../2026-09-20-celeba-poc1/README.md). Its exact external-input records
are copied here as `external-inputs.json` and `external-inputs.sha256`; follow the
parent's acquisition and checksum instructions. The environment directory pins
`uv.lock` and project metadata. Per-attempt `git_dirty` can be null when the Git
status timeout expires on the mounted Windows checkout; the source commit is
still recorded, and a separate status check verified cleanliness before execution.

## Fetch, verify, and restore

From the repository root:

```bash
git lfs pull --include='research/sequential-visual-memory/2026-09-21-fly-only-diagnostics/**'
cd research/sequential-visual-memory/2026-09-21-fly-only-diagnostics
sha256sum --check checksums.sha256
```

Restore into a working data home that does not already contain this study:

```bash
export FLYSTATE_HOME=/path/to/working-data-home
test ! -e "$FLYSTATE_HOME/runs/diagnostics/2026-09-21-fly-only-diagnostics"
mkdir -p "$FLYSTATE_HOME/runs/diagnostics"
cp -a snapshot/runs/diagnostics/2026-09-21-fly-only-diagnostics \
  "$FLYSTATE_HOME/runs/diagnostics/"
```

Read results through the linked issue table or each attempt's `report.json` and
Parquet tables. The repository README documents `flystate diagnose run` and the
safe NumPy formula for replaying `model/weights.npz`. Inspecting saved results
requires no retraining. Regenerating pixel/encoded features requires the pinned
external photographs; neural features are preserved in the parent study's trace
caches. Run new experiments under fresh names rather than replacing an attempt.
