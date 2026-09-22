# fly-connectome-lab

`fly-connectome-lab` is an early-stage open-source research project for reproducible computational experiments built on Drosophila connectomes, starting with MaleCNS v1.0.

## Status

POC 1 — Sequential Visual Memory is implemented for CelebA, with a CPU experiment CLI and a local results viewer. Persistent, reset, and reset-concatenation runs share reproducible artifact contracts. Other datasets and lesion experiments remain future work.

## What is a connectome?

A connectome is a structural wiring diagram of a nervous system: which neurons connect to which other neurons, and how those connections are organized. In this project, a connectome is used as a biological structural prior for reproducible machine-learning experiments.

MaleCNS is a structural connectome release. It is not a complete biological simulation of a living fly brain, and work in this repository should not describe it as one.

## Planned experiments

### 1. Sequential Visual Memory

This experiment line is planned around a frozen MaleCNS recurrent reservoir. The initial goal is to present faces or medical images as sequences of partial observations, using moving-window glimpses for 2D images and sequential slices for 3D medical data. Early comparisons will focus on persistent MaleCNS state versus reset-between-observations controls, initially using CelebA, LFW, OrganMNIST3D, and NoduleMNIST3D.

### 2. Brain Lesion and Functional Recovery

This experiment line is planned as an extension of Sequential Visual Memory. It will introduce controlled neuron and synapse lesions, measure functional degradation, and evaluate recovery through readout retraining, homeostatic compensation, and constrained plasticity of surviving connections. Recovery experiments must never silently create connections that do not exist in the biological connectome.

## Start the results viewer

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). Run these commands in a Linux or WSL terminal from the repository root:

```bash
uv sync --frozen
export FLYSTATE_HOME="$HOME/data/flystate"
uv run flystate paths --json
uv run flystate serve
```

Open **http://127.0.0.1:8765** in your browser. When using WSL, open that address in your Windows browser while the WSL command keeps running. Stop the server with **Ctrl+C**. If the port is occupied, use `uv run flystate serve --port 9000` and open `http://127.0.0.1:9000`. `uv run flystate serve --json` emits one startup descriptor on stdout; the process stays running until stopped.

`FLYSTATE_HOME` must point to the installation that contains your existing `runs/`, `cache/`, and `data/` directories. Set it in each terminal, or persist it in your shell profile. Without it, the CLI uses the current directory; an empty dashboard usually means the wrong storage home is selected. `flystate paths --json` shows the resolved location. Starting the viewer does not download data, run simulations, fit models, or change recorded artifacts. Existing results can be browsed immediately; there is no Node.js build step or external CDN.

### Using the interface

| Section | What you can inspect |
| --- | --- |
| **Overview** | Run counts and accuracy curves by class count and evaluation split. Toggle 95% Wilson bands, inspect point values, and export a chart as SVG. The dashed line marks chance accuracy. |
| **Experiments** | Automatically discover all manifest-backed attempts, including nested studies, running jobs, and failures. Search names/paths/parameters and filter by study, kind, or memory policy. Open a run, choose a completed test/validation evaluation or the training validation curve, then inspect top-1/top-5 accuracy, final-observation confusion counts, timings, configuration, and provenance. |
| **Prediction explorer** | Inside a run: choose an observation, filter correct/incorrect predictions, search an exact sample ID, and page through results. The table shows the input photo beside training examples of the actual person and the person selected by the model. |
| **Episode inspector** | Click **View episode**. Use **Play/Pause** or the observation slider to replay the aligned image and window trajectory alongside photographs of the top-five predicted identities and their model probabilities. The first choice is highlighted, and candidates matching the actual person are marked. These candidate photos are training examples, not generated images or retrieved test images. Technical class indices remain available in a disclosure. Switch between voltage and spike-trace features, inspect the heatmap, and click a heatmap row to select its observation. Drag the spatial view to rotate, scroll to zoom, and hover to inspect a neuron. |
| **Comparisons** | Saved paired accuracy differences, bootstrap confidence intervals, and McNemar p-values. These reports retain the compatibility checks and split used by the CLI comparison. |
| **Evidence library** | Select calibration curves and firing rates, CPU scaling and sustained-load measurements, pixel controls, or saved Markdown research reports. Expand source JSON to inspect or download the original values. |
| **Trace caches** | Cache identity, build status, image/observation/feature dimensions, threads, and batch size. |

The experiment register checks for changes every five seconds while the tab is
visible, and when it regains focus. New attempts and status changes appear without
restarting the server or adding experiment IDs to code. Search and filter choices
are preserved. **Refresh artifacts** remains available for an immediate manual refresh.
Reload the browser once after upgrading viewer code; subsequent experiments are
discovered automatically.

Discovery recursively reads `manifest.json` files under `$FLYSTATE_HOME/runs/`,
regardless of study name or nesting depth. Keep new attempt outputs under that
working directory. Legacy training runs retain their detailed episode pages;
diagnostic and unfamiliar experiment kinds receive a generic evidence page.
Recorded recognition scores, convergence measurements, cohort audits, failures,
parameters, and available files appear according to the saved content. Prediction
tables show the input photograph and training-only examples of predicted people
when the matching prepared cache exists. A numerical diagnosis without recognition
scores is labeled accordingly. Unknown future kinds remain browsable without a
new per-experiment viewer configuration.

Evidence tables are paginated, text previews are limited to 2 MiB, and oversized
Parquet row groups/pages are rejected with an explicit preview-limit message.
Numeric model arrays are listed without loading them into the catalog. Missing or
partially written evidence produces a warning without hiding other attempts.
Archived Git snapshots become visible after restoring them into the working data
home using their study README; the viewer does not scan external datasets or
silently restore snapshots.
 Pages have bookmarkable URL fragments, including individual runs. Charts include expandable numeric tables; confusion matrices include nonzero counts. Reports are rendered as plain text, so embedded HTML is not executed.

The episode heatmap includes the cached readout population, not every neuron in the connectome. Spatial positions are used only when the installed brain file matches the run's SHA-256. Neurons with missing coordinates stay in the heatmap and are explicitly counted as missing from the spatial view. Missing aligned images or matching brain geometry do not prevent inspecting metrics and cached activity. Episode reads verify the selected committed trace chunk; prepared images are verified against their recorded content digests.

The viewer binds to loopback only and provides read-only endpoints. Keep experiment execution in the CLI. A class-count filter is a browsing convenience, not proof that two runs are scientifically comparable. Use saved paired comparisons for that assessment. Validation curves describe model selection; test curves describe held-out evaluation.

### Generate results for a new installation

The viewer can start with empty storage. To populate it, first install the published connectome and register your own complete, licensed CelebA dataset:

```bash
uv run flystate brain download
uv run flystate dataset register celeba /path/to/celeba-root
uv run flystate dataset validate celeba --full
uv run flystate doctor
uv run flystate experiment validate configs/celeba-smoke.yaml
```

The dataset root must contain the CelebA images and annotation files in the supported layout; `dataset validate` reports missing files. Acquisition and usage terms are described in [THIRD_PARTY_DATA.md](THIRD_PARTY_DATA.md). The viewer does not require access to the original dataset when the matching prepared image cache is present.

For an initial smoke experiment, build traces and train a readout:

```bash
uv run flystate trace build configs/celeba-smoke.yaml
uv run flystate train configs/celeba-smoke.yaml --json
```

Copy the emitted `run_id` and use it in these commands:

```bash
uv run flystate evaluate RUN_ID --split val
uv run flystate evaluate RUN_ID --split test
```

Use validation results to develop the protocol; run the test evaluation only after freezing that protocol. Completed training files remain immutable, and evaluations are appended as new directories.

For the full three-policy comparison:

```bash
uv run flystate trace build configs/celeba-persistent.yaml
uv run flystate train configs/celeba-persistent.yaml --json
uv run flystate trace build configs/celeba-reset.yaml
uv run flystate train configs/celeba-reset.yaml --json
uv run flystate train configs/celeba-reset.yaml --set memory.mode=reset_concat --set name=celeba-reset-concat --json
```

`reset_concat` reuses the matching completed reset trace cache. Evaluate each emitted run ID on the same split, then compare the appropriate pair:

```bash
uv run flystate compare PERSISTENT_RUN_ID RESET_RUN_ID
uv run flystate compare PERSISTENT_RUN_ID CONCAT_RUN_ID
```

The comparison command selects the latest completed test evaluations by default. Use `--eval-a` and `--eval-b` to select explicit evaluation IDs. For additional evidence, run `flystate brain benchmark --help`, `flystate brain calibrate --help`, and `flystate dataset design-check --help` to choose an appropriate measurement configuration. The viewer discovers their saved reports automatically.

### Troubleshooting

- **Empty results:** verify `FLYSTATE_HOME` with `flystate paths --json`, then refresh.
- **No test curve:** select validation or create a completed test evaluation for the frozen run.
- **Artifact warning or HTTP 409:** inspect the reported missing, incomplete, or inconsistent artifact; the viewer does not silently repair experiment evidence.
- **Image unavailable:** retain the matching `cache/preprocess/` directory with its `images.npy`, index, and metadata. Neural traces live separately under `cache/features/`.
- **Spatial view unavailable:** install the same brain file used by the run. A different brain or malformed geometry is not substituted.
- **Browser cannot connect:** keep `flystate serve` running, check its reported port, and use the loopback address. The server is not a remote hosting service.


## Development

```bash
uv sync --frozen
uv run flystate --help
uv run flystate paths --json
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest
```

The normal test suite uses only synthetic local data. Browser regression tests are opt-in and require a separately installed Chromium binary:

```bash
uv run playwright install chromium
FLYSTATE_BROWSER_TESTS=1 uv run pytest tests/integration/test_viewer.py
```

Install the browser once before testing; the tests themselves download nothing and intercept HTTP requests through the local ASGI test client. The default suite runs the read-only API, artifact integrity, and CLI tests without requiring Chromium.

All generated data lives under `FLYSTATE_HOME` (default: the working directory).

## Project scope

The goal of `fly-connectome-lab` is to support reproducible experiments with Drosophila connectomes while keeping biological-connectivity assumptions explicit and auditable.

## Data and licenses

This repository is licensed under Apache License 2.0. Third-party datasets, downloaded MaleCNS data, and external software dependencies are not covered by this repository's Apache-2.0 license unless their own upstream licenses explicitly say so.

See [THIRD_PARTY_DATA.md](THIRD_PARTY_DATA.md) for dataset, software, attribution, citation, and redistribution notes.

## References and citations

See [REFERENCES.md](REFERENCES.md) for curated references, upstream publications, and citation guidance.

## Reproducibility

Completed computed evidence is versioned under [research/](research/README.md), with a separate directory for each study. The [first CelebA study](research/sequential-visual-memory/2026-09-20-celeba-poc1/README.md) preserves all five trained models, six neural trace caches, predictions, metrics, configurations, calibration, and reports (about 512 MiB). Large numerical payloads use Git LFS; metadata remains readable JSON, YAML, and Markdown.

Follow the study README to fetch LFS objects, verify the SHA-256 inventory, and restore results into a new `FLYSTATE_HOME` for the viewer. No retraining is needed. External CelebA photographs and downloaded connectome files are excluded: the study records exact source links, versions, byte counts, and checksums, with instructions for obtaining and verifying the original inputs and regenerating photographs.

Normal working data remains outside Git or in ignored paths. Add future completed research campaigns as sibling study directories; preserve earlier evidence unchanged. Research archives are not included in Python distributions.

## Upstream projects and tools

- MaleCNS official project: <https://male-cns.janelia.org/>
- MaleCNS download page: <https://male-cns.janelia.org/download/>
- MaleCNS analysis and supplemental repository: <https://github.com/flyconnectome/2025malecns>
- fly.ai / flybrain: <https://github.com/alextitonis/fly.ai>
- ConnecTorch: <https://github.com/us/connectorch>
- MedMNIST: <https://medmnist.com/> and <https://github.com/MedMNIST/MedMNIST>

## Maintainer

Maintainer: <https://github.com/akhmialeuski>

## Fly-only recognition diagnostics

Research roadmap [#39](https://github.com/akhmialeuski/fly-connectome-lab/issues/39)
tracks the sequential investigation. The first campaign
[#40](https://github.com/akhmialeuski/fly-connectome-lab/issues/40) compares information
at the pixels, fixed encoder, and neural readout stages. Pixel/encoder classifiers
are diagnostic controls; the experiment continues to use the fly connectome.
These are exploratory training/validation measurements. Previously inspected test
images and the newly reserved photographs are excluded from probe fitting/scoring.

Set `FLYSTATE_HOME` to the existing data home and use the original run's effective
configuration to retain its exact cohort and trace identity:

```bash
export FLYSTATE_HOME=/path/to/flystate-home
uv run flystate diagnose run "$FLYSTATE_HOME/runs/<original-run>/config.yaml" \
  --phase audit --output runs/diagnostics/<study>/audit --json
uv run flystate diagnose run "$FLYSTATE_HOME/runs/<original-run>/config.yaml" \
  --representation neural --history last --features both --components 60 \
  --output runs/diagnostics/<study>/neural-last-pca60 --json
```

Available representations are `pixels`, `encoded`, and `neural`; histories are
`last` and `all`. Neural feature blocks are `both`, `spike_trace`, and `voltage`.
Use `--components 0` for a classifier without PCA. `--label-mode permuted`
permutes training labels only; `--label-mode memorization` measures a small
training-only subset and must not be interpreted as generalization.

Every attempt requires a fresh output directory. It records configuration,
source hashes, environment, status (including exceptions), resource measurements,
and a SHA-256 inventory. Successful probes also save CV scores, training and
validation predictions, feature statistics, and numeric model coefficients.
`model/weights.npz` can be opened with `numpy.load(..., allow_pickle=False)`:
standardize with `(x - scaler_mean) / scaler_scale`; when `has_pca` is true in
`model/model.json`, subtract `pca_mean` and multiply by `pca_components.T`.
Compute logits with `z @ coef.T + intercept`, then softmax for multiclass or
sigmoid for the positive binary class. Columns follow the saved `classes` array.
No photograph or input feature matrix is saved in these diagnostic results.

Audits preserve original splits, report exact duplicates separately from
perceptual review candidates, and reserve up to three unused photographs per
identity after conservative screening. Incomplete reserve coverage does not
constitute a balanced confirmation test. Follow the issue's preregistered order
and record failures as well as successes; use a new attempt name for any retry.

For a preregistered learning curve, select nested training photographs without changing
the original cohort, validation set, encoder, or simulation seed:

```bash
uv run flystate diagnose run "$FLYSTATE_HOME/runs/<original-run>/config.yaml" \
  --representation neural --history last --features both --components 60 \
  --train-per-class 4 --subset-seed 0 --max-iterations 50000 \
  --output runs/diagnostics/<new-study>/neural-four-photos-seed0 --json
```

`--train-per-class` requires at least two training photographs per class and true
labels. `--subset-seed` controls only a named per-label training draw; prefixes of
the same draw give nested subsets. Selected rows retain the original fitting
order, so selecting all training photographs reproduces the unsampled endpoint.
`training-subset.json` records membership hashes, CV fold membership, and actual
PCA dimensions before fitting. CV folds are bounded by per-class support and PCA
by each training fold's rank limit. A smaller subset therefore changes the
available fold sizes and dimensions. Validation and old test images never enter
the training draw. Register the candidate sizes, seeds, and numerical budget
before running; archive unsuccessful attempts too. The first controlled study
is tracked in [#42](https://github.com/akhmialeuski/fly-connectome-lab/issues/42).

For a failed true-label probe, `--phase convergence` reconstructs the original
training folds and identifies the first fold/C that exhausts 5,000 iterations.
It records the regularized objective, exact gradient, warnings, and numeric fold
coefficients, then independently fits that same problem with a 20,000-iteration
cap and, only if necessary, 50,000. It never scores validation predictions.
These fold exports are explicitly diagnostic and are not selected readouts.
A missing reproduced failure or another stopping limit is reported as such.

```bash
uv run flystate diagnose run "$FLYSTATE_HOME/runs/<original-run>/config.yaml" \
  --phase convergence --representation encoded --history all --components 60 \
  --output runs/diagnostics/<study>/encoded-convergence --json
```

Only after numerical diagnosis and a recorded protocol amendment, a fresh probe
may use `--max-iterations 50000`. This changes the iteration cap; the 1e-6 tolerance,
solver, regularization, folds, and preprocessing remain the same. Other solver
limits remain unchanged. The default cap stays 5,000, and an unconverged full probe
still fails rather than publishing an accuracy. Preserve the original failed
attempt and give any retry a new output directory.

### Training-only signal stability

`flystate diagnose stability CONFIG --membership MEMBERSHIP.json --output NEW_PATH --json`
measures paired image/noise responses and float16 storage loss without fitting a
classifier. Membership JSON must contain ordered training-only `sample_ids` and
their canonical `sample_ids_sha256`. The command requires a persistent,
noise-enabled baseline and its verified trace cache; exact native replay is a
mandatory gate before noise interventions. See the frozen
[signal-stability protocol](research/sequential-visual-memory/2026-09-22-signal-stability/README.md).

### Experiments and diagnostics in the viewer

The **Experiments** menu contains sequential image-patch training runs and retains
the episode, observation, and memory-state viewer. **Diagnostics** contains probes,
cohort audits, numerical checks, and unfamiliar non-training records. Both lists
are discovered automatically and refresh every five seconds; no study-name list
is maintained. Existing links remain valid. Each entry and detail page explains
its idea and recorded result. Diagnostic readout accuracy is not evidence of a
memory benefit; pixel and encoded-current controls do not simulate neural memory.
Summaries are derived from saved metadata without modifying scientific artifacts.
Unknown experiment kinds explicitly state when a specific hypothesis is unavailable.
