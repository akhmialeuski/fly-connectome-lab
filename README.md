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
| **Experiments** | Search runs and filter by memory policy. Open a run, choose a completed test/validation evaluation or the training validation curve, then inspect top-1/top-5 accuracy, final-observation confusion counts, timings, configuration, and provenance. |
| **Prediction explorer** | Inside a run: choose an observation, filter correct/incorrect predictions, search an exact sample ID, and page through results. The table shows the input photo beside training examples of the actual person and the person selected by the model. |
| **Episode inspector** | Click **View episode**. Use **Play/Pause** or the observation slider to replay the aligned image and window trajectory alongside photographs of the top-five predicted identities and their model probabilities. The first choice is highlighted, and candidates matching the actual person are marked. These candidate photos are training examples, not generated images or retrieved test images. Technical class indices remain available in a disclosure. Switch between voltage and spike-trace features, inspect the heatmap, and click a heatmap row to select its observation. Drag the spatial view to rotate, scroll to zoom, and hover to inspect a neuron. |
| **Comparisons** | Saved paired accuracy differences, bootstrap confidence intervals, and McNemar p-values. These reports retain the compatibility checks and split used by the CLI comparison. |
| **Evidence library** | Select calibration curves and firing rates, CPU scaling and sustained-load measurements, pixel controls, or saved Markdown research reports. Expand source JSON to inspect or download the original values. |
| **Trace caches** | Cache identity, build status, image/observation/feature dimensions, threads, and batch size. |

Use **Refresh artifacts** after creating new results from another terminal. Pages have bookmarkable URL fragments, including individual runs. Charts include expandable numeric tables; confusion matrices include nonzero counts. Reports are rendered as plain text, so embedded HTML is not executed.

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

To keep experiments reproducible and the repository lightweight, datasets, generated traces, model checkpoints, caches, uploaded images, and experimental artifacts should be stored outside Git or in explicitly ignored paths. This repository is intended to track documentation, code, and configuration needed to reproduce experiments, not the resulting bulk data.

## Upstream projects and tools

- MaleCNS official project: <https://male-cns.janelia.org/>
- MaleCNS download page: <https://male-cns.janelia.org/download/>
- MaleCNS analysis and supplemental repository: <https://github.com/flyconnectome/2025malecns>
- fly.ai / flybrain: <https://github.com/alextitonis/fly.ai>
- ConnecTorch: <https://github.com/us/connectorch>
- MedMNIST: <https://medmnist.com/> and <https://github.com/MedMNIST/MedMNIST>

## Maintainer

Maintainer: <https://github.com/akhmialeuski>
