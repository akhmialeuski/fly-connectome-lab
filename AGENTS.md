# AGENTS.md

Instructions for AI coding agents working in this repository. Read this file completely before changing anything. Resolve specification defects from evidence and cover corrections with regression tests.

## Project

`fly-connectome-lab` runs reproducible machine-learning experiments on the MaleCNS v1.0 *Drosophila* connectome. All code lives in one Python package, `flystate`, with one command-line entry point, `flystate`.

POC 1 — Sequential Visual Memory is implemented. The user has now explicitly authorized a local web interface for exploring and visualizing its results. This extends the earlier POC 1 scope to a read-only results API and browser viewer. The scientific experiment protocol and completed artifacts remain unchanged.

The complete T01-T16 milestone is assigned. Work in dependency order using stacked pull requests, without merging them. Laptop measurements and T16 are included in the assignment. Do not wait for earlier pull requests to merge before implementing their dependants.

## Hard scope limits for POC 1

The original POC 1 web-interface restriction is superseded by the user's explicit viewer request. Continue to exclude LFW, MedMNIST, VGGFace2, Fashion-MNIST, lesion experiments, ConnecTorch, PyTorch, GPU or CuPy code, Rust, Docker, and Jupyter notebooks. Serve the viewer on loopback, preserve artifact immutability, and ship browser assets locally without CDN dependencies.

## Environment

- Python 3.12 only (`requires-python = ">=3.12,<3.13"`), managed with uv. Use `uv add`, `uv sync`, `uv run`. Never call pip directly. `uv.lock` is committed.
- Runtime dependencies allowed in POC 1: `flybrain==0.1.0` (brings numpy, scipy, numba), `typer`, `pydantic`, `pyyaml`, `structlog`, `psutil`, `threadpoolctl`, `pillow`, `scikit-learn`, `zarr>=3`, `pyarrow`. Any other runtime dependency needs one line of justification in the PR description.
- Dev dependencies include `pytest`, coverage, `ruff`, `pyright`, `mypy`, `flake8`, type stubs, and `pre-commit`.
- Target machine: a laptop CPU (Intel Core Ultra 5 235H, 14 threads, no NVIDIA GPU). Everything must run on CPU.

## Definition of done

A change is done only when all of these pass locally and in CI:

```bash
uv sync --frozen
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest
```

Every new CLI command has `--help` text, supports `--json` (exactly one JSON object on stdout, logs on stderr) and is covered by at least one test that invokes it through `typer.testing.CliRunner`.

## Repository layout

```text
src/flystate/
  __init__.py      # __version__ from package metadata
  settings.py      # FLYSTATE_HOME and the directory layout
  log.py           # structlog configuration
  hashing.py       # canonical JSON, sha256 helpers, stable_int
  doctor.py        # environment checks behind `flystate doctor`
  cli/             # Typer app, one module per command group, no scientific logic
  experiments/     # Pydantic config models, YAML loading, overrides, hashing
  brain/           # flybrain wrapper, brain files, benchmark, calibration
  datasets/        # CelebA adapter, registry, subsets, splits, face alignment, fingerprints
  episodes/        # Episode, EpisodeBuilder, trajectories
  encoders/        # visual encoder
  traces/          # trace builder and Zarr feature cache
  readouts/        # per-timestep PCA + logistic readouts
  evaluation/      # metrics, statistics, compare, design check
  storage/         # run directories, manifests, Parquet IO
configs/           # experiment YAML files (committed)
tests/unit/
tests/integration/
tests/conftest.py  # shared fixtures, including the synthetic brain and synthetic CelebA
docs/              # only when an issue asks for a document
```

## Local data directories

All generated data lives under `FLYSTATE_HOME` (environment variable, default: the current working directory). Never write anywhere else and never write into `src/`, `configs/` or `tests/`.

```text
$FLYSTATE_HOME/data/brain/         flybrain files brain.npz and weights.npz
$FLYSTATE_HOME/data/datasets/      registry.json with the paths of registered datasets
$FLYSTATE_HOME/cache/features/     trace caches, one directory per cache key
$FLYSTATE_HOME/cache/preprocess/   aligned image caches
$FLYSTATE_HOME/runs/               runs, evaluations, benchmarks, calibrations, design checks, comparisons
```

`data/`, `cache/`, `runs/`, `models/` and `uploads/` are git-ignored. Never commit datasets, brain files, caches, runs, model weights or images. Tests set `FLYSTATE_HOME` to `tmp_path`.

## Data and network rules

- Tests and CI never download anything: no brain files, no datasets. Use the fixtures from `tests/conftest.py`: `synthetic_brain_dir` (a small brain.npz and weights.npz that `FlyBrain(data=...)` loads) and `synthetic_celeba_dir` (a CelebA-shaped directory with generated images and annotation files).
- Tests that need the real brain are marked `@pytest.mark.connectome` and are skipped unless the environment variable `FLYSTATE_REAL_BRAIN=1` is set.
- Always pass the brain directory explicitly: `FlyBrain(data=paths.brain, ...)`, `flybrain.download(paths.brain, ...)`. `flybrain.data.DATA` is read from `FLY_DATA` at import time; do not rely on it.
- CelebA may be used for non-commercial research only. Never copy CelebA images into Git, docs, test fixtures or run directories. Runs store `sample_id` values only.

## Verified facts about flybrain 0.1.0

Checked against the released wheel source (`flybrain/brain.py`, `data.py`, `reservoir.py`). Do not assume anything else about its behaviour.

- Constructor: `FlyBrain(data=None, seed=64, device=None, batch=1, dt=None, sensory_input=True, refractory=0.0)`. flystate always passes `data=paths.brain`, `device="cpu"`, `sensory_input=False`, `refractory=0.0`.
- State: `brain.v` is a float32 array of shape `(n, batch)`; `brain.fired` holds flat int64 indices into `brain.v` of the spikes of the last step (row = neuron, column = fly, `flat = row * batch + column`); `brain.steps` counts steps.
- One step does, in this order: `current = brain.synaptic_input(brain.fired) * brain.gain` (gain 3.0); `v *= brain.decay` (exp(-dt/tau), tau 0.1 s); `v += current + brain.tonic` (0.14 at dt 0.02 s); noise `v += (rng.random((n, batch)) < noise_hz * dt) * noise_amp` (1.2 Hz, 0.22); eye drive and injections; neurons with `v >= 1.0` spike and are set to 0.
- Noise comes from ONE `np.random.default_rng(seed)` for the whole batch. The noise a fly receives therefore depends on its position in the batch and on how many flies are in the batch. flystate must not use this noise stream.
- `brain.reset(seed)` sets every voltage to 0. This is not the rest state: a silent neuron settles near `0.14 / (1 - exp(-0.2)) ≈ 0.772`.
- `brain.step(inject=[(idx, amount)])` accepts a scalar or one value per fly for an index array. A per-neuron array of shape `(len(idx), batch)` raises a NumPy broadcasting error. Per-neuron currents must be added to `brain.v[idx, :]` directly.
- CPU propagation (`flybrain.brain._propagate`, numba) is event-driven over the CSC columns of the neurons that fired. Each numba thread accumulates into its own length-n float32 buffer and the buffers are summed in thread order. Results are deterministic for a fixed numba thread count; a different thread count may change the float summation order. A batch is processed by a Python loop over flies, so batching does not share matrix reads on CPU.
- `brain.cells(["descending_neuron"])` and `brain.cells(["visual_projection"])` select whole superclasses. `brain.npz` also contains `ids` (MaleCNS bodyId), `cell_type`, `side`, `superclass`, `positions`, `visual` (photoreceptor indices), `azimuth` (one number per photoreceptor, no elevation) and motor groups named `group_<name>_<L|R>`. Superclass sizes in the released file: `visual_projection` 9,201, `descending_neuron` 1,314; there are 6,006 photoreceptors.
- `flybrain.reservoir.Trace` averages over the batch by default (`aggregate="mean"`). flystate does not use it and keeps its own per-fly traces.
- Prebuilt files and their sha256 values are in `flybrain.data.FILES`. The real network has 166,700 neurons and 25,582,938 connections.

## Reproducibility rules

- Every random choice takes its seed from the experiment config. Child seeds are derived with `numpy.random.SeedSequence([seed, stable_int(key)])`, where `stable_int(key)` is the first 8 bytes of `sha256(key.encode("utf-8"))` read as a big-endian unsigned integer (`flystate.hashing.stable_int`). Never use Python `hash()`, the global `np.random.*` functions or unseeded generators.
- The brain noise of an episode depends only on (config seed, sample_id), never on batch position, batch size or processing order.
- Simulation commands (`brain benchmark`, `brain calibrate`, `trace build`) set the numba thread count explicitly with `numba.set_num_threads`, record it, and run inside `threadpoolctl.threadpool_limits(limits=1, user_api="blas")` so that BLAS threads do not compete with numba threads. `threadpoolctl` is a direct dependency from T05 on.
- Hashes are sha256 over canonical JSON: `json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`.
- Never use pickle, joblib or `np.load(..., allow_pickle=True)`. Arrays go to `.npz` (loaded with `allow_pickle=False`), Zarr or Parquet; metadata goes to JSON or YAML.

## Logging and results

- Logging is configured only by `flystate.log.configure_logging`. Human-readable lines go to stderr. Machine-readable JSON Lines go to the `events.jsonl` of the current run or cache directory. Every event has `timestamp` (UTC, ISO 8601), `level` and `event`, plus bound context such as `run_id`, `cache_key` and `phase`.
- Metrics and predictions never live only in logs. They are written to Parquet files as specified in the issues.
- No `print()` in library code. CLI output goes through the helpers in `flystate.cli.common`.

## Run directory contract

`flystate train` creates a run directory. `flystate evaluate` adds a subdirectory under `evals/`. After the run status becomes `completed`, existing files are never modified or deleted.

```text
runs/<run_id>/
  config.original.yaml     # the YAML file as given
  config.yaml              # effective config with every default written out
  manifest.json            # schema_version, run_id, status, hashes, versions, seeds, threads
  environment.json         # python, platform, cpu model, ram, threads, package versions
  summary.json             # key metrics
  logs/events.jsonl
  model/weights.npz
  model/model.json
  metrics/validation.parquet
  evals/<eval_id>/eval.json
  evals/<eval_id>/metrics.parquet
  evals/<eval_id>/predictions.parquet
  evals/<eval_id>/confusion.parquet
  evals/<eval_id>/timings.parquet
```

`run_id = f"{utc:%Y%m%d-%H%M%S}-{config.name}-{config_hash[:6]}"`.

## Code style

- English only in code, comments, docstrings, CLI messages and logs.
- Type hints on every function (ruff `ANN` rules). pyright runs in `standard` mode.
- Science code is small pure functions on NumPy arrays; file and console I/O stay at the edges. No global mutable state.
- `pathlib.Path` everywhere; no hard-coded absolute paths.
- Every public function has a docstring that states array shapes, dtypes and units.
- All functions and methods, including tests, use reStructuredText parameter and return documentation. Prefer keyword arguments where the called API supports them. Logger messages are positional.
- Exit codes: 0 success, 1 runtime error, 2 invalid configuration or arguments, 3 incompatible runs in `compare`.

## Pull requests

- One issue per PR. PR title `[Txx] <issue title>`; the body starts with `Closes #<issue number>`.
- Do not modify `LICENSE`, `CITATION.cff`, `THIRD_PARTY_DATA.md` or `REFERENCES.md` unless the issue asks for it.
- No drive-by refactors or unrelated formatting changes.
- Record source-specification corrections in the ignored local review file. Never include that file in a commit or pull request.
- Resolve Numba threads before computing trace keys; thread count affects floating-point sums and belongs in cache identity. Batch size does not.
- Include selected image content in preprocessing cache identity. Never reuse cached images solely because annotations are unchanged.
- Fit preprocessing inside every cross-validation fold. Design decisions use validation data; final test results are reported only after freezing the protocol.
- Completed run files are immutable; fresh evaluation subdirectories may be appended.
