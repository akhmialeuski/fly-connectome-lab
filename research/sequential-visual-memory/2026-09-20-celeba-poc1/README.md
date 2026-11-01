# CelebA sequential visual memory — first frozen comparison

Study ID: `2026-09-20-celeba-poc1`. Protocol frozen on 2026-09-20. Evidence archived on 2026-09-21.

This study compares persistent MaleCNS state against resets between image windows and a reset-concatenation readout. The reservoir is fixed; per-observation standardization, PCA, and logistic readouts are learned. All five trained models and their observed results are preserved, so inspecting the experiment does not require repeating training or simulation.

## Protocol and outcome

The frozen protocol uses 128 × 128 aligned RGB faces, sixteen raster windows of 32 × 32 pixels, injection amplitude 0.05, ten simulation steps per window, seed 0, four Numba threads, and batch size one. Features combine spike traces and voltage from 1,314 descending neurons. Main runs use 100 identities and 2,000 images (1,400 train, 300 validation, 300 test); smoke runs use 20 identities and 400 images (280 train, 60 validation, 60 test).

| Run ID | Cohort | Memory policy | Final test accuracy |
| --- | --- | --- | ---: |
| `20260920-071344-celeba-smoke-9406ee` | Smoke | persistent | 8.33% |
| `20260920-071430-celeba-smoke-400cac` | Smoke | reset | 3.33% |
| `20260920-075340-celeba-persistent-541c94` | Main | persistent | 1.67% |
| `20260920-080049-celeba-reset-72aac7` | Main | reset | 0.67% |
| `20260920-080640-celeba-reset-concat-c29bf2` | Main | reset_concat | 1.33% |

The main chance baseline is 1%. The persistent-minus-reset difference is 1 percentage point, with a paired 95% bootstrap interval of approximately −0.67 to 3.00 points and McNemar p = 0.453125. This single-seed study does not establish a persistent-memory advantage. Calibration selected a detectable signal and latency budget; it did not establish an accuracy plateau across simulation-step settings. Full evidence and limitations are in [the final report](snapshot/runs/reports/t16-first-comparison.md).

[experiment.json](experiment.json) maps each run to its configuration hash, dataset fingerprint, trace key, trained model digest, and evaluation directories. Four trace caches belong to the frozen protocol; two earlier eight-window engineering caches are retained and explicitly labeled. Reset-concat reuses the main reset trace, rather than duplicating its simulation output.

## What is preserved

- `snapshot/runs/`: original effective and original YAML, manifests, environments, logs, fitted readout metadata and all numerical coefficients, validation curves, evaluation predictions, confusion counts, and timings. The report subdirectories preserve comparisons, calibration, CPU measurements, pixel controls, and Markdown findings.
- `snapshot/cache/features/`: all six computed Zarr traces, sample/split indexes, build metadata, per-chunk hashes, and execution logs.
- `preprocessing/`: exact selected sample indexes, fitted alignment templates, and metadata with expected aligned-image hashes. `images.npy` is deliberately absent because it contains transformed CelebA photographs.
- `provenance/acquisition/`: historical diagnostics, protocol freeze, timing measurements, parity checks, and experiment orchestration records. Original input downloads and screenshots are excluded.
- `environment/`: `uv.lock` and `pyproject.toml` from training commit `5c3fb076e84abd58f8320e52382ca6647034541c`. Every run also retains its measured package and CPU environment.

The snapshot is about 512 MiB before checkout overhead. Binary model coefficients and large feature chunks use Git LFS. JSON and reports are small, readable ordinary Git files. Historical absolute paths in original records describe the machine that produced them; restoration uses the relative paths and identities in this study, without editing those records.

## Retrieve and verify the computed results

Run from the repository root, with Git LFS installed:

```bash
git lfs install
git lfs pull --include='research/sequential-visual-memory/2026-09-20-celeba-poc1/**'
git lfs fsck
archive="$(pwd)/research/sequential-visual-memory/2026-09-20-celeba-poc1"
(cd "$archive" && sha256sum --check checksums.sha256)
```

Every check must report `OK`. A pointer file, partial download, or changed byte fails verification. Preserve the inventory and the Git commit containing it when distributing or backing up the archive.

Restore only into a **new, nonexistent directory**, so no existing experiment is overwritten:

```bash
export FLYSTATE_HOME="$HOME/data/flystate-restored-poc1"
mkdir -p "$(dirname "$FLYSTATE_HOME")"
mkdir "$FLYSTATE_HOME" && cp -a "$archive/snapshot/." "$FLYSTATE_HOME/"
uv sync --frozen
uv run flystate serve
```

If `mkdir` reports that the target already exists, choose a different target; do not copy over existing results. Open http://127.0.0.1:8765. Metrics, predictions, reports, and cached activity work immediately. Photographs need the externally acquired CelebA files and regenerated image cache; spatial neuron positions need the matching connectome below. Neither is needed to preserve or inspect the trained coefficients.

## Obtain the exact external inputs

[external-inputs.json](external-inputs.json) records the six original filenames, byte counts, SHA-256 values, versions, and direct source links. [external-inputs.sha256](external-inputs.sha256) is executable verification evidence. These files were verified against the actual bytes used, not merely an upstream filename.

Create a staging directory outside the repository:

```bash
inputs="$HOME/data/flystate-inputs-poc1"
mkdir -p "$inputs/brain" "$inputs/celeba"
curl --fail --location --retry 3 \
  'https://github.com/alextitonis/fly.ai/releases/download/brain-v1/brain.npz' \
  --output "$inputs/brain/brain.npz"
curl --fail --location --retry 3 \
  'https://github.com/alextitonis/fly.ai/releases/download/brain-v1/weights.npz' \
  --output "$inputs/brain/weights.npz"
```

These are the `brain-v1` files consumed by `flybrain==0.1.0`, not an arbitrary newer MaleCNS export. The project also offers `flystate brain download`; for this historical study, the explicitly pinned URLs and recorded SHA-256 remain authoritative.

Download these four official CelebA files into `$inputs/celeba/`, retaining their exact names:

| File | Exact Google Drive source |
| --- | --- |
| `img_align_celeba.zip` | [Aligned JPEG archive](https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view) |
| `identity_CelebA.txt` | [Identity annotations](https://drive.google.com/file/d/1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS/view) |
| `list_landmarks_align_celeba.txt` | [Aligned-image landmarks](https://drive.google.com/file/d/0B7EVK8r0v71pd0FJY3Blby1HUTQ/view) |
| `list_eval_partition.txt` | [Official partitions](https://drive.google.com/file/d/0B7EVK8r0v71pY0NSMzRuSXJEVkk/view) |

Use the browser download action when Drive requires confirmation. Source file IDs and MD5 values are independently listed by the [torchvision CelebA loader](https://github.com/pytorch/vision/blob/main/torchvision/datasets/celeba.py); PyTorch is not required by this project. This study uses the aligned JPEG archive, not the PNG, in-the-wild, HQ, CSV-mirror, or re-encoded variants.

The [official CelebA project](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) provides the access and non-commercial research terms. It states that identity annotations are released on request; if the identity link is inaccessible, contact the authors through that page and request `identity_CelebA.txt` for research. A changed mirror is not an accepted substitute. Upstream access and permanent availability cannot be guaranteed by this repository; a source outage leaves a documented missing dependency, while the calculated results remain archived.

Before extracting or registering anything, verify **all six inputs**:

```bash
(cd "$inputs" && sha256sum --check "$archive/external-inputs.sha256")
```

Stop on any mismatch. Do not rewrite the pinned checksums to accommodate a different download. After successful verification:

```bash
mkdir -p "$FLYSTATE_HOME/data/brain"
cp "$inputs/brain/brain.npz" "$inputs/brain/weights.npz" "$FLYSTATE_HOME/data/brain/"
unzip -q "$inputs/celeba/img_align_celeba.zip" -d "$inputs/celeba"
uv run flystate dataset register celeba "$inputs/celeba"
uv run flystate dataset validate celeba --full
```

The original archive contains 202,599 images and 10,177 identities. The experiment uses its recorded deterministic subsets, not the entire dataset as training input. Class labels map to original identity numbers through each archived trace's `index.parquet`; the official train/validation/test partition file is preserved as an external input even though this study uses configured per-identity splits.

## Regenerate photographs without retraining

Rebuild image caches from the **archived effective configurations**, leaving the saved traces and learned readouts unchanged:

```bash
uv run flystate dataset prepare "$archive/snapshot/runs/20260920-071344-celeba-smoke-9406ee/config.yaml"
uv run flystate dataset prepare "$archive/snapshot/runs/20260920-075340-celeba-persistent-541c94/config.yaml"
```

These two caches cover all five frozen-protocol runs. Compare the returned cache keys with the corresponding directories in `preprocessing/`, then verify rebuilt image, index, and template hashes:

```bash
(cd "$FLYSTATE_HOME" && sha256sum --check "$archive/preprocessed-artifacts.sha256")
```

Only the two frozen-protocol image caches are required by this check. Earlier engineering preprocessing metadata remains archived separately. Do not copy metadata-only `preprocessing/` directories into the working cache before rebuilding: those archived directories intentionally lack images. If regeneration differs, inspect the pinned original inputs and dependency versions; do not replace the historical hashes.

`FLYSTATE_HOME` may now be viewed with `flystate serve`, including training-example photographs and matching readout-neuron geometry. Experiments and evaluations can be executed in the restored working home, but newly generated results are not the original observations. Store a later research campaign under a new sibling study directory.
