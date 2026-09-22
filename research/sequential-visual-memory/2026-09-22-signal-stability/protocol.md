# Training-only signal stability and storage precision

Status: preregistered, not executed. Next bounded P1 step in #39 after the cached-feature diagnostics (#40) and fixed-cohort training-size study (#42). The neural learning curve stayed near chance while input controls improved. That observation does not establish whether noise, sampling, quantization, or dynamics causes the loss.

## Frozen design

Reuse the exact 40 training photographs (two per identity, 20 identities) from `smoke-neural-last-n02-seed0`. The ordered IDs and their canonical JSON SHA-256 are below. Never select images by neural responses or scores. No validation, historical test, or reserved images enter this measurement.

Keep the original persistent smoke geometry, encoder and encoder seed, neuron identities, readout feature definitions, 16 image patches, observation duration, drive amplitude, graph, and four Numba threads. Limit BLAS to one thread. Compute the original noise-enabled warmed rest state once with seed zero and reuse it for every condition, including the noise-off intervention. Changing warmup along with episode noise would confound the comparison.

Preserve native float32 features before storage. The initial native episode-noise condition uses the original seed and sample identifiers; its float16 cast must exactly match the archived trace rows before interpreting any intervention. Record both maximum and RMS quantization error, relative to signal differences and per-block variability, at every observation. Casting an old float16 cache to float32 is not this experiment.

Run 163 episodes in this declared order:

1. Native episode-specific noise on all 40 images, with the original seed and identifiers: baseline replay and storage-precision check.
2. All 40 images under one common noise stream, seed zero. Use one fixed noise identifier for every image while retaining the real sample ID in evidence.
3. Repeat those 40 images under a second common noise stream, seed one. Only the episode-noise seed changes; rest, encoder, trajectory, and readout seeds stay fixed.
4. All 40 images with episode noise disabled, from the same baseline warmed rest state.
5. Three zero-current episodes: common-noise seed zero, common-noise seed one, and noise off. Zero current means no visual or positional encoder injection, not a black image encoded with position features.

The common noise identifier is `signal-stability-common-noise`. Noise follows the existing `EpisodeBrain.begin` SeedSequence rule. Record the identifiers and seed derivations explicitly; real sample IDs never enter a learned feature. Native-noise and matched-noise conditions answer different questions and must not be pooled as independent replicates.

## Measurements and limits

For voltage and external spike-filter features separately and together, save per-observation native float32 arrays, neuronal index arrays, spike summaries, rest state, configurations, timing/RSS, and checksums. Compare same-image differences across matched noise seeds with different-image differences under the same noise seed. Distinguish within-identity from between-identity pairs and avoid treating dependent pairs as independent samples. Report descriptive per-observation distributions, not a significance claim from pair counts. Compare stimulus-minus-zero-current responses under each matching stream.

This stage fits no classifier and no neural parameters. It cannot establish held-out recognition or a memory benefit. Small float16 rounding error does not by itself prove classifier predictions are unaffected. If replay fails, stop scientific interpretation and diagnose the discrepancy before changing the protocol. Do not silently select a better seed, population, duration, or image subset.

## Engineering gate and reporting

Implement a dedicated immutable diagnostic command, with explicit `--help` and JSON output, before executing real data. Test native replay, independent noise controls, common-rest reuse, finite outputs, image ordering, and failure preservation on synthetic fixtures. Unknown/new studies must remain automatically discoverable in Diagnostics, with a concise recorded idea and results. Freeze the implementation commit before the first real simulation.

Publish each attempt, failure, and conclusion here; preserve results in a new sibling study directory and verify remote LFS restoration. This issue is a measurement protocol, not evidence that any root cause has already been identified.

## Frozen membership and case counts

```json
{
  "status": "preregistered; not executed",
  "sample_ids": [
    "celeba-003287",
    "celeba-005983",
    "celeba-012018",
    "celeba-021473",
    "celeba-021991",
    "celeba-022223",
    "celeba-024703",
    "celeba-029819",
    "celeba-039277",
    "celeba-042011",
    "celeba-045394",
    "celeba-047269",
    "celeba-047881",
    "celeba-055592",
    "celeba-056683",
    "celeba-059692",
    "celeba-059878",
    "celeba-062293",
    "celeba-067989",
    "celeba-075840",
    "celeba-077585",
    "celeba-091038",
    "celeba-101562",
    "celeba-103551",
    "celeba-105382",
    "celeba-110990",
    "celeba-117252",
    "celeba-117517",
    "celeba-140855",
    "celeba-142219",
    "celeba-150057",
    "celeba-158442",
    "celeba-160335",
    "celeba-161260",
    "celeba-170257",
    "celeba-177712",
    "celeba-183442",
    "celeba-186450",
    "celeba-189229",
    "celeba-196138"
  ],
  "sample_ids_sha256": "b64d3053b08ac88de2e3378d9c6875daec1801e3955d33aff87547e96a509f78",
  "selection": "Exact 2-per-identity seed-0 training subset from the completed training-size study; no validation, historical test, or reserve rows",
  "cases": {
    "native_episode_noise": 40,
    "matched_noise_seed_0": 40,
    "matched_noise_seed_1": 40,
    "noise_off": 40,
    "zero_current_matched_seed_0": 1,
    "zero_current_matched_seed_1": 1,
    "zero_current_noise_off": 1
  },
  "total_episodes": 163,
  "rest_seed": 0,
  "numba_threads": 4,
  "blas_threads": 1
}
```
