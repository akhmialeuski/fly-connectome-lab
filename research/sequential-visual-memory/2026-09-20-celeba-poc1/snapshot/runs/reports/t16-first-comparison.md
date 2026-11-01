# T16: First frozen laptop comparison

The first POC1 comparison is complete. The frozen configuration does not provide convincing evidence that persistent state improves final face identification over reset state. Gate B decision: diagnose and revise the configuration on validation before beginning the viewer milestone.

This is a completed single-seed result, including a negative/inconclusive scientific outcome; no final-test tuning or source-code changes were made during the experiment.

## Frozen protocol

- Commit: `5c3fb076e84abd58f8320e52382ca6647034541c`; frozen at `2026-09-20T07:04:15.128275+00:00`.
- Code PRs: #19 through #33; configuration PR: #34. All remain stacked and unmerged.
- Official aligned CelebA: 202,599 images and 10,177 identities validated; selected 20 images per identity, with 14 train, three validation, and three test images.
- Smoke: 20 identities, 400 total images, 60 test images. Main: 100 identities, 2,000 total images, 300 test images.
- Sixteen 32x32 raster windows over aligned 128x128 images; amplitude 0.05; ten neural steps per observation; four Numba threads; batch size one; seed zero.
- DN spike-trace and voltage features; float16 trace storage and float32 readout inputs; fold-local StandardScaler/PCA, full SVD, five-fold train-only C selection, and logistic tolerance 1e-6.
- Models, trace caches, predictions, JSON reports, and all dataset files remain outside Git.

Configuration hashes:

- `celeba-persistent.yaml`: `541c9465b7a1ba18dfc583cc7b47b6a644d7a730860e0e272fe90e768cf95acf`
- `celeba-reset.yaml`: `72aac79cd2a174ad4cd716ad1cb61626e6f70ffac7a47908fe00da8883d149b6`
- `celeba-smoke.yaml`: `9406ee5b0f5b2ba794b71bad4def5d906248af35165f49a8fceb7f0c96fe82b3`

## Calibration and design checks

The smallest tested admissible amplitude is 0.05. Paired calibration gives first response at step 2, half-peak response at step 8, and DN activity 1.346 Hz. Only k=10 among {3,5,10} satisfies the prescribed half-peak latency rule; an empirical accuracy plateau across k was not established.

[Full raster calibration](https://github.com/akhmialeuski/fly-connectome-lab/issues/12#issuecomment-5748284097).

The original random-eight design failed the 10 pp validation gate. The frozen raster-sixteen pixel controls pass on both subsets. Geometry and observation count changed together, so their separate causal effects are not identified. Some individual windows are informative; the control does not prove that every single window is insufficient.

| Subset | All windows | Last window | Mean single window | Whole image | Gap (pp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| smoke | 53.33% | 16.67% | 31.35% | 53.33% | 36.67 |
| main | 42.67% | 3.33% | 21.52% | 41.67% | 39.33 |

Both design checks use validation data and the prescribed pixel logistic tolerance 1e-4.

## Laptop benchmark and trace wall times

Intel Core Ultra 5 235H, 14 logical CPUs, 16.47 GB RAM, WSL2. Best measured grid setting: 4 threads, batch 1, 2.3778 ms per episode-step. The five-minute sustained run changed from 2.5592 to 3.7707 ms per episode-step, ratio 1.4734. Slowdown was observed; a thermal cause was not established.

[Full benchmark evidence](https://github.com/akhmialeuski/fly-connectome-lab/issues/6#issuecomment-5739850685).

| Final trace build | Images | Observations | Full CLI seconds | Background numerical workload |
| --- | ---: | ---: | ---: | --- |
| smoke-persistent | 400 | 16 | 272.03 | Main pixel design check (partial overlap for main persistent) |
| smoke-reset | 400 | 16 | 272.49 | Main pixel design check (partial overlap for main persistent) |
| main-persistent | 2000 | 16 | 1155.47 | Main pixel design check (partial overlap for main persistent) |
| main-reset | 2000 | 16 | 1088.48 | None from this agent |

Times include CLI startup, cache preparation/verification, warmup, simulation, and storage. They are observed command wall times, not isolated kernel benchmark measurements. Reset-concat reuses the main reset cache and requires no additional simulation.

## Final test outcomes

| Comparison | Accuracy A | Accuracy B | Difference (pp) | 95% paired CI (pp) | McNemar p |
| --- | ---: | ---: | ---: | --- | ---: |
| Smoke persistent vs reset | 8.33% | 3.33% | 5.00 | [-3.33, 13.33] | 0.453125 |
| Main persistent vs reset | 1.67% | 0.67% | 1.00 | [-0.67, 3.00] | 0.453125 |
| Main persistent vs reset-concat | 1.67% | 1.33% | 0.33 | [-1.67, 2.33] | 1 |

Intervals use 10,000 paired image bootstrap resamples with seed zero. Final comparisons use exact McNemar tests because discordant counts are below 25. The main persistent/reset result represents five correct predictions versus two out of 300 images, with no jointly correct final predictions. Uniform-chance accuracy is 1% for the main task and 5% for smoke.

All three main readouts are close to chance. Final validation accuracy is 0.33% for persistent, 1.00% for reset, and 0.67% for reset-concat; selected final train-only CV accuracy is about 1.14% for each. Weak performance is therefore visible before final-test inspection. The usable identity signal in the current neural representation/readout pipeline is inadequate, but these measurements do not isolate encoding/noise, PCA compression, or population selection as the root cause.

Gate B decision: revise and diagnose the configuration using validation before building the viewer. Treat the current test set as already inspected when designing any subsequent confirmatory experiment. This result does not establish the absence of memory in every MaleCNS configuration, and supports neither multi-seed generalization nor a claim about biological fly memory.

## Integrity and verification

- Both controls have bitwise-identical first-observation features for all 400 smoke and 2,000 main images.
- First-observation prediction records match exactly for all 60 smoke and 300 main test images.
- All five trained runs record the clean frozen source commit and completed status. Reset-concat uses the same reset cache.
- Comparisons verify complete unique sample-observation pairs, matching labels, numerical provenance, trajectory and dataset identity, and permitted configuration differences.
- Full local suite: 270 passed, one optional connectome test skipped; package line and branch coverage is 100%. Static checks, reST docstrings, pre-commit, builds, and GitHub CI passed. Real CLI runs provide additional connectome validation.

Run identities:

- smoke-persistent: `20260920-071344-celeba-smoke-9406ee`.
- smoke-reset: `20260920-071430-celeba-smoke-400cac`.
- main-persistent: `20260920-075340-celeba-persistent-541c94`.
- main-reset: `20260920-080049-celeba-reset-72aac7`.
- main-concat: `20260920-080640-celeba-reset-concat-c29bf2`.

## Smoke per-observation comparison

Split: test. Bootstrap resamples: 10000; seed: 0.

| t | Accuracy A | Accuracy B | Difference (pp) | 95% CI (pp) | A only | B only | p | Method |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 1 | 10.00% | 10.00% | 0.00 | [0.00, 0.00] | 0 | 0 | 1 | exact |
| 2 | 8.33% | 1.67% | 6.67 | [0.00, 15.00] | 5 | 1 | 0.21875 | exact |
| 3 | 5.00% | 1.67% | 3.33 | [-3.33, 10.00] | 3 | 1 | 0.625 | exact |
| 4 | 5.00% | 1.67% | 3.33 | [-3.33, 10.00] | 3 | 1 | 0.625 | exact |
| 5 | 6.67% | 5.00% | 1.67 | [-6.67, 10.00] | 4 | 3 | 1 | exact |
| 6 | 3.33% | 5.00% | -1.67 | [-8.33, 3.33] | 1 | 2 | 1 | exact |
| 7 | 1.67% | 8.33% | -6.67 | [-15.00, 0.00] | 1 | 5 | 0.21875 | exact |
| 8 | 3.33% | 8.33% | -5.00 | [-13.33, 3.33] | 2 | 5 | 0.453125 | exact |
| 9 | 6.67% | 1.67% | 5.00 | [-1.67, 13.33] | 4 | 1 | 0.375 | exact |
| 10 | 3.33% | 6.67% | -3.33 | [-11.67, 5.00] | 2 | 4 | 0.6875 | exact |
| 11 | 5.00% | 8.33% | -3.33 | [-11.67, 5.00] | 2 | 4 | 0.6875 | exact |
| 12 | 3.33% | 5.00% | -1.67 | [-10.00, 5.00] | 2 | 3 | 1 | exact |
| 13 | 3.33% | 3.33% | 0.00 | [-6.67, 6.67] | 2 | 2 | 1 | exact |
| 14 | 8.33% | 5.00% | 3.33 | [-5.00, 13.33] | 5 | 3 | 0.726562 | exact |
| 15 | 5.00% | 8.33% | -3.33 | [-13.33, 5.00] | 3 | 5 | 0.726562 | exact |
| 16 | 8.33% | 3.33% | 5.00 | [-3.33, 13.33] | 5 | 2 | 0.453125 | exact |

Final memory effect (persistent minus reset): 5.00 pp; 95% paired interval [-3.33, 13.33] pp; McNemar p=0.453125.

## Main persistent versus reset

Split: test. Bootstrap resamples: 10000; seed: 0.

| t | Accuracy A | Accuracy B | Difference (pp) | 95% CI (pp) | A only | B only | p | Method |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 1 | 2.00% | 2.00% | 0.00 | [0.00, 0.00] | 0 | 0 | 1 | exact |
| 2 | 0.67% | 0.67% | 0.00 | [-1.33, 1.33] | 2 | 2 | 1 | exact |
| 3 | 2.33% | 1.33% | 1.00 | [-1.00, 3.33] | 7 | 4 | 0.548828 | exact |
| 4 | 0.67% | 0.33% | 0.33 | [-0.67, 1.67] | 2 | 1 | 1 | exact |
| 5 | 1.00% | 0.33% | 0.67 | [-0.67, 2.00] | 3 | 1 | 0.625 | exact |
| 6 | 1.67% | 0.33% | 1.33 | [0.33, 2.67] | 4 | 0 | 0.125 | exact |
| 7 | 0.67% | 0.33% | 0.33 | [-0.67, 1.67] | 2 | 1 | 1 | exact |
| 8 | 1.00% | 1.33% | -0.33 | [-2.00, 1.33] | 3 | 4 | 1 | exact |
| 9 | 1.00% | 1.33% | -0.33 | [-2.00, 1.33] | 3 | 4 | 1 | exact |
| 10 | 1.00% | 1.33% | -0.33 | [-2.00, 1.33] | 3 | 4 | 1 | exact |
| 11 | 0.67% | 1.67% | -1.00 | [-3.00, 0.67] | 2 | 5 | 0.453125 | exact |
| 12 | 1.00% | 1.00% | 0.00 | [-1.67, 1.67] | 3 | 3 | 1 | exact |
| 13 | 1.67% | 0.67% | 1.00 | [-0.67, 2.67] | 5 | 2 | 0.453125 | exact |
| 14 | 1.00% | 1.67% | -0.67 | [-2.33, 1.00] | 2 | 4 | 0.6875 | exact |
| 15 | 1.67% | 1.33% | 0.33 | [-1.67, 2.33] | 5 | 4 | 1 | exact |
| 16 | 1.67% | 0.67% | 1.00 | [-0.67, 3.00] | 5 | 2 | 0.453125 | exact |

Final memory effect (persistent minus reset): 1.00 pp; 95% paired interval [-0.67, 3.00] pp; McNemar p=0.453125.

## Main persistent versus reset-concat

Split: test. Bootstrap resamples: 10000; seed: 0.

| t | Accuracy A | Accuracy B | Difference (pp) | 95% CI (pp) | A only | B only | p | Method |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 1 | 2.00% | 2.00% | 0.00 | [0.00, 0.00] | 0 | 0 | 1 | exact |
| 2 | 0.67% | 1.00% | -0.33 | [-1.67, 1.00] | 2 | 3 | 1 | exact |
| 3 | 2.33% | 0.33% | 2.00 | [0.33, 4.00] | 7 | 1 | 0.0703125 | exact |
| 4 | 0.67% | 0.67% | 0.00 | [-1.00, 1.00] | 1 | 1 | 1 | exact |
| 5 | 1.00% | 0.33% | 0.67 | [-0.67, 2.00] | 3 | 1 | 0.625 | exact |
| 6 | 1.67% | 0.33% | 1.33 | [-0.33, 3.00] | 5 | 1 | 0.21875 | exact |
| 7 | 0.67% | 1.67% | -1.00 | [-2.67, 0.67] | 2 | 5 | 0.453125 | exact |
| 8 | 1.00% | 0.67% | 0.33 | [-1.00, 1.67] | 3 | 2 | 1 | exact |
| 9 | 1.00% | 0.00% | 1.00 | [0.00, 2.33] | 3 | 0 | 0.25 | exact |
| 10 | 1.00% | 0.67% | 0.33 | [-1.00, 1.67] | 3 | 2 | 1 | exact |
| 11 | 0.67% | 1.00% | -0.33 | [-1.67, 1.00] | 2 | 3 | 1 | exact |
| 12 | 1.00% | 2.00% | -1.00 | [-3.00, 1.00] | 3 | 6 | 0.507812 | exact |
| 13 | 1.67% | 0.67% | 1.00 | [-0.67, 2.67] | 5 | 2 | 0.453125 | exact |
| 14 | 1.00% | 0.00% | 1.00 | [0.00, 2.33] | 3 | 0 | 0.25 | exact |
| 15 | 1.67% | 1.33% | 0.33 | [-1.67, 2.33] | 5 | 4 | 1 | exact |
| 16 | 1.67% | 1.33% | 0.33 | [-1.67, 2.33] | 5 | 4 | 1 | exact |

<details>
<summary>Full final smoke validation design-check JSON</summary>

```json
{
  "baselines": {
    "all_windows": {
      "C": 1.0,
      "accuracy": 0.5333333333333333,
      "ci_high": 0.653720939585615,
      "ci_low": 0.40893427027157897,
      "ci_method": "wilson",
      "cv_scores": {
        "0.01": 0.5071428571428571,
        "0.1": 0.5142857142857142,
        "1.0": 0.5178571428571428,
        "10.0": 0.5178571428571428
      }
    },
    "first_window": {
      "C": 10.0,
      "accuracy": 0.13333333333333333,
      "ci_high": 0.24165159676499612,
      "ci_low": 0.06914109480586975,
      "ci_method": "wilson",
      "cv_scores": {
        "0.01": 0.16428571428571428,
        "0.1": 0.16071428571428573,
        "1.0": 0.15714285714285717,
        "10.0": 0.16785714285714284
      }
    },
    "last_window": {
      "C": 0.01,
      "accuracy": 0.16666666666666666,
      "ci_high": 0.2803161316361453,
      "ci_low": 0.09313176979191454,
      "ci_method": "wilson",
      "cv_scores": {
        "0.01": 0.11428571428571428,
        "0.1": 0.09642857142857145,
        "1.0": 0.08571428571428572,
        "10.0": 0.07857142857142857
      }
    },
    "single_window_mean": {
      "C": [
        10.0,
        10.0,
        0.01,
        0.01,
        0.01,
        0.1,
        0.1,
        0.1,
        0.01,
        0.1,
        1.0,
        0.01,
        0.01,
        0.1,
        1.0,
        0.01
      ],
      "accuracy": 0.31354166666666666,
      "ci_high": 0.365625,
      "ci_low": 0.26458333333333334,
      "ci_method": "image_bootstrap"
    },
    "whole_image": {
      "C": 0.1,
      "accuracy": 0.5333333333333333,
      "ci_high": 0.653720939585615,
      "ci_low": 0.40893427027157897,
      "ci_method": "wilson",
      "cv_scores": {
        "0.01": 0.5071428571428571,
        "0.1": 0.5178571428571428,
        "1.0": 0.5178571428571428,
        "10.0": 0.5142857142857142
      }
    }
  },
  "config_hash": "9406ee5b0f5b2ba794b71bad4def5d906248af35165f49a8fceb7f0c96fe82b3",
  "config_name": "celeba-smoke",
  "created_utc": "2026-09-20T07:03:05.345496+00:00",
  "dataset_fingerprint": "92adc8cc97da3de2e8af26e08ceb7cd37f5dbaa92fb60bc50c46f43527de794d",
  "evaluation_split": "val",
  "gap_pp": 36.66666666666667,
  "message": "memory gap meets threshold",
  "n_classes": 20,
  "n_eval": 60,
  "n_train": 280,
  "output": "/home/anatolk/data/flystate/runs/design-checks/20260920-070305-celeba-smoke-33c778bb.json",
  "schema_version": 1,
  "status": "ok",
  "trajectory_hash": "a96a2d52f7cf2db48c7e4ebcfa3d519dd617630cf71fed36aa1bfd164b5f9b1d",
  "window_baselines": [
    {
      "C": 10.0,
      "accuracy": 0.13333333333333333,
      "ci_high": 0.24165159676499612,
      "ci_low": 0.06914109480586975,
      "ci_method": "wilson",
      "cv_scores": {
        "0.01": 0.16428571428571428,
        "0.1": 0.16071428571428573,
        "1.0": 0.15714285714285717,
        "10.0": 0.16785714285714284
      }
    },
    {
      "C": 10.0,
      "accuracy": 0.4166666666666667,
      "ci_high": 0.5427191897735988,
      "ci_low": 0.30064278558341617,
      "ci_method": "wilson",
      "cv_scores": {
        "0.01": 0.35,
        "0.1": 0.34285714285714286,
        "1.0": 0.35,
        "10.0": 0.35357142857142865
      }
    },
    {
      "C": 0.01,
      "accuracy": 0.4666666666666667,
      "ci_high": 0.591065729728421,
      "ci_low": 0.34627906041438494,
      "ci_method": "wilson",
      "cv_scores": {
        "0.01": 0.35357142857142854,
        "0.1": 0.35,
        "1.0": 0.35,
        "10.0": 0.33571428571428574
      }
    },
    {
      "C": 0.01,
      "accuracy": 0.21666666666666667,
      "ci_high": 0.3362002757319478,
      "ci_low": 0.1312304404819031,
      "ci_method": "wilson",
      "cv_scores": {
        "0.01": 0.17857142857142858,
        "0.1": 0.1642857142857143,
        "1.0": 0.15714285714285714,
        "10.0": 0.1392857142857143
      }
    },
    {
      "C": 0.01,
      "accuracy": 0.16666666666666666,
      "ci_high": 0.2803161316361453,
      "ci_low": 0.09313176979191454,
      "ci_method": "wilson",
      "cv_scores": {
        "0.01": 0.2,
        "0.1": 0.18571428571428572,
        "1.0": 0.16785714285714287,
        "10.0": 0.17142857142857143
      }
    },
    {
      "C": 0.1,
      "accuracy": 0.5333333333333333,
      "ci_high": 0.653720939585615,
      "ci_low": 0.40893427027157897,
      "ci_method": "wilson",
      "cv_scores": {
        "0.01": 0.5,
        "0.1": 0.5214285714285716,
        "1.0": 0.5214285714285715,
        "10.0": 0.5178571428571429
      }
    },
    {
      "C": 0.1,
      "accuracy": 0.5666666666666667,
      "ci_high": 0.6842760319531322,
      "ci_low": 0.44103438776125564,
      "ci_method": "wilson",
      "cv_scores": {
        "0.01": 0.5107142857142857,
        "0.1": 0.5321428571428573,
        "1.0": 0.525,
        "10.0": 0.5321428571428573
      }
    },
    {
      "C": 0.1,
      "accuracy": 0.2833333333333333,
      "ci_high": 0.40767285164391176,
      "ci_low": 0.1850682842843271,
      "ci_method": "wilson",
      "cv_scores": {
        "0.01": 0.22142857142857145,
        "0.1": 0.22857142857142856,
        "1.0": 0.21428571428571427,
        "10.0": 0.19642857142857142
      }
    },
    {
      "C": 0.01,
      "accuracy": 0.08333333333333333,
      "ci_high": 0.18068942018334405,
      "ci_low": 0.03612045660173077,
      "ci_method": "wilson",
      "cv_scores": {
        "0.01": 0.16428571428571428,
        "0.1": 0.15357142857142858,
        "1.0": 0.1392857142857143,
        "10.0": 0.13214285714285715
      }
    },
    {
      "C": 0.1,
      "accuracy": 0.5,
      "ci_high": 0.6226497575844423,
      "ci_low": 0.3773502424155577,
      "ci_method": "wilson",
      "cv_scores": {
        "0.01": 0.4428571428571429,
        "0.1": 0.46071428571428574,
        "1.0": 0.4571428571428572,
        "10.0": 0.4571428571428572
      }
    },
    {
      "C": 1.0,
      "accuracy": 0.43333333333333335,
      "ci_high": 0.5589656122387443,
      "ci_low": 0.31572396804686764,
      "ci_method": "wilson",
      "cv_scores": {
        "0.01": 0.3928571428571429,
        "0.1": 0.4107142857142857,
        "1.0": 0.43571428571428567,
        "10.0": 0.42857142857142855
      }
    },
    {
      "C": 0.01,
      "accuracy": 0.16666666666666666,
      "ci_high": 0.2803161316361453,
      "ci_low": 0.09313176979191454,
      "ci_method": "wilson",
      "cv_scores": {
        "0.01": 0.14285714285714285,
        "0.1": 0.1392857142857143,
        "1.0": 0.1285714285714286,
        "10.0": 0.11428571428571428
      }
    },
    {
      "C": 0.01,
      "accuracy": 0.13333333333333333,
      "ci_high": 0.24165159676499612,
      "ci_low": 0.06914109480586975,
      "ci_method": "wilson",
      "cv_scores": {
        "0.01": 0.10357142857142858,
        "0.1": 0.07857142857142856,
        "1.0": 0.07857142857142858,
        "10.0": 0.09285714285714286
      }
    },
    {
      "C": 0.1,
      "accuracy": 0.43333333333333335,
      "ci_high": 0.5589656122387443,
      "ci_low": 0.31572396804686764,
      "ci_method": "wilson",
      "cv_scores": {
        "0.01": 0.3571428571428571,
        "0.1": 0.3857142857142858,
        "1.0": 0.3821428571428571,
        "10.0": 0.38571428571428573
      }
    },
    {
      "C": 1.0,
      "accuracy": 0.31666666666666665,
      "ci_high": 0.4423376702617676,
      "ci_low": 0.2130586755236652,
      "ci_method": "wilson",
      "cv_scores": {
        "0.01": 0.375,
        "0.1": 0.37142857142857144,
        "1.0": 0.37857142857142856,
        "10.0": 0.37857142857142856
      }
    },
    {
      "C": 0.01,
      "accuracy": 0.16666666666666666,
      "ci_high": 0.2803161316361453,
      "ci_low": 0.09313176979191454,
      "ci_method": "wilson",
      "cv_scores": {
        "0.01": 0.11428571428571428,
        "0.1": 0.09642857142857145,
        "1.0": 0.08571428571428572,
        "10.0": 0.07857142857142857
      }
    }
  ]
}
```

</details>

<details>
<summary>Full final main validation design-check JSON</summary>

```json
{
  "baselines": {
    "all_windows": {
      "C": 0.1,
      "accuracy": 0.4266666666666667,
      "ci_high": 0.4832140599409689,
      "ci_low": 0.3719735758386311,
      "ci_method": "wilson",
      "cv_scores": {
        "0.01": 0.4078571428571428,
        "0.1": 0.42000000000000004,
        "1.0": 0.41571428571428576,
        "10.0": 0.4121428571428572
      }
    },
    "first_window": {
      "C": 0.1,
      "accuracy": 0.08333333333333333,
      "ci_high": 0.12012160196406073,
      "ci_low": 0.05708087405639427,
      "ci_method": "wilson",
      "cv_scores": {
        "0.01": 0.06357142857142857,
        "0.1": 0.06785714285714285,
        "1.0": 0.06428571428571428,
        "10.0": 0.062142857142857146
      }
    },
    "last_window": {
      "C": 0.01,
      "accuracy": 0.03333333333333333,
      "ci_high": 0.060261825804054814,
      "ci_low": 0.01820494733885482,
      "ci_method": "wilson",
      "cv_scores": {
        "0.01": 0.04714285714285714,
        "0.1": 0.03571428571428571,
        "1.0": 0.04071428571428572,
        "10.0": 0.03785714285714285
      }
    },
    "single_window_mean": {
      "C": [
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        1.0,
        1.0,
        0.01,
        0.01,
        1.0,
        1.0,
        0.1,
        0.1,
        1.0,
        0.1,
        0.01
      ],
      "accuracy": 0.21520833333333333,
      "ci_high": 0.235,
      "ci_low": 0.195625,
      "ci_method": "image_bootstrap"
    },
    "whole_image": {
      "C": 0.1,
      "accuracy": 0.4166666666666667,
      "ci_high": 0.47316440353973044,
      "ci_low": 0.3622760916643605,
      "ci_method": "wilson",
      "cv_scores": {
        "0.01": 0.40928571428571425,
        "0.1": 0.41571428571428576,
        "1.0": 0.4142857142857143,
        "10.0": 0.4135714285714286
      }
    }
  },
  "config_hash": "541c9465b7a1ba18dfc583cc7b47b6a644d7a730860e0e272fe90e768cf95acf",
  "config_name": "celeba-persistent",
  "created_utc": "2026-09-20T07:28:16.232088+00:00",
  "dataset_fingerprint": "b42902f2ef40b6ed3ccb19916e94efcb30457c29275ed8d00bedd626e09d86fc",
  "evaluation_split": "val",
  "gap_pp": 39.333333333333336,
  "message": "memory gap meets threshold",
  "n_classes": 100,
  "n_eval": 300,
  "n_train": 1400,
  "output": "/home/anatolk/data/flystate/runs/design-checks/20260920-072816-celeba-persistent-105233cc.json",
  "schema_version": 1,
  "status": "ok",
  "trajectory_hash": "1ffaded8f57dde948d9bd7741d15043629dc3da904ccb3c26d217299f49bdf9a",
  "window_baselines": [
    {
      "C": 0.1,
      "accuracy": 0.08333333333333333,
      "ci_high": 0.12012160196406073,
      "ci_low": 0.05708087405639427,
      "ci_method": "wilson",
      "cv_scores": {
        "0.01": 0.06357142857142857,
        "0.1": 0.06785714285714285,
        "1.0": 0.06428571428571428,
        "10.0": 0.062142857142857146
      }
    },
    {
      "C": 0.1,
      "accuracy": 0.21666666666666667,
      "ci_high": 0.2667098486240773,
      "ci_low": 0.1737878350698321,
      "ci_method": "wilson",
      "cv_scores": {
        "0.01": 0.1807142857142857,
        "0.1": 0.21000000000000002,
        "1.0": 0.205,
        "10.0": 0.2
      }
    },
    {
      "C": 0.1,
      "accuracy": 0.24333333333333335,
      "ci_high": 0.2949351839778819,
      "ci_low": 0.1982215412507184,
      "ci_method": "wilson",
      "cv_scores": {
        "0.01": 0.17214285714285715,
        "0.1": 0.20857142857142857,
        "1.0": 0.19285714285714287,
        "10.0": 0.19142857142857142
      }
    },
    {
      "C": 0.1,
      "accuracy": 0.06,
      "ci_high": 0.09283944515290843,
      "ci_low": 0.038286369524692054,
      "ci_method": "wilson",
      "cv_scores": {
        "0.01": 0.0707142857142857,
        "0.1": 0.07928571428571428,
        "1.0": 0.0757142857142857,
        "10.0": 0.0657142857142857
      }
    },
    {
      "C": 0.1,
      "accuracy": 0.12,
      "ci_high": 0.16165781382701602,
      "ci_low": 0.08795084430363892,
      "ci_method": "wilson",
      "cv_scores": {
        "0.01": 0.09142857142857143,
        "0.1": 0.10071428571428573,
        "1.0": 0.095,
        "10.0": 0.08785714285714286
      }
    },
    {
      "C": 1.0,
      "accuracy": 0.49666666666666665,
      "ci_high": 0.5529280602079671,
      "ci_low": 0.4404895596001963,
      "ci_method": "wilson",
      "cv_scores": {
        "0.01": 0.4171428571428571,
        "0.1": 0.4442857142857143,
        "1.0": 0.4492857142857143,
        "10.0": 0.44571428571428573
      }
    },
    {
      "C": 1.0,
      "accuracy": 0.4766666666666667,
      "ci_high": 0.533121676782833,
      "ci_low": 0.42080166187431245,
      "ci_method": "wilson",
      "cv_scores": {
        "0.01": 0.41500000000000004,
        "0.1": 0.4392857142857142,
        "1.0": 0.4407142857142857,
        "10.0": 0.4364285714285714
      }
    },
    {
      "C": 0.01,
      "accuracy": 0.12666666666666668,
      "ci_high": 0.16908117004926165,
      "ci_low": 0.09369224846506603,
      "ci_method": "wilson",
      "cv_scores": {
        "0.01": 0.12214285714285714,
        "0.1": 0.10571428571428572,
        "1.0": 0.10214285714285713,
        "10.0": 0.09928571428571428
      }
    },
    {
      "C": 0.01,
      "accuracy": 0.07,
      "ci_high": 0.10463601042593322,
      "ci_low": 0.04623694482717634,
      "ci_method": "wilson",
      "cv_scores": {
        "0.01": 0.08857142857142856,
        "0.1": 0.07928571428571426,
        "1.0": 0.07857142857142857,
        "10.0": 0.07285714285714286
      }
    },
    {
      "C": 1.0,
      "accuracy": 0.4166666666666667,
      "ci_high": 0.47316440353973044,
      "ci_low": 0.3622760916643605,
      "ci_method": "wilson",
      "cv_scores": {
        "0.01": 0.31714285714285717,
        "0.1": 0.355,
        "1.0": 0.35642857142857143,
        "10.0": 0.35
      }
    },
    {
      "C": 1.0,
      "accuracy": 0.39,
      "ci_high": 0.44625143203840895,
      "ci_low": 0.33653002163099116,
      "ci_method": "wilson",
      "cv_scores": {
        "0.01": 0.3242857142857143,
        "0.1": 0.3392857142857143,
        "1.0": 0.3464285714285714,
        "10.0": 0.3464285714285714
      }
    },
    {
      "C": 0.1,
      "accuracy": 0.07333333333333333,
      "ci_high": 0.10853134160571733,
      "ci_low": 0.04892399383922859,
      "ci_method": "wilson",
      "cv_scores": {
        "0.01": 0.07428571428571429,
        "0.1": 0.07999999999999999,
        "1.0": 0.06571428571428571,
        "10.0": 0.05428571428571429
      }
    },
    {
      "C": 0.1,
      "accuracy": 0.03,
      "ci_high": 0.056022539141640945,
      "ci_low": 0.01586185380943231,
      "ci_method": "wilson",
      "cv_scores": {
        "0.01": 0.04071428571428572,
        "0.1": 0.047857142857142855,
        "1.0": 0.045,
        "10.0": 0.04285714285714285
      }
    },
    {
      "C": 1.0,
      "accuracy": 0.32666666666666666,
      "ci_high": 0.3816377910762407,
      "ci_low": 0.2760784389482685,
      "ci_method": "wilson",
      "cv_scores": {
        "0.01": 0.24857142857142858,
        "0.1": 0.25857142857142856,
        "1.0": 0.2607142857142857,
        "10.0": 0.2607142857142857
      }
    },
    {
      "C": 0.1,
      "accuracy": 0.28,
      "ci_high": 0.33334388521084696,
      "ci_low": 0.23221902212795326,
      "ci_method": "wilson",
      "cv_scores": {
        "0.01": 0.2678571428571429,
        "0.1": 0.28,
        "1.0": 0.26642857142857146,
        "10.0": 0.2671428571428572
      }
    },
    {
      "C": 0.01,
      "accuracy": 0.03333333333333333,
      "ci_high": 0.060261825804054814,
      "ci_low": 0.01820494733885482,
      "ci_method": "wilson",
      "cv_scores": {
        "0.01": 0.04714285714285714,
        "0.1": 0.03571428571428571,
        "1.0": 0.04071428571428572,
        "10.0": 0.03785714285714285
      }
    }
  ]
}
```

</details>

Local artifacts: `/home/anatolk/data/flystate/runs/reports/` and `/home/anatolk/data/flystate/acquisition/t16-*`. Original Knowledge Base notes were not modified; private source corrections are excluded from Git and every PR.
