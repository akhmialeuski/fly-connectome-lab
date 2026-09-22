## Completed diagnostic archive: independent remote restore verified

Archive commit: `f6a5d004db87fe7d33f56648edc5e701f906d016`, PR #41. All 556 tracked study files match the SHA-256 inventory. The 14 new Git LFS payloads (339,223,770 bytes) were fetched from origin into a separate empty object store and independently hashed; every byte matches the archived inventory. Earlier smoke and main payloads were verified in separate restores.

The completed cached campaign preserves two cohort audits, 26 full-probe attempts (22 fitted models and four original failures), four numerical diagnoses, and eight numerical fold parameter sets. All 12,280 stored prediction rows replay from exported coefficients; all eight saved objective/gradient measurements replay too. Eight new attempt comments were read back from GitHub and matched to their saved bodies. Numerical-source and archive-commit CI both pass.

This completes the cached diagnostic and numerical evidence for #40. It does not complete the broader #39 roadmap: training-size curves (#42), paired noise/quantization diagnostics, signal delivery, and evidence-gated dynamics/plasticity remain. No improved fly recognition or memory advantage is claimed.

Remote payload verification:

```json
{
  "commit": "f6a5d004db87fe7d33f56648edc5e701f906d016",
  "remote": "origin",
  "verification": "Independent Git LFS fetch into a separate empty object store; every payload matches the archived SHA-256 inventory.",
  "objects": [
    {
      "path": "snapshot/runs/diagnostics/2026-09-21-fly-only-diagnostics/convergence-main-01-pixels-persistent-all-both-pca60-true/fold-1-budget-20000/weights.npz",
      "sha256": "5dc5c1ba3e02ca9d060f2845a4abc94b9c1121cdd02508bdc35b078975ebe088",
      "bytes": 24823984
    },
    {
      "path": "snapshot/runs/diagnostics/2026-09-21-fly-only-diagnostics/convergence-main-01-pixels-persistent-all-both-pca60-true/fold-1-budget-5000/weights.npz",
      "sha256": "4ef00773d0ad34fc494c2cc075ebfc045bbb69d7fb288c8d7723b6583293e37e",
      "bytes": 24823984
    },
    {
      "path": "snapshot/runs/diagnostics/2026-09-21-fly-only-diagnostics/convergence-main-02-encoded-persistent-all-both-pca60-true/fold-1-budget-20000/weights.npz",
      "sha256": "2362bd253a6e48983e8a9871ba3a2b6a049c5178c7b8715972d8799ad423ed53",
      "bytes": 31275184
    },
    {
      "path": "snapshot/runs/diagnostics/2026-09-21-fly-only-diagnostics/convergence-main-02-encoded-persistent-all-both-pca60-true/fold-1-budget-5000/weights.npz",
      "sha256": "1d189333277c94e6ebf02ff95b9c99fff007067dfb94121cc8b1fb5fef1b00fe",
      "bytes": 31275184
    },
    {
      "path": "snapshot/runs/diagnostics/2026-09-21-fly-only-diagnostics/convergence-smoke-02-pixels-persistent-all-both-pca60-true/fold-1-budget-20000/weights.npz",
      "sha256": "21d4cb0b0ed909c761ebcb942bf538c48aec847bed22913cbd16ccb8a2b53133",
      "bytes": 24784304
    },
    {
      "path": "snapshot/runs/diagnostics/2026-09-21-fly-only-diagnostics/convergence-smoke-02-pixels-persistent-all-both-pca60-true/fold-1-budget-5000/weights.npz",
      "sha256": "6432196385b9fc14f44f1636968f0e34a80237cd00f56a8c83e5eef5b597953f",
      "bytes": 24784304
    },
    {
      "path": "snapshot/runs/diagnostics/2026-09-21-fly-only-diagnostics/convergence-smoke-04-encoded-persistent-all-both-pca60-true/fold-5-budget-20000/weights.npz",
      "sha256": "c3e998f9253a38960570a5beb7c27c5e9f676408f515403194a670cd050fe78b",
      "bytes": 31235504
    },
    {
      "path": "snapshot/runs/diagnostics/2026-09-21-fly-only-diagnostics/convergence-smoke-04-encoded-persistent-all-both-pca60-true/fold-5-budget-5000/weights.npz",
      "sha256": "7384532c2f67b6317f64d55ac72cda80ce9ac036f4c6e0cd2305b62953062edc",
      "bytes": 31235504
    },
    {
      "path": "snapshot/runs/diagnostics/2026-09-21-fly-only-diagnostics/retry50000-main-01-pixels-persistent-all-both-pca60-true/model/weights.npz",
      "sha256": "7324cb8c5c1ff6bda3664f73a44087bf0c495806400aca0b84c0cb73cb5bdfe2",
      "bytes": 24823984
    },
    {
      "path": "snapshot/runs/diagnostics/2026-09-21-fly-only-diagnostics/retry50000-main-01-pixels-persistent-all-both-pca60-true/train-predictions.parquet",
      "sha256": "eab6d08660db46cae30ec51a1c03ee855746156fc030907aefa5d4d2fc982ab1",
      "bytes": 1433421
    },
    {
      "path": "snapshot/runs/diagnostics/2026-09-21-fly-only-diagnostics/retry50000-main-02-encoded-persistent-all-both-pca60-true/model/weights.npz",
      "sha256": "acac8cca976b942f7d8f4e15e03cd8ed38e860556013b37393a853bee8f4c5f1",
      "bytes": 31275184
    },
    {
      "path": "snapshot/runs/diagnostics/2026-09-21-fly-only-diagnostics/retry50000-main-02-encoded-persistent-all-both-pca60-true/train-predictions.parquet",
      "sha256": "af072c30c57e9bbc673e8ec7d5d600297ab70aa5cfd4214151a3f8898707f8a2",
      "bytes": 1433421
    },
    {
      "path": "snapshot/runs/diagnostics/2026-09-21-fly-only-diagnostics/retry50000-smoke-02-pixels-persistent-all-both-pca60-true/model/weights.npz",
      "sha256": "88c878b9476cbb811730a4bad61fb1a3eb59ead6822551d4b3167babebe34f02",
      "bytes": 24784304
    },
    {
      "path": "snapshot/runs/diagnostics/2026-09-21-fly-only-diagnostics/retry50000-smoke-04-encoded-persistent-all-both-pca60-true/model/weights.npz",
      "sha256": "1bac556f1508b87467746b442a26f4a7a0ceed778d86b005d38a5fb0d9ccb5f7",
      "bytes": 31235504
    }
  ],
  "total_bytes": 339223770
}
```
