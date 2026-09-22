### Sequential execution: `smoke-pixels-all-n08-seed0`

Exit code: 0. Completed.

```json
{
  "parameters": {
    "c_grid": [
      0.01,
      0.1,
      1.0,
      10.0
    ],
    "cv_folds": 5,
    "features": "both",
    "history": "all",
    "kind": "identity_probe",
    "label_mode": "true",
    "max_iterations": 50000,
    "pca_components": 60,
    "representation": "pixels",
    "seed": 0,
    "subset_seed": 0,
    "tolerance": 1e-06,
    "train_per_class": 8
  },
  "scores": {
    "train": {
      "accuracy": 1.0,
      "log_loss": 0.005635498991763422,
      "n": 160,
      "top5_accuracy": 1.0
    },
    "validation": {
      "accuracy": 0.43333333333333335,
      "log_loss": 3.286300549622899,
      "n": 60,
      "top5_accuracy": 0.75
    }
  },
  "cv_scores": {
    "0.01": 0.35625,
    "0.1": 0.375,
    "1.0": 0.3625,
    "10.0": 0.375
  },
  "selected_C": 0.1,
  "membership_sha256": "a13298d47caa4ab6b6ca7e9dec729d44e2409a4034455c6e83f383e6bd705c7c",
  "effective_cv_folds": 5,
  "fold_pca_dimensions": [
    60,
    60,
    60,
    60,
    60
  ],
  "inference": "exploratory; final-test and reserve images were not scored"
}
```

Working evidence: `$FLYSTATE_HOME/runs/diagnostics/2026-09-22-training-size-curve/smoke-pixels-all-n08-seed0/`. Full coefficients, predictions, provenance, and checksums will be archived in the sibling study directory. No final-test predictions were scored. This remains exploratory.
