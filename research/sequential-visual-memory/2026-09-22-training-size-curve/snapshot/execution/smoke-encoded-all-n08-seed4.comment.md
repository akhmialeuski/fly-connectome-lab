### Sequential execution: `smoke-encoded-all-n08-seed4`

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
    "representation": "encoded",
    "seed": 0,
    "subset_seed": 4,
    "tolerance": 1e-06,
    "train_per_class": 8
  },
  "scores": {
    "train": {
      "accuracy": 1.0,
      "log_loss": 0.027382646644435137,
      "n": 160,
      "top5_accuracy": 1.0
    },
    "validation": {
      "accuracy": 0.5,
      "log_loss": 1.9969814686880492,
      "n": 60,
      "top5_accuracy": 0.8333333333333334
    }
  },
  "cv_scores": {
    "0.01": 0.375,
    "0.1": 0.36875,
    "1.0": 0.34375,
    "10.0": 0.35
  },
  "selected_C": 0.01,
  "membership_sha256": "f9c4971526ca995837d7d088f6b4ec212cc4a0d68fe79c3b44792f8e4d53f1be",
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

Working evidence: `$FLYSTATE_HOME/runs/diagnostics/2026-09-22-training-size-curve/smoke-encoded-all-n08-seed4/`. Full coefficients, predictions, provenance, and checksums will be archived in the sibling study directory. No final-test predictions were scored. This remains exploratory.
