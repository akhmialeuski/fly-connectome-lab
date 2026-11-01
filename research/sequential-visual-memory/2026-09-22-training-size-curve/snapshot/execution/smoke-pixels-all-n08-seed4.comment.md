### Sequential execution: `smoke-pixels-all-n08-seed4`

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
    "subset_seed": 4,
    "tolerance": 1e-06,
    "train_per_class": 8
  },
  "scores": {
    "train": {
      "accuracy": 1.0,
      "log_loss": 9.13931511183163e-05,
      "n": 160,
      "top5_accuracy": 1.0
    },
    "validation": {
      "accuracy": 0.5166666666666667,
      "log_loss": 3.58476565158563,
      "n": 60,
      "top5_accuracy": 0.85
    }
  },
  "cv_scores": {
    "0.01": 0.35,
    "0.1": 0.375,
    "1.0": 0.36875,
    "10.0": 0.38125
  },
  "selected_C": 10.0,
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

Working evidence: `$FLYSTATE_HOME/runs/diagnostics/2026-09-22-training-size-curve/smoke-pixels-all-n08-seed4/`. Full coefficients, predictions, provenance, and checksums will be archived in the sibling study directory. No final-test predictions were scored. This remains exploratory.
