### Sequential execution: `smoke-encoded-all-n04-seed2`

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
    "subset_seed": 2,
    "tolerance": 1e-06,
    "train_per_class": 4
  },
  "scores": {
    "train": {
      "accuracy": 1.0,
      "log_loss": 0.000348872756791149,
      "n": 80,
      "top5_accuracy": 1.0
    },
    "validation": {
      "accuracy": 0.38333333333333336,
      "log_loss": 3.0307209482384945,
      "n": 60,
      "top5_accuracy": 0.65
    }
  },
  "cv_scores": {
    "0.01": 0.3125,
    "0.1": 0.3125,
    "1.0": 0.325,
    "10.0": 0.32499999999999996
  },
  "selected_C": 1.0,
  "membership_sha256": "fe0c78a9bfe9e9a732b01d49f5411ad90af24e2a710cb0616f50f26c9c5b1c7f",
  "effective_cv_folds": 4,
  "fold_pca_dimensions": [
    59,
    59,
    59,
    59
  ],
  "inference": "exploratory; final-test and reserve images were not scored"
}
```

Working evidence: `$FLYSTATE_HOME/runs/diagnostics/2026-09-22-training-size-curve/smoke-encoded-all-n04-seed2/`. Full coefficients, predictions, provenance, and checksums will be archived in the sibling study directory. No final-test predictions were scored. This remains exploratory.
