### Sequential execution: `smoke-pixels-all-n04-seed2`

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
    "subset_seed": 2,
    "tolerance": 1e-06,
    "train_per_class": 4
  },
  "scores": {
    "train": {
      "accuracy": 1.0,
      "log_loss": 0.028049850199989335,
      "n": 80,
      "top5_accuracy": 1.0
    },
    "validation": {
      "accuracy": 0.3333333333333333,
      "log_loss": 2.637857477377564,
      "n": 60,
      "top5_accuracy": 0.6333333333333333
    }
  },
  "cv_scores": {
    "0.01": 0.3125,
    "0.1": 0.3125,
    "1.0": 0.3125,
    "10.0": 0.3125
  },
  "selected_C": 0.01,
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

Working evidence: `$FLYSTATE_HOME/runs/diagnostics/2026-09-22-training-size-curve/smoke-pixels-all-n04-seed2/`. Full coefficients, predictions, provenance, and checksums will be archived in the sibling study directory. No final-test predictions were scored. This remains exploratory.
