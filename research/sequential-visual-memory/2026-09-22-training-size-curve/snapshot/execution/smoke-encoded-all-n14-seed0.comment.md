### Sequential execution: `smoke-encoded-all-n14-seed0`

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
    "subset_seed": 0,
    "tolerance": 1e-06,
    "train_per_class": 14
  },
  "scores": {
    "train": {
      "accuracy": 1.0,
      "log_loss": 0.03716270845779768,
      "n": 280,
      "top5_accuracy": 1.0
    },
    "validation": {
      "accuracy": 0.5166666666666667,
      "log_loss": 2.2990518797658344,
      "n": 60,
      "top5_accuracy": 0.7833333333333333
    }
  },
  "cv_scores": {
    "0.01": 0.4428571428571429,
    "0.1": 0.43928571428571433,
    "1.0": 0.4428571428571429,
    "10.0": 0.4428571428571429
  },
  "selected_C": 0.01,
  "membership_sha256": "2330962f0b162cbe5680a86016b07fff3ca4f1c801441eb1e2632927ffd95e2b",
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

Working evidence: `$FLYSTATE_HOME/runs/diagnostics/2026-09-22-training-size-curve/smoke-encoded-all-n14-seed0/`. Full coefficients, predictions, provenance, and checksums will be archived in the sibling study directory. No final-test predictions were scored. This remains exploratory.
