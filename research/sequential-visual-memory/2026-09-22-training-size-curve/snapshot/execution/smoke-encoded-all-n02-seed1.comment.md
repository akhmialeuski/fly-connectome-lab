### Sequential execution: `smoke-encoded-all-n02-seed1`

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
    "subset_seed": 1,
    "tolerance": 1e-06,
    "train_per_class": 2
  },
  "scores": {
    "train": {
      "accuracy": 1.0,
      "log_loss": 0.015990739422790883,
      "n": 40,
      "top5_accuracy": 1.0
    },
    "validation": {
      "accuracy": 0.21666666666666667,
      "log_loss": 2.7170341794270265,
      "n": 60,
      "top5_accuracy": 0.55
    }
  },
  "cv_scores": {
    "0.01": 0.07500000000000001,
    "0.1": 0.07500000000000001,
    "1.0": 0.07500000000000001,
    "10.0": 0.07500000000000001
  },
  "selected_C": 0.01,
  "membership_sha256": "ea141dd52a5a0f65f3f59106ceba116dfe4ae1cc117d34437c5c2c25fe3f716a",
  "effective_cv_folds": 2,
  "fold_pca_dimensions": [
    19,
    19
  ],
  "inference": "exploratory; final-test and reserve images were not scored"
}
```

Working evidence: `$FLYSTATE_HOME/runs/diagnostics/2026-09-22-training-size-curve/smoke-encoded-all-n02-seed1/`. Full coefficients, predictions, provenance, and checksums will be archived in the sibling study directory. No final-test predictions were scored. This remains exploratory.
