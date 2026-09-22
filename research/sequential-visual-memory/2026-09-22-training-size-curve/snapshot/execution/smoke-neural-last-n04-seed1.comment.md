### Sequential execution: `smoke-neural-last-n04-seed1`

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
    "history": "last",
    "kind": "identity_probe",
    "label_mode": "true",
    "max_iterations": 50000,
    "pca_components": 60,
    "representation": "neural",
    "seed": 0,
    "subset_seed": 1,
    "tolerance": 1e-06,
    "train_per_class": 4
  },
  "scores": {
    "train": {
      "accuracy": 1.0,
      "log_loss": 0.00571161516856152,
      "n": 80,
      "top5_accuracy": 1.0
    },
    "validation": {
      "accuracy": 0.016666666666666666,
      "log_loss": 4.780654423136626,
      "n": 60,
      "top5_accuracy": 0.23333333333333334
    }
  },
  "cv_scores": {
    "0.01": 0.0125,
    "0.1": 0.0125,
    "1.0": 0.025,
    "10.0": 0.025
  },
  "selected_C": 1.0,
  "membership_sha256": "2df566a3c5f9d38f89a1f8b529644589e795a252e03e88e9b45d5d5a9277c2a2",
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

Working evidence: `$FLYSTATE_HOME/runs/diagnostics/2026-09-22-training-size-curve/smoke-neural-last-n04-seed1/`. Full coefficients, predictions, provenance, and checksums will be archived in the sibling study directory. No final-test predictions were scored. This remains exploratory.
