### Sequential execution: `smoke-pixels-all-n14-seed0`

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
    "train_per_class": 14
  },
  "scores": {
    "train": {
      "accuracy": 1.0,
      "log_loss": 0.0010508032318843432,
      "n": 280,
      "top5_accuracy": 1.0
    },
    "validation": {
      "accuracy": 0.5166666666666667,
      "log_loss": 3.5528943899931558,
      "n": 60,
      "top5_accuracy": 0.8
    }
  },
  "cv_scores": {
    "0.01": 0.46428571428571425,
    "0.1": 0.46428571428571425,
    "1.0": 0.4714285714285714,
    "10.0": 0.46428571428571425
  },
  "selected_C": 1.0,
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

Working evidence: `$FLYSTATE_HOME/runs/diagnostics/2026-09-22-training-size-curve/smoke-pixels-all-n14-seed0/`. Full coefficients, predictions, provenance, and checksums will be archived in the sibling study directory. No final-test predictions were scored. This remains exploratory.
