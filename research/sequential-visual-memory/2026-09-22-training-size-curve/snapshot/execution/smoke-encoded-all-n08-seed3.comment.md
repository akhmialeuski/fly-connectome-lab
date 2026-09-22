### Sequential execution: `smoke-encoded-all-n08-seed3`

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
    "subset_seed": 3,
    "tolerance": 1e-06,
    "train_per_class": 8
  },
  "scores": {
    "train": {
      "accuracy": 1.0,
      "log_loss": 0.0005647489366736576,
      "n": 160,
      "top5_accuracy": 1.0
    },
    "validation": {
      "accuracy": 0.4,
      "log_loss": 3.852959681118113,
      "n": 60,
      "top5_accuracy": 0.7166666666666667
    }
  },
  "cv_scores": {
    "0.01": 0.36875,
    "0.1": 0.36875,
    "1.0": 0.38125,
    "10.0": 0.375
  },
  "selected_C": 1.0,
  "membership_sha256": "498523892d24ecf1037b7254a0c949c75d05939ec1919873fb39214a1eef5bf8",
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

Working evidence: `$FLYSTATE_HOME/runs/diagnostics/2026-09-22-training-size-curve/smoke-encoded-all-n08-seed3/`. Full coefficients, predictions, provenance, and checksums will be archived in the sibling study directory. No final-test predictions were scored. This remains exploratory.
