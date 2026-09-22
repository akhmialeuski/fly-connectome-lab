### Sequential execution: `smoke-neural-last-n04-seed3`

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
    "subset_seed": 3,
    "tolerance": 1e-06,
    "train_per_class": 4
  },
  "scores": {
    "train": {
      "accuracy": 1.0,
      "log_loss": 0.0052341403307695,
      "n": 80,
      "top5_accuracy": 1.0
    },
    "validation": {
      "accuracy": 0.06666666666666667,
      "log_loss": 4.413738541845436,
      "n": 60,
      "top5_accuracy": 0.25
    }
  },
  "cv_scores": {
    "0.01": 0.05,
    "0.1": 0.05,
    "1.0": 0.0625,
    "10.0": 0.0625
  },
  "selected_C": 1.0,
  "membership_sha256": "4c36b476a2ea6119a083631839ed625da1f9317ffa7563c8bb69360acc764a9a",
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

Working evidence: `$FLYSTATE_HOME/runs/diagnostics/2026-09-22-training-size-curve/smoke-neural-last-n04-seed3/`. Full coefficients, predictions, provenance, and checksums will be archived in the sibling study directory. No final-test predictions were scored. This remains exploratory.
