### Sequential execution: `smoke-pixels-all-n04-seed0`

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
    "train_per_class": 4
  },
  "scores": {
    "train": {
      "accuracy": 1.0,
      "log_loss": 0.0037672010765487122,
      "n": 80,
      "top5_accuracy": 1.0
    },
    "validation": {
      "accuracy": 0.4166666666666667,
      "log_loss": 2.8532624016492916,
      "n": 60,
      "top5_accuracy": 0.75
    }
  },
  "cv_scores": {
    "0.01": 0.3,
    "0.1": 0.3125,
    "1.0": 0.3,
    "10.0": 0.3
  },
  "selected_C": 0.1,
  "membership_sha256": "32570d7cf4abe80ff79b80734a56d91b57a233199b2a9d03bbd19201acda2edf",
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

Working evidence: `$FLYSTATE_HOME/runs/diagnostics/2026-09-22-training-size-curve/smoke-pixels-all-n04-seed0/`. Full coefficients, predictions, provenance, and checksums will be archived in the sibling study directory. No final-test predictions were scored. This remains exploratory.
