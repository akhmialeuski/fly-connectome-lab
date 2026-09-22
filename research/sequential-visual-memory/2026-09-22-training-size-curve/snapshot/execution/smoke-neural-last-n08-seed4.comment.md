### Sequential execution: `smoke-neural-last-n08-seed4`

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
    "subset_seed": 4,
    "tolerance": 1e-06,
    "train_per_class": 8
  },
  "scores": {
    "train": {
      "accuracy": 1.0,
      "log_loss": 0.01787496043318305,
      "n": 160,
      "top5_accuracy": 1.0
    },
    "validation": {
      "accuracy": 0.03333333333333333,
      "log_loss": 4.362759361217287,
      "n": 60,
      "top5_accuracy": 0.26666666666666666
    }
  },
  "cv_scores": {
    "0.01": 0.01875,
    "0.1": 0.0125,
    "1.0": 0.025,
    "10.0": 0.025
  },
  "selected_C": 1.0,
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

Working evidence: `$FLYSTATE_HOME/runs/diagnostics/2026-09-22-training-size-curve/smoke-neural-last-n08-seed4/`. Full coefficients, predictions, provenance, and checksums will be archived in the sibling study directory. No final-test predictions were scored. This remains exploratory.
