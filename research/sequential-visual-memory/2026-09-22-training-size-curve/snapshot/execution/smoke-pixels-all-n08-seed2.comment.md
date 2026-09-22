### Sequential execution: `smoke-pixels-all-n08-seed2`

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
    "train_per_class": 8
  },
  "scores": {
    "train": {
      "accuracy": 1.0,
      "log_loss": 0.005537367182573726,
      "n": 160,
      "top5_accuracy": 1.0
    },
    "validation": {
      "accuracy": 0.4666666666666667,
      "log_loss": 3.1816819732157873,
      "n": 60,
      "top5_accuracy": 0.7666666666666667
    }
  },
  "cv_scores": {
    "0.01": 0.40625,
    "0.1": 0.4125,
    "1.0": 0.4125,
    "10.0": 0.40625
  },
  "selected_C": 0.1,
  "membership_sha256": "2db78ded0ce680b54721aaa38f593a6b9d161102df57a9dbc9335884654a454f",
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

Working evidence: `$FLYSTATE_HOME/runs/diagnostics/2026-09-22-training-size-curve/smoke-pixels-all-n08-seed2/`. Full coefficients, predictions, provenance, and checksums will be archived in the sibling study directory. No final-test predictions were scored. This remains exploratory.
