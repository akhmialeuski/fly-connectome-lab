### Sequential execution: `smoke-neural-last-n08-seed2`

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
    "subset_seed": 2,
    "tolerance": 1e-06,
    "train_per_class": 8
  },
  "scores": {
    "train": {
      "accuracy": 0.975,
      "log_loss": 0.6424859498271525,
      "n": 160,
      "top5_accuracy": 1.0
    },
    "validation": {
      "accuracy": 0.016666666666666666,
      "log_loss": 3.2790341661165456,
      "n": 60,
      "top5_accuracy": 0.25
    }
  },
  "cv_scores": {
    "0.01": 0.0625,
    "0.1": 0.05625,
    "1.0": 0.0625,
    "10.0": 0.05625
  },
  "selected_C": 0.01,
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

Working evidence: `$FLYSTATE_HOME/runs/diagnostics/2026-09-22-training-size-curve/smoke-neural-last-n08-seed2/`. Full coefficients, predictions, provenance, and checksums will be archived in the sibling study directory. No final-test predictions were scored. This remains exploratory.
