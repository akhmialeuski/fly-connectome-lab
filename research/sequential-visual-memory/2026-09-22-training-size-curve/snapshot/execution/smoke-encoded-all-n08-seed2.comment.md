### Sequential execution: `smoke-encoded-all-n08-seed2`

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
    "subset_seed": 2,
    "tolerance": 1e-06,
    "train_per_class": 8
  },
  "scores": {
    "train": {
      "accuracy": 1.0,
      "log_loss": 0.0005171477157160097,
      "n": 160,
      "top5_accuracy": 1.0
    },
    "validation": {
      "accuracy": 0.43333333333333335,
      "log_loss": 3.5749368252679385,
      "n": 60,
      "top5_accuracy": 0.7333333333333333
    }
  },
  "cv_scores": {
    "0.01": 0.4125,
    "0.1": 0.40625,
    "1.0": 0.41875,
    "10.0": 0.41875
  },
  "selected_C": 1.0,
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

Working evidence: `$FLYSTATE_HOME/runs/diagnostics/2026-09-22-training-size-curve/smoke-encoded-all-n08-seed2/`. Full coefficients, predictions, provenance, and checksums will be archived in the sibling study directory. No final-test predictions were scored. This remains exploratory.
