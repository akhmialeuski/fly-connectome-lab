### Sequential execution: `smoke-pixels-all-n02-seed0`

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
    "train_per_class": 2
  },
  "scores": {
    "train": {
      "accuracy": 1.0,
      "log_loss": 0.02056842275479378,
      "n": 40,
      "top5_accuracy": 1.0
    },
    "validation": {
      "accuracy": 0.38333333333333336,
      "log_loss": 2.5451594055019005,
      "n": 60,
      "top5_accuracy": 0.6166666666666667
    }
  },
  "cv_scores": {
    "0.01": 0.15,
    "0.1": 0.15,
    "1.0": 0.15,
    "10.0": 0.15
  },
  "selected_C": 0.01,
  "membership_sha256": "b64d3053b08ac88de2e3378d9c6875daec1801e3955d33aff87547e96a509f78",
  "effective_cv_folds": 2,
  "fold_pca_dimensions": [
    19,
    19
  ],
  "inference": "exploratory; final-test and reserve images were not scored"
}
```

Working evidence: `$FLYSTATE_HOME/runs/diagnostics/2026-09-22-training-size-curve/smoke-pixels-all-n02-seed0/`. Full coefficients, predictions, provenance, and checksums will be archived in the sibling study directory. No final-test predictions were scored. This remains exploratory.
