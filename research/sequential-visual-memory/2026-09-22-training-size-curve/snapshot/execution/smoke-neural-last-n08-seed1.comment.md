### Sequential execution: `smoke-neural-last-n08-seed1`

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
    "train_per_class": 8
  },
  "scores": {
    "train": {
      "accuracy": 0.99375,
      "log_loss": 0.6218218412107666,
      "n": 160,
      "top5_accuracy": 1.0
    },
    "validation": {
      "accuracy": 0.05,
      "log_loss": 3.2025264969311538,
      "n": 60,
      "top5_accuracy": 0.2
    }
  },
  "cv_scores": {
    "0.01": 0.075,
    "0.1": 0.0625,
    "1.0": 0.0625,
    "10.0": 0.06875
  },
  "selected_C": 0.01,
  "membership_sha256": "9b1bdfc3f16fc484808980213e87296f8ba6ec61dbbaa42ddaa9a0a29ece274f",
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

Working evidence: `$FLYSTATE_HOME/runs/diagnostics/2026-09-22-training-size-curve/smoke-neural-last-n08-seed1/`. Full coefficients, predictions, provenance, and checksums will be archived in the sibling study directory. No final-test predictions were scored. This remains exploratory.
