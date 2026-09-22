### Sequential execution: `smoke-07-neural-reset-all-both-pca60-true`

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
    "pca_components": 60,
    "representation": "neural",
    "seed": 0,
    "tolerance": 1e-06
  },
  "scores": {
    "train": {
      "accuracy": 0.9964285714285714,
      "log_loss": 0.3368155386654945,
      "n": 280,
      "top5_accuracy": 1.0
    },
    "validation": {
      "accuracy": 0.06666666666666667,
      "log_loss": 3.2118424992715973,
      "n": 60,
      "top5_accuracy": 0.2833333333333333
    }
  },
  "cv_scores": {
    "0.01": 0.04642857142857143,
    "0.1": 0.04642857142857143,
    "1.0": 0.04642857142857143,
    "10.0": 0.04642857142857143
  },
  "selected_C": 0.01,
  "inference": "exploratory; final-test and reserve images were not scored"
}
```

Working evidence: `$FLYSTATE_HOME/runs/diagnostics/2026-09-21-fly-only-diagnostics/smoke-07-neural-reset-all-both-pca60-true/`. Full coefficients, predictions, provenance, and checksums will be archived in the sibling study directory. No final-test predictions were scored. This remains exploratory.
