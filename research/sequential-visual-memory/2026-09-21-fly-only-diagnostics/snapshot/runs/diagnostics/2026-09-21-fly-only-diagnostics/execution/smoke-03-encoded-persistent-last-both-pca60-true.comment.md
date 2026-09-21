### Sequential execution: `smoke-03-encoded-persistent-last-both-pca60-true`

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
    "pca_components": 60,
    "representation": "encoded",
    "seed": 0,
    "tolerance": 1e-06
  },
  "scores": {
    "train": {
      "accuracy": 0.6928571428571428,
      "log_loss": 1.3441650964191147,
      "n": 280,
      "top5_accuracy": 0.9178571428571428
    },
    "validation": {
      "accuracy": 0.18333333333333332,
      "log_loss": 3.13308995502531,
      "n": 60,
      "top5_accuracy": 0.4666666666666667
    }
  },
  "cv_scores": {
    "0.01": 0.11428571428571428,
    "0.1": 0.09642857142857143,
    "1.0": 0.09642857142857142,
    "10.0": 0.09285714285714286
  },
  "selected_C": 0.01,
  "inference": "exploratory; final-test and reserve images were not scored"
}
```

Working evidence: `$FLYSTATE_HOME/runs/diagnostics/2026-09-21-fly-only-diagnostics/smoke-03-encoded-persistent-last-both-pca60-true/`. Full coefficients, predictions, provenance, and checksums will be archived in the sibling study directory. No final-test predictions were scored. This remains exploratory.
