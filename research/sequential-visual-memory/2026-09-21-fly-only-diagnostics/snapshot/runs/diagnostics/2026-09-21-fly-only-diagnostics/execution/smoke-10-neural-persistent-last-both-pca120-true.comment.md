### Sequential execution: `smoke-10-neural-persistent-last-both-pca120-true`

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
    "pca_components": 120,
    "representation": "neural",
    "seed": 0,
    "tolerance": 1e-06
  },
  "scores": {
    "train": {
      "accuracy": 1.0,
      "log_loss": 0.476007927988048,
      "n": 280,
      "top5_accuracy": 1.0
    },
    "validation": {
      "accuracy": 0.03333333333333333,
      "log_loss": 3.2542255656025048,
      "n": 60,
      "top5_accuracy": 0.21666666666666667
    }
  },
  "cv_scores": {
    "0.01": 0.04642857142857142,
    "0.1": 0.039285714285714285,
    "1.0": 0.03928571428571428,
    "10.0": 0.04285714285714286
  },
  "selected_C": 0.01,
  "inference": "exploratory; final-test and reserve images were not scored"
}
```

Working evidence: `$FLYSTATE_HOME/runs/diagnostics/2026-09-21-fly-only-diagnostics/smoke-10-neural-persistent-last-both-pca120-true/`. Full coefficients, predictions, provenance, and checksums will be archived in the sibling study directory. No final-test predictions were scored. This remains exploratory.
