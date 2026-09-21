### Sequential execution: `smoke-09-neural-persistent-last-both-pca20-true`

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
    "pca_components": 20,
    "representation": "neural",
    "seed": 0,
    "tolerance": 1e-06
  },
  "scores": {
    "train": {
      "accuracy": 0.2857142857142857,
      "log_loss": 2.3010821894094633,
      "n": 280,
      "top5_accuracy": 0.7285714285714285
    },
    "validation": {
      "accuracy": 0.016666666666666666,
      "log_loss": 3.1162023837336994,
      "n": 60,
      "top5_accuracy": 0.23333333333333334
    }
  },
  "cv_scores": {
    "0.01": 0.06071428571428571,
    "0.1": 0.06071428571428571,
    "1.0": 0.04285714285714286,
    "10.0": 0.04642857142857142
  },
  "selected_C": 0.01,
  "inference": "exploratory; final-test and reserve images were not scored"
}
```

Working evidence: `$FLYSTATE_HOME/runs/diagnostics/2026-09-21-fly-only-diagnostics/smoke-09-neural-persistent-last-both-pca20-true/`. Full coefficients, predictions, provenance, and checksums will be archived in the sibling study directory. No final-test predictions were scored. This remains exploratory.
