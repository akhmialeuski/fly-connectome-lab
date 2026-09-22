### Sequential execution: `main-04-neural-reset-all-both-pca60-true`

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
      "accuracy": 0.5314285714285715,
      "log_loss": 1.8987496231248555,
      "n": 1400,
      "top5_accuracy": 0.8014285714285714
    },
    "validation": {
      "accuracy": 0.006666666666666667,
      "log_loss": 6.365913203876382,
      "n": 300,
      "top5_accuracy": 0.043333333333333335
    }
  },
  "cv_scores": {
    "0.01": 0.009285714285714286,
    "0.1": 0.011428571428571429,
    "1.0": 0.011428571428571429,
    "10.0": 0.010714285714285714
  },
  "selected_C": 0.1,
  "inference": "exploratory; final-test and reserve images were not scored"
}
```

Working evidence: `$FLYSTATE_HOME/runs/diagnostics/2026-09-21-fly-only-diagnostics/main-04-neural-reset-all-both-pca60-true/`. Full coefficients, predictions, provenance, and checksums will be archived in the sibling study directory. No final-test predictions were scored. This remains exploratory.
