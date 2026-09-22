### Sequential execution: `main-03-neural-persistent-last-both-pca60-true`

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
    "representation": "neural",
    "seed": 0,
    "tolerance": 1e-06
  },
  "scores": {
    "train": {
      "accuracy": 0.5678571428571428,
      "log_loss": 1.8359547354511225,
      "n": 1400,
      "top5_accuracy": 0.8142857142857143
    },
    "validation": {
      "accuracy": 0.0033333333333333335,
      "log_loss": 11.259212296658706,
      "n": 300,
      "top5_accuracy": 0.05333333333333334
    }
  },
  "cv_scores": {
    "0.01": 0.010714285714285714,
    "0.1": 0.008571428571428572,
    "1.0": 0.008571428571428572,
    "10.0": 0.011428571428571429
  },
  "selected_C": 10.0,
  "inference": "exploratory; final-test and reserve images were not scored"
}
```

Working evidence: `$FLYSTATE_HOME/runs/diagnostics/2026-09-21-fly-only-diagnostics/main-03-neural-persistent-last-both-pca60-true/`. Full coefficients, predictions, provenance, and checksums will be archived in the sibling study directory. No final-test predictions were scored. This remains exploratory.
