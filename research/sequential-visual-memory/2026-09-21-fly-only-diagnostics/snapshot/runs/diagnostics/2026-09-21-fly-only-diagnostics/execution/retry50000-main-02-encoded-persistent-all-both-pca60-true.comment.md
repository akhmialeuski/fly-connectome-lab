### Sequential execution: `retry50000-main-02-encoded-persistent-all-both-pca60-true`

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
    "tolerance": 1e-06
  },
  "scores": {
    "train": {
      "accuracy": 1.0,
      "log_loss": 0.11951795816768375,
      "n": 1400,
      "top5_accuracy": 1.0
    },
    "validation": {
      "accuracy": 0.3566666666666667,
      "log_loss": 4.31201868002641,
      "n": 300,
      "top5_accuracy": 0.62
    }
  },
  "cv_scores": {
    "0.01": 0.3392857142857143,
    "0.1": 0.32999999999999996,
    "1.0": 0.3221428571428572,
    "10.0": 0.32071428571428573
  },
  "selected_C": 0.01,
  "inference": "exploratory; final-test and reserve images were not scored"
}
```

Working evidence: `$FLYSTATE_HOME/runs/diagnostics/2026-09-21-fly-only-diagnostics/retry50000-main-02-encoded-persistent-all-both-pca60-true/`. Full coefficients, predictions, provenance, and checksums will be archived in the sibling study directory. No final-test predictions were scored. This remains exploratory.
