### Sequential execution: `retry50000-main-01-pixels-persistent-all-both-pca60-true`

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
    "tolerance": 1e-06
  },
  "scores": {
    "train": {
      "accuracy": 1.0,
      "log_loss": 0.15575416288307922,
      "n": 1400,
      "top5_accuracy": 1.0
    },
    "validation": {
      "accuracy": 0.37333333333333335,
      "log_loss": 3.9607010573913293,
      "n": 300,
      "top5_accuracy": 0.6366666666666667
    }
  },
  "cv_scores": {
    "0.01": 0.3635714285714286,
    "0.1": 0.35642857142857143,
    "1.0": 0.35,
    "10.0": 0.34857142857142853
  },
  "selected_C": 0.01,
  "inference": "exploratory; final-test and reserve images were not scored"
}
```

Working evidence: `$FLYSTATE_HOME/runs/diagnostics/2026-09-21-fly-only-diagnostics/retry50000-main-01-pixels-persistent-all-both-pca60-true/`. Full coefficients, predictions, provenance, and checksums will be archived in the sibling study directory. No final-test predictions were scored. This remains exploratory.
