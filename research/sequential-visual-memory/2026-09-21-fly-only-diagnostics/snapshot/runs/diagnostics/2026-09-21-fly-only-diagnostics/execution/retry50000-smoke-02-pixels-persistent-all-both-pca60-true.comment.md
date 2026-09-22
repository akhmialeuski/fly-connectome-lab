### Sequential execution: `retry50000-smoke-02-pixels-persistent-all-both-pca60-true`

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
      "log_loss": 0.0010508032318843432,
      "n": 280,
      "top5_accuracy": 1.0
    },
    "validation": {
      "accuracy": 0.5166666666666667,
      "log_loss": 3.5528943899931558,
      "n": 60,
      "top5_accuracy": 0.8
    }
  },
  "cv_scores": {
    "0.01": 0.46428571428571425,
    "0.1": 0.46428571428571425,
    "1.0": 0.4714285714285714,
    "10.0": 0.46428571428571425
  },
  "selected_C": 1.0,
  "inference": "exploratory; final-test and reserve images were not scored"
}
```

Working evidence: `$FLYSTATE_HOME/runs/diagnostics/2026-09-21-fly-only-diagnostics/retry50000-smoke-02-pixels-persistent-all-both-pca60-true/`. Full coefficients, predictions, provenance, and checksums will be archived in the sibling study directory. No final-test predictions were scored. This remains exploratory.
