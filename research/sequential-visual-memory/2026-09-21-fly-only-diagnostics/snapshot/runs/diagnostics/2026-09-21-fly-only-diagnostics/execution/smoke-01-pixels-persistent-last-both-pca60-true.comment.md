### Sequential execution: `smoke-01-pixels-persistent-last-both-pca60-true`

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
    "representation": "pixels",
    "seed": 0,
    "tolerance": 1e-06
  },
  "scores": {
    "train": {
      "accuracy": 0.6071428571428571,
      "log_loss": 1.6103523916191353,
      "n": 280,
      "top5_accuracy": 0.8857142857142857
    },
    "validation": {
      "accuracy": 0.16666666666666666,
      "log_loss": 2.982128832785252,
      "n": 60,
      "top5_accuracy": 0.5
    }
  },
  "cv_scores": {
    "0.01": 0.11428571428571428,
    "0.1": 0.10357142857142856,
    "1.0": 0.08214285714285714,
    "10.0": 0.08571428571428572
  },
  "selected_C": 0.01,
  "inference": "exploratory; final-test and reserve images were not scored"
}
```

Working evidence: `$FLYSTATE_HOME/runs/diagnostics/2026-09-21-fly-only-diagnostics/smoke-01-pixels-persistent-last-both-pca60-true/`. Full coefficients, predictions, provenance, and checksums will be archived in the sibling study directory. No final-test predictions were scored. This remains exploratory.
