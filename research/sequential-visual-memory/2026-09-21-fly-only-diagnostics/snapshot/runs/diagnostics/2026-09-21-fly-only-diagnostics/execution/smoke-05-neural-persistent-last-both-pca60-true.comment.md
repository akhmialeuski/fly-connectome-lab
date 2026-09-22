### Sequential execution: `smoke-05-neural-persistent-last-both-pca60-true`

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
      "accuracy": 1.0,
      "log_loss": 0.009309138935379999,
      "n": 280,
      "top5_accuracy": 1.0
    },
    "validation": {
      "accuracy": 0.06666666666666667,
      "log_loss": 8.452999095156091,
      "n": 60,
      "top5_accuracy": 0.31666666666666665
    }
  },
  "cv_scores": {
    "0.01": 0.05714285714285714,
    "0.1": 0.05714285714285714,
    "1.0": 0.049999999999999996,
    "10.0": 0.06071428571428571
  },
  "selected_C": 10.0,
  "inference": "exploratory; final-test and reserve images were not scored"
}
```

Working evidence: `$FLYSTATE_HOME/runs/diagnostics/2026-09-21-fly-only-diagnostics/smoke-05-neural-persistent-last-both-pca60-true/`. Full coefficients, predictions, provenance, and checksums will be archived in the sibling study directory. No final-test predictions were scored. This remains exploratory.
