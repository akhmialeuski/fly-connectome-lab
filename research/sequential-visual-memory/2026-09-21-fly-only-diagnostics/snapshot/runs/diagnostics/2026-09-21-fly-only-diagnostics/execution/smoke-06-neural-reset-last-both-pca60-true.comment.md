### Sequential execution: `smoke-06-neural-reset-last-both-pca60-true`

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
      "log_loss": 0.010149434828901686,
      "n": 280,
      "top5_accuracy": 1.0
    },
    "validation": {
      "accuracy": 0.03333333333333333,
      "log_loss": 8.782860483866006,
      "n": 60,
      "top5_accuracy": 0.18333333333333332
    }
  },
  "cv_scores": {
    "0.01": 0.06071428571428571,
    "0.1": 0.05714285714285714,
    "1.0": 0.06428571428571428,
    "10.0": 0.07142857142857142
  },
  "selected_C": 10.0,
  "inference": "exploratory; final-test and reserve images were not scored"
}
```

Working evidence: `$FLYSTATE_HOME/runs/diagnostics/2026-09-21-fly-only-diagnostics/smoke-06-neural-reset-last-both-pca60-true/`. Full coefficients, predictions, provenance, and checksums will be archived in the sibling study directory. No final-test predictions were scored. This remains exploratory.
