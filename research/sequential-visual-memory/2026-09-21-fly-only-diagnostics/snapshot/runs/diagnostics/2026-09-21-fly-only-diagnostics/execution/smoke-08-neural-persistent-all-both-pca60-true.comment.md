### Sequential execution: `smoke-08-neural-persistent-all-both-pca60-true`

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
      "accuracy": 1.0,
      "log_loss": 0.05639290089667429,
      "n": 280,
      "top5_accuracy": 1.0
    },
    "validation": {
      "accuracy": 0.06666666666666667,
      "log_loss": 4.314781847250141,
      "n": 60,
      "top5_accuracy": 0.25
    }
  },
  "cv_scores": {
    "0.01": 0.046428571428571416,
    "0.1": 0.05357142857142857,
    "1.0": 0.05357142857142857,
    "10.0": 0.05357142857142857
  },
  "selected_C": 0.1,
  "inference": "exploratory; final-test and reserve images were not scored"
}
```

Working evidence: `$FLYSTATE_HOME/runs/diagnostics/2026-09-21-fly-only-diagnostics/smoke-08-neural-persistent-all-both-pca60-true/`. Full coefficients, predictions, provenance, and checksums will be archived in the sibling study directory. No final-test predictions were scored. This remains exploratory.
