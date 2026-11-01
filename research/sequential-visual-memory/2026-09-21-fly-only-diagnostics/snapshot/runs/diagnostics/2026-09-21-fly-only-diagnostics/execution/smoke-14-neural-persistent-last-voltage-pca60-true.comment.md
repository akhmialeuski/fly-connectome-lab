### Sequential execution: `smoke-14-neural-persistent-last-voltage-pca60-true`

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
    "features": "voltage",
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
      "log_loss": 0.10719360054924842,
      "n": 280,
      "top5_accuracy": 1.0
    },
    "validation": {
      "accuracy": 0.08333333333333333,
      "log_loss": 5.97418625855689,
      "n": 60,
      "top5_accuracy": 0.25
    }
  },
  "cv_scores": {
    "0.01": 0.04285714285714286,
    "0.1": 0.05357142857142856,
    "1.0": 0.06071428571428571,
    "10.0": 0.05
  },
  "selected_C": 1.0,
  "inference": "exploratory; final-test and reserve images were not scored"
}
```

Working evidence: `$FLYSTATE_HOME/runs/diagnostics/2026-09-21-fly-only-diagnostics/smoke-14-neural-persistent-last-voltage-pca60-true/`. Full coefficients, predictions, provenance, and checksums will be archived in the sibling study directory. No final-test predictions were scored. This remains exploratory.
