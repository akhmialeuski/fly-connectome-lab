### Sequential execution: `smoke-16-neural-persistent-last-both-pca60-permuted`

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
    "label_mode": "permuted",
    "pca_components": 60,
    "representation": "neural",
    "seed": 0,
    "tolerance": 1e-06
  },
  "scores": {
    "train": {
      "accuracy": 0.9821428571428571,
      "log_loss": 0.4460968922998055,
      "n": 280,
      "top5_accuracy": 0.9964285714285714
    },
    "validation": {
      "accuracy": 0.05,
      "log_loss": 4.167912514973774,
      "n": 60,
      "top5_accuracy": 0.26666666666666666
    }
  },
  "cv_scores": {
    "0.01": 0.03928571428571428,
    "0.1": 0.05,
    "1.0": 0.04642857142857142,
    "10.0": 0.04642857142857142
  },
  "selected_C": 0.1,
  "inference": "exploratory; final-test and reserve images were not scored"
}
```

Working evidence: `$FLYSTATE_HOME/runs/diagnostics/2026-09-21-fly-only-diagnostics/smoke-16-neural-persistent-last-both-pca60-permuted/`. Full coefficients, predictions, provenance, and checksums will be archived in the sibling study directory. No final-test predictions were scored. This remains exploratory.
