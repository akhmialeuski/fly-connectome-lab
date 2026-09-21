### Sequential execution: `smoke-13-neural-persistent-last-spike_trace-pca60-true`

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
    "features": "spike_trace",
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
      "accuracy": 0.7464285714285714,
      "log_loss": 1.4835101921021903,
      "n": 280,
      "top5_accuracy": 0.9678571428571429
    },
    "validation": {
      "accuracy": 0.08333333333333333,
      "log_loss": 3.1099733433432055,
      "n": 60,
      "top5_accuracy": 0.3333333333333333
    }
  },
  "cv_scores": {
    "0.01": 0.04642857142857142,
    "0.1": 0.03214285714285714,
    "1.0": 0.024999999999999998,
    "10.0": 0.025
  },
  "selected_C": 0.01,
  "inference": "exploratory; final-test and reserve images were not scored"
}
```

Working evidence: `$FLYSTATE_HOME/runs/diagnostics/2026-09-21-fly-only-diagnostics/smoke-13-neural-persistent-last-spike_trace-pca60-true/`. Full coefficients, predictions, provenance, and checksums will be archived in the sibling study directory. No final-test predictions were scored. This remains exploratory.
