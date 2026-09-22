### Sequential execution: `smoke-17-encoded-persistent-all-both-pca60-memorization`

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
    "label_mode": "memorization",
    "pca_components": 60,
    "representation": "encoded",
    "seed": 0,
    "tolerance": 1e-06
  },
  "scores": {
    "memorization_train": {
      "accuracy": 1.0,
      "log_loss": 0.010726632070261735,
      "n": 20,
      "top5_accuracy": 1.0
    }
  },
  "cv_scores": {
    "0.01": 0.15000000000000002,
    "0.1": 0.15000000000000002,
    "1.0": 0.15000000000000002,
    "10.0": 0.15000000000000002
  },
  "selected_C": 0.01,
  "inference": "exploratory; final-test and reserve images were not scored"
}
```

Working evidence: `$FLYSTATE_HOME/runs/diagnostics/2026-09-21-fly-only-diagnostics/smoke-17-encoded-persistent-all-both-pca60-memorization/`. Full coefficients, predictions, provenance, and checksums will be archived in the sibling study directory. No final-test predictions were scored. This remains exploratory.
