### Sequential execution: `smoke-neural-last-n02-seed3`

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
    "max_iterations": 50000,
    "pca_components": 60,
    "representation": "neural",
    "seed": 0,
    "subset_seed": 3,
    "tolerance": 1e-06,
    "train_per_class": 2
  },
  "scores": {
    "train": {
      "accuracy": 1.0,
      "log_loss": 0.18001993075380063,
      "n": 40,
      "top5_accuracy": 1.0
    },
    "validation": {
      "accuracy": 0.05,
      "log_loss": 10.023246148387576,
      "n": 60,
      "top5_accuracy": 0.2833333333333333
    }
  },
  "cv_scores": {
    "0.01": 0.1,
    "0.1": 0.1,
    "1.0": 0.1,
    "10.0": 0.1
  },
  "selected_C": 0.01,
  "membership_sha256": "93e479d45398fd3a57d47a2b18f60910e9e751bf91394b9d7181a8558a67f9cb",
  "effective_cv_folds": 2,
  "fold_pca_dimensions": [
    19,
    19
  ],
  "inference": "exploratory; final-test and reserve images were not scored"
}
```

Working evidence: `$FLYSTATE_HOME/runs/diagnostics/2026-09-22-training-size-curve/smoke-neural-last-n02-seed3/`. Full coefficients, predictions, provenance, and checksums will be archived in the sibling study directory. No final-test predictions were scored. This remains exploratory.
