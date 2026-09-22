### Sequential execution: `smoke-neural-last-n02-seed0`

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
    "subset_seed": 0,
    "tolerance": 1e-06,
    "train_per_class": 2
  },
  "scores": {
    "train": {
      "accuracy": 1.0,
      "log_loss": 0.18289486070237665,
      "n": 40,
      "top5_accuracy": 1.0
    },
    "validation": {
      "accuracy": 0.05,
      "log_loss": 11.249484875987742,
      "n": 60,
      "top5_accuracy": 0.26666666666666666
    }
  },
  "cv_scores": {
    "0.01": 0.025,
    "0.1": 0.025,
    "1.0": 0.025,
    "10.0": 0.025
  },
  "selected_C": 0.01,
  "membership_sha256": "b64d3053b08ac88de2e3378d9c6875daec1801e3955d33aff87547e96a509f78",
  "effective_cv_folds": 2,
  "fold_pca_dimensions": [
    19,
    19
  ],
  "inference": "exploratory; final-test and reserve images were not scored"
}
```

Working evidence: `$FLYSTATE_HOME/runs/diagnostics/2026-09-22-training-size-curve/smoke-neural-last-n02-seed0/`. Full coefficients, predictions, provenance, and checksums will be archived in the sibling study directory. No final-test predictions were scored. This remains exploratory.
