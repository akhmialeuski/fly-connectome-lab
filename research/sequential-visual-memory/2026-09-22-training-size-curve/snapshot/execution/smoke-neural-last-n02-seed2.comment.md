### Sequential execution: `smoke-neural-last-n02-seed2`

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
    "subset_seed": 2,
    "tolerance": 1e-06,
    "train_per_class": 2
  },
  "scores": {
    "train": {
      "accuracy": 1.0,
      "log_loss": 0.18004027113056092,
      "n": 40,
      "top5_accuracy": 1.0
    },
    "validation": {
      "accuracy": 0.03333333333333333,
      "log_loss": 12.624061794012071,
      "n": 60,
      "top5_accuracy": 0.11666666666666667
    }
  },
  "cv_scores": {
    "0.01": 0.05,
    "0.1": 0.05,
    "1.0": 0.05,
    "10.0": 0.05
  },
  "selected_C": 0.01,
  "membership_sha256": "26e2a4ac13e231a45695beb342a6e4be0b5fe355ca7f73ffaa6b27b1a69a26da",
  "effective_cv_folds": 2,
  "fold_pca_dimensions": [
    19,
    19
  ],
  "inference": "exploratory; final-test and reserve images were not scored"
}
```

Working evidence: `$FLYSTATE_HOME/runs/diagnostics/2026-09-22-training-size-curve/smoke-neural-last-n02-seed2/`. Full coefficients, predictions, provenance, and checksums will be archived in the sibling study directory. No final-test predictions were scored. This remains exploratory.
