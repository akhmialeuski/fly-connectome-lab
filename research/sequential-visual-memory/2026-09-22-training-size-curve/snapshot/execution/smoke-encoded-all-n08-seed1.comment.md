### Sequential execution: `smoke-encoded-all-n08-seed1`

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
    "max_iterations": 50000,
    "pca_components": 60,
    "representation": "encoded",
    "seed": 0,
    "subset_seed": 1,
    "tolerance": 1e-06,
    "train_per_class": 8
  },
  "scores": {
    "train": {
      "accuracy": 1.0,
      "log_loss": 0.02713963810228607,
      "n": 160,
      "top5_accuracy": 1.0
    },
    "validation": {
      "accuracy": 0.48333333333333334,
      "log_loss": 2.349578219211012,
      "n": 60,
      "top5_accuracy": 0.7333333333333333
    }
  },
  "cv_scores": {
    "0.01": 0.375,
    "0.1": 0.375,
    "1.0": 0.375,
    "10.0": 0.35625
  },
  "selected_C": 0.01,
  "membership_sha256": "9b1bdfc3f16fc484808980213e87296f8ba6ec61dbbaa42ddaa9a0a29ece274f",
  "effective_cv_folds": 5,
  "fold_pca_dimensions": [
    60,
    60,
    60,
    60,
    60
  ],
  "inference": "exploratory; final-test and reserve images were not scored"
}
```

Working evidence: `$FLYSTATE_HOME/runs/diagnostics/2026-09-22-training-size-curve/smoke-encoded-all-n08-seed1/`. Full coefficients, predictions, provenance, and checksums will be archived in the sibling study directory. No final-test predictions were scored. This remains exploratory.
