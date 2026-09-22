### Sequential execution: `smoke-pixels-all-n04-seed4`

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
    "representation": "pixels",
    "seed": 0,
    "subset_seed": 4,
    "tolerance": 1e-06,
    "train_per_class": 4
  },
  "scores": {
    "train": {
      "accuracy": 1.0,
      "log_loss": 0.02752741815222342,
      "n": 80,
      "top5_accuracy": 1.0
    },
    "validation": {
      "accuracy": 0.48333333333333334,
      "log_loss": 2.2564686903020834,
      "n": 60,
      "top5_accuracy": 0.7166666666666667
    }
  },
  "cv_scores": {
    "0.01": 0.2375,
    "0.1": 0.22499999999999998,
    "1.0": 0.21249999999999997,
    "10.0": 0.21249999999999997
  },
  "selected_C": 0.01,
  "membership_sha256": "b6db7bcbafd42aaa96a9ae7a8b0cf5fced5aed2fa2a1816fd225979873a5b255",
  "effective_cv_folds": 4,
  "fold_pca_dimensions": [
    59,
    59,
    59,
    59
  ],
  "inference": "exploratory; final-test and reserve images were not scored"
}
```

Working evidence: `$FLYSTATE_HOME/runs/diagnostics/2026-09-22-training-size-curve/smoke-pixels-all-n04-seed4/`. Full coefficients, predictions, provenance, and checksums will be archived in the sibling study directory. No final-test predictions were scored. This remains exploratory.
