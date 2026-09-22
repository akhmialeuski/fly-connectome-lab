### Sequential execution: `smoke-encoded-all-n04-seed3`

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
    "subset_seed": 3,
    "tolerance": 1e-06,
    "train_per_class": 4
  },
  "scores": {
    "train": {
      "accuracy": 1.0,
      "log_loss": 0.01807043463504135,
      "n": 80,
      "top5_accuracy": 1.0
    },
    "validation": {
      "accuracy": 0.26666666666666666,
      "log_loss": 3.0358432330918097,
      "n": 60,
      "top5_accuracy": 0.6166666666666667
    }
  },
  "cv_scores": {
    "0.01": 0.4,
    "0.1": 0.4,
    "1.0": 0.4,
    "10.0": 0.4
  },
  "selected_C": 0.01,
  "membership_sha256": "4c36b476a2ea6119a083631839ed625da1f9317ffa7563c8bb69360acc764a9a",
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

Working evidence: `$FLYSTATE_HOME/runs/diagnostics/2026-09-22-training-size-curve/smoke-encoded-all-n04-seed3/`. Full coefficients, predictions, provenance, and checksums will be archived in the sibling study directory. No final-test predictions were scored. This remains exploratory.
