### Sequential execution: `smoke-neural-last-n02-seed4`

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
    "subset_seed": 4,
    "tolerance": 1e-06,
    "train_per_class": 2
  },
  "scores": {
    "train": {
      "accuracy": 1.0,
      "log_loss": 0.18107432746546212,
      "n": 40,
      "top5_accuracy": 1.0
    },
    "validation": {
      "accuracy": 0.1,
      "log_loss": 9.8932741905352,
      "n": 60,
      "top5_accuracy": 0.25
    }
  },
  "cv_scores": {
    "0.01": 0.0,
    "0.1": 0.0,
    "1.0": 0.0,
    "10.0": 0.0
  },
  "selected_C": 0.01,
  "membership_sha256": "628736838a59cda532df6c4cfd32f6a4ee0ac41722aa17cbc2f1398d27798286",
  "effective_cv_folds": 2,
  "fold_pca_dimensions": [
    19,
    19
  ],
  "inference": "exploratory; final-test and reserve images were not scored"
}
```

Working evidence: `$FLYSTATE_HOME/runs/diagnostics/2026-09-22-training-size-curve/smoke-neural-last-n02-seed4/`. Full coefficients, predictions, provenance, and checksums will be archived in the sibling study directory. No final-test predictions were scored. This remains exploratory.
