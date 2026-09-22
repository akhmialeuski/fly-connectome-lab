### Sequential execution: `convergence-main-02-encoded-persistent-all-both-pca60-true`

Exit code: 0. Completed.

```json
{
  "parameters": {
    "budgets": [
      5000,
      20000,
      50000
    ],
    "c_grid": [
      0.01,
      0.1,
      1.0,
      10.0
    ],
    "features": "both",
    "history": "all",
    "kind": "convergence_diagnostic",
    "label_mode": "true",
    "pca_components": 60,
    "representation": "encoded",
    "seed": 0,
    "tolerance": 1e-06
  },
  "failure_reproduced": true,
  "first_failure": {
    "C": 0.01,
    "fold": 1,
    "projected_features": 60,
    "projected_std_max": 92.88748678962887,
    "projected_std_min": 11.417624031551785,
    "training_rows": 1120
  },
  "scan_fit_count": 1,
  "budget_measurements": [
    {
      "C": 0.01,
      "coefficient_norm": 2.0280614296120176,
      "converged": false,
      "elapsed_seconds": 16.3093401939841,
      "fold": 1,
      "gradient_infinity_norm": 1.847579226583188e-05,
      "gradient_threshold_met": false,
      "iterations": [
        5000
      ],
      "max_iterations": 5000,
      "objective": 0.27720065987175846,
      "warnings": [
        {
          "category": "ConvergenceWarning",
          "message": "lbfgs failed to converge after 5000 iteration(s) (status=1):\nSTOP: TOTAL NO. OF ITERATIONS REACHED LIMIT\n\nIncrease the number of iterations to improve the convergence (max_iter=5000).\nYou might also want to scale the data as shown in:\n    https://scikit-learn.org/stable/modules/preprocessing.html\nPlease also refer to the documentation for alternative solver options:\n    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression"
        }
      ]
    },
    {
      "C": 0.01,
      "coefficient_norm": 2.028083025477201,
      "converged": true,
      "elapsed_seconds": 33.32682386797387,
      "fold": 1,
      "gradient_infinity_norm": 5.818982900234609e-07,
      "gradient_threshold_met": true,
      "iterations": [
        10396
      ],
      "max_iterations": 20000,
      "objective": 0.27719978902469755,
      "warnings": []
    }
  ],
  "validation_predictions_scored": 0,
  "inference": "Numerical training diagnostic; no recognition accuracy is estimated."
}
```

Working evidence: `$FLYSTATE_HOME/runs/diagnostics/2026-09-21-fly-only-diagnostics/convergence-main-02-encoded-persistent-all-both-pca60-true/`. Full coefficients, predictions, provenance, and checksums will be archived in the sibling study directory. No final-test predictions were scored. This remains exploratory.
