### Sequential execution: `convergence-smoke-04-encoded-persistent-all-both-pca60-true`

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
    "fold": 5,
    "projected_features": 60,
    "projected_std_max": 97.61153203869783,
    "projected_std_min": 12.921832303208822,
    "training_rows": 224
  },
  "scan_fit_count": 17,
  "budget_measurements": [
    {
      "C": 0.01,
      "coefficient_norm": 0.5762304994075913,
      "converged": false,
      "elapsed_seconds": 1.015411455067806,
      "fold": 5,
      "gradient_infinity_norm": 1.3233689978527086e-05,
      "gradient_threshold_met": false,
      "iterations": [
        5000
      ],
      "max_iterations": 5000,
      "objective": 0.10554078860267485,
      "warnings": [
        {
          "category": "ConvergenceWarning",
          "message": "lbfgs failed to converge after 5000 iteration(s) (status=1):\nSTOP: TOTAL NO. OF ITERATIONS REACHED LIMIT\n\nIncrease the number of iterations to improve the convergence (max_iter=5000).\nYou might also want to scale the data as shown in:\n    https://scikit-learn.org/stable/modules/preprocessing.html\nPlease also refer to the documentation for alternative solver options:\n    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression"
        }
      ]
    },
    {
      "C": 0.01,
      "coefficient_norm": 0.576231564524851,
      "converged": true,
      "elapsed_seconds": 1.0959709630114958,
      "fold": 5,
      "gradient_infinity_norm": 5.59082797777985e-07,
      "gradient_threshold_met": true,
      "iterations": [
        5411
      ],
      "max_iterations": 20000,
      "objective": 0.10554076982631135,
      "warnings": []
    }
  ],
  "validation_predictions_scored": 0,
  "inference": "Numerical training diagnostic; no recognition accuracy is estimated."
}
```

Working evidence: `$FLYSTATE_HOME/runs/diagnostics/2026-09-21-fly-only-diagnostics/convergence-smoke-04-encoded-persistent-all-both-pca60-true/`. Full coefficients, predictions, provenance, and checksums will be archived in the sibling study directory. No final-test predictions were scored. This remains exploratory.
