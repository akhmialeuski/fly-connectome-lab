### Sequential execution: `smoke-02-pixels-persistent-all-both-pca60-true`

Exit code: 1. Failed attempt retained; no result is substituted.

```json
{
  "created_utc": "2026-09-21T15:17:50.714443+00:00",
  "elapsed_seconds": 14.972199787967838,
  "error": "lbfgs failed to converge after 5000 iteration(s) (status=1):\nSTOP: TOTAL NO. OF ITERATIONS REACHED LIMIT\n\nIncrease the number of iterations to improve the convergence (max_iter=5000).\nYou might also want to scale the data as shown in:\n    https://scikit-learn.org/stable/modules/preprocessing.html\nPlease also refer to the documentation for alternative solver options:\n    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression",
  "error_type": "ConvergenceWarning",
  "git_commit": "aa36121ac96c51361e33d0b8267445a3ab31030b",
  "git_dirty": null,
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
    "pca_components": 60,
    "representation": "pixels",
    "seed": 0,
    "tolerance": 1e-06
  },
  "rss_end_bytes": 627589120,
  "rss_high_water_bytes": 1090117632,
  "schema_version": 1,
  "status": "failed"
}
```

Working evidence: `$FLYSTATE_HOME/runs/diagnostics/2026-09-21-fly-only-diagnostics/smoke-02-pixels-persistent-all-both-pca60-true/`. Full coefficients, predictions, provenance, and checksums will be archived in the sibling study directory. No final-test predictions were scored. This remains exploratory.
