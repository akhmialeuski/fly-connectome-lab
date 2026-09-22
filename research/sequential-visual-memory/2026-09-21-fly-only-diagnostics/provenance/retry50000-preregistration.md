## Numerical diagnosis outcome and preregistered full retries

All four original failures reproduce as iteration-limit stops. Independently refitting the same training fold and C with only a larger iteration cap converges and meets the original gradient threshold of 1e-6. No validation predictions were scored in this diagnosis.

| Case | Fold | C | Iterations to convergence | Gradient infinity norm |
| --- | --- | --- | --- | --- |
| convergence-smoke-02-pixels-persistent-all-both-pca60-true | 1 | 0.01 | 7010 | 6.92781922e-07 |
| convergence-smoke-04-encoded-persistent-all-both-pca60-true | 5 | 0.01 | 5411 | 5.59082798e-07 |
| convergence-main-01-pixels-persistent-all-both-pca60-true | 1 | 0.01 | 6117 | 8.62112901e-07 |
| convergence-main-02-encoded-persistent-all-both-pca60-true | 1 | 0.01 | 10396 | 5.8189829e-07 |

Decision registered before these new full probes: execute the four fresh attempt IDs below in order, using code f821470ac263a584eb2a42e05108030d7fbb2630, the original five training-only CV folds, scaler and PCA60, C grid [0.01, 0.1, 1, 10], tolerance 1e-6, seed 0, and a common maximum of 50,000 iterations. Only the iteration cap changes. Other optimizer limits remain unchanged. Any new failure is retained and investigated; no failed C candidate is removed and no accuracy is substituted. Original failures remain immutable. Full-probe convergence is not guaranteed by convergence of the first failed fold.

These are diagnostic input controls, not replacement recognition models. Validation remains exploratory; final-test and reserved photos are not scored. This result explains the original numerical stop, not the fly representation failure or the broader cause of conditioning.

```json
[
  {
    "id": "retry50000-smoke-02-pixels-persistent-all-both-pca60-true",
    "config": "/home/anatolk/data/flystate/runs/20260920-071344-celeba-smoke-9406ee/config.yaml",
    "phase": "probe",
    "representation": "pixels",
    "history": "all",
    "features": "both",
    "components": 60,
    "label-mode": "true",
    "max-iterations": 50000
  },
  {
    "id": "retry50000-smoke-04-encoded-persistent-all-both-pca60-true",
    "config": "/home/anatolk/data/flystate/runs/20260920-071344-celeba-smoke-9406ee/config.yaml",
    "phase": "probe",
    "representation": "encoded",
    "history": "all",
    "features": "both",
    "components": 60,
    "label-mode": "true",
    "max-iterations": 50000
  },
  {
    "id": "retry50000-main-01-pixels-persistent-all-both-pca60-true",
    "config": "/home/anatolk/data/flystate/runs/20260920-075340-celeba-persistent-541c94/config.yaml",
    "phase": "probe",
    "representation": "pixels",
    "history": "all",
    "features": "both",
    "components": 60,
    "label-mode": "true",
    "max-iterations": 50000
  },
  {
    "id": "retry50000-main-02-encoded-persistent-all-both-pca60-true",
    "config": "/home/anatolk/data/flystate/runs/20260920-075340-celeba-persistent-541c94/config.yaml",
    "phase": "probe",
    "representation": "encoded",
    "history": "all",
    "features": "both",
    "components": 60,
    "label-mode": "true",
    "max-iterations": 50000
  }
]
```
