## Preregistered numerical follow-up after the original smoke/main queue

The initial all-history pixel and encoder probes failed the fixed 5,000-iteration
budget at tolerance 1e-6. This is a training optimizer failure, not an accuracy
result. Do not overwrite those failures or change the original queue.

1. For each distinct failed representation/history/PCA/label configuration,
   reconstruct only its training rows and the same shuffled CV folds. Scan folds
   and sorted C values in their original order to identify the first failing fit.
   Capture convergence warnings without treating that fit as a usable model.
2. Freeze that exact training-fold problem and record float64 objective, gradient
   infinity norm (including intercept), iteration count, PCA scale range, and
   numeric scaler/PCA/classifier coefficients at 5,000 iterations.
3. Independently refit that same fold and C from the same initialization with a
   20,000-iteration cap; only if still unconverged, try 50,000. Keep tolerance,
   solver, regularization, preprocessing, labels, and class weights unchanged.
   Never inspect validation predictions during this numerical diagnosis.
4. If the same objective converges under the increased budget, register full
   fresh-ID retries of the failed probes with a common 50,000-iteration cap and
   the original 1e-6 tolerance. Do not discard C candidates or loosen tolerance.
   If it still fails, preserve the failure and investigate conditioning before
   choosing another algorithm. Do not silently produce an accuracy anyway.

The distinction to test is whether a finite iteration budget, rather than absent
identity signal, prevented a valid control. Improved validation accuracy is not
an optimizer selection criterion. This protocol is posted before executing the follow-up; it is not evidence that any retry converges.

The four cases are smoke-02, smoke-04, main-01, and main-02. All have true training labels. A cap increase changes only `max_iter`; other solver stopping limits remain unchanged and any different stopping reason is retained explicitly. A nonreproduced failure or a different exhausted limit requires a reported decision, not an automatic claim of resolution.
