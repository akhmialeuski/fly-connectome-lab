### Diagnostic numerical failures must be separated from recognition quality

In the preregistered matched-tolerance campaign (#40), both smoke all-history controls (pixels and encoded input, PCA60) reached the L-BFGS limit of 5,000 iterations at tolerance 1e-6. Their attempts are preserved as failures and have no reported validation accuracy:

- [Pixel-all failure](https://github.com/akhmialeuski/fly-connectome-lab/issues/40#issuecomment-5762887728).
- [Encoded-all failure](https://github.com/akhmialeuski/fly-connectome-lab/issues/40#issuecomment-5762913493).

The corresponding last-observation probes completed: pixels 10/60 validation correct (16.67%), encoded input 11/60 (18.33%), chance 5%. That one-observation comparison does not locate the all-history bottleneck.

The older pixel design checks used tolerance 1e-4, whereas the new stage-matched protocol uses the neural readout's 1e-6 tolerance at every stage. An earlier successful pixel score therefore does not imply the new stricter numerical check must converge within the same budget. These failures are optimizer stopping failures, not evidence that the representations contain no identity information.

Installed sklearn source confirms L-BFGS-B uses the stated `maxiter`, gradient tolerance `gtol=1e-6`, `ftol=64*float64_eps`, and L2 strength `1/(C*n_train)`. The original queue continues unchanged. After the registered attempts, a separate training-only convergence investigation will identify failing folds/C values and measure objective/gradient/iteration behavior before any retry policy is changed. No tolerance relaxation or replacement solver has been applied.
