### Numerical diagnostic code freeze

The training-only convergence investigation is implemented at commit `f821470ac263a584eb2a42e05108030d7fbb2630`, following the [registered numerical protocol](https://github.com/akhmialeuski/fly-connectome-lab/issues/40#issuecomment-5763354124).

Validation: **313 passed, 2 skipped**, 99% aggregate coverage; Ruff, Pyright, mypy, flake8, reST checks, pre-commit, and source/wheel builds pass. Binary and multiclass objective gradients are checked independently against finite differences. Tests cover iteration exhaustion, preservation of failed-fold coefficients, increased-budget convergence on the same training problem, unchanged default behavior, fold isolation, and a CLI diagnosis that never scores validation predictions.

The first 24 attempts (two audits plus 22 smoke/main probes) are archived separately with their original bytes. The original scientific queue used the original 5,000-iteration budget throughout. The new measurement command does not change those records or silently retry them.

Four numerical diagnoses now follow in order: smoke pixel-all, smoke encoded-all, main pixel-all, main encoded-all. Each scans the same original training folds/C values, reproduces the first failure at 5,000 iterations, and increases only the iteration cap on that fixed problem to 20,000, then 50,000 if necessary. Objective, gradient, warnings, sample IDs, and fold coefficients are retained. A completed diagnostic is not itself a converged recognition model. Full recognition retries require the evidence-based decision defined in the protocol.
