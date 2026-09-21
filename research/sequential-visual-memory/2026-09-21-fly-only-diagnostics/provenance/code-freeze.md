### Scientific execution started

Code commit: `aa36121ac96c51361e33d0b8267445a3ab31030b` on `feat/fly-only-diagnostics`. The checkout was clean before starting real-data attempts. The code and preregistration are pushed; the implementation PR is a draft stacked on the existing archive branch.

Validation: full regression run **295 passed, 2 skipped**, 99% aggregate coverage; the final targeted run **42 passed**, including optional-PCA fold isolation and diagnostic integration. Ruff, Pyright, mypy, flake8, reST checks, pre-commit, frozen dependency synchronization, and source/wheel builds pass. The two skipped tests require separately enabled real-connectome fixtures; the research itself uses the explicitly installed real brain.

Execution now follows the registered order: P0 smoke audit, P0 main audit, then smoke probes 01–18. Attempts run serially after verification finished. Per-attempt comments retain failures and do not substitute an unregistered retry.
