## Training-size protocol and membership freeze

The next bounded study is #42: 20 fixed identities, the same 60 validation photographs, and nested 2/4/8/14 training photographs per identity. Pixels/all, encoded/all, and fixed-neural/last are compared with identical fitting settings. Five subset seeds are retained; each representation's identical 14-photo endpoint is fitted once, giving 48 unique attempts.

Before any fit, all sample memberships and the ordered schedule were generated under `$FLYSTATE_HOME/runs/diagnostics/2026-09-22-training-size-curve/execution/`:

- `schedule.json`: SHA-256 `ebbd67b6325093e23bb7d707c2919025bae120408f0afef3033e64d8d53eea56`.
- `frozen-memberships.json`: SHA-256 `ce1324af47551dd8b9eba4b05ced2c33c4c3a12bb204cb80a2ffce7ff6847900`.
- Per-label subset stream: `SeedSequence([subset_seed, stable_int("learning-curve-subset:label:{label}")])`; sorted original training IDs are permuted, prefixes selected, and original fitting order restored.
- Paired identity-cluster bootstrap will use 2,000 resamples, seed 0, and namespace `learning-curve-identity-bootstrap`. With the same three validation photographs per identity, bootstrapping the 20 per-identity mean paired correctness differences is equivalent to retaining all three photographs in each sampled identity cluster. It estimates a conditional exploratory interval, separately from the five subset-seed realizations.

Implementation validation: 322 tests passed, two optional tests skipped; the new subset module has 100% statement and branch coverage. Full-endpoint model arrays reproduce the original unsampled synthetic fit exactly, and smaller subsets retain the same validation IDs and labels. Ruff, Pyright, mypy, flake8, reST checks, pre-commit, and source/wheel builds pass.

No real-data learning-curve models have been fitted yet. First complete and review the remaining #40 control, publish its archive, then freeze the rebased execution commit and begin this ordered schedule. Same-image noise and quantization diagnostics remain subsequent separate requirements of #39.
