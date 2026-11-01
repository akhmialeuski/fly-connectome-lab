Implementation freeze before real-data execution: `5652dc6112aaa329a508c8d7fd49910783d09bd5` on `feat/signal-stability`.

Synthetic native replay, image reordering, common-rest reuse, explicit zero-current cases, finite-value failure handling, membership isolation, and command JSON/help checks passed. New Python statements and branches have 100% test coverage. Full regression: 325 passed, two conditional skips; pre-commit, Pyright, mypy, Flake8 and package builds passed.

The ordered training membership remains SHA-256 `b64d3053b08ac88de2e3378d9c6875daec1801e3955d33aff87547e96a509f78`. The next command will execute the preregistered 163 episodes in one immutable attempt, preserving all seven case arrays and stopping before interventions if native float16 replay differs. No classifier or synaptic parameters are trained, and no held-out photographs are scored.
