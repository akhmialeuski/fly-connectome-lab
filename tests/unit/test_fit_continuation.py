"""L-BFGS continuation when SciPy's function-evaluation cap stops a fit before its budget."""

import warnings
from typing import Any, cast

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from flystate.readouts.fitting import EVALUATION_LIMIT, _fit

BUDGET: int = 100
FEATURES: np.ndarray = np.zeros(shape=(4, 2))
LABELS: np.ndarray = np.array([0, 1, 0, 1])
OTHER_FAILURE: str = 'ITERATIONS REACHED LIMIT'


class ScriptedEstimator:
    """Stand-in estimator whose fits report scripted iterations and convergence warnings."""

    def __init__(self, script: list[tuple[int, str | None]]) -> None:
        """Store the per-call iterations and optional warning text.

        :param script: One (iterations, warning message or none) pair per fit call.
        :type script: list[tuple[int, str | None]]
        """
        self.script = script
        self.max_iter = BUDGET
        self.warm_start = False
        self.calls: list[tuple[int, bool]] = []
        self.n_iter_ = np.zeros(shape=1, dtype=np.int32)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ScriptedEstimator':  # noqa: N803
        """Record the call settings and emit the scripted outcome.

        :param X: Ignored features.
        :type X: np.ndarray
        :param y: Ignored labels.
        :type y: np.ndarray
        :returns: This estimator.
        :rtype: ScriptedEstimator
        """
        iterations, message = self.script[len(self.calls)]
        self.calls.append((self.max_iter, self.warm_start))
        self.n_iter_ = np.array([iterations], dtype=np.int32)
        if message is not None:
            warnings.warn(message=message, category=ConvergenceWarning, stacklevel=2)
        return self

    def set_params(self, **params: Any) -> 'ScriptedEstimator':
        """Apply parameter updates as scikit-learn does.

        :param params: Parameter values by name.
        :type params: Any
        :returns: This estimator.
        :rtype: ScriptedEstimator
        """
        for name, value in params.items():
            setattr(self, name, value)
        return self


def _run(script: list[tuple[int, str | None]]) -> ScriptedEstimator:
    """Fit a scripted estimator through the continuation helper.

    :param script: Scripted outcomes per call.
    :type script: list[tuple[int, str | None]]
    :returns: The estimator after fitting.
    :rtype: ScriptedEstimator
    """
    estimator = ScriptedEstimator(script=script)
    _fit(classifier=cast(LogisticRegression, estimator), x=FEATURES, y=LABELS)
    return estimator


def test_converged_fit_uses_a_single_plain_call() -> None:
    """Leave a fit that converges at once on the original single-call path."""
    estimator = _run(script=[(40, None)])
    assert estimator.calls == [(BUDGET, False)]
    assert int(estimator.n_iter_[0]) == 40


def test_evaluation_cap_is_continued_within_the_budget() -> None:
    """Continue with a warm start for the remaining iterations and report their total."""
    estimator = _run(script=[(30, EVALUATION_LIMIT), (25, None)])
    assert estimator.calls == [(BUDGET, False), (BUDGET - 30, True)]
    assert int(estimator.n_iter_[0]) == 55
    assert estimator.max_iter == BUDGET and estimator.warm_start is False


@pytest.mark.parametrize(
    argnames='script',
    argvalues=[
        [(30, OTHER_FAILURE)],
        [(60, EVALUATION_LIMIT), (40, EVALUATION_LIMIT)],
    ],
)
def test_other_failures_and_an_exhausted_budget_still_raise(
    script: list[tuple[int, str | None]],
) -> None:
    """Raise for any other convergence failure and once the whole budget is used.

    :param script: Scripted outcomes per call.
    :type script: list[tuple[int, str | None]]
    """
    with pytest.raises(expected_exception=ConvergenceWarning):
        _run(script=script)
