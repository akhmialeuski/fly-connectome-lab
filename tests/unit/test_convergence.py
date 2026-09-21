"""Check the numerical objective independently before diagnosing real optimization failures."""

import numpy as np
import pytest

from flystate.diagnostics.convergence import measure_fit, objective_gradient


class TestObjectiveGradient:
    """Validate binary and multiclass derivatives with nonconsecutive class labels."""

    @pytest.mark.parametrize('class_count', [2, 3])
    def test_finite_differences(self, class_count: int) -> None:
        """Create a fixed problem, compute gradients, and compare every parameter to differences.

        :param class_count: Binary or multiclass likelihood.
        :type class_count: int
        """
        generator = np.random.default_rng(seed=4)
        x = generator.normal(size=(12, 4))
        classes = np.asarray(a=[2, 7, 11][:class_count])
        y = classes[np.arange(len(x)) % class_count]
        coef = generator.normal(size=(1 if class_count == 2 else class_count, x.shape[1]))
        intercept = generator.normal(size=len(coef))
        _, weights_gradient, bias_gradient = objective_gradient(
            x=x, y=y, classes=classes, coef=coef, intercept=intercept, c_value=0.3
        )
        epsilon = 1e-6
        for array, gradient in ((coef, weights_gradient), (intercept, bias_gradient)):
            for index in np.ndindex(array.shape):
                original = array[index]
                array[index] = original + epsilon
                positive = objective_gradient(
                    x=x, y=y, classes=classes, coef=coef, intercept=intercept, c_value=0.3
                )[0]
                array[index] = original - epsilon
                negative = objective_gradient(
                    x=x, y=y, classes=classes, coef=coef, intercept=intercept, c_value=0.3
                )[0]
                array[index] = original
                assert gradient[index] == pytest.approx(
                    expected=(positive - negative) / (2 * epsilon), abs=1e-8
                )

    @pytest.mark.parametrize('case', ['empty', 'label', 'class', 'shape', 'nan', 'regularization'])
    def test_invalid_problem(self, case: str) -> None:
        """Reject malformed problems rather than reporting meaningless convergence statistics.

        :param case: Invalid shape, identity, or numerical input.
        :type case: str
        """
        x = np.ones(shape=(4, 2))
        y = np.asarray(a=[0, 1, 0, 1])
        classes = np.asarray(a=[0, 1])
        coef = np.zeros(shape=(1, 2))
        intercept = np.zeros(shape=1)
        if case == 'empty':
            x = x[:0]
        elif case == 'label':
            y[0] = 7
        elif case == 'class':
            classes = classes[::-1]
        elif case == 'shape':
            coef = coef.T
        elif case == 'nan':
            intercept[0] = np.nan
        with pytest.raises(expected_exception=ValueError):
            objective_gradient(
                x=x,
                y=y,
                classes=classes,
                coef=coef,
                intercept=intercept,
                c_value=0 if case == 'regularization' else 1,
            )


class TestMeasureFit:
    """Keep the same objective while making iteration exhaustion observable."""

    def test_budget_exhaustion_and_convergence(self) -> None:
        """Fit two budgets, then verify warnings and improvements in objective and gradient."""
        generator = np.random.default_rng(seed=8)
        x = generator.normal(size=(40, 5))
        y = np.arange(len(x)) % 4
        _, short = measure_fit(x=x, y=y, c_value=0.1, seed=0, budget=1)
        _, long = measure_fit(x=x, y=y, c_value=0.1, seed=0, budget=1000)
        assert not short['converged'] and short['warnings']
        assert long['converged'] and not long['warnings']
        assert long['objective'] < short['objective']
        assert long['gradient_infinity_norm'] < short['gradient_infinity_norm']
        assert long['gradient_threshold_met']
