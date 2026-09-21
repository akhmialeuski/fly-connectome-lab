"""Shared fold-local scaler/PCA/logistic fitting for pixels and brain readouts."""

import warnings
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

MAX_LOGISTIC_ITERATIONS: int = 5000
LOGISTIC_TOLERANCE: float = 1e-4


def _projection(x: NDArray, components: int | None, seed: int) -> tuple[Pipeline, NDArray]:
    """Fit preprocessing on one training fold only.

    :param x: Finite float64 training features of shape (N,F).
    :type x: NDArray
    :param components: Maximum PCA components, or null for scaling without PCA.
    :type components: Optional[int]
    :param seed: Explicit sklearn random seed.
    :type seed: int
    :returns: Fitted transform and projected training matrix.
    :rtype: tuple[Pipeline, NDArray]
    """
    steps: list[tuple[str, StandardScaler | PCA]] = [('scaler', StandardScaler())]
    if components is not None:
        steps.append(
            (
                'pca',
                PCA(
                    n_components=min(components, len(x) - 1, x.shape[1]),
                    svd_solver='full',
                    random_state=seed,
                ),
            )
        )
    projection = Pipeline(steps=steps)
    return projection, projection.fit_transform(X=x)


def _classifier(c_value: float, seed: int, tolerance: float) -> LogisticRegression:
    """Construct the shared deterministic regularized classifier.

    :param c_value: Positive inverse regularization strength.
    :type c_value: float
    :param seed: Explicit estimator seed.
    :type seed: int
    :param tolerance: Positive optimizer convergence tolerance.
    :type tolerance: float
    :returns: Unfitted logistic regression estimator.
    :rtype: LogisticRegression
    """
    return LogisticRegression(
        C=c_value, max_iter=MAX_LOGISTIC_ITERATIONS, tol=tolerance, random_state=seed
    )


def fit_classifier(
    x_train: NDArray,
    y_train: NDArray,
    pca_components: int | None,
    c_grid: Sequence[float],
    cv_folds: int,
    seed: int,
    tolerance: float = LOGISTIC_TOLERANCE,
) -> tuple[Pipeline, dict[str, float]]:
    """Choose C using fold-local transforms, then refit on all training rows.

    :param x_train: Finite numeric feature matrix (N,F), converted to float64.
    :type x_train: NDArray
    :param y_train: Integer class labels (N,), with at least two samples per class.
    :type y_train: NDArray
    :param pca_components: Maximum PCA dimension per fold, or null to retain scaled features.
    :type pca_components: Optional[int]
    :param c_grid: Positive finite candidate inverse regularization strengths.
    :type c_grid: Sequence[float]
    :param cv_folds: Requested stratified fold count, at least two.
    :type cv_folds: int
    :param seed: Explicit estimator and fold-shuffle seed.
    :type seed: int
    :param tolerance: Positive optimizer tolerance; pixels default to 1e-4.
    :type tolerance: float
    :returns: Fitted scaler/PCA/classifier pipeline and mean CV accuracy for each C.
    :rtype: tuple[Pipeline, dict[str, float]]
    :raises ValueError: If features, class support, or hyperparameters are invalid.
    :raises ConvergenceWarning: If a classifier does not converge within the configured budget.
    """
    if x_train.ndim != 2 or x_train.shape[1] < 1 or not np.isfinite(x_train).all():
        raise ValueError('Training features must be a finite nonempty matrix.')
    if y_train.shape != (len(x_train),) or y_train.dtype.kind not in 'iu':
        raise ValueError('Training labels must be an integer vector matching the feature rows.')
    classes, counts = np.unique(ar=y_train, return_counts=True)
    if len(classes) < 2 or counts.min() < 2:
        raise ValueError(
            'Cross-validation requires at least two classes and two samples per class.'
        )
    candidates = sorted(set(c_grid))
    if (
        not np.isfinite(tolerance)
        or tolerance <= 0
        or (pca_components is not None and pca_components < 1)
        or cv_folds < 2
        or not candidates
        or any(not np.isfinite(value) or value <= 0 for value in candidates)
    ):
        raise ValueError('PCA, C candidates, and fold count must be valid positive settings.')
    x = np.asarray(a=x_train, dtype=np.float64)
    folds = StratifiedKFold(
        n_splits=min(cv_folds, int(counts.min())), shuffle=True, random_state=seed
    )
    scores: dict[float, list[float]] = {value: [] for value in candidates}
    with threadpool_limits(limits=1, user_api='blas'), warnings.catch_warnings():
        warnings.simplefilter(action='error', category=ConvergenceWarning)
        for train, held_out in folds.split(X=x, y=y_train):
            projection, train_features = _projection(
                x=x[train], components=pca_components, seed=seed
            )
            held_out_features = projection.transform(X=x[held_out])
            for value in candidates:
                classifier = _classifier(c_value=value, seed=seed, tolerance=tolerance)
                classifier.fit(X=train_features, y=y_train[train])
                scores[value].append(
                    float(classifier.score(X=held_out_features, y=y_train[held_out]))
                )
        means = {value: float(np.mean(a=values)) for value, values in scores.items()}
        best = min(candidates, key=lambda value: (-means[value], value))
        projection, transformed = _projection(x=x, components=pca_components, seed=seed)
        classifier = _classifier(c_value=best, seed=seed, tolerance=tolerance)
        classifier.fit(X=transformed, y=y_train)
    pipeline = Pipeline(steps=[*projection.steps, ('classifier', classifier)])
    return pipeline, {str(value): means[value] for value in candidates}
