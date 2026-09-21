"""Training-only measurements of the original logistic objective and stopping budget."""

import warnings
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit, logsumexp
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from flystate.diagnostics.artifacts import attempt, export_classifier
from flystate.diagnostics.data import load_representation
from flystate.experiments.config import ExperimentConfig
from flystate.log import get_logger
from flystate.readouts.fitting import _classifier, _projection
from flystate.settings import Paths
from flystate.storage.json import write_json

CONVERGENCE_BUDGETS: tuple[int, ...] = (5000, 20000, 50000)
DIAGNOSTIC_TOLERANCE: float = 1e-6


def objective_gradient(
    x: NDArray,
    y: NDArray,
    classes: NDArray,
    coef: NDArray,
    intercept: NDArray,
    c_value: float,
) -> tuple[float, NDArray, NDArray]:
    """Evaluate the unweighted sklearn L2 logistic objective and its exact gradients.

    The objective is mean negative log likelihood plus ||coef||^2/(2*C*N).
    Intercepts are unpenalized; binary models use one logit for classes[1].

    :param x: Finite projected float64 training features (N,F).
    :type x: NDArray
    :param y: Training labels (N,).
    :type y: NDArray
    :param classes: Sorted distinct integer class labels (K,).
    :type classes: NDArray
    :param coef: Float64 classifier coefficients (K,F), or (1,F) for binary data.
    :type coef: NDArray
    :param intercept: Float64 intercepts (K,), or (1,) for binary data.
    :type intercept: NDArray
    :param c_value: Positive inverse regularization strength.
    :type c_value: float
    :returns: Objective in nats per sample, coefficient gradient, and intercept gradient.
    :rtype: tuple[float, NDArray, NDArray]
    :raises ValueError: If shapes, class identities, or numeric values are invalid.
    """
    if (
        x.ndim != 2
        or min(x.shape) < 1
        or not np.isfinite(x).all()
        or y.shape != (len(x),)
        or classes.ndim != 1
        or len(classes) < 2
        or np.any(classes[1:] <= classes[:-1])
        or not np.isin(y, classes).all()
        or not np.isfinite(c_value)
        or c_value <= 0
    ):
        raise ValueError('Invalid logistic objective inputs.')
    rows = 1 if len(classes) == 2 else len(classes)
    if (
        coef.shape != (rows, x.shape[1])
        or intercept.shape != (rows,)
        or not np.isfinite(coef).all()
        or not np.isfinite(intercept).all()
    ):
        raise ValueError('Invalid logistic parameter shapes or values.')
    labels = np.searchsorted(a=classes, v=y)
    logits = x @ coef.T + intercept
    if len(classes) == 2:
        loss = np.mean(a=np.logaddexp(0, logits[:, 0]) - labels * logits[:, 0])
        residual = (expit(logits[:, 0]) - labels)[:, None]
    else:
        normalization = np.asarray(a=logsumexp(a=logits, axis=1), dtype=np.float64)
        loss = np.mean(a=normalization - logits[np.arange(len(x)), labels])
        residual = np.exp(logits - normalization[:, None])
        residual[np.arange(len(x)), labels] -= 1
    strength = 1 / (c_value * len(x))
    loss += 0.5 * strength * np.sum(a=coef**2)
    return (
        float(loss),
        residual.T @ x / len(x) + strength * coef,
        residual.mean(axis=0),
    )


def measure_fit(
    x: NDArray, y: NDArray, c_value: float, seed: int, budget: int
) -> tuple[LogisticRegression, dict[str, Any]]:
    """Fit one training problem while preserving stopping warnings as diagnostic evidence.

    :param x: Already fitted/projected training features (N,F), float64.
    :type x: NDArray
    :param y: Training labels (N,).
    :type y: NDArray
    :param c_value: Fixed inverse regularization strength.
    :type c_value: float
    :param seed: Original estimator seed.
    :type seed: int
    :param budget: Positive iteration cap, independent of validation accuracy.
    :type budget: int
    :returns: Fitted diagnostic estimator and objective/gradient/stopping measurements.
    :rtype: tuple[LogisticRegression, dict[str, Any]]
    :raises ValueError: If the iteration budget or estimator inputs are invalid.
    """
    if budget < 1:
        raise ValueError('The diagnostic iteration budget must be positive.')
    classifier = _classifier(
        c_value=c_value, seed=seed, tolerance=DIAGNOSTIC_TOLERANCE, max_iterations=budget
    )
    started = perf_counter()
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter(action='always', category=ConvergenceWarning)
        classifier.fit(X=x, y=y)
    elapsed = perf_counter() - started
    loss, weight_gradient, bias_gradient = objective_gradient(
        x=x,
        y=y,
        classes=classifier.classes_,
        coef=classifier.coef_,
        intercept=np.asarray(a=classifier.intercept_, dtype=np.float64),
        c_value=c_value,
    )
    gradient_norm = float(max(np.abs(weight_gradient).max(), np.abs(bias_gradient).max()))
    return classifier, {
        'C': c_value,
        'max_iterations': budget,
        'iterations': classifier.n_iter_.tolist(),
        'converged': not any(issubclass(w.category, ConvergenceWarning) for w in recorded),
        'warnings': [
            {'category': w.category.__name__, 'message': str(w.message)} for w in recorded
        ],
        'objective': loss,
        'gradient_infinity_norm': gradient_norm,
        'gradient_threshold_met': gradient_norm <= DIAGNOSTIC_TOLERANCE,
        'coefficient_norm': float(np.linalg.norm(x=classifier.coef_)),
        'elapsed_seconds': elapsed,
    }


def preserve_fit(
    projection: Pipeline,
    classifier: LogisticRegression,
    measurement: dict[str, Any],
    directory: Path,
) -> None:
    """Save diagnostic fold coefficients with an explicit non-production purpose.

    :param projection: Original fold-local scaler and optional PCA.
    :type projection: Pipeline
    :param classifier: Estimator returned even when its budget was exhausted.
    :type classifier: LogisticRegression
    :param measurement: Objective and stopping measurements for this estimator.
    :type measurement: dict[str, Any]
    :param directory: New numeric export directory.
    :type directory: Path
    """
    model = Pipeline(steps=[*projection.steps, ('classifier', classifier)])
    metadata = export_classifier(model=model, directory=directory)
    metadata.update(
        purpose='training-fold convergence diagnostic; not a selected readout',
        converged=measurement['converged'],
    )
    write_json(path=directory / 'model.json', value=metadata)
    write_json(path=directory / 'measurement.json', value=measurement)


def diagnose_convergence(
    cfg: ExperimentConfig,
    paths: Paths,
    output: Path,
    representation: str,
    history: str,
    features: str,
    components: int | None,
    budgets: Sequence[int] = CONVERGENCE_BUDGETS,
) -> dict[str, Any]:
    """Reproduce the first failing training fold/C and increase only its iteration cap.

    :param cfg: Original cohort and cross-validation configuration.
    :type cfg: ExperimentConfig
    :param paths: Working data home.
    :type paths: Paths
    :param output: Fresh diagnostic attempt directory.
    :type output: Path
    :param representation: Original pixels, encoded, or neural representation.
    :type representation: str
    :param history: Original last/all history policy.
    :type history: str
    :param features: Original neural feature selection.
    :type features: str
    :param components: Original PCA cap, or null for scaling alone.
    :type components: Optional[int]
    :param budgets: Strictly increasing positive iteration caps; first reproduces the failure.
    :type budgets: Sequence[int]
    :returns: Training-only scan, first-failure identity, and fixed-problem budget measurements.
    :rtype: dict[str, Any]
    :raises ValueError: If budgets or PCA settings are invalid.
    """
    parameters = {
        'kind': 'convergence_diagnostic',
        'representation': representation,
        'history': history,
        'features': features,
        'pca_components': components,
        'budgets': list(budgets),
        'tolerance': DIAGNOSTIC_TOLERANCE,
        'seed': cfg.seed,
        'label_mode': 'true',
        'c_grid': cfg.readout.c_grid,
    }
    with attempt(paths=paths, cfg=cfg, output=output, parameters=parameters) as directory:
        if (
            not budgets
            or any(b < 1 for b in budgets)
            or list(budgets) != sorted(set(budgets))
            or (components is not None and components < 1)
        ):
            raise ValueError('Budgets must increase strictly and PCA components must be positive.')
        data = load_representation(
            cfg=cfg, paths=paths, representation=representation, history=history, features=features
        )
        training = [i for i, sample in enumerate(data.samples) if sample.split == 'train']
        x = data.x[training].astype(np.float64)
        y = np.asarray(a=[data.samples[i].label for i in training], dtype=np.int64)
        _, counts = np.unique(ar=y, return_counts=True)
        fold_count = min(cfg.readout.cv_folds, int(counts.min()))
        folds = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=cfg.seed)
        report: dict[str, Any] = {
            'parameters': parameters,
            'provenance': data.provenance,
            'effective_cv_folds': fold_count,
            'scan': [],
            'first_failure': None,
            'budget_measurements': [],
            'validation_predictions_scored': 0,
            'inference': 'Numerical training diagnostic; no recognition accuracy is estimated.',
        }
        with threadpool_limits(limits=1, user_api='blas'):
            for fold, (fit_rows, _) in enumerate(folds.split(X=x, y=y), start=1):
                projection, projected = _projection(
                    x=x[fit_rows], components=components, seed=cfg.seed
                )
                for c_value in sorted(set(cfg.readout.c_grid)):
                    classifier, measurement = measure_fit(
                        x=projected,
                        y=y[fit_rows],
                        c_value=c_value,
                        seed=cfg.seed,
                        budget=budgets[0],
                    )
                    measurement['fold'] = fold
                    report['scan'].append(measurement)
                    write_json(path=directory / 'scan.json', value=report['scan'])
                    get_logger(name='convergence').info(
                        'training_fit_measured',
                        fold=fold,
                        c_value=c_value,
                        converged=measurement['converged'],
                        iterations=measurement['iterations'],
                    )
                    if measurement['converged']:
                        continue
                    scales = projected.std(axis=0)
                    report['first_failure'] = {
                        'fold': fold,
                        'C': c_value,
                        'training_rows': len(fit_rows),
                        'projected_features': projected.shape[1],
                        'projected_std_min': float(scales.min()),
                        'projected_std_max': float(scales.max()),
                        'training_sample_ids': [
                            data.samples[training[i]].sample_id for i in fit_rows
                        ],
                    }
                    report['budget_measurements'].append(measurement)
                    preserve_fit(
                        projection=projection,
                        classifier=classifier,
                        measurement=measurement,
                        directory=directory / f'fold-{fold}-budget-{budgets[0]}',
                    )
                    for budget in budgets[1:]:
                        classifier, measurement = measure_fit(
                            x=projected,
                            y=y[fit_rows],
                            c_value=c_value,
                            seed=cfg.seed,
                            budget=budget,
                        )
                        measurement['fold'] = fold
                        report['budget_measurements'].append(measurement)
                        preserve_fit(
                            projection=projection,
                            classifier=classifier,
                            measurement=measurement,
                            directory=directory / f'fold-{fold}-budget-{budget}',
                        )
                        write_json(path=directory / 'report.json', value=report)
                        if measurement['converged']:
                            break
                    break
                if report['first_failure'] is not None:
                    break
        report['failure_reproduced'] = report['first_failure'] is not None
        write_json(path=directory / 'report.json', value=report)
    return report
