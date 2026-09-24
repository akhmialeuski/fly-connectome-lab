"""Label-free memory curve: held-out R^2 of each window's input from the final state (T33)."""

import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from flystate.datasets.preprocess import prepare_dataset
from flystate.diagnostics.input_access import _features
from flystate.experiments.config import load_config
from flystate.settings import get_paths

REPO = Path(__file__).resolve().parents[4]
RECORDS = Path(os.environ['FLYSTATE_HOME']) / 'runs/diagnostics/2026-09-25-rate-access/record'
COHORT = REPO / 'research/sequential-visual-memory/2026-09-24-input-access/cohort.json'


def main(recordings: list[str]) -> None:
    """Print the window-recall curve of the central brain and descending neurons.

    :param recordings: Recording directory names under the T33 record directory.
    :type recordings: list[str]
    """
    cfg = load_config(path=REPO / 'configs/celeba-smoke.yaml')
    paths = get_paths()
    prepared = prepare_dataset(cfg=cfg, paths=paths)
    rows_by_id = {sample.sample_id: row for row, sample in enumerate(prepared.samples)}
    cohort = json.loads(COHORT.read_text(encoding='utf-8'))
    rows = [rows_by_id[item['sample_id']] for item in cohort['samples']]
    features, _ = _features(cfg=cfg, paths=paths, prepared=prepared, rows=rows)
    current = features['encoded_current'].reshape(len(rows), 16, -1)
    target_pca = PCA(n_components=10, random_state=0).fit(current.reshape(-1, current.shape[2]))
    targets = np.stack([target_pca.transform(current[:, w]) for w in range(16)], axis=1)
    for name in recordings:
        data = np.load(RECORDS / name / 'responses.npz')
        for population in ('central_brain', 'descending'):
            state = data[f'state_{population}'][:, -1].astype(np.float64)
            state = (state - state.mean(0)) / (state.std(0) + 1e-12)
            reduced = PCA(n_components=100, random_state=0).fit_transform(state)
            recall = []
            for window in range(16):
                target = targets[:, window]
                predicted = np.zeros_like(target)
                for train, test in KFold(5, shuffle=True, random_state=0).split(reduced):
                    model = Ridge(alpha=1.0).fit(reduced[train], target[train])
                    predicted[test] = model.predict(reduced[test])
                residual = ((target - predicted) ** 2).sum()
                recall.append(1 - residual / ((target - target.mean(0)) ** 2).sum())
            print(name, population, ' '.join(f'{value:.2f}' for value in recall))


if __name__ == '__main__':
    main(recordings=sys.argv[1:])
