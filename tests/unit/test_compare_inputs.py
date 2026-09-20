"""Validate comparison input grids independently of numerical training."""

from typing import Any

import pytest

from flystate.evaluation.compare import IncompatibleRunsError, _prediction_map, flatten_config


class TestComparisonInputs:
    """Reject malformed paired observations before any bootstrap computation."""

    def test_flatten(self) -> None:
        """Keep sequence-valued leaves intact while flattening nested configuration mappings."""
        assert flatten_config(values={'a': {'b': [1, 2]}, 'c': 3}) == {'a.b': [1, 2], 'c': 3}

    @pytest.mark.parametrize(
        'change',
        [
            {'sample_id': 2},
            {'t': 0},
            {'y_true': '1'},
            {'correct': 1},
            {'correct': False},
        ],
    )
    def test_invalid_row(self, change: dict[str, Any]) -> None:
        """Reject invalid record types, observation ranges, and inconsistent correctness.

        :param change: Corruption applied to a valid prediction row.
        :type change: dict[str, Any]
        """
        row = {'sample_id': 'a', 't': 1, 'y_true': 0, 'y_pred': 0, 'correct': True}
        with pytest.raises(expected_exception=IncompatibleRunsError):
            _prediction_map(
                metadata={'timesteps': 1, 'n_samples': 1}, predictions=[{**row, **change}], name='A'
            )

    def test_invalid_grid(self) -> None:
        """Reject missing records, duplicate pairs, changing labels, and invalid dimensions."""
        row = {'sample_id': 'a', 't': 1, 'y_true': 0, 'y_pred': 0, 'correct': True}
        cases: list[tuple[dict[str, Any], list[dict[str, Any]]]] = [
            ({'timesteps': 0, 'n_samples': 1}, [row]),
            ({'timesteps': 1, 'n_samples': 1}, [row, row]),
            ({'timesteps': 2, 'n_samples': 1}, [row]),
            ({'timesteps': 2, 'n_samples': 1}, [row, {**row, 't': 2, 'y_true': 1, 'y_pred': 1}]),
            ({'timesteps': 1, 'n_samples': 1}, [{}]),
        ]
        for metadata, records in cases:
            with pytest.raises(expected_exception=IncompatibleRunsError):
                _prediction_map(metadata=metadata, predictions=records, name='B')
