"""Automatic discovery of nested, unfamiliar, and evolving research attempts."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from flystate.diagnostics.probes import run_probe
from flystate.experiments.config import ExperimentConfig
from flystate.hashing import sha256_file
from flystate.settings import get_paths
from flystate.storage.json import write_json
from flystate.storage.parquet import write_table
from flystate.viewer import experiments
from flystate.viewer.app import create_app
from flystate.viewer.experiments import (
    MAX_TEXT_BYTES,
    ExperimentStore,
    bounded_document,
    nested_resource,
)


class TestExperimentDiscovery:
    """Keep every manifest-backed attempt visible without a study-name allowlist."""

    def test_recorded_outcomes_are_discovered(self) -> None:
        """Expose contrast decisions, frozen selections and photograph counts, skipping junk."""
        paths = get_paths()
        reports = {
            'analyze': {
                'contrasts': [
                    {
                        'name': 'W1',
                        'difference_pp': 1.5,
                        'interval_95_pp': [-1.0, 4.0],
                        'decision': 'no practically relevant difference',
                        'per_cohort_pp': {'s4': 2.0},
                    },
                    {'name': 'broken'},
                    'not a contrast',
                ]
            },
            'select': {
                'families': {'fly': {'selected_alpha': 0.75}, 'bad': {'selected_alpha': 'x'}}
            },
            'record': {'episodes': 400},
            'rules': {'decisions': {'S1': True, 'S2': False, 'note': 'ignored'}},
            'flag': {'episodes': True},
        }
        for name, report in reports.items():
            directory = paths.runs / 'outcomes' / name
            write_json(
                path=directory / 'manifest.json',
                value={'status': 'completed', 'parameters': {'kind': f'{name}_kind'}},
            )
            write_json(path=directory / 'report.json', value=report)
        entries = {
            entry['name']: entry
            for entry in ExperimentStore(root=paths.runs).discover(legacy_runs=[])
        }
        assert entries['analyze']['decisions'] == [
            {
                'name': 'W1',
                'difference_pp': 1.5,
                'interval_95_pp': [-1.0, 4.0],
                'decision': 'no practically relevant difference',
            }
        ]
        assert entries['select']['selection'] == {'fly': 0.75}
        assert entries['record']['episodes'] == 400
        assert entries['rules']['checks'] == {'S1': True, 'S2': False}
        assert 'episodes' not in entries['flag']
        assert experiments.recorded_outcome(payload={'contrasts': 'x', 'families': []}) == {}

    def test_analysis_gate_discovery(self) -> None:
        """Expose a saved decision and case count for an unfamiliar analysis kind."""
        paths = get_paths()
        directory = paths.runs / 'future-analysis' / 'decision'
        write_json(
            path=directory / 'manifest.json',
            value={'status': 'completed', 'parameters': {'kind': 'custom_analysis'}},
        )
        write_json(
            path=directory / 'report.json',
            value={'gate': 'do_not_advance', 'fits': {'A': {}, 'B': {}}},
        )
        entry = ExperimentStore(root=paths.runs).discover(legacy_runs=[])[0]
        assert entry['gate'] == 'do_not_advance'
        assert entry['case_count'] == 2

    def test_nested_attempts(self) -> None:
        """Create an unfamiliar nested attempt, read it, then observe a new completed result."""
        paths = get_paths()
        relative = 'future studies/arbitrary/deep/attempt one'
        directory = paths.runs / relative
        write_json(
            path=directory / 'manifest.json',
            value={'status': 'running', 'parameters': {'kind': 'future_kind'}},
        )
        client = TestClient(app=create_app(paths=paths), base_url='http://127.0.0.1')
        catalog = client.get(url='/api/catalog').json()
        entry = next(row for row in catalog['experiments'] if row['id'] == relative)
        assert entry['status'] == 'running' and entry['kind'] == 'future_kind'
        write_json(
            path=directory / 'manifest.json',
            value={'status': 'completed', 'parameters': {'kind': 'future_kind'}},
        )
        write_json(
            path=directory / 'report.json',
            value={
                'scores': {'validation': {'accuracy': 0.5, 'n': 4}},
                'conclusion': 'Recorded study conclusion.',
            },
        )
        updated = next(
            row
            for row in client.get(url='/api/catalog').json()['experiments']
            if row['id'] == relative
        )
        assert updated['status'] == 'completed'
        assert updated['result_summary'] == 'Recorded study conclusion.'
        assert updated['scores']['validation']['accuracy'] == 0.5
        assert updated['revision'] != entry['revision']
        detail = client.get(url='/api/experiments/detail', params={'path': relative})
        assert detail.status_code == 200
        assert detail.json()['documents']['report.json']['scores']['validation']['n'] == 4

    def test_damage_and_boundaries(self, tmp_path: Path) -> None:
        """Retain corrupt entries, reject escaped paths, and bound metadata and evidence reads.

        :param tmp_path: Isolated files outside the runs boundary.
        :type tmp_path: Path
        """
        paths = get_paths()
        directory = paths.runs / 'new' / 'study' / 'broken'
        write_json(
            path=directory / 'manifest.json',
            value={'status': 'failed', 'parameters': {}, 'error': 'optimizer stopped'},
        )
        write_json(path=directory / 'report.json', value={'scores': []})
        (directory / 'config.json').write_text(data='{"dataset": "invalid"}')
        store = ExperimentStore(root=paths.runs)
        entry = store.discover(legacy_runs=[])[0]
        assert entry['status'] == 'failed' and len(entry['warnings']) == 2
        (directory / 'report.json').write_text(data='incomplete JSON')
        detail = store.detail(relative='new/study/broken')
        assert detail['warnings'] and 'manifest.json' in detail['documents']
        (directory / 'manifest.json').write_text(data='incomplete')
        assert store.discover(legacy_runs=[])[0]['status'] == 'unreadable'
        write_json(path=directory / 'manifest.json', value={'parameters': []})
        assert store.discover(legacy_runs=[])[0]['status'] == 'unreadable'
        outside = tmp_path / 'outside.json'
        outside.write_text(data='{"private":true}')
        (directory / 'escape.json').symlink_to(target=outside)
        (directory / 'external-directory').symlink_to(target=tmp_path, target_is_directory=True)
        assert any(
            'escapes' in warning
            for warning in store.detail(relative='new/study/broken')['warnings']
        )
        for relative in (
            '../outside.json',
            '/outside.json',
            'new//study',
            'new/./study',
            'new\\study',
            '\x00',
        ):
            with pytest.raises(expected_exception=ValueError):
                nested_resource(root=paths.runs, relative=relative)
        with pytest.raises(expected_exception=FileNotFoundError):
            store.detail(relative='absent')
        client = TestClient(app=create_app(paths=paths), base_url='http://localhost')
        for name in ('escape.json', '../broken/manifest.json', '/outside.json'):
            response = client.get(
                url='/api/experiments/evidence', params={'path': 'new/study/broken', 'name': name}
            )
            assert response.status_code == 409
        large = directory / 'large.log'
        large.write_text(data='x' * (MAX_TEXT_BYTES + 1))
        for name in ('large.log', 'external-directory', 'model.npz'):
            if name == 'model.npz':
                (directory / name).write_bytes(data=b'not loaded as an array')
            with pytest.raises(expected_exception=ValueError):
                store.evidence(relative='new/study/broken', name=name, offset=0, limit=10)
        for offset, limit in ((-1, 10), (0, 0), (0, 201)):
            assert (
                client.get(
                    url='/api/experiments/evidence',
                    params={
                        'path': 'new/study/broken',
                        'name': 'manifest.json',
                        'offset': offset,
                        'limit': limit,
                    },
                ).status_code
                == 422
            )
            with pytest.raises(expected_exception=ValueError):
                store.evidence(
                    relative='new/study/broken', name='manifest.json', offset=offset, limit=limit
                )
        (directory / 'config.json').write_text(data='x' * (MAX_TEXT_BYTES + 1))
        assert any(
            'limit' in warning for warning in store.detail(relative='new/study/broken')['warnings']
        )

    def test_streamed_pages(self) -> None:
        """Page across Parquet batches, preserve row order, and leave every artifact unchanged."""
        paths = get_paths()
        directory = paths.runs / 'unknown' / 'attempt'
        write_json(path=directory / 'manifest.json', value={})
        write_table(
            path=directory / 'measurements.parquet', rows=[{'value': i} for i in range(600)]
        )
        before = {p.name: (p.stat().st_mtime_ns, sha256_file(path=p)) for p in directory.iterdir()}
        store = ExperimentStore(root=paths.runs)
        for offset, expected in (
            (0, list(range(10))),
            (253, list(range(253, 263))),
            (590, list(range(590, 600))),
            (700, []),
        ):
            page = store.evidence(
                relative='unknown/attempt', name='measurements.parquet', offset=offset, limit=10
            )
            assert page['total'] == 600
            assert [row['value'] for row in page['rows']] == expected
        text = store.evidence(relative='unknown/attempt', name='manifest.json', offset=0, limit=10)
        assert json.loads(s=text['text']) == {}
        after = {p.name: (p.stat().st_mtime_ns, sha256_file(path=p)) for p in directory.iterdir()}
        assert after == before

    def test_probe_photographs(self, tiny_experiment: ExperimentConfig) -> None:
        """Run an offline probe, inspect identity references, and serve only recorded sample images.

        :param tiny_experiment: Synthetic source photographs and encoder configuration.
        :type tiny_experiment: ExperimentConfig
        """
        paths = get_paths()
        relative = 'any-study/probe'
        run_probe(
            cfg=tiny_experiment,
            paths=paths,
            output=paths.runs / relative,
            representation='encoded',
            history='all',
            features='both',
            components=10,
            label_mode='true',
        )
        client = TestClient(app=create_app(paths=paths), base_url='http://localhost')
        before = {
            str(p): (p.stat().st_mtime_ns, sha256_file(path=p))
            for p in paths.home.rglob('*')
            if p.is_file()
        }
        detail = client.get(url='/api/experiments/detail', params={'path': relative}).json()
        assert len(detail['identities']) == tiny_experiment.dataset.subset.n_identities
        page = client.get(
            url='/api/experiments/evidence',
            params={'path': relative, 'name': 'validation-predictions.parquet', 'limit': 2},
        ).json()
        assert page['total'] == 8 and len(page['rows']) == 2
        for sample_id in (detail['identities'][0]['sample_id'], page['rows'][0]['sample_id']):
            image = client.get(
                url='/api/experiments/image', params={'path': relative, 'sample_id': sample_id}
            )
            assert image.status_code == 200 and image.content.startswith(b'\x89PNG')
        assert (
            client.get(
                url='/api/experiments/image', params={'path': relative, 'sample_id': 'not-a-member'}
            ).status_code
            == 404
        )
        after = {
            str(p): (p.stat().st_mtime_ns, sha256_file(path=p))
            for p in paths.home.rglob('*')
            if p.is_file()
        }
        assert after == before

    def test_preview_and_reference_limits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Reject oversized table previews and directories, then bound photos and retain bad tables.

        :param monkeypatch: Isolated preview-budget overrides.
        :type monkeypatch: pytest.MonkeyPatch
        """
        paths = get_paths()
        directory = paths.runs / 'study' / 'attempt'
        write_json(path=paths.runs / 'manifest.json', value={})
        write_json(path=directory / 'manifest.json', value={})
        (directory / 'folder').mkdir()
        store = ExperimentStore(root=paths.runs)
        assert len(store.discover(legacy_runs=[])) == 1
        with pytest.raises(expected_exception=ValueError, match='regular file'):
            bounded_document(path=directory)
        with pytest.raises(expected_exception=ValueError, match='regular file'):
            store.evidence(relative='study/attempt', name='folder', offset=0, limit=10)
        write_table(
            path=directory / 'samples.parquet',
            rows=[
                {'sample_id': str(i), 'label': i, 'identity': i, 'split': 'train'}
                for i in range(501)
            ],
        )
        client = TestClient(app=create_app(paths=paths), base_url='http://localhost')
        detail = client.get(url='/api/experiments/detail', params={'path': 'study/attempt'}).json()
        assert len(detail['identities']) == 500
        assert any('limited' in warning for warning in detail['warnings'])
        with monkeypatch.context() as scoped:
            scoped.setattr(target=experiments, name='MAX_ROW_GROUP_BYTES', value=1)
            with pytest.raises(expected_exception=ValueError, match='row groups'):
                store.evidence(relative='study/attempt', name='samples.parquet', offset=0, limit=10)
        with monkeypatch.context() as scoped:
            scoped.setattr(target=experiments, name='MAX_TEXT_BYTES', value=1)
            with pytest.raises(expected_exception=ValueError, match='Table page'):
                store.evidence(relative='study/attempt', name='samples.parquet', offset=0, limit=10)
        (directory / 'samples.parquet').write_bytes(data=b'incomplete table')
        broken = client.get(url='/api/experiments/detail', params={'path': 'study/attempt'}).json()
        assert broken['identities'] == []
        assert any('Identity references unavailable' in warning for warning in broken['warnings'])
