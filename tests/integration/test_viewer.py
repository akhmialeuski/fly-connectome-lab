"""Exercise the results API against a complete, synthetic offline experiment."""

import hashlib
import io
import json
import os
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from playwright.sync_api import Route, expect, sync_playwright
from typer.testing import CliRunner

from flystate.cli.main import app
from flystate.diagnostics.probes import run_probe
from flystate.evaluation.evaluate import evaluate
from flystate.experiments.config import ExperimentConfig, effective_yaml
from flystate.hashing import sha256_file
from flystate.readouts.training import train
from flystate.settings import get_paths
from flystate.storage.json import write_json
from flystate.traces.builder import build_trace
from flystate.traces.store import TraceStore
from flystate.viewer.app import create_app
from flystate.viewer.repository import Repository, document, resource


@pytest.fixture
def viewer_run(tiny_experiment: ExperimentConfig) -> dict[str, Any]:
    """Build, train, and evaluate synthetic data for real artifact access.

    :param tiny_experiment: Installed small synthetic experiment.
    :type tiny_experiment: ExperimentConfig
    :returns: Run and evaluation identifiers with one sample identifier.
    :rtype: dict[str, Any]
    """
    paths = get_paths()
    build_trace(cfg=tiny_experiment, paths=paths)
    trained = train(
        cfg=tiny_experiment, paths=paths, original_yaml=effective_yaml(cfg=tiny_experiment)
    )
    evaluated = evaluate(run_dir=Path(trained['run_dir']), paths=paths, split='test')
    repo = Repository(paths=paths)
    page = repo.predictions(
        run_id=trained['run_id'],
        eval_id=evaluated['eval_id'],
        t=1,
        correct=None,
        sample_id=None,
        offset=0,
        limit=1,
    )
    return {**trained, **evaluated, 'sample_id': page['rows'][0]['sample_id']}


class TestViewer:
    """Validate artifact identity, query behavior, and immutable local access."""

    def test_end_to_end_read_only(self, viewer_run: dict[str, Any]) -> None:
        """Snapshot artifacts, visit every data route, and verify bytes and mtimes unchanged.

        :param viewer_run: Completed synthetic run and evaluation.
        :type viewer_run: dict[str, Any]
        """
        paths = get_paths()
        report = paths.runs / 'reports' / 'summary.md'
        report.parent.mkdir()
        report.write_text('# Results\n<script>unsafe()</script>', encoding='utf-8')
        write_json(path=paths.runs / 'comparisons' / 'pair.json', value={'rows': []})
        before = {
            str(p): (p.stat().st_mtime_ns, sha256_file(path=p))
            for p in paths.home.rglob('*')
            if p.is_file()
        }
        client = TestClient(app=create_app(paths=paths), base_url='http://127.0.0.1')
        base = f'/api/runs/{viewer_run["run_id"]}'
        eval_base = f'{base}/evaluations/{viewer_run["eval_id"]}'
        catalog = client.get(url='/api/catalog').json()
        assert len(catalog['runs']) == len(catalog['caches']) == 1
        assert catalog['warnings'] == []
        detail = client.get(url=base).json()
        assert detail['evaluations'][0]['status'] == 'completed'
        assert len(detail['validation']) == 4
        manifest = detail['manifest']
        store = TraceStore(directory=paths.features / manifest['cache_key'])
        by_sample = {row['sample_id']: row for row in store.index}
        assert len(detail['identities']) == 4
        for identity in detail['identities']:
            reference = by_sample[identity['sample_id']]
            assert reference['split'] == 'train'
            assert reference['label'] == identity['label']
            assert reference['identity'] == identity['identity']
            assert identity['sample_id'] not in {
                row['sample_id'] for row in store.index if row['split'] == 'test'
            }
        assert client.get(url=eval_base).json()['meta']['split'] == 'test'
        page = client.get(url=eval_base + '/predictions', params={'t': 4, 'limit': 2}).json()
        assert page['total'] == 4 and len(page['rows']) == 2
        later = client.get(url=eval_base + '/predictions', params={'t': 4, 'offset': 2}).json()
        assert not {r['sample_id'] for r in page['rows']} & {r['sample_id'] for r in later['rows']}
        filtered = client.get(url=eval_base + '/predictions', params={'correct': True}).json()
        assert all(row['correct'] for row in filtered['rows'])
        sample_id = viewer_run['sample_id']
        one = client.get(url=eval_base + '/predictions', params={'sample_id': sample_id}).json()
        assert one['total'] == 4
        episode = client.get(url=f'{base}/samples/{sample_id}').json()
        assert len(episode['boxes']) == len(episode['activity']['voltage']) == 4
        assert episode['neurons']['available']
        assert len(episode['neurons']['ids']) == len(episode['activity']['voltage'][0])
        image = client.get(url=f'{base}/samples/{sample_id}/image')
        assert image.headers['content-type'] == 'image/png'
        assert Image.open(fp=io.BytesIO(image.content)).size == (128, 128)
        assert (
            client.get(url='/api/reports/reports/summary.md').json()['text'].startswith('# Results')
        )
        assert client.get(url='/api/reports/comparisons/pair.json').json() == {'rows': []}
        assert client.get(url='/').status_code == 200
        for asset in ('app.js', 'charts.js', 'episode.js', 'identities.js', 'app.css'):
            assert client.get(url=f'/{asset}').status_code == 200
        assert client.get(url=base).headers['cache-control'] == 'no-store'
        assert 'frame-ancestors' in client.get(url='/').headers['content-security-policy']
        assert client.post(url=eval_base).status_code == 405
        assert (
            client.get(url='/api/catalog', headers={'Host': 'external.example'}).status_code == 400
        )
        for params in ({'limit': 201}, {'limit': 0}, {'offset': -1}, {'t': 0}):
            assert client.get(url=eval_base + '/predictions', params=params).status_code == 422
        after = {
            str(p): (p.stat().st_mtime_ns, sha256_file(path=p))
            for p in paths.home.rglob('*')
            if p.is_file()
        }
        assert after == before

    def test_partial_and_corrupt_artifacts(self, viewer_run: dict[str, Any]) -> None:
        """Damage discovery entries, preserve healthy entries, and reject inconsistent evaluations.

        :param viewer_run: Completed synthetic run and evaluation.
        :type viewer_run: dict[str, Any]
        """
        paths = get_paths()
        repo = Repository(paths=paths)
        directory = Path(viewer_run['run_dir'])
        write_json(path=paths.runs / 'broken' / 'manifest.json', value={})
        write_json(path=paths.features / 'broken' / 'build.json', value=[])
        write_json(path=paths.runs / 'comparisons' / 'broken.json', value=[])
        write_json(path=directory / 'evals' / 'broken' / 'eval.json', value={})
        write_json(path=paths.runs / 'comparisons' / 'latest.json', value={})
        (paths.runs / 'comparisons' / 'ignored.txt').write_text('ignored')
        assert len(repo.catalog()['warnings']) == 3
        assert len(repo.run(run_id=directory.name)['warnings']) == 1
        for kind, name in (('unknown', 'x.json'), ('reports', 'x.txt')):
            with pytest.raises(ValueError, match='Unsupported'):
                repo.report(kind=kind, name=name)
        eval_dir = Path(viewer_run['eval_dir'])
        metadata = document(path=eval_dir / 'eval.json')
        write_json(path=eval_dir / 'eval.json', value={**metadata, 'status': 'failed'})
        client = TestClient(app=create_app(paths=paths), base_url='http://localhost')
        assert (
            client.get(url=f'/api/runs/{directory.name}/evaluations/{eval_dir.name}').status_code
            == 409
        )
        assert client.get(url='/api/runs/missing').status_code == 404
        (directory / 'summary.json').unlink()
        (directory / 'metrics' / 'validation.parquet').unlink()
        partial_run = repo.run(run_id=directory.name)
        assert partial_run['validation'] == [] and 'summary' not in partial_run
        assert repo.catalog()['runs'][0]['summary'] == {}
        (directory / 'config.yaml').write_text('invalid: true')
        assert len(repo.catalog()['warnings']) == 4

    def test_sample_integrity_and_missing_geometry(self, viewer_run: dict[str, Any]) -> None:
        """Verify missing samples, unavailable geometry, then reject damaged committed traces.

        :param viewer_run: Completed synthetic run and evaluation.
        :type viewer_run: dict[str, Any]
        """
        paths = get_paths()
        repo = Repository(paths=paths)
        run_id, sample_id = viewer_run['run_id'], viewer_run['sample_id']
        with pytest.raises(FileNotFoundError, match='does not belong'):
            repo.sample(run_id=run_id, sample_id='missing')
        paths.brain.joinpath('brain.npz').unlink()
        assert not repo.sample(run_id=run_id, sample_id=sample_id)['neurons']['available']
        manifest = document(path=Path(viewer_run['run_dir']) / 'manifest.json')
        store = TraceStore(directory=paths.features / manifest['cache_key'], mode='a')
        row = next(r['row'] for r in store.index if r['sample_id'] == sample_id)
        chunk = row // store.chunk_size
        store.array(name='done')[chunk] = False
        with pytest.raises(ValueError, match='not completed'):
            repo.sample(run_id=run_id, sample_id=sample_id)
        store.array(name='done')[chunk] = True
        store.array(name='features')[row, 0, 0] = 1000
        with pytest.raises(ValueError, match='integrity'):
            repo.sample(run_id=run_id, sample_id=sample_id)
        write_json(
            path=Path(viewer_run['run_dir']) / 'manifest.json',
            value={**manifest, 'dataset_fingerprint': 'wrong'},
        )
        with pytest.raises(ValueError, match='dataset identity'):
            repo.sample(run_id=run_id, sample_id=sample_id)
        degraded = repo.run(run_id=run_id)
        assert degraded['identities'] == []
        assert 'Identity reference images unavailable' in degraded['warnings'][0]

    def test_images_and_integrity_memo(self, viewer_run: dict[str, Any]) -> None:
        """Check cached hashes, detect edits, and report absent or malformed prepared images.

        :param viewer_run: Completed synthetic run and evaluation.
        :type viewer_run: dict[str, Any]
        """
        paths = get_paths()
        repo = Repository(paths=paths)
        run_id, sample_id = viewer_run['run_id'], viewer_run['sample_id']
        assert repo.image(run_id=run_id, sample_id=sample_id) == repo.image(
            run_id=run_id, sample_id=sample_id
        )
        with pytest.raises(FileNotFoundError, match='Aligned image'):
            repo.image(run_id=run_id, sample_id='missing')
        prepared = next(paths.preprocess.glob('*/meta.json')).parent
        image_file = prepared / 'images.npy'
        with image_file.open(mode='ab') as stream:
            stream.write(b'corrupt')
        with pytest.raises(ValueError, match='integrity'):
            repo.image(run_id=run_id, sample_id=sample_id)
        np.save(file=image_file, arr=np.zeros(shape=(40, 128, 128, 3), dtype=np.float32))
        meta = document(path=prepared / 'meta.json')
        meta['artifacts']['images.npy'] = sha256_file(path=image_file)
        write_json(path=prepared / 'meta.json', value=meta)
        with pytest.raises(ValueError, match='RGB geometry'):
            repo.image(run_id=run_id, sample_id=sample_id)
        meta['fingerprint'] = 'unrelated'
        write_json(path=prepared / 'meta.json', value=meta)
        with pytest.raises(FileNotFoundError, match='Aligned image'):
            repo.image(run_id=run_id, sample_id=sample_id)

    def test_boundaries_and_empty_installation(self, tmp_path: Path) -> None:
        """Reject traversal, escaping symlinks, and non-object JSON without creating storage.

        :param tmp_path: Test-owned filesystem boundary.
        :type tmp_path: Path
        """
        paths = get_paths()
        assert Repository(paths=paths).catalog() == {
            'runs': [],
            'experiments': [],
            'reports': [],
            'caches': [],
            'warnings': [],
        }
        assert not paths.home.exists()
        root = tmp_path / 'root'
        root.mkdir()
        for name in ('..', '.', 'a/b', '\\outside', 'a b'):
            with pytest.raises(ValueError, match='identifier'):
                resource(root, name)
        (root / 'escape').symlink_to(target=tmp_path)
        with pytest.raises(ValueError, match='escapes'):
            resource(root, 'escape')
        with pytest.raises(FileNotFoundError):
            resource(root, 'missing')
        for value in ('[]', '{"value": NaN}', 'bad json'):
            path = root / 'invalid.json'
            path.write_text(value)
            with pytest.raises(ValueError):
                document(path=path)

    def test_geometry_and_trace_boundaries(self, viewer_run: dict[str, Any]) -> None:
        """Retain neurons without positions, reject shape mismatch, and block internal symlinks.

        :param viewer_run: Completed synthetic run and evaluation.
        :type viewer_run: dict[str, Any]
        """
        paths = get_paths()
        repo = Repository(paths=paths)
        run_id, sample_id = viewer_run['run_id'], viewer_run['sample_id']
        manifest_path = Path(viewer_run['run_dir']) / 'manifest.json'
        manifest = document(path=manifest_path)
        brain_path = paths.brain / 'brain.npz'
        with np.load(file=brain_path, allow_pickle=False) as source:
            arrays = {name: source[name] for name in source.files}
        selected = np.flatnonzero(a=arrays['superclass'] == 'descending_neuron')
        arrays['positions'] = arrays['positions'].astype(np.float32)
        arrays['positions'][selected[0]] = np.nan
        np.savez(file=brain_path, **arrays)
        manifest['brain_files_sha256']['brain.npz'] = sha256_file(path=brain_path)
        write_json(path=manifest_path, value=manifest)
        result = repo.sample(run_id=run_id, sample_id=sample_id)
        assert result['neurons']['available']
        assert result['neurons']['missing_positions'] == 1
        assert 0 not in result['neurons']['feature_indices']
        assert len(result['neurons']['ids']) == len(selected)
        arrays['positions'] = arrays['positions'][:, :2]
        np.savez(file=brain_path, **arrays)
        manifest['brain_files_sha256']['brain.npz'] = sha256_file(path=brain_path)
        write_json(path=manifest_path, value=manifest)
        assert (
            'geometry does not match'
            in repo.sample(run_id=run_id, sample_id=sample_id)['neurons']['reason']
        )
        cache = paths.features / manifest['cache_key']
        link = cache / 'escape'
        link.symlink_to(target=paths.brain, target_is_directory=True)
        with pytest.raises(ValueError, match='escapes'):
            repo.sample(run_id=run_id, sample_id=sample_id)
        link.unlink()
        link.symlink_to(target=cache / 'build.json')
        assert repo.sample(run_id=run_id, sample_id=sample_id)['sample']['sample_id'] == sample_id
        store = TraceStore(directory=cache, mode='a')
        row = next(r['row'] for r in store.index if r['sample_id'] == sample_id)
        chunk = row // store.chunk_size
        store.array(name='features')[row, 0, 0] = np.nan
        start, stop = store.bounds(chunk=chunk)
        hashes_path = cache / 'chunks' / f'{chunk}.json'
        hashes = document(path=hashes_path)
        hashes['features'] = hashlib.sha256(
            string=np.asarray(a=store.array(name='features')[start:stop]).tobytes()
        ).hexdigest()
        write_json(path=hashes_path, value=hashes)
        with pytest.raises(ValueError, match='invalid values'):
            repo.sample(run_id=run_id, sample_id=sample_id)

    def test_serve_cli(self) -> None:
        """Invoke help and JSON startup through Typer, checking the loopback server contract."""
        runner = CliRunner()
        assert runner.invoke(app=app, args=['serve', '--help']).exit_code == 0
        assert runner.invoke(app=app, args=['serve', '--port', '1']).exit_code == 2
        with patch(target='flystate.cli.serve.uvicorn.run') as serve:
            result = runner.invoke(app=app, args=['serve', '--port', '9000', '--json'])
        assert result.exit_code == 0, result.output
        assert len(result.stdout.splitlines()) == 1
        assert json.loads(s=result.stdout)['url'] == 'http://127.0.0.1:9000'
        assert serve.call_args.kwargs['host'] == '127.0.0.1'
        assert serve.call_args.kwargs['port'] == 9000


@pytest.mark.skipif(
    os.environ.get('FLYSTATE_BROWSER_TESTS') != '1', reason='Opt-in installed Chromium'
)
class TestViewerBrowser:
    """Navigate the actual bundled application with offline synthetic HTTP responses."""

    def test_browser_workflows(self, viewer_run: dict[str, Any]) -> None:
        """Load real pages, filter predictions, replay an episode, and check responsive layout.

        :param viewer_run: Completed synthetic run and evaluation.
        :type viewer_run: dict[str, Any]
        """
        paths = get_paths()
        write_json(
            path=paths.runs / 'comparisons' / 'pair.json',
            value={
                'run_a': viewer_run['run_id'],
                'run_b': 'control',
                'split': 'test',
                'bootstrap': 100,
                'delta_mem_pp': {'diff_pp': 25, 'ci_low_pp': -10, 'ci_high_pp': 50, 'p': 0.5},
                'rows': [
                    {
                        't': 1,
                        'acc_a': 0.5,
                        'acc_b': 0.25,
                        'diff_pp': 25,
                        'ci_low_pp': -10,
                        'ci_high_pp': 50,
                        'p': 0.5,
                    }
                ],
            },
        )
        write_json(
            path=paths.runs / 'calibrations' / 'probe.json',
            value={
                'amplitudes': [
                    {
                        'amplitude': 0.1,
                        'd': [0, 1],
                        'rate_all_hz': 2,
                        'rate_input_hz': 3,
                        'rate_readout_hz': 1,
                        'latency_first': 1,
                        'latency_half': 2,
                    }
                ],
                'recommendation': {'amplitude': 0.1, 'steps_per_observation': 2},
            },
        )
        write_json(
            path=paths.runs / 'benchmarks' / 'cpu.json',
            value={
                'grid': [
                    {
                        'batch': 1,
                        'threads': 1,
                        'ms_per_episode_step': 2,
                        'cpu_percent': 10,
                        'rss_mb': 100,
                    }
                ],
                'sustained': {
                    'ms_per_episode_step_first_30s': 2,
                    'ms_per_episode_step_last_30s': 3,
                    'throttle_ratio': 1.5,
                },
            },
        )
        write_json(
            path=paths.runs / 'design-checks' / 'pixels.json',
            value={
                'baselines': {'whole_image': {'accuracy': 0.5}},
                'evaluation_split': 'val',
                'n_classes': 4,
                'n_eval': 4,
                'gap_pp': 10,
                'message': 'Memory gap passes.',
                'window_baselines': [{'accuracy': 0.25, 'ci_low': 0.1, 'ci_high': 0.5}],
            },
        )
        report = paths.runs / 'reports' / 'report.md'
        report.parent.mkdir()
        report.write_text('# Evidence\n<script>window.injected=true</script>', encoding='utf-8')
        errors: list[str] = []
        with (
            TestClient(app=create_app(paths=paths), base_url='http://127.0.0.1') as client,
            sync_playwright() as playwright,
        ):
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={'width': 1440, 'height': 1000})
            page.on(event='pageerror', f=lambda error: errors.append(str(error)))
            page.route(url='**/*', handler=partial(fulfill_local, client=client))
            page.goto(url='http://127.0.0.1/')
            expect(actual=page.locator('svg.chart')).to_have_count(count=1)
            page.get_by_label(text='Evaluation split').select_option(value='val')
            expect(
                actual=page.get_by_text(text='no completed evaluation for this split', exact=False)
            ).to_be_visible()
            page.get_by_label(text='Evaluation split').select_option(value='test')
            expect(actual=page.locator('svg.chart')).to_have_count(count=1)
            page.get_by_role(role='link', name='02 Experiments').click()
            page.get_by_label(text='Search experiments').fill(value='no match')
            expect(actual=page.get_by_text(text='No matching experiments.')).to_be_visible()
            page.get_by_label(text='Search experiments').fill(value='tiny')
            page.get_by_role(role='link', name='tiny', exact=True).click()
            expect(actual=page.get_by_text(text='Idea and results', exact=True)).to_be_visible()
            expect(actual=page.get_by_role(role='button', name='View episode')).to_have_count(
                count=4
            )
            page.get_by_label(text='Prediction observation').select_option(value='1')
            page.get_by_role(role='button', name='View episode').first.click()
            expect(actual=page.get_by_text(text='Readout activity', exact=True)).to_be_visible()
            expect(
                actual=page.get_by_text(text='The network sees one highlighted', exact=False)
            ).to_be_visible()
            page.get_by_label(text='Episode observation').fill(value='3')
            expect(
                actual=page.get_by_text(text='After 3 of 4 image windows,', exact=False)
            ).to_be_visible()
            expect(actual=page.locator('.candidate')).to_have_count(count=4)
            expect(actual=page.locator('.candidate img')).to_have_count(count=4)
            expect(actual=page.get_by_text(text="Model's first choice", exact=True)).to_be_visible()
            expected = client.get(
                url=f'/api/runs/{viewer_run["run_id"]}/evaluations/'
                f'{viewer_run["eval_id"]}/predictions',
                params={'t': 3, 'sample_id': viewer_run['sample_id']},
            ).json()['rows'][0]
            assert page.locator('.first-choice').get_attribute(name='data-label') == str(
                expected['y_pred']
            )
            assert (
                page.locator('.first-choice .candidate-probability').inner_text()
                == f'{expected["p_pred"]:.2%}'
            )
            page.locator('.candidate img').first.evaluate(expression='image => image.decode()')
            page.get_by_label(text='Neural feature', exact=True).select_option(value='voltage')
            page.get_by_role(role='button', name='Play', exact=True).click()
            page.get_by_role(role='button', name='Pause', exact=True).click()
            page.get_by_label(text='Close episode').click()
            page.get_by_label(text='Sample ID', exact=True).fill(value=viewer_run['sample_id'])
            page.get_by_role(role='button', name='Find sample').click()
            expect(actual=page.get_by_role(role='button', name='View episode')).to_have_count(
                count=1
            )
            page.get_by_role(role='link', name='04 Comparisons').click()
            expect(
                actual=page.get_by_text(text='Difference across observations', exact=True)
            ).to_be_visible()
            expect(actual=page.get_by_text(text='Final memory effect', exact=True)).to_be_visible()
            assert 'NaN' not in page.locator('#content').inner_text()
            page.get_by_role(role='link', name='05 Evidence library').click()
            expect(actual=page.locator('.report-text')).to_contain_text(expected='<script>')
            assert page.evaluate(expression='window.injected') is None
            for category, title in (
                ('calibrations', 'Signal propagation'),
                ('benchmarks', 'CPU scaling'),
                ('design-checks', 'Pixel controls'),
            ):
                page.get_by_label(text='Report category', exact=True).select_option(value=category)
                expect(actual=page.get_by_text(text=title, exact=True)).to_be_visible()
            page.get_by_role(role='link', name='06 Trace caches').click()
            expect(
                actual=page.get_by_text(text='Stored neural recordings', exact=True)
            ).to_be_visible()
            page.set_viewport_size(viewport_size={'width': 390, 'height': 844})
            assert page.evaluate(expression='document.documentElement.scrollWidth <= innerWidth')
            assert errors == []
            browser.close()

    def test_research_live_discovery(self, tiny_experiment: ExperimentConfig) -> None:
        """Display a nested probe with photos, then discover a new failure without losing filters.

        :param tiny_experiment: Offline photographs and deterministic encoder configuration.
        :type tiny_experiment: ExperimentConfig
        """
        paths = get_paths()
        relative = 'future/study/attempt-a'
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
        client = TestClient(app=create_app(paths=paths), base_url='http://127.0.0.1')
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={'width': 1280, 'height': 900})
            errors: list[str] = []
            page.on(event='pageerror', f=lambda error: errors.append(str(error)))
            page.route(url='**/*', handler=partial(fulfill_local, client=client))
            page.goto(url='http://127.0.0.1/#runs')
            expect(
                actual=page.get_by_role(role='link', name='attempt-a', exact=True)
            ).to_have_count(count=0)
            page.get_by_role(role='link', name='03 Diagnostics').click()
            summaries = page.evaluate(
                expression="""async () => {
                    const {interpretation} = await import('/interpretation.js');
                    return [
                        {kind: 'training', mode: 'reset'},
                        {kind: 'training', mode: 'reset_concat'},
                        {kind: 'identity_probe', parameters: {representation: 'pixels'}},
                        {kind: 'identity_probe', parameters: {representation: 'neural',
                            history: 'last', train_per_class: 2, subset_seed: 0}},
                        {kind: 'identity_probe', parameters: {representation: 'neural',
                            history: 'all', label_mode: 'shuffled'}},
                        {kind: 'convergence_diagnostic', report: {
                            budget_measurements: [{converged: false}, {converged: true}]}},
                        {kind: 'cohort_audit', report: {reserve_count: 0}},
                        {kind: 'future_kind', status: 'running'},
                        {kind: 'new_kind', status: 'completed',
                            parameters: {hypothesis: 'Recorded idea'},
                            result_summary: 'Recorded outcome'},
                        {kind: 'new_kind', status: 'failed',
                            result_summary: 'Must not hide failure'},
                        {kind: 'identity_probe', status: 'completed', classes: 20,
                            parameters: {representation: 'neural', history: 'last',
                                features: 'voltage', pca_components: 60},
                            scores: {validation: {accuracy: 0.1}}},
                        {kind: 'readout_ablation_analysis', status: 'completed', report: {
                            gate: 'do_not_advance', fits: {
                                B0: {scores: {validation: {accuracy: 0.0333333333}}},
                                B2: {scores: {validation: {accuracy: 0.1}}}
                            }}},
                        {kind: 'future_analysis', status: 'completed', gate: 'do_not_advance',
                            case_count: 8}
                    ].map(interpretation);
                }"""
            )
            assert 'Reset neural state' in summaries[0]['idea']
            assert 'combine' in summaries[1]['idea']
            assert 'without simulating neural memory' in summaries[2]['idea']
            assert 'final observation' in summaries[3]['idea']
            assert '2 photographs' in summaries[3]['idea']
            assert 'combined observations' in summaries[4]['idea']
            assert 'shuffled' in summaries[4]['idea']
            assert '1 of 2' in summaries[5]['result']
            assert '0 reserved' in summaries[6]['result']
            assert 'No specific hypothesis' in summaries[7]['idea']
            assert 'No completed recognition result' in summaries[7]['result']
            assert summaries[8] == {'idea': 'Recorded idea', 'result': 'Recorded outcome'}
            assert 'Attempt failed' in summaries[9]['result']
            assert 'voltage block' in summaries[10]['idea']
            assert '60 components' in summaries[10]['idea']
            assert '10.00%' in summaries[10]['result']
            assert 'readout ablation' in summaries[11]['idea']
            assert 'do not advance' in summaries[11]['result']
            assert '10.00% (B2)' in summaries[11]['result']
            assert '8 recorded cases' in summaries[12]['result']
            expect(
                actual=page.get_by_role(role='link', name='attempt-a', exact=True)
            ).to_be_visible()
            page.get_by_label(text='Search experiments').fill(value='attempt')
            page.get_by_label(text='Experiment study').select_option(value='future/study')
            write_json(
                path=paths.runs / 'future/study/attempt-b/manifest.json',
                value={
                    'status': 'failed',
                    'parameters': {'kind': 'future_kind'},
                    'error': '<script>unsafe()</script>',
                    'error_type': 'RecordedFailure',
                },
            )
            expect(
                actual=page.get_by_role(role='link', name='attempt-b', exact=True)
            ).to_be_visible(timeout=12000)
            expect(actual=page.get_by_label(text='Search experiments')).to_have_value(
                value='attempt'
            )
            expect(actual=page.get_by_label(text='Experiment study')).to_have_value(
                value='future/study'
            )
            page.get_by_role(role='link', name='attempt-b', exact=True).click()
            expect(actual=page.get_by_role(role='alert')).to_contain_text(
                expected='RecordedFailure'
            )
            assert page.evaluate(expression='window.unsafe') is None
            page.get_by_role(role='link', name='03 Diagnostics').click()
            page.get_by_role(role='link', name='attempt-a', exact=True).click()
            expect(actual=page.get_by_text(text='Idea and results', exact=True)).to_be_visible()
            expect(
                actual=page.get_by_text(
                    text=(
                        'Test identity information in encoded input currents, '
                        'without simulating neural memory.'
                    ),
                    exact=True,
                )
            ).to_be_visible()
            expect(
                actual=page.get_by_text(text='Recorded recognition scores', exact=True)
            ).to_be_visible()
            expect(actual=page.get_by_text(text="Model's first choice", exact=True)).to_be_visible()
            expected = client.get(
                url='/api/experiments/evidence',
                params={
                    'path': relative,
                    'name': 'validation-predictions.parquet',
                    'limit': 1,
                },
            ).json()['rows'][0]
            assert page.locator('.first-choice').get_attribute(name='data-label') == str(
                expected['y_pred']
            )
            page.locator('.first-choice img').evaluate(expression='image => image.decode()')
            page.get_by_label(text='Experiment evidence').select_option(value='manifest.json')
            expect(actual=page.locator('.report-text')).to_contain_text(expected='identity_probe')
            page.set_viewport_size(viewport_size={'width': 390, 'height': 844})
            assert page.evaluate(expression='document.documentElement.scrollWidth <= innerWidth')
            assert errors == []
            browser.close()


def fulfill_local(route: Route, client: TestClient) -> None:
    """Serve browser requests directly through the API without external network access.

    :param route: Intercepted browser request.
    :type route: Route
    :param client: In-process ASGI test client.
    :type client: TestClient
    """
    assert route.request.url.startswith('http://127.0.0.1/')
    response = client.request(method=route.request.method, url=route.request.url)
    route.fulfill(
        status=response.status_code, headers=dict(response.headers), body=response.content
    )
