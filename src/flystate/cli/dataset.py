"""Local CelebA registration, validation, inspection, and metadata commands."""

from collections import Counter
from dataclasses import asdict
from pathlib import Path
from statistics import median
from typing import Annotated, Literal, Never

import numpy as np
import typer
from PIL import Image, ImageDraw

from flystate.cli.common import CONFIG_ERROR, RUNTIME_ERROR, emit
from flystate.datasets import registry
from flystate.datasets.align import align_face, map_landmarks, similarity_transform
from flystate.datasets.celeba import OFFICIAL_COUNTS, CelebAAdapter, sample_id_for
from flystate.datasets.errors import DatasetError
from flystate.datasets.preprocess import prepare_dataset
from flystate.evaluation.design_check import design_check
from flystate.experiments.config import ConfigError, load_config
from flystate.log import get_logger
from flystate.settings import get_paths, output_path

app = typer.Typer(no_args_is_help=True, help='Register and inspect local CelebA data.')
LICENSE_LINE: str = 'CelebA: non-commercial research purposes only'


def _name(name: str) -> None:
    """Reject unsupported dataset names as command-line argument errors.

    :param name: Requested dataset name.
    :type name: str
    :raises typer.BadParameter: If the name is not celeba.
    """
    if name != 'celeba':
        raise typer.BadParameter('Only celeba is supported in POC 1.')


def _failure(error: Exception, as_json: bool, code: int = RUNTIME_ERROR) -> Never:
    """Keep operational failures on stderr and preserve the JSON stdout contract.

    :param error: Operational exception.
    :type error: Exception
    :param as_json: Emit one JSON error object when true.
    :type as_json: bool
    :param code: Exit status for the failure category.
    :type code: int
    :returns: Never returns normally.
    :rtype: Never
    :raises typer.Exit: Always exits with the runtime error status.
    """
    get_logger(name='dataset').error('dataset_failed', detail=str(error))
    if as_json:
        emit(result={'error': str(error)}, as_json=True)
    raise typer.Exit(code=code) from error


@app.command(name='register')
def register_command(
    name: str, root: Path, as_json: Annotated[bool, typer.Option('--json')] = False
) -> None:
    """Register the local root containing official CelebA files.

    :param name: Dataset name.
    :type name: str
    :param root: Existing local root directory.
    :type root: Path
    :param as_json: Emit one JSON object.
    :type as_json: bool
    """
    _name(name=name)
    try:
        result = registry.register(paths=get_paths(), name=name, root=root)
    except (DatasetError, OSError) as error:
        _failure(error=error, as_json=as_json)
    emit(result=result, as_json=as_json)


@app.command(name='validate')
def validate_command(
    name: str,
    full: Annotated[bool, typer.Option('--full', help='Check every JPEG filename.')] = False,
    expected_counts: Annotated[str, typer.Option('--expected-counts', hidden=True)] = 'official',
    as_json: Annotated[bool, typer.Option('--json')] = False,
) -> None:
    """Validate annotations, official counts, and optionally image existence.

    :param name: Registered dataset name.
    :type name: str
    :param full: Check all JPEG directory entries.
    :type full: bool
    :param expected_counts: Official cardinality, or none for synthetic tests.
    :type expected_counts: str
    :param as_json: Emit one JSON object.
    :type as_json: bool
    :raises typer.BadParameter: If the count policy is unknown.
    :raises typer.Exit: If validation reports a problem.
    """
    _name(name=name)
    if expected_counts not in {'official', 'none'}:
        raise typer.BadParameter('Expected counts must be official or none.')
    paths = get_paths()
    try:
        record = registry.get(paths=paths, name=name)
        adapter = CelebAAdapter(
            root=Path(record['path']),
            expected=None if expected_counts == 'none' else OFFICIAL_COUNTS,
        )
        report = adapter.validate(full=full)
        registry.set_validation(paths=paths, name=name, report=report, full=full)
    except (DatasetError, OSError, KeyError, TypeError) as error:
        _failure(error=error, as_json=as_json)
    emit(result=report, as_json=as_json)
    if not report['ok']:
        raise typer.Exit(code=RUNTIME_ERROR)


@app.command(name='list')
def list_command(as_json: Annotated[bool, typer.Option('--json')] = False) -> None:
    """List registered local datasets and their recorded validation status.

    :param as_json: Emit one JSON object.
    :type as_json: bool
    """
    try:
        result = registry.entries(paths=get_paths())
    except DatasetError as error:
        _failure(error=error, as_json=as_json)
    emit(result={'datasets': result}, as_json=as_json)


@app.command(name='info')
def info_command(name: str, as_json: Annotated[bool, typer.Option('--json')] = False) -> None:
    """Describe identity coverage, partitions, local registration, and license.

    :param name: Registered dataset name.
    :type name: str
    :param as_json: Emit one JSON object.
    :type as_json: bool
    """
    _name(name=name)
    try:
        entry = registry.get(paths=get_paths(), name=name)
        records = CelebAAdapter(root=Path(entry['path']), expected=None).records()
        counts = Counter(record.identity for record in records)
        partitions = Counter(record.partition for record in records)
        result = {
            'images': len(records),
            'identities': len(counts),
            'images_per_identity': {
                'min': min(counts.values()),
                'median': median(counts.values()),
                'max': max(counts.values()),
                'at_least_20': sum(count >= 20 for count in counts.values()),
            },
            'partitions': {str(key): partitions[key] for key in (0, 1, 2)},
            'license': LICENSE_LINE,
            'registration': entry,
        }
    except (DatasetError, OSError, ValueError, KeyError, TypeError) as error:
        _failure(error=error, as_json=as_json)
    emit(result=result, as_json=as_json)


@app.command(name='inspect')
def inspect_command(
    name: str,
    sample: str,
    config: Annotated[Path | None, typer.Option('--config')] = None,
    out: Annotated[
        Path | None, typer.Option('--out', help='PNG path inside FLYSTATE_HOME.')
    ] = None,
    as_json: Annotated[bool, typer.Option('--json')] = False,
) -> None:
    """Inspect one filename or sample identifier and optionally draw its landmarks.

    :param name: Registered dataset name.
    :type name: str
    :param sample: Six-digit JPEG filename or celeba-NNNNNN identifier.
    :type sample: str
    :param config: Optional experiment configuration for aligned inspection.
    :type config: Optional[Path]
    :param out: Optional PNG export under the experiment home.
    :type out: Optional[Path]
    :param as_json: Emit one JSON object.
    :type as_json: bool
    """
    _name(name=name)
    paths = get_paths()
    try:
        entry = registry.get(paths=paths, name=name)
        adapter = CelebAAdapter(root=Path(entry['path']), expected=None)
        filename = (
            f'{sample.removeprefix("celeba-")}.jpg' if sample.startswith('celeba-') else sample
        )
        identifier = sample_id_for(filename=filename)
        record = next((item for item in adapter.records() if item.filename == filename), None)
        if record is None:
            raise DatasetError(f'Unknown CelebA sample: {sample}.')
        result: dict[str, object] = {**asdict(obj=record), 'sample_id': identifier}
        prepared = None
        if config is not None:
            prepared = prepare_dataset(cfg=load_config(path=config), paths=paths)
            result['preprocess_key'] = prepared.key
        if out is not None:
            destination = output_path(path=out, paths=paths)
            destination.parent.mkdir(parents=True, exist_ok=True)
            pixels = adapter.load_image(filename=filename)
            points = np.asarray(a=record.landmarks, dtype=np.float64).reshape(5, 2)
            if prepared is not None:
                transform = similarity_transform(src=points, dst=prepared.template)
                pixels = align_face(
                    image=pixels,
                    landmarks=points,
                    template=prepared.template,
                    size=prepared.images.shape[1],
                )
                points = map_landmarks(landmarks=points, transform=transform)
            image = Image.fromarray(obj=pixels)
            draw = ImageDraw.Draw(im=image)
            if prepared is not None:
                for x, y in prepared.template:
                    draw.ellipse(xy=(x - 3, y - 3, x + 3, y + 3), outline='lime')
            for x, y in points:
                draw.ellipse(xy=(x - 2, y - 2, x + 2, y + 2), fill='red')
            image.save(fp=destination, format='PNG')
            result['output'] = str(destination)
    except ConfigError as error:
        _failure(error=error, as_json=as_json, code=CONFIG_ERROR)
    except (DatasetError, OSError, ValueError, KeyError, TypeError) as error:
        _failure(error=error, as_json=as_json)
    emit(result=result, as_json=as_json)


@app.command(name='prepare')
def prepare_command(config: Path, as_json: Annotated[bool, typer.Option('--json')] = False) -> None:
    """Select, split, and align images into a verified reusable preprocessing cache.

    :param config: Experiment YAML path.
    :type config: Path
    :param as_json: Emit one JSON object.
    :type as_json: bool
    """
    try:
        prepared = prepare_dataset(cfg=load_config(path=config), paths=get_paths())
    except ConfigError as error:
        _failure(error=error, as_json=as_json, code=CONFIG_ERROR)
    except (DatasetError, OSError, ValueError) as error:
        _failure(error=error, as_json=as_json)
    emit(
        result={
            'key': prepared.key,
            'directory': str(prepared.directory),
            'counts': dict(Counter(sample.split for sample in prepared.samples)),
            'fingerprint': prepared.fingerprint,
        },
        as_json=as_json,
    )


@app.command(name='design-check')
def design_check_command(
    config: Path,
    split: Annotated[Literal['val', 'test'], typer.Option('--split')] = 'val',
    as_json: Annotated[bool, typer.Option('--json')] = False,
) -> None:
    """Measure whether all windows identify faces better than the last window.

    :param config: Experiment YAML path.
    :type config: Path
    :param split: Validation by default; test only for a frozen protocol.
    :type split: Literal['val', 'test']
    :param as_json: Emit exactly the saved JSON report.
    :type as_json: bool
    """
    try:
        result = design_check(cfg=load_config(path=config), paths=get_paths(), split=split)
    except ConfigError as error:
        _failure(error=error, as_json=as_json, code=CONFIG_ERROR)
    except (DatasetError, OSError, ValueError, Warning) as error:
        _failure(error=error, as_json=as_json)
    if as_json:
        emit(result=result, as_json=True)
    else:
        typer.echo(message='baseline             accuracy       95% interval       n_eval')
        for name, baseline in result['baselines'].items():
            typer.echo(
                message=f'{name:21} {baseline["accuracy"]:8.2%} '
                f'[{baseline["ci_low"]:.2%}, {baseline["ci_high"]:.2%}] '
                f'{result["n_eval"]:8}'
            )
        typer.echo(
            message=f'Gap: {result["gap_pp"]:.2f} pp; {result["status"]}: {result["message"]}'
        )
        typer.echo(message=f'Report: {result["output"]}')
