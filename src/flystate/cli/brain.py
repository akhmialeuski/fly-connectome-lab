"""Brain acquisition and inspection commands."""

import math
from pathlib import Path
from typing import Annotated, Never
from zipfile import BadZipFile

import flybrain
import typer

from flystate.brain import files
from flystate.brain.benchmark import default_threads, run_benchmark
from flystate.brain.calibrate import calibrate
from flystate.cli.common import CONFIG_ERROR, RUNTIME_ERROR, emit
from flystate.datasets.errors import DatasetError
from flystate.experiments.config import ConfigError, load_config
from flystate.log import get_logger
from flystate.settings import get_paths

app = typer.Typer(no_args_is_help=True, help='Download and inspect connectome files.')


def _report(check_hash: bool, require_published: bool, as_json: bool) -> bool:
    """Print one integrity and metadata report with a meaningful exit status.

    :param check_hash: Compute file digests.
    :type check_hash: bool
    :param require_published: Require production neuron and connection counts.
    :type require_published: bool
    :param as_json: Emit machine-readable output.
    :type as_json: bool
    :returns: Whether all requested integrity checks pass.
    :rtype: bool
    """
    paths = get_paths()
    verification = files.verify_brain_files(brain_dir=paths.brain, check_hash=check_hash)
    valid = all(item['exists'] and item['sha256_ok'] is not False for item in verification.values())
    present = all(item['exists'] for item in verification.values())
    stats = files.brain_stats(brain_dir=paths.brain) if present else {}
    if require_published:
        valid = (
            valid
            and stats.get('neurons') == files.EXPECTED_NEURONS
            and (stats.get('connections') == files.EXPECTED_CONNECTIONS)
        )
    emit(
        result={
            'directory': str(paths.brain),
            'files': verification,
            'flybrain_version': flybrain.__version__,
            'stats': stats,
            'valid': valid,
        },
        as_json=as_json,
    )
    return valid


def _failure(error: Exception, as_json: bool, code: int = RUNTIME_ERROR) -> Never:
    """Report an expected file operation failure consistently.

    :param error: Operational exception.
    :type error: Exception
    :param as_json: Emit JSON in addition to the stderr log.
    :type as_json: bool
    :param code: Command failure exit status.
    :type code: int
    :returns: Never returns normally.
    :rtype: Never
    :raises typer.Exit: Always exits with the runtime error code.
    """
    get_logger(name='brain').error('brain_files_failed', detail=str(error))
    if as_json:
        emit(result={'error': str(error)}, as_json=True)
    raise typer.Exit(code=code) from error


@app.command(name='download')
def download_command(
    force: Annotated[bool, typer.Option('--force', help='Replace existing brain files.')] = False,
    as_json: Annotated[bool, typer.Option('--json')] = False,
) -> None:
    """Download and verify the published MaleCNS brain into FLYSTATE_HOME.

    :param force: Replace existing files.
    :type force: bool
    :param as_json: Emit one JSON report.
    :type as_json: bool
    """
    try:
        files.download_brain(paths=get_paths(), force=force)
        valid = _report(check_hash=True, require_published=True, as_json=as_json)
    except (OSError, ValueError, RuntimeError, KeyError, BadZipFile, EOFError) as error:
        _failure(error=error, as_json=as_json)
    if not valid:
        raise typer.Exit(code=RUNTIME_ERROR)


@app.command(name='info')
def info_command(
    no_hash: Annotated[bool, typer.Option('--no-hash', help='Skip SHA-256 computation.')] = False,
    as_json: Annotated[bool, typer.Option('--json')] = False,
) -> None:
    """Inspect installed brain files without downloading or creating a simulator.

    :param no_hash: Skip digest verification.
    :type no_hash: bool
    :param as_json: Emit one JSON report.
    :type as_json: bool
    """
    try:
        valid = _report(check_hash=not no_hash, require_published=False, as_json=as_json)
    except (OSError, ValueError, RuntimeError, KeyError, BadZipFile, EOFError) as error:
        _failure(error=error, as_json=as_json)
    if not valid:
        raise typer.Exit(code=RUNTIME_ERROR)


def _positive_csv(value: str) -> tuple[int, ...]:
    """Parse an ordered, unique grid of positive integer arguments.

    :param value: Comma-separated positive integers.
    :type value: str
    :returns: Deduplicated values in input order.
    :rtype: tuple[int, ...]
    :raises typer.BadParameter: If a token is absent, noninteger, or nonpositive.
    """
    try:
        values = tuple(dict.fromkeys(int(token.strip()) for token in value.split(',')))
        if min(values) < 1:
            raise ValueError('Values must be positive.')
        return values
    except ValueError as error:
        raise typer.BadParameter('Expected comma-separated positive integers.') from error


@app.command(name='benchmark')
def benchmark_command(
    threads: Annotated[
        str | None, typer.Option('--threads', help='Comma-separated thread grid.')
    ] = None,
    batch: Annotated[str, typer.Option('--batch', help='Comma-separated batch-size grid.')] = '1,4',
    warmup: Annotated[int, typer.Option('--warmup', min=0)] = 50,
    steps: Annotated[int, typer.Option('--steps', min=1)] = 300,
    sustained_seconds: Annotated[float, typer.Option('--sustained-seconds', min=0.001)] = 300,
    input_neurons: Annotated[int, typer.Option('--input-neurons', min=1)] = 3872,
    seed: Annotated[int, typer.Option('--seed', min=0)] = 0,
    brain_dir: Annotated[
        Path | None, typer.Option('--brain-dir', help='Explicit installed brain.')
    ] = None,
    as_json: Annotated[bool, typer.Option('--json')] = False,
) -> None:
    """Measure CPU throughput, sustained performance, and experiment budgets.

    :param threads: Optional comma-separated Numba thread grid.
    :type threads: Optional[str]
    :param batch: Comma-separated batch-size grid.
    :type batch: str
    :param warmup: Untimed input steps per grid cell.
    :type warmup: int
    :param steps: Timed steps per grid cell.
    :type steps: int
    :param sustained_seconds: Duration for testing the winning pair.
    :type sustained_seconds: float
    :param input_neurons: Distinct stimulated visual projection neurons.
    :type input_neurons: int
    :param seed: Input and episode seed.
    :type seed: int
    :param brain_dir: Override the installed connectome directory.
    :type brain_dir: Optional[Path]
    :param as_json: Emit one JSON report on stdout.
    :type as_json: bool
    """
    thread_grid = _positive_csv(value=threads) if threads is not None else default_threads()
    batch_grid = _positive_csv(value=batch)
    paths = get_paths()
    try:
        report = run_benchmark(
            paths=paths,
            brain_dir=brain_dir or paths.brain,
            threads=thread_grid,
            batches=batch_grid,
            warmup=warmup,
            steps=steps,
            sustained_seconds=sustained_seconds,
            input_neurons=input_neurons,
            seed=seed,
        )
    except (OSError, ValueError, RuntimeError, KeyError, BadZipFile, EOFError) as error:
        _failure(error=error, as_json=as_json)
    if as_json:
        emit(result=report, as_json=True)
    else:
        rows = ['threads  batch  ms/step  ms/episode-step  CPU %  mean Hz  RSS MB']
        rows.extend(
            f'{row["threads"]:7d}  {row["batch"]:5d}  {row["ms_per_step"]:7.3f}  '
            f'{row["ms_per_episode_step"]:15.3f}  {row["cpu_percent"]:5.1f}  '
            f'{row["mean_rate_hz"]:7.3f}  {row["rss_mb"]:6.1f}'
            for row in report['grid']
        )
        emit(
            result={
                'grid': '\n' + '\n'.join(rows),
                **{
                    key: report[key]
                    for key in (
                        'best',
                        'sustained',
                        'episode_estimates',
                        'trace_build_estimates',
                        'report_path',
                    )
                },
            },
            as_json=False,
        )


@app.command(name='calibrate')
def calibrate_command(
    config: Path,
    amplitudes: Annotated[str, typer.Option('--amplitudes')] = '0.05,0.1,0.2,0.4,0.8',
    probes: Annotated[int, typer.Option('--probes', min=1)] = 32,
    steps: Annotated[int, typer.Option('--steps', min=1)] = 10,
    spontaneous_steps: Annotated[int, typer.Option('--spontaneous-steps', min=1)] = 50,
    brain_dir: Annotated[Path | None, typer.Option('--brain-dir')] = None,
    as_json: Annotated[bool, typer.Option('--json')] = False,
) -> None:
    """Measure firing rates and matched-noise response latency on validation images.

    :param config: Experiment YAML file.
    :type config: Path
    :param amplitudes: Comma-separated voltage amplitudes in (0,1].
    :type amplitudes: str
    :param probes: Validation images to probe.
    :type probes: int
    :param steps: Measurement length in simulation steps.
    :type steps: int
    :param spontaneous_steps: Unstimulated measurement length in steps.
    :type spontaneous_steps: int
    :param brain_dir: Optional explicit installed brain directory.
    :type brain_dir: Optional[Path]
    :param as_json: Emit one JSON report identical to the saved artifact.
    :type as_json: bool
    """
    paths = get_paths()
    try:
        grid = tuple(float(token.strip()) for token in amplitudes.split(','))
        if any(not math.isfinite(value) or not 0 < value <= 1 for value in grid):
            raise ValueError('Amplitudes must be finite values in (0,1].')
    except ValueError as error:
        _failure(error=error, as_json=as_json, code=CONFIG_ERROR)
    try:
        report = calibrate(
            cfg=load_config(path=config),
            paths=paths,
            brain_dir=brain_dir or paths.brain,
            amplitudes=grid,
            probes=probes,
            steps=steps,
            spontaneous_steps=spontaneous_steps,
        )
    except ConfigError as error:
        _failure(error=error, as_json=as_json, code=CONFIG_ERROR)
    except (
        DatasetError,
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        BadZipFile,
        EOFError,
    ) as error:
        _failure(error=error, as_json=as_json)
    if as_json:
        emit(result=report, as_json=True)
    else:
        typer.echo(message='amplitude  all Hz  input Hz  readout Hz  first step  half-peak step')
        for row in report['amplitudes']:
            typer.echo(
                message=f'{row["amplitude"]:9.3f} {row["rate_all_hz"]:7.3f} '
                f'{row["rate_input_hz"]:9.3f} {row["rate_readout_hz"]:11.3f} '
                f'{row["latency_first"]!s:>11} {row["latency_half"]!s:>15}'
            )
        emit(
            result={key: report[key] for key in ('warnings', 'recommendation', 'output')},
            as_json=False,
        )
