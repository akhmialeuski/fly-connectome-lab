"""Explicit exploratory audits and controlled identity-signal probes."""

from pathlib import Path
from typing import Annotated

import typer

from flystate.cli.common import CONFIG_ERROR, INTERRUPTED, RUNTIME_ERROR, emit
from flystate.cli.drive_sweep import drive_decode_command, drive_record_command
from flystate.cli.identity import identity_command
from flystate.cli.identity_analysis import identity_analysis_command
from flystate.cli.input_access import input_access_command
from flystate.cli.input_loss_selection import input_loss_selection_command
from flystate.cli.matched_neural import matched_neural_command
from flystate.cli.noise import noise_trace_command
from flystate.cli.noise_analysis import noise_analysis_command
from flystate.cli.stability import stability_command
from flystate.cli.temporal import temporal_command
from flystate.cli.temporal_analysis import temporal_analysis_command
from flystate.diagnostics.audit import audit_cohort
from flystate.diagnostics.convergence import diagnose_convergence
from flystate.diagnostics.probes import run_probe
from flystate.experiments.config import ConfigError, load_config
from flystate.log import get_logger
from flystate.readouts.fitting import MAX_LOGISTIC_ITERATIONS
from flystate.settings import get_paths

app = typer.Typer(
    no_args_is_help=True, help='Audit cohorts and run development-only signal probes.'
)

app.command(name='stability')(stability_command)
app.command(
    name='input-access',
    help='Fit both frozen training-only pixel/current identity-access probes.',
)(input_access_command)
app.command(
    name='input-loss-selection',
    help='Select input-control regularization by fit-only out-of-fold log loss.',
)(input_loss_selection_command)
app.command(
    name='matched-neural',
    help='Compare original fly descending-state OOF identity scores with encoded input.',
)(matched_neural_command)
app.command(
    name='drive-record',
    help='Record T32 population spike counts under a scaled encoder drive.',
)(drive_record_command)
app.command(
    name='drive-decode',
    help='Decode identity from T32 recordings with the T30 fit-only OOF readout.',
)(drive_decode_command)
app.command(name='noise-trace')(noise_trace_command)
app.command(name='noise-analyze')(noise_analysis_command)
app.command(
    name='identity',
    help='Record one frozen training-only identity response condition.',
)(identity_command)
app.command(
    name='identity-analyze',
    help='Verify three identity conditions and compute the frozen training-only gate.',
)(identity_analysis_command)
app.command(
    name='temporal',
    help='Record one frozen stimulus, blank, or noise-control response in a fresh attempt.',
)(temporal_command)
app.command(
    name='temporal-analyze',
    help='Verify and compare all frozen stimulus and corrected blank attempts.',
)(temporal_analysis_command)


@app.command(name='run')
def diagnose_command(
    config: Path,
    output: Annotated[Path, typer.Option('--output')],
    phase: Annotated[str, typer.Option('--phase')] = 'probe',
    representation: Annotated[str, typer.Option('--representation')] = 'neural',
    history: Annotated[str, typer.Option('--history')] = 'last',
    features: Annotated[str, typer.Option('--features')] = 'both',
    components: Annotated[int, typer.Option('--components')] = 60,
    label_mode: Annotated[str, typer.Option('--label-mode')] = 'true',
    max_iterations: Annotated[int, typer.Option('--max-iterations')] = MAX_LOGISTIC_ITERATIONS,
    train_per_class: Annotated[int | None, typer.Option('--train-per-class')] = None,
    subset_seed: Annotated[int, typer.Option('--subset-seed')] = 0,
    trace_source: Annotated[Path | None, typer.Option('--trace-source')] = None,
    trace_precision: Annotated[str | None, typer.Option('--trace-precision')] = None,
    overrides: Annotated[list[str] | None, typer.Option('--set')] = None,
    as_json: Annotated[bool, typer.Option('--json')] = False,
) -> None:
    """Run an audit or one probe in a fresh output directory; zero components disables PCA.

    :param config: Existing effective cohort/brain configuration.
    :type config: Path
    :param output: New directory relative to FLYSTATE_HOME or inside it.
    :type output: Path
    :param phase: Audit, probe, or training-only convergence diagnosis.
    :type phase: str
    :param representation: Pixels, encoded input, or neural features.
    :type representation: str
    :param history: Last observation or all observations.
    :type history: str
    :param features: Both, spike_trace, or voltage.
    :type features: str
    :param components: PCA component cap; zero retains scaled original features.
    :type components: int
    :param label_mode: True, permuted, or memorization labels.
    :type label_mode: str
    :param max_iterations: Probe iteration cap; the original protocol uses 5000.
    :type max_iterations: int
    :param train_per_class: Optional nested balanced training subset; probe phase only.
    :type train_per_class: Optional[int]
    :param subset_seed: Independent nonnegative subset draw seed.
    :type subset_seed: int
    :param trace_source: Optional completed native trace to fit.
    :type trace_source: Optional[Path]
    :param trace_precision: Float32 or float16 treatment of the native trace.
    :type trace_precision: Optional[str]
    :param overrides: Optional original experiment overrides.
    :type overrides: Optional[list[str]]
    :param as_json: Emit exactly one JSON result.
    :type as_json: bool
    :raises typer.Exit: If the attempt fails or arguments are invalid.
    """
    try:
        cfg = load_config(path=config, overrides=overrides or ())
        paths = get_paths()
        if (
            phase not in {'audit', 'probe', 'convergence'}
            or components < 0
            or max_iterations < 1
            or subset_seed < 0
            or (train_per_class is None and subset_seed != 0)
            or (
                train_per_class is not None
                and (train_per_class < 2 or phase != 'probe' or label_mode != 'true')
            )
            or (phase == 'convergence' and label_mode != 'true')
            or (trace_source is None) != (trace_precision is None)
            or (trace_source is not None and (phase != 'probe' or representation != 'neural'))
            or (trace_precision is not None and trace_precision not in {'float32', 'float16'})
        ):
            raise ConfigError(
                'Use audit/probe/convergence, nonnegative components, positive iterations, '
                'true labels for convergence diagnosis, '
                'and training subsets only for true-label probes.'
            )
        if phase == 'audit':
            result = audit_cohort(cfg=cfg, paths=paths, output=output)
        elif phase == 'convergence':
            result = diagnose_convergence(
                cfg=cfg,
                paths=paths,
                output=output,
                representation=representation,
                history=history,
                features=features,
                components=components or None,
            )
        else:
            result = run_probe(
                cfg=cfg,
                paths=paths,
                output=output,
                representation=representation,
                history=history,
                features=features,
                components=components or None,
                label_mode=label_mode,
                max_iterations=max_iterations,
                train_per_class=train_per_class,
                subset_seed=subset_seed,
                trace_source=trace_source,
                trace_precision=trace_precision,
            )
    except (Exception, KeyboardInterrupt) as error:
        code = (
            CONFIG_ERROR
            if isinstance(error, ConfigError)
            else INTERRUPTED
            if isinstance(error, KeyboardInterrupt)
            else RUNTIME_ERROR
        )
        get_logger(name='diagnose').error('diagnostic_failed', detail=str(error))
        if as_json:
            emit(result={'error': str(error)}, as_json=True)
        raise typer.Exit(code=code) from error
    emit(result=result, as_json=as_json)
