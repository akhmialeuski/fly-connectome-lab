"""Explicit exploratory audits and controlled identity-signal probes."""

from pathlib import Path
from typing import Annotated

import typer

from flystate.cli.common import CONFIG_ERROR, INTERRUPTED, RUNTIME_ERROR, emit
from flystate.diagnostics.audit import audit_cohort
from flystate.diagnostics.probes import run_probe
from flystate.experiments.config import ConfigError, load_config
from flystate.log import get_logger
from flystate.settings import get_paths

app = typer.Typer(
    no_args_is_help=True, help='Audit cohorts and run development-only signal probes.'
)


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
    overrides: Annotated[list[str] | None, typer.Option('--set')] = None,
    as_json: Annotated[bool, typer.Option('--json')] = False,
) -> None:
    """Run an audit or one probe in a fresh output directory; zero components disables PCA.

    :param config: Existing effective cohort/brain configuration.
    :type config: Path
    :param output: New directory relative to FLYSTATE_HOME or inside it.
    :type output: Path
    :param phase: Audit or probe.
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
    :param overrides: Optional original experiment overrides.
    :type overrides: Optional[list[str]]
    :param as_json: Emit exactly one JSON result.
    :type as_json: bool
    :raises typer.Exit: If the attempt fails or arguments are invalid.
    """
    try:
        cfg = load_config(path=config, overrides=overrides or ())
        paths = get_paths()
        if phase not in {'audit', 'probe'} or components < 0:
            raise ConfigError('Phase must be audit/probe and components must be nonnegative.')
        result = (
            audit_cohort(cfg=cfg, paths=paths, output=output)
            if phase == 'audit'
            else run_probe(
                cfg=cfg,
                paths=paths,
                output=output,
                representation=representation,
                history=history,
                features=features,
                components=components or None,
                label_mode=label_mode,
            )
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
