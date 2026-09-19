"""Inspect observation order and pixels with a local episode contact sheet."""

from pathlib import Path
from typing import Annotated

import typer
from PIL import Image, ImageDraw, ImageFont

from flystate.cli.common import CONFIG_ERROR, RUNTIME_ERROR, emit
from flystate.datasets.errors import DatasetError
from flystate.datasets.preprocess import prepare_dataset
from flystate.episodes.episode import EpisodeBuilder
from flystate.episodes.trajectory import TrajectoryError
from flystate.experiments.config import ConfigError, load_config
from flystate.log import get_logger
from flystate.settings import get_paths, output_path

app = typer.Typer(no_args_is_help=True, help='Inspect deterministic moving-window episodes.')
CAPTION_HEIGHT: int = 16


@app.command(name='show')
def show_command(
    config: Path,
    sample_id: str,
    out: Annotated[Path, typer.Option('--out', help='PNG destination inside FLYSTATE_HOME.')],
    scale: Annotated[int, typer.Option('--scale', min=1, max=10)] = 3,
    as_json: Annotated[bool, typer.Option('--json')] = False,
) -> None:
    """Render a numbered aligned face and its ordered observation windows.

    :param config: Experiment YAML path.
    :type config: Path
    :param sample_id: Selected image identifier.
    :type sample_id: str
    :param out: PNG output path under the experiment home.
    :type out: Path
    :param scale: Integer display magnification.
    :type scale: int
    :param as_json: Emit one metadata object on stdout.
    :type as_json: bool
    :raises typer.Exit: If configuration, sample selection, trajectory, or export fails.
    """
    try:
        cfg = load_config(path=config)
        paths = get_paths()
        destination = output_path(path=out, paths=paths)
        prepared = prepare_dataset(cfg=cfg, paths=paths)
        row = next(
            (
                index
                for index, sample in enumerate(prepared.samples)
                if sample.sample_id == sample_id
            ),
            None,
        )
        if row is None:
            raise DatasetError(f'Sample {sample_id!r} is not in the configured subset.')
        builder = EpisodeBuilder(episodes=cfg.episodes, image_size=cfg.dataset.preprocess.size)
        episode = builder.build(sample=prepared.samples[row], image=prepared.images[row])
        size, window, steps = cfg.dataset.preprocess.size, cfg.episodes.window, cfg.episodes.steps
        canvas = Image.new(
            mode='RGB',
            size=((size + steps * window) * scale, max(size, window + CAPTION_HEIGHT) * scale),
            color='white',
        )
        face = Image.fromarray(obj=prepared.images[row]).resize(
            size=(size * scale, size * scale), resample=Image.Resampling.NEAREST
        )
        canvas.paste(im=face, box=(0, 0))
        draw = ImageDraw.Draw(im=canvas)
        font = ImageFont.load_default(size=10 * scale)
        for index, box in enumerate(episode.boxes):
            x0, y0, x1, y1 = (int(value) * scale for value in box)
            draw.rectangle(xy=(x0, y0, x1 - 1, y1 - 1), outline='yellow', width=scale)
            draw.text(
                xy=(x0 + scale, y0 + scale),
                text=str(index + 1),
                font=font,
                fill='yellow',
                stroke_width=scale,
                stroke_fill='black',
            )
            crop = Image.fromarray(obj=episode.observations[index]).resize(
                size=(window * scale, window * scale), resample=Image.Resampling.NEAREST
            )
            left = (size + index * window) * scale
            canvas.paste(im=crop, box=(left, 0))
            draw.text(
                xy=(left + scale, (window + 1) * scale),
                text=str(index + 1),
                font=font,
                fill='black',
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(fp=destination, format='PNG')
    except (ConfigError, DatasetError, TrajectoryError, OSError, ValueError) as error:
        get_logger(name='episode').error('episode_failed', detail=str(error))
        if as_json:
            emit(result={'error': str(error)}, as_json=True)
        raise typer.Exit(
            code=CONFIG_ERROR if isinstance(error, ConfigError) else RUNTIME_ERROR
        ) from error
    emit(
        result={
            'sample_id': episode.sample_id,
            'split': episode.split,
            'target': episode.target,
            'boxes': episode.boxes.tolist(),
            'positions': episode.positions.tolist(),
            'output': str(destination),
        },
        as_json=as_json,
    )
