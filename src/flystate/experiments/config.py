"""Immutable experiment configuration, YAML overrides, and content identities."""

import math
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    ValidationError,
    model_serializer,
    model_validator,
)

from flystate.hashing import sha256_obj

NonnegativeInt = Annotated[int, Field(ge=0, strict=True)]
PositiveInt = Annotated[int, Field(ge=1, strict=True)]
PositiveFloat = Annotated[float, Field(gt=0)]
NonnegativeFloat = Annotated[float, Field(ge=0)]
FeatureName = Literal['spike_trace', 'voltage']
UnitFraction = Annotated[float, Field(gt=0, le=1)]


class ConfigError(Exception):
    """An experiment document or override cannot be validated."""


class FrozenConfig(BaseModel):
    """Reject unknown fields and nonfinite values in immutable configurations."""

    model_config = ConfigDict(
        extra='forbid', frozen=True, allow_inf_nan=False, validate_default=True
    )


class SubsetConfig(FrozenConfig):
    """Balanced classification identities and disjoint calibration identities."""

    n_identities: Annotated[int, Field(ge=2, strict=True)] = 20
    images_per_identity: Annotated[int, Field(ge=5, strict=True)] = 20
    calibration_identities: NonnegativeInt = 0
    selection_seed: NonnegativeInt = 0


class SplitConfig(FrozenConfig):
    """Train, validation, and test fractions applied separately to each identity."""

    kind: Literal['per_identity'] = 'per_identity'
    fractions: tuple[PositiveFloat, PositiveFloat, PositiveFloat] = (0.70, 0.15, 0.15)

    @model_validator(mode='after')
    def validate_sum(self) -> Self:
        """Require fractions to sum to one within absolute tolerance.

        :returns: Validated split specification.
        :rtype: Self
        :raises ValueError: If the fractions do not sum to one.
        """
        if not math.isclose(sum(self.fractions), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError('Split fractions must sum to one.')
        return self


class PreprocessConfig(FrozenConfig):
    """Five-landmark similarity alignment in output pixel coordinates."""

    align: Literal['landmarks5'] = 'landmarks5'
    size: Annotated[int, Field(ge=32, le=512, strict=True)] = 128
    eye_center: tuple[NonnegativeFloat, NonnegativeFloat] = (64.0, 52.0)
    interocular: PositiveFloat = 40.0
    color: Literal['rgb'] = 'rgb'

    @model_validator(mode='after')
    def validate_geometry(self) -> Self:
        """Keep the requested eye centers inside the aligned image.

        :returns: Validated alignment geometry.
        :rtype: Self
        :raises ValueError: If either eye falls outside the output image.
        """
        x, y = self.eye_center
        if x - self.interocular / 2 < 0 or x + self.interocular / 2 >= self.size or y >= self.size:
            raise ValueError('Both target eye centers must be inside the aligned image.')
        return self


class DatasetConfig(FrozenConfig):
    """Dataset selection, identity split, and alignment specification."""

    name: Literal['celeba'] = 'celeba'
    subset: SubsetConfig = Field(default_factory=SubsetConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)


class TrajectoryConfig(FrozenConfig):
    """Seeded observation placement and maximum shared window area."""

    strategy: Literal['random', 'raster', 'permuted'] = 'random'
    max_overlap: Annotated[float, Field(ge=0, lt=1)] = 0.25
    seed: NonnegativeInt = 0


class EpisodesConfig(FrozenConfig):
    """Classification windows measured in pixels and observations."""

    task: Literal['classification'] = 'classification'
    window: PositiveInt = 32
    steps: Annotated[int, Field(ge=1, le=16, strict=True)] = 8
    trajectory: TrajectoryConfig = Field(default_factory=TrajectoryConfig)

    @model_validator(mode='after')
    def validate_grid(self) -> Self:
        """Require a square grid with at least two rows for grid trajectories.

        :returns: Validated episode specification.
        :rtype: Self
        :raises ValueError: If the number of observations cannot form the grid.
        """
        if self.trajectory.strategy != 'random' and (
            self.steps < 4 or math.isqrt(self.steps) ** 2 != self.steps
        ):
            raise ValueError('Raster and permuted trajectories require steps = g*g with g >= 2.')
        return self


class PositionConfig(FrozenConfig):
    """Gaussian position channels for horizontal and vertical window coordinates."""

    enabled: bool = True
    neurons_per_axis: Annotated[int, Field(ge=2, le=64, strict=True)] = 16


class EncoderConfig(FrozenConfig):
    """Seeded sparse RGB projection into a named neuronal population."""

    kind: Literal['sparse_projection'] = 'sparse_projection'
    target_population: Annotated[str, Field(min_length=1)] = 'visual_projection'
    neurons_per_channel: PositiveInt = 1280
    nonzeros_per_row: Annotated[int, Field(ge=1, le=64, strict=True)] = 8
    position: PositionConfig = Field(default_factory=PositionConfig)
    amplitude: Annotated[float, Field(gt=0, le=1)] = 0.1
    seed: NonnegativeInt = 0


class NoiseConfig(FrozenConfig):
    """Independent per-neuron noise, in hertz and voltage units."""

    enabled: bool = True
    rate_hz: NonnegativeFloat = 1.2
    amplitude: NonnegativeFloat = 0.22


class RateConfig(FrozenConfig):
    """Graded leaky-tanh dynamics on flybrain's effective graph; all values are dimensionless.

    The update is ``x <- (1 - leak) x + leak tanh(gain W x + input_scale * encoded_current)``, with
    ``driven_leak`` replacing ``leak`` for the encoder-driven neurons when it is given.
    """

    gain: PositiveFloat = 1.0
    leak: UnitFraction = 0.02
    driven_leak: UnitFraction | None = 1.0
    input_scale: PositiveFloat = 20.0


class BrainConfig(FrozenConfig):
    """CPU simulation timing, warmup, noise, and execution resources."""

    backend: Literal['flybrain', 'rate'] = 'flybrain'
    rate: RateConfig | None = None
    dt_s: PositiveFloat = 0.02
    sensory_input: Literal[False] = False
    warmup_steps: NonnegativeInt = 25
    steps_per_observation: Annotated[int, Field(ge=1, le=50, strict=True)] = 5
    noise: NoiseConfig = Field(default_factory=NoiseConfig)
    threads: PositiveInt | None = None
    batch_size: PositiveInt = 4

    @model_validator(mode='after')
    def validate_noise(self) -> Self:
        """Require a valid Bernoulli probability for one simulation step.

        :returns: Validated simulation configuration.
        :rtype: Self
        :raises ValueError: If the noise probability exceeds one.
        """
        if self.noise.rate_hz * self.dt_s > 1:
            raise ValueError('brain.noise.rate_hz * brain.dt_s must not exceed one.')
        if (self.backend == 'rate') != (self.rate is not None):
            raise ValueError('brain.rate must be given exactly when brain.backend is rate.')
        if self.backend == 'rate' and (self.noise.enabled or self.warmup_steps):
            raise ValueError('The rate backend is noise-free and starts at rest without warmup.')
        return self

    @model_serializer(mode='wrap')
    def omit_absent_rate(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Leave ``rate`` out of spiking configurations so their hashes and cache keys stay put.

        :param handler: Pydantic's default serializer for this model.
        :type handler: SerializerFunctionWrapHandler
        :returns: Serialized fields, without ``rate`` when it is not set.
        :rtype: dict[str, Any]
        """
        data = handler(self)
        if self.rate is None:
            data.pop('rate', None)
        return data


class MemoryConfig(FrozenConfig):
    """State retention or concatenated reset controls."""

    mode: Literal['persistent', 'reset', 'reset_concat'] = 'persistent'


class ReadoutConfig(FrozenConfig):
    """Readout features, trace time constant, and training-only model selection."""

    population: Annotated[str, Field(min_length=1)] = 'descending_neuron'
    features: Annotated[tuple[FeatureName, ...], Field(min_length=1)] = ('spike_trace', 'voltage')
    trace_tau_s: PositiveFloat = 0.1
    pca_components: PositiveInt = 60
    c_grid: Annotated[tuple[PositiveFloat, ...], Field(min_length=1)] = (0.01, 0.1, 1.0, 10.0)
    cv_folds: Annotated[int, Field(ge=2, strict=True)] = 5

    @model_validator(mode='after')
    def validate_features(self) -> Self:
        """Reject repeated feature blocks.

        :returns: Validated readout specification.
        :rtype: Self
        :raises ValueError: If the feature list contains duplicates.
        """
        if len(set(self.features)) != len(self.features):
            raise ValueError('Readout features must be unique.')
        return self


class EvaluationConfig(FrozenConfig):
    """Paired bootstrap sample count and independent resampling seed."""

    bootstrap_samples: Annotated[int, Field(ge=100, strict=True)] = 10000
    bootstrap_seed: NonnegativeInt = 0


class DesignCheckConfig(FrozenConfig):
    """Minimum validation memory gap, in percentage points."""

    min_memory_gap_pp: NonnegativeFloat = 10.0


class ExperimentConfig(FrozenConfig):
    """Complete, deeply immutable POC 1 experiment specification."""

    name: Annotated[str, Field(pattern=r'^[a-z0-9][a-z0-9-]{0,40}$')]
    seed: NonnegativeInt = 0
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    episodes: EpisodesConfig = Field(default_factory=EpisodesConfig)
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    brain: BrainConfig = Field(default_factory=BrainConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    readout: ReadoutConfig = Field(default_factory=ReadoutConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    design_check: DesignCheckConfig = Field(default_factory=DesignCheckConfig)

    @model_validator(mode='after')
    def validate_feasibility(self) -> Self:
        """Check window geometry, sparse projection, splits, and CV feasibility.

        :returns: Validated experiment specification.
        :rtype: Self
        :raises ValueError: If coupled settings cannot support the experiment.
        """
        if self.episodes.window > self.dataset.preprocess.size:
            raise ValueError('episodes.window must not exceed dataset.preprocess.size.')
        if self.encoder.nonzeros_per_row > self.episodes.window**2:
            raise ValueError('encoder.nonzeros_per_row must not exceed window pixel count.')
        count = self.dataset.subset.images_per_identity
        train = round(self.dataset.split.fractions[0] * count)
        val = round(self.dataset.split.fractions[1] * count)
        if min(train, val, count - train - val) < 1:
            raise ValueError('Every identity must have at least one train, val, and test image.')
        if train < self.readout.cv_folds:
            raise ValueError('Training images per identity must be at least readout.cv_folds.')
        if self.brain.backend == 'rate' and self.readout.features != ('voltage',):
            raise ValueError('The rate backend records only its graded state as voltage.')
        return self


def _merge_defaults(defaults: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    """Merge supplied values while retaining unknown fields for final validation.

    :param defaults: Complete default field mapping.
    :type defaults: dict[str, Any]
    :param data: Supplied partial mapping.
    :type data: dict[str, Any]
    :returns: Independent recursively merged mapping.
    :rtype: dict[str, Any]
    """
    result = deepcopy(x=defaults)
    for key, value in data.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_defaults(defaults=result[key], data=value)
        else:
            result[key] = deepcopy(x=value)
    return result


def apply_overrides(data: dict[str, object], overrides: Sequence[str]) -> dict[str, object]:
    """Apply YAML values to known dotted schema paths without mutating the input.

    :param data: Partial unvalidated experiment document.
    :type data: dict[str, object]
    :param overrides: Ordered dotted.path=value assignments.
    :type overrides: Sequence[str]
    :returns: Mapping including defaulted fields and supplied overrides.
    :rtype: dict[str, object]
    :raises ConfigError: If an override has invalid syntax or an unknown path.
    """
    schema = ExperimentConfig(name='defaults').model_dump(mode='json')
    defaults = {key: value for key, value in schema.items() if key != 'name'}
    result = _merge_defaults(defaults=defaults, data=data)
    for override in overrides:
        dotted, separator, raw = override.partition('=')
        if not separator or not dotted or not raw.strip():
            raise ConfigError(f'Expected dotted.path=value, received {override!r}.')
        parts = dotted.split('.')
        known: Any = schema
        target: Any = result
        for index, part in enumerate(parts):
            if not isinstance(known, dict) or part not in known:
                raise ConfigError(f'Unknown override path: {dotted}.')
            if not isinstance(target, dict):
                raise ConfigError(f'Override parent is not a mapping: {dotted}.')
            if index == len(parts) - 1:
                try:
                    target[part] = yaml.safe_load(stream=raw)
                except yaml.YAMLError as error:
                    raise ConfigError(f'Invalid YAML override {dotted}: {error}') from error
            else:
                known = known[part]
                target = target.get(part)
    return result


def load_config(path: Path, overrides: Sequence[str] = ()) -> ExperimentConfig:
    """Load, override, and validate one YAML experiment document.

    :param path: UTF-8 YAML input path.
    :type path: Path
    :param overrides: Ordered schema field assignments.
    :type overrides: Sequence[str]
    :returns: Immutable validated experiment.
    :rtype: ExperimentConfig
    :raises ConfigError: If reading, parsing, overriding, or validation fails.
    """
    try:
        data = yaml.safe_load(stream=path.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ConfigError('Experiment YAML must contain a mapping.')
        return ExperimentConfig.model_validate(obj=apply_overrides(data=data, overrides=overrides))
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as error:
        raise ConfigError(f'Cannot load experiment {path}: {error}') from error


def effective_yaml(cfg: ExperimentConfig) -> str:
    """Serialize all effective values in schema order.

    :param cfg: Validated experiment.
    :type cfg: ExperimentConfig
    :returns: Complete safe YAML document.
    :rtype: str
    """
    return yaml.safe_dump(data=cfg.model_dump(mode='json'), sort_keys=False)


def config_hash(cfg: ExperimentConfig) -> str:
    """Hash the canonical effective configuration.

    :param cfg: Validated experiment.
    :type cfg: ExperimentConfig
    :returns: SHA-256 hexadecimal digest.
    :rtype: str
    """
    return sha256_obj(obj=cfg.model_dump(mode='json'))


def section_hash(cfg: ExperimentConfig, section: str) -> str:
    """Hash exactly one known top-level configuration field.

    :param cfg: Validated experiment.
    :type cfg: ExperimentConfig
    :param section: Top-level field name.
    :type section: str
    :returns: SHA-256 hexadecimal digest.
    :rtype: str
    :raises ConfigError: If the requested field does not exist.
    """
    data = cfg.model_dump(mode='json')
    if section not in data:
        raise ConfigError(f'Unknown configuration section: {section}.')
    return sha256_obj(obj=data[section])
