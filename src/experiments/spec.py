"""A stage's experiment spec: which runs each experiment compares, resolved into parse configs.

Each experiment changes parser options on its control, set by its ``control`` key:
- ``base``: the spec's ``base_config``, whatever winners are chosen; parsing never waits on gold or review.
- an experiment id, e.g. ``E2``: that experiment's winner; until it is chosen the experiment waits (``NoWinnerYet``).
- unset: the *current best*, the winner of the latest earlier experiment that names one, else ``base_config``.
A variant's run directory carries a hash of its parse settings, so changing a winner a control depends on gives new
directories instead of clashing with old runs.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from pipeline.config import RunConfig

Role = Literal['reference', 'control', 'variant']


class NoWinnerYet(LookupError):
    """The experiment builds on another whose winner is not chosen yet."""


@dataclass(frozen=True, slots=True)
class MetricInfo:
    """A ``pages.csv`` column and how to read it."""

    column: str
    label: str
    target: str = ''  # A text quality, B layout, C serializer
    better: Literal['lower', 'higher'] = 'lower'
    gold: bool = False
    description: str = ''


@dataclass(frozen=True, slots=True)
class Variant:
    name: str
    options: Mapping[str, Any] = field(default_factory=dict)  # parser options set on the current best
    source: str | None = None  # 'E2.egret_large': reuse that experiment's run instead of parsing again


@dataclass(frozen=True, slots=True)
class Experiment:
    id: str
    title: str
    question: str
    pipeline_stage: str
    target: str
    metrics: tuple[str, ...]
    variants: tuple[Variant, ...]
    split: str | None = None  # None: the spec's split
    winner: str | None = None  # variant name (or 'control') chosen after review
    control: str | None = None  # 'base', an earlier experiment's id, or None for the latest earlier winner
    notes: str = ''


@dataclass(frozen=True, slots=True)
class Arm:
    """One row of an experiment's table: a parse run and its role."""

    role: Role
    name: str
    raw: Mapping[str, Any]  # the run config as YAML would hold it

    @property
    def config(self) -> RunConfig:
        return RunConfig.from_dict(self.raw)

    @property
    def run_dir(self) -> Path:
        return Path(self.raw['output'])

    @property
    def options(self) -> dict[str, Any]:
        """Parser options as set in the config, flattened one level: ``{'overrides.force_backend_text': True}``."""
        return flatten(self.raw['parser'].get('options') or {})

    def resolved_options(self) -> dict[str, Any]:
        """Every parser option the run used, defaults included, once parsed; else the config's options."""
        manifest = self.run_dir / 'run.jsonl'
        if manifest.is_file():
            with open(manifest, encoding='utf-8') as handle:
                for record in map(json.loads, handle):
                    if record['status'] == 'ok':
                        return flatten(record['parser']['config'])
        return self.options

    @property
    def parsed(self) -> bool:
        return (self.run_dir / 'run.jsonl').is_file()


@dataclass(frozen=True)
class Spec:
    path: Path
    stage: str
    title: str
    aim: str
    base_config: Path
    references: tuple[Path, ...]
    output: Path  # parent of the variants' run directories
    configs: Path  # where generated run configs are written
    split: str
    strata: Path | None
    metrics: Mapping[str, MetricInfo]
    experiments: tuple[Experiment, ...]

    @classmethod
    def load(cls, path: Path) -> Spec:
        raw = yaml.safe_load(path.read_text(encoding='utf-8'))
        return cls(
            path=path,
            stage=str(raw['stage']),
            title=raw['title'],
            aim=raw.get('aim', ''),
            base_config=Path(raw['base_config']),
            references=tuple(Path(p) for p in raw.get('references') or ()),
            output=Path(raw['output']),
            configs=Path(raw['configs']),
            split=raw.get('split', 'screen'),
            strata=Path(raw['strata']) if raw.get('strata') else None,
            metrics={
                column: MetricInfo(column, **{'label': column, **(info or {})})
                for column, info in (raw.get('metrics') or {}).items()
            },
            experiments=tuple(_experiment(item) for item in raw['experiments']),
        )

    def experiment(self, experiment_id: str) -> Experiment:
        for experiment in self.experiments:
            if experiment.id == experiment_id:
                return experiment
        raise KeyError(f'{experiment_id!r} not in {self.path}; known: {[e.id for e in self.experiments]}')

    def split_of(self, experiment: Experiment) -> str:
        return experiment.split or self.split

    def info(self, column: str) -> MetricInfo:
        return self.metrics.get(column) or MetricInfo(column, column)

    def reference_arms(self) -> list[Arm]:
        return [_arm_from_file('reference', path) for path in self.references]

    def best_from(self, experiment: Experiment) -> Experiment | None:
        """The experiment whose winner is ``experiment``'s control, or None when that is the base config.

        Raises ``NoWinnerYet`` when the control names an experiment that has no winner yet.
        """
        index = self.experiments.index(experiment)
        if experiment.control == 'base':
            return None
        if experiment.control:
            source = self.experiment(experiment.control)
            if self.experiments.index(source) >= index:
                raise ValueError(f'{experiment.id}: control {source.id} must be an earlier experiment')
            if not source.winner:
                raise NoWinnerYet(f'{experiment.id} builds on the winner of {source.id}, which is not chosen yet')
            return source
        return next((e for e in reversed(self.experiments[:index]) if e.winner), None)

    def control(self, experiment: Experiment) -> Arm:
        """The run ``experiment`` changes options on: see ``best_from``."""
        earlier = self.best_from(experiment)
        if earlier is None:
            return _arm_from_file('control', self.base_config)
        assert earlier.winner
        arm = self.control(earlier) if earlier.winner == 'control' else self.variant(earlier, earlier.winner)
        return Arm('control', arm.name, arm.raw)

    def variant(self, experiment: Experiment, name: str) -> Arm:
        variant = next((v for v in experiment.variants if v.name == name), None)
        if variant is None:
            raise KeyError(f'{experiment.id} has no variant {name!r}')
        if variant.source:
            source_id, _, source_name = variant.source.partition('.')
            source = self.variant(self.experiment(source_id), source_name)
            return Arm('variant', name, source.raw)
        raw = copy.deepcopy(dict(self.control(experiment).raw))
        raw['parser']['options'] = merge(raw['parser'].get('options') or {}, variant.options)
        stem = f'{experiment.id.lower()}_{name}_{settings_hash(raw)}'
        raw['name'], raw['output'] = stem, (self.output / stem).as_posix()
        return Arm('variant', name, raw)

    def arms(self, experiment: Experiment) -> list[Arm]:
        """Every row of the experiment's table: references, the control, then the variants."""
        variants = [self.variant(experiment, v.name) for v in experiment.variants]
        return [*self.reference_arms(), self.control(experiment), *variants]

    def changed(self, experiment: Experiment) -> list[str]:
        """Options some variant sets differently from the control: the independent variables."""
        control = self.control(experiment).options
        keys: dict[str, None] = {}
        for v in experiment.variants:
            for key, value in self.variant(experiment, v.name).options.items():
                if key not in control or control[key] != value:
                    keys[key] = None
        return list(keys)

    def config_path(self, arm: Arm) -> Path:
        return self.configs / f'{arm.raw["name"]}.yaml'


def flatten(options: Mapping[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in options.items():
        if isinstance(value, Mapping) and value:
            flat |= {f'{key}.{inner}': item for inner, item in value.items()}
        else:
            flat[key] = value
    return flat


def merge(base: Mapping[str, Any], changes: Mapping[str, Any]) -> dict[str, Any]:
    """``changes`` over ``base``; mapping values (``overrides``) merge one level deep."""
    merged = dict(base)
    for key, value in changes.items():
        old = merged.get(key)
        merged[key] = {**old, **value} if isinstance(old, Mapping) and isinstance(value, Mapping) else value
    return merged


def settings_hash(raw: Mapping[str, Any]) -> str:
    settings = RunConfig.from_dict(raw).settings
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:6]


def _arm_from_file(role: Role, path: Path) -> Arm:
    raw = yaml.safe_load(path.read_text(encoding='utf-8'))
    return Arm(role, raw['name'], raw)


def _experiment(raw: Mapping[str, Any]) -> Experiment:
    variants = []
    for name, options in (raw.get('variants') or {}).items():
        options = dict(options or {})
        source = options.pop('from', None)
        variants.append(Variant(str(name), options, source))
    return Experiment(
        id=raw['id'],
        title=raw['title'],
        question=raw.get('question', ''),
        pipeline_stage=raw.get('pipeline_stage', ''),
        target=raw.get('target', ''),
        metrics=tuple(raw.get('metrics') or ()),
        variants=tuple(variants),
        split=raw.get('split'),
        winner=raw.get('winner'),
        control=raw.get('control'),
        notes=raw.get('notes', ''),
    )
