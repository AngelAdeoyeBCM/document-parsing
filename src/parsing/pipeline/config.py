from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    """A named component and the keyword options used to build it."""

    name: str
    options: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, raw: str | Mapping[str, Any]) -> ComponentSpec:
        if isinstance(raw, str):
            return cls(raw)
        return cls(raw['name'], dict(raw.get('options') or {}))


DEFAULT_RENDERERS = (ComponentSpec('json'), ComponentSpec('proof'))


@dataclass(frozen=True, slots=True)
class InputSpec:
    directory: Path  # root of the extracted corpus
    manifest: Path | None = None  # relative paths to parse, one per line; None parses everything
    limit: int | None = None


@dataclass(frozen=True, slots=True)
class TriageSpec:
    policy: str = 'tail'  # how flagged pages become skipped pages
    keep_first: int = 1  # cover pages are never dropped
    min_hits: int = 3


@dataclass(frozen=True, slots=True)
class RouteSpec:
    """Documents whose text layer averages under ``max_chars_per_page`` get the parser with ``options`` overridden."""

    name: str
    max_chars_per_page: float = 50
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunConfig:
    name: str
    input: InputSpec
    output: Path
    parser: ComponentSpec
    triage: TriageSpec | None = None
    routes: tuple[RouteSpec, ...] = ()
    postprocess: tuple[ComponentSpec, ...] = ()
    renderers: tuple[ComponentSpec, ...] = DEFAULT_RENDERERS

    @classmethod
    def load(cls, path: Path) -> RunConfig:
        return cls.from_dict(yaml.safe_load(path.read_text(encoding='utf-8')))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> RunConfig:
        source = raw['input']
        triage = raw.get('triage')
        renderers = raw.get('renderers')
        return cls(
            name=raw['name'],
            input=InputSpec(
                directory=Path(source['directory']),
                manifest=_path(source.get('manifest')),
                limit=source.get('limit'),
            ),
            output=Path(raw['output']),
            parser=ComponentSpec.parse(raw['parser']),
            triage=TriageSpec(
                policy=triage.get('policy', 'tail'),
                keep_first=triage.get('keep_first', 1),
                min_hits=triage.get('min_hits', 3),
            )
            if triage
            else None,
            routes=tuple(
                RouteSpec(route['name'], route.get('max_chars_per_page', 50), dict(route.get('options') or {}))
                for route in raw.get('routes') or ()
            ),
            postprocess=tuple(ComponentSpec.parse(item) for item in raw.get('postprocess') or ()),
            renderers=tuple(ComponentSpec.parse(item) for item in renderers) if renderers else DEFAULT_RENDERERS,
        )


def _path(value: str | None) -> Path | None:
    return Path(value) if value else None
