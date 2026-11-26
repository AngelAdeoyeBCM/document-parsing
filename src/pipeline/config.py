from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
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


@dataclass(frozen=True, slots=True)
class InputSpec:
    directory: Path  # root of the extracted corpus
    manifest: Path | None = None  # relative paths to parse, one per line; None parses everything
    limit: int | None = None


@dataclass(frozen=True, slots=True)
class TriageSpec:
    policy: str = 'tail'  # how flagged pages become skipped pages
    keep_first: int = 1  # cover pages are never dropped
    min_share: float = 0.3  # share of a prose page's text that must be disclosure wording


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

    @property
    def settings(self) -> dict[str, Any]:
        """What decides the parsed output (not the input), as plain JSON values."""
        settings = {
            'parser': asdict(self.parser),
            'triage': asdict(self.triage) if self.triage else None,
            'routes': [asdict(route) for route in self.routes],
            'postprocess': [asdict(spec) for spec in self.postprocess],
        }
        return json.loads(json.dumps(settings, default=str))

    @classmethod
    def load(cls, path: Path) -> RunConfig:
        return cls.from_dict(yaml.safe_load(path.read_text(encoding='utf-8')))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> RunConfig:
        source = raw['input']
        triage = raw.get('triage')
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
                min_share=triage.get('min_share', 0.3),
            )
            if triage
            else None,
            routes=tuple(
                RouteSpec(route['name'], route.get('max_chars_per_page', 50), dict(route.get('options') or {}))
                for route in raw.get('routes') or ()
            ),
            postprocess=tuple(ComponentSpec.parse(item) for item in raw.get('postprocess') or ()),
        )


def _path(value: str | None) -> Path | None:
    return Path(value) if value else None
