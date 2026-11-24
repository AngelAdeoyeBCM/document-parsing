from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from inventory import Triage

from ..postprocess import ChartFilter, DocumentProcessor
from ..render import JsonRenderer, MarkdownRenderer, ProofRenderer, Renderer
from .config import ComponentSpec, TriageSpec

PROCESSORS: Mapping[str, type[DocumentProcessor]] = {'chart_filter': ChartFilter}
RENDERERS: Mapping[str, type[Renderer]] = {'json': JsonRenderer, 'markdown': MarkdownRenderer, 'proof': ProofRenderer}


def build_triage(spec: TriageSpec) -> Triage:
    return Triage(
        classifiers=tuple((classifier.name, dict(classifier.options)) for classifier in spec.classifiers),
        policy=spec.policy,
        keep_first=spec.keep_first,
    )


def build_renderer(spec: ComponentSpec) -> Renderer:
    return _build(RENDERERS, spec)


def build_processor(spec: ComponentSpec) -> DocumentProcessor:
    return _build(PROCESSORS, spec)


def _build(table: Mapping[str, Callable[..., Any]], spec: ComponentSpec) -> Any:
    try:
        factory = table[spec.name]
    except KeyError:
        raise KeyError(f'unknown component {spec.name!r}; available: {sorted(table)}') from None
    return factory(**spec.options)
