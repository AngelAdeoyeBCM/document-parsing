from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ..postprocess import ChartFilter, DocumentProcessor
from ..render import JsonRenderer, MarkdownRenderer, ProofRenderer, Renderer
from ..triage import (
    CompositeClassifier,
    FlaggedPolicy,
    PageClassifier,
    RepetitionClassifier,
    RuleClassifier,
    SkipPolicy,
    TailPolicy,
)
from .config import ComponentSpec

CLASSIFIERS: Mapping[str, type[PageClassifier]] = {'rules': RuleClassifier, 'repetition': RepetitionClassifier}
POLICIES: Mapping[str, type[SkipPolicy]] = {'tail': TailPolicy, 'flagged': FlaggedPolicy}
PROCESSORS: Mapping[str, type[DocumentProcessor]] = {'chart_filter': ChartFilter}
RENDERERS: Mapping[str, type[Renderer]] = {'json': JsonRenderer, 'markdown': MarkdownRenderer, 'proof': ProofRenderer}


def build_classifier(specs: Sequence[ComponentSpec]) -> PageClassifier:
    classifiers = [_build(CLASSIFIERS, spec) for spec in specs]
    return classifiers[0] if len(classifiers) == 1 else CompositeClassifier(classifiers)


def build_policy(name: str, keep_first: int) -> SkipPolicy:
    return _build(POLICIES, ComponentSpec(name, {'keep_first': keep_first}))


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
