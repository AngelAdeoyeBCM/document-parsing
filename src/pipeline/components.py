from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from triage import Triage

from .config import ComponentSpec, TriageSpec

# Disabled with PostprocessStage; restore with it:
# from parsing.postprocess import ChartFilter, DocumentProcessor
# PROCESSORS: Mapping[str, type[DocumentProcessor]] = {'chart_filter': ChartFilter}


def build_triage(spec: TriageSpec) -> Triage:
    return Triage(policy=spec.policy, keep_first=spec.keep_first, min_share=spec.min_share)


# def build_processor(spec: ComponentSpec) -> DocumentProcessor:
#     return _build(PROCESSORS, spec)


def _build(table: Mapping[str, Callable[..., Any]], spec: ComponentSpec) -> Any:
    try:
        factory = table[spec.name]
    except KeyError:
        raise KeyError(f'unknown component {spec.name!r}; available: {sorted(table)}') from None
    return factory(**spec.options)
