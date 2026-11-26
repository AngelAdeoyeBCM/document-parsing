from __future__ import annotations

import importlib
from collections.abc import Mapping

from .base import Parser

# Imported on demand so the baseline never loads Docling or torch.
_PARSERS = {
    'pypdf': 'parsing.parsers.pypdf_parser:PypdfParser',
    'docling': 'parsing.parsers.docling_standard:DoclingStandardParser',
    'docling-vlm': 'parsing.parsers.docling_vlm:DoclingVlmParser',
}


def available() -> list[str]:
    return sorted(_PARSERS)


def parser_class(name: str) -> type[Parser]:
    try:
        target = _PARSERS[name]
    except KeyError:
        raise KeyError(f'unknown parser {name!r}; available: {available()}') from None
    module, _, attribute = target.partition(':')
    return getattr(importlib.import_module(module), attribute)


def get_parser(name: str, options: Mapping[str, object] | None = None) -> Parser:
    """Build a parser by name, with ``options`` passed to its config class."""
    cls = parser_class(name)
    return cls(cls.config_cls(**(options or {})))
