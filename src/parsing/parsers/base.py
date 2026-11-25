from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from importlib import metadata
from types import MappingProxyType
from typing import ClassVar

from docling_core.types.doc import DoclingDocument

from inventory import PdfSource

from ..core import ParsedDocument, ParserInfo


@dataclass(frozen=True, slots=True)
class ParserConfig:
    """Settings for one parser; each backend subclasses this with its own fields."""


class Parser(ABC):
    """Converts one PDF into a ParsedDocument."""

    name: ClassVar[str]
    package: ClassVar[str]  # distribution whose version is recorded with the output
    config_cls: ClassVar[type[ParserConfig]] = ParserConfig

    def __init__(self, config: ParserConfig | None = None) -> None:
        config = config if config is not None else self.config_cls()
        if not isinstance(config, self.config_cls):
            raise TypeError(f'{type(self).__name__} expects {self.config_cls.__name__}, got {type(config).__name__}')
        self.config = config

    @property
    def info(self) -> ParserInfo:
        return ParserInfo(
            name=self.name, version=metadata.version(self.package), config=MappingProxyType(asdict(self.config))
        )

    def parse(self, source: PdfSource, skip: frozenset[int] = frozenset(), texts: Sequence[str] = ()) -> ParsedDocument:
        """Parse every page of ``source`` except the 1-based page numbers in ``skip``.

        ``texts`` is the inventory's stored pypdf text layer, for parsers that can reuse it instead of re-reading.
        """
        start = time.perf_counter()
        document = self._convert(source, skip, texts)
        return ParsedDocument(
            source=source, parser=self.info, document=document, skipped_pages=skip, seconds=time.perf_counter() - start
        )

    @abstractmethod
    def _convert(self, source: PdfSource, skip: frozenset[int], texts: Sequence[str]) -> DoclingDocument:
        """The kept pages as a DoclingDocument, numbered as in the source PDF."""
