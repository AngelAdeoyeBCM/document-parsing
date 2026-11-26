from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar

from docling_core.types.doc import DoclingDocument

from inventory import PdfSource

from ..core import ParsedDocument, ParserInfo


@dataclass(frozen=True, slots=True)
class ParserConfig:
    """Settings for one parser; each backend subclasses this with its own fields."""


class Parser(ABC):
    """Converts one PDF file into a DoclingDocument, timed and labelled with the parser that made it."""

    name: ClassVar[str]
    package: ClassVar[str]  # distribution whose version is recorded with the output
    config_cls: ClassVar[type[ParserConfig]] = ParserConfig

    def __init__(self, config: ParserConfig | None = None) -> None:
        config = config if config is not None else self.config_cls()
        if not isinstance(config, self.config_cls):
            raise TypeError(f'{type(self).__name__} expects {self.config_cls.__name__}, got {type(config).__name__}')
        self.config = config
        self.info = ParserInfo(self.name, metadata.version(self.package), MappingProxyType(asdict(config)))

    def parse(self, source: PdfSource, pdf: Path | None = None) -> ParsedDocument:
        """Convert ``pdf``, by default the source PDF itself, as the document of ``source``.

        A caller that drops trailing pages (tail triage) passes the truncated copy instead: the pages kept are the
        leading ones, so they keep their source page numbers. Only the conversion is timed.
        """
        start = time.perf_counter()
        document = self._convert(pdf or source.path)
        return ParsedDocument(source=source, parser=self.info, document=document, seconds=time.perf_counter() - start)

    @abstractmethod
    def _convert(self, path: Path) -> DoclingDocument:
        """Every page of the PDF at ``path`` as a DoclingDocument."""
