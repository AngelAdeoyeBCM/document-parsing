"""What Docling's standard and VLM pipelines share: device, PDF backend, page images and the converter."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from docling.backend.docling_parse_backend import ThreadedDoclingParseDocumentBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument

from .base import Parser, ParserConfig

_BACKENDS = {'docling_parse': ThreadedDoclingParseDocumentBackend, 'pypdfium2': PyPdfiumDocumentBackend}


@dataclass(frozen=True, slots=True)
class DoclingConfig(ParserConfig):
    device: str = 'cuda'  # 'cpu', 'cuda' or 'cuda:N'
    num_threads: int = 4
    backend: Literal['docling_parse', 'pypdfium2'] = 'docling_parse'  # how the PDF's text and pages are read
    images_scale: float = 2.0
    picture_images: bool = True  # keep figure crops; saved beside the document JSON


class DoclingParser(Parser):
    """Shared Docling conversion; variants differ only in their pipeline options.

    The converter and its models are loaded when the parser is built, so no document's time includes them and a
    bad option fails before any document is parsed.
    """

    package = 'docling'
    config: DoclingConfig

    def __init__(self, config: DoclingConfig | None = None) -> None:
        super().__init__(config)
        self._converter = DocumentConverter(format_options={InputFormat.PDF: self._format_option()})
        self._converter.initialize_pipeline(InputFormat.PDF)

    @abstractmethod
    def _format_option(self) -> PdfFormatOption:
        """Docling pipeline and backend for this variant."""

    @property
    def _accelerator(self) -> AcceleratorOptions:
        return AcceleratorOptions(device=self.config.device, num_threads=self.config.num_threads)

    @property
    def _backend(self) -> type:
        return _BACKENDS[self.config.backend]

    def _convert(self, path: Path) -> DoclingDocument:
        return self._converter.convert(path).document
