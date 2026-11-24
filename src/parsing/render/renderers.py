from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Literal

from docling_core.types.doc import ImageRefMode

from ..core import ParsedDocument

ImageMode = Literal['referenced', 'embedded', 'placeholder']


class Renderer(ABC):
    """Writes one output file per document; any figures or page images go to ``<doc_id>_artifacts/``."""

    suffix: ClassVar[str]

    def write(self, document: ParsedDocument, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f'{document.source.doc_id}{self.suffix}'
        # Docling resolves a relative artifacts dir against the output file's folder and links images relatively.
        self._save(document, path, Path(f'{document.source.doc_id}_artifacts'))
        return path

    @abstractmethod
    def _save(self, document: ParsedDocument, path: Path, artifacts: Path) -> None: ...


class DoclingRenderer(Renderer):
    """Renderer backed by one of the DoclingDocument ``save_as_*`` exporters."""

    def __init__(self, image_mode: ImageMode = 'referenced') -> None:
        self.image_mode = ImageRefMode(image_mode)


class JsonRenderer(DoclingRenderer):
    """Lossless DoclingDocument JSON; the format evaluation reads back."""

    suffix = '.json'

    def _save(self, document: ParsedDocument, path: Path, artifacts: Path) -> None:
        document.document.save_as_json(path, artifacts, image_mode=self.image_mode, ensure_ascii=False)


class MarkdownRenderer(DoclingRenderer):
    suffix = '.md'

    def __init__(self, image_mode: ImageMode = 'referenced', page_breaks: bool = True) -> None:
        super().__init__(image_mode)
        self.page_breaks = page_breaks  # markers a chunker can use to recover page numbers

    def _save(self, document: ParsedDocument, path: Path, artifacts: Path) -> None:
        document.document.save_as_markdown(
            path,
            artifacts,
            image_mode=self.image_mode,
            page_break_placeholder='<!-- page break -->' if self.page_breaks else None,
        )
