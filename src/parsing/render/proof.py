from __future__ import annotations

from pathlib import Path

import pypdfium2 as pdfium
from docling_core.transforms.visualizer.layout_visualizer import LayoutVisualizer
from docling_core.transforms.visualizer.reading_order_visualizer import ReadingOrderVisualizer
from docling_core.types.doc import DoclingDocument, ImageRef
from PIL.Image import Image

from ..core import ParsedDocument, pdf_text_in_picture
from .proof_html import ProofHtml
from .renderers import Renderer


class ProofRenderer(Renderer):
    """Linked side-by-side HTML of every source page and its parse; optionally Docling's box overlays as PNGs."""

    suffix = '.html'

    def __init__(
        self, scale: float = 1.5, zoom: float = 0.5, overlays: bool = False, reading_order: bool = False
    ) -> None:
        self.scale = scale
        self.zoom = zoom
        self.overlays = overlays
        layout = LayoutVisualizer()
        self._visualizer = ReadingOrderVisualizer(base_visualizer=layout) if reading_order else layout

    def _save(self, document: ParsedDocument, path: Path, artifacts: Path) -> None:
        images, picture_words = self._read_pdf(document)
        path.write_text(ProofHtml(document, images, self.scale, self.zoom, picture_words).render(), encoding='utf-8')
        if self.overlays:
            out = path.parent / artifacts
            out.mkdir(parents=True, exist_ok=True)
            for page_no, image in self._visualizer.get_visualization(doc=self._with_images(document, images)).items():
                image.save(out / f'layout_p{page_no:03d}.png')

    def _read_pdf(self, document: ParsedDocument) -> tuple[dict[int, Image], dict[str, int]]:
        """Every source page image (skipped ones too) and the PDF word count inside each picture."""
        pdf = pdfium.PdfDocument(document.source.path)
        try:
            images = {n: pdf[n - 1].render(scale=self.scale).to_pil() for n in range(1, len(pdf) + 1)}
            doc = document.document
            words = {p.self_ref: len(pdf_text_in_picture(pdf, doc, p).split()) for p in doc.pictures if p.prov}
            return images, words
        finally:
            pdf.close()

    def _with_images(self, document: ParsedDocument, images: dict[int, Image]) -> DoclingDocument:
        doc = document.document.model_copy(deep=True)
        for page_no, page in doc.pages.items():
            page.image = ImageRef.from_pil(images[page_no], dpi=round(72 * self.scale))
        return doc
