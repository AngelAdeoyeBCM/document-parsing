"""A docling ``BaseVisualizer`` that draws over another visualizer's images, or on a transparent layer."""

from __future__ import annotations

from abc import abstractmethod

from docling_core.transforms.visualizer.base import BaseVisualizer
from docling_core.types.doc import BoundingBox, DoclingDocument, ProvenanceItem
from PIL import Image as PILImage
from PIL.Image import Image
from typing_extensions import override


class OverlayVisualizer(BaseVisualizer):
    """Draws on top of ``base_visualizer``'s images (e.g. docling's layout boxes), else on copies of the page images.

    With ``layer``, it draws on transparent images the size of the page images instead, to stack over them in a
    page and show or hide.
    """

    base_visualizer: BaseVisualizer | None = None
    layer: bool = False

    @override
    def get_visualization(self, *, doc: DoclingDocument, **kwargs) -> dict[int | None, Image]:
        if self.base_visualizer is not None:
            images = self.base_visualizer.get_visualization(doc=doc, **kwargs)
        else:
            pages = {n: page.image.pil_image for n, page in doc.pages.items() if page.image and page.image.pil_image}
            images = {
                n: PILImage.new('RGBA', image.size, (0, 0, 0, 0)) if self.layer else image.copy()
                for n, image in pages.items()
            }
        self._draw(doc, images)
        return images

    @abstractmethod
    def _draw(self, doc: DoclingDocument, images: dict[int | None, Image]) -> None: ...


def pixel_box(doc: DoclingDocument, prov: ProvenanceItem, image: Image) -> BoundingBox:
    """The provenance box in the page image's pixels, origin top-left."""
    size = doc.pages[prov.page_no].size
    box = prov.bbox.to_top_left_origin(page_height=size.height)
    sx, sy = image.width / size.width, image.height / size.height
    return BoundingBox(l=box.l * sx, t=box.t * sy, r=box.r * sx, b=box.b * sy)
