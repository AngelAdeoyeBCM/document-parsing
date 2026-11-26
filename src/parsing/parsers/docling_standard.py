"""Docling's standard PDF pipeline: layout, OCR, tables and reading-order models over the PDF text layer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import reduce
from typing import Any, Literal

from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    HeadingHierarchyOptions,
    LayoutObjectDetectionOptions,
    NemotronOcrOptions,
    OcrAutoOptions,
    OcrMode,
    OcrOptions,
    PdfPipelineOptions,
    PictureDescriptionApiOptions,
    RapidOcrOptions,
    ScriptPositionOptions,
    TableFormerMode,
    TesseractCliOcrOptions,
    TesseractOcrOptions,
)
from docling.document_converter import PdfFormatOption

from .docling_common import DoclingConfig, DoclingParser

_OCR_ENGINES = {
    'auto': OcrAutoOptions,
    'easyocr': EasyOcrOptions,
    'nemotron': NemotronOcrOptions,  # Python 3.12 + CUDA 13 only, so it runs from .venv-nemotron
    'rapidocr': RapidOcrOptions,
    'tesseract': TesseractCliOcrOptions,
    'tesserocr': TesseractOcrOptions,
}


@dataclass(frozen=True, slots=True)
class DoclingStandardConfig(DoclingConfig):
    # Layout: which detection model labels the page's blocks
    layout_model: Literal[
        'layout_heron_default', 'layout_heron_101', 'layout_egret_medium', 'layout_egret_large', 'layout_egret_xlarge'
    ] = 'layout_egret_xlarge'  # only preset that kept every copper caption and chart axis intact

    # OCR
    do_ocr: bool = False  # research PDFs are born-digital; OCR mostly costs time
    ocr_engine: Literal['auto', 'easyocr', 'nemotron', 'rapidocr', 'tesseract', 'tesserocr'] = 'auto'
    ocr_mode: Literal['default', 'full_page', 'layout_regions', 'pdf_aware_layout_regions'] = 'default'
    ocr_lang: str | None = None  # engine default when unset (rapidocr 'ch', easyocr fr/de/es/en)
    replace_broken_text_layer: bool = False
    rapidocr_backend: Literal['torch', 'onnxruntime', 'openvino', 'paddle'] = 'torch'  # onnxruntime here is CPU-only
    rapidocr_model_size: Literal['tiny', 'small', 'medium'] = 'small'  # PP-OCRv6 det/rec size
    rapidocr_use_cls: bool | None = None  # 180-degree classifier; flips short crops like '99%' into '%66'
    tessdata_path: str | None = None  # tesseract/tesserocr language data dir; TESSDATA_PREFIX when unset

    # Tables (structure itself is on; switch it off with overrides: {do_table_structure: false})
    table_mode: Literal['fast', 'accurate'] = 'accurate'
    cell_matching: bool = True

    # Assemble: reading order and document structure
    use_reading_order_separators: bool = True
    heading_hierarchy: bool = False
    script_positions_enabled: bool = False

    # Pictures: enrichment and descriptions
    use_embedded_picture_images: bool = False
    classify_pictures: bool = False
    extract_charts: bool = False  # chart-to-table with Docling's default chart VLM
    describe_url: str | None = None  # OpenAI-compatible chat completions URL for picture descriptions
    describe_model: str | None = None
    describe_prompt: str = (
        'Describe this figure from a financial research report: chart type, what is plotted, units, '
        'period covered and the main takeaway. At most 100 words, only what is visible.'
    )

    # Any other PdfPipelineOptions setting by dotted path, applied last, e.g. {'do_table_structure': False}
    overrides: Mapping[str, Any] = field(default_factory=dict)


class DoclingStandardParser(DoclingParser):
    """Docling's layout, TableFormer and reading-order models over the PDF text layer."""

    name = 'docling'
    config_cls = DoclingStandardConfig
    config: DoclingStandardConfig

    def _format_option(self) -> PdfFormatOption:
        options = self._pipeline_options()
        set_options(options, self.config.overrides)
        return PdfFormatOption(pipeline_options=options, backend=self._backend)

    def _pipeline_options(self) -> PdfPipelineOptions:
        c = self.config
        options = PdfPipelineOptions(
            accelerator_options=self._accelerator,
            images_scale=c.images_scale,
            generate_picture_images=c.picture_images,
            layout_options=LayoutObjectDetectionOptions.from_preset(c.layout_model),
            do_ocr=c.do_ocr,
            ocr_options=self._ocr_options(),
            do_table_structure=True,
            use_reading_order_separators=c.use_reading_order_separators,
            heading_hierarchy_options=HeadingHierarchyOptions(enabled=c.heading_hierarchy),
            script_position_options=ScriptPositionOptions(enabled=c.script_positions_enabled),
            use_embedded_picture_images=c.use_embedded_picture_images,
            do_picture_classification=c.classify_pictures,
            do_chart_extraction=c.extract_charts,
        )
        options.table_structure_options.mode = TableFormerMode(c.table_mode)
        options.table_structure_options.do_cell_matching = c.cell_matching
        if c.describe_url:
            options.enable_remote_services = True
            options.do_picture_description = True
            options.picture_description_options = PictureDescriptionApiOptions(
                url=c.describe_url, params={'model': c.describe_model}, prompt=c.describe_prompt
            )
        return options

    def _ocr_options(self) -> OcrOptions:
        c = self.config
        ocr: dict[str, Any] = {'mode': OcrMode(c.ocr_mode), 'replace_broken_text_layer': c.replace_broken_text_layer}
        if c.ocr_lang:
            ocr['lang'] = [c.ocr_lang]
        if c.ocr_engine == 'rapidocr':
            ocr |= {'backend': c.rapidocr_backend, 'model_size': c.rapidocr_model_size, 'use_cls': c.rapidocr_use_cls}
        if c.tessdata_path and c.ocr_engine in ('tesseract', 'tesserocr'):
            ocr['path'] = c.tessdata_path
        return _OCR_ENGINES[c.ocr_engine](**ocr)


def set_options(target: object, values: Mapping[str, object]) -> None:
    """Set dotted option paths, e.g. ``{'layout_options.create_orphan_clusters': False}``; unknown names raise."""
    for path, value in values.items():
        *parents, name = path.split('.')
        owner = reduce(getattr, parents, target)
        if not hasattr(owner, name):
            raise AttributeError(f'{type(owner).__name__} has no option {name!r} (in {path!r})')
        setattr(owner, name, value)
