from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal

from docling.backend.docling_parse_backend import ThreadedDoclingParseDocumentBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel import vlm_model_specs
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    HeadingHierarchyOptions,
    LayoutObjectDetectionOptions,
    NemotronOcrOptions,
    OcrAutoOptions,
    OcrMode,
    PdfPipelineOptions,
    PictureDescriptionApiOptions,
    RapidOcrOptions,
    ScriptPositionOptions,
    TableFormerMode,
    TesseractCliOcrOptions,
    TesseractOcrOptions,
    VlmPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling_core.types.doc import DoclingDocument

from .model import ModelConfig, ModelParser

_BACKENDS = {'docling_parse': ThreadedDoclingParseDocumentBackend, 'pypdfium2': PyPdfiumDocumentBackend}
_OCR_ENGINES = {
    'auto': OcrAutoOptions,
    'easyocr': EasyOcrOptions,
    'nemotron': NemotronOcrOptions,  # Python 3.12 + CUDA 13 only, so it runs from .venv-nemotron
    'rapidocr': RapidOcrOptions,
    'tesseract': TesseractCliOcrOptions,
    'tesserocr': TesseractOcrOptions,
}


@dataclass(frozen=True, slots=True)
class DoclingConfig(ModelConfig):
    backend: Literal['docling_parse', 'pypdfium2'] = 'docling_parse'
    images_scale: float = 2.0
    picture_images: bool = True  # keep figure crops so renderers can save them
    num_threads: int = 4


class DoclingParser(ModelParser):
    """Shared Docling conversion; variants differ only in their pipeline options."""

    package = 'docling'
    config: DoclingConfig

    @cached_property
    def _converter(self) -> DocumentConverter:
        return DocumentConverter(format_options={InputFormat.PDF: self._format_option()})

    @abstractmethod
    def _format_option(self) -> PdfFormatOption:
        """Docling pipeline and backend for this variant."""

    @property
    def _accelerator(self) -> AcceleratorOptions:
        return AcceleratorOptions(device=self.config.device, num_threads=self.config.num_threads)

    @property
    def _backend(self) -> type:
        return _BACKENDS[self.config.backend]

    def _convert_pdf(self, path: Path) -> DoclingDocument:
        return self._converter.convert(path).document


@dataclass(frozen=True, slots=True)
class DoclingStandardConfig(DoclingConfig):
    layout_model: Literal[
        'layout_heron_default', 'layout_heron_101', 'layout_egret_medium', 'layout_egret_large', 'layout_egret_xlarge'
    ] = 'layout_egret_xlarge'  # only preset that kept every copper caption and chart axis intact
    do_ocr: bool = False  # research PDFs are born-digital; OCR mostly costs time
    ocr_engine: Literal['auto', 'easyocr', 'nemotron', 'rapidocr', 'tesseract', 'tesserocr'] = 'auto'
    rapidocr_backend: Literal['torch', 'onnxruntime', 'openvino', 'paddle'] = 'torch'  # onnxruntime here is CPU-only
    rapidocr_model_size: Literal['tiny', 'small', 'medium'] = 'small'  # PP-OCRv6 det/rec size
    rapidocr_use_cls: bool | None = None  # 180-degree classifier; flips short crops like '99%' into '%66'
    ocr_lang: str | None = None  # engine default when unset (rapidocr 'ch', easyocr fr/de/es/en)
    tessdata_path: str | None = None  # tesseract/tesserocr language data dir; TESSDATA_PREFIX when unset
    ocr_mode: Literal['default', 'full_page', 'layout_regions', 'pdf_aware_layout_regions'] = 'default'
    replace_broken_text_layer: bool = False
    table_mode: Literal['fast', 'accurate'] = 'accurate'
    cell_matching: bool = True
    heading_hierarchy: bool = False
    script_positions_enabled: bool = False
    use_reading_order_separators: bool = True
    classify_pictures: bool = False
    extract_charts: bool = False  # chart-to-table with Docling's default chart VLM
    describe_url: str | None = None  # OpenAI-compatible chat completions URL for picture descriptions
    describe_model: str | None = None
    describe_prompt: str = (
        'Describe this figure from a financial research report: chart type, what is plotted, units, '
        'period covered and the main takeaway. At most 100 words, only what is visible.'
    )


class DoclingStandardParser(DoclingParser):
    """Docling's layout, TableFormer and reading-order models over the PDF text layer."""

    name = 'docling'
    config_cls = DoclingStandardConfig
    config: DoclingStandardConfig

    def _format_option(self) -> PdfFormatOption:
        c = self.config
        ocr = {'mode': OcrMode(c.ocr_mode), 'replace_broken_text_layer': c.replace_broken_text_layer}
        if c.ocr_engine == 'rapidocr':
            ocr['backend'] = c.rapidocr_backend
            ocr['model_size'] = c.rapidocr_model_size
            ocr['use_cls'] = c.rapidocr_use_cls
        if c.ocr_lang:
            ocr['lang'] = [c.ocr_lang]
        if c.tessdata_path and c.ocr_engine in ('tesseract', 'tesserocr'):
            ocr['path'] = c.tessdata_path
        options = PdfPipelineOptions(
            accelerator_options=self._accelerator,
            do_ocr=c.do_ocr,
            ocr_options=_OCR_ENGINES[c.ocr_engine](**ocr),
            do_table_structure=True,
            layout_options=LayoutObjectDetectionOptions.from_preset(c.layout_model),
            images_scale=c.images_scale,
            generate_picture_images=c.picture_images,
            heading_hierarchy_options=HeadingHierarchyOptions(enabled=c.heading_hierarchy),
            script_position_options=ScriptPositionOptions(enabled=c.script_positions_enabled),
            use_reading_order_separators=c.use_reading_order_separators,
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
        return PdfFormatOption(pipeline_options=options, backend=self._backend)


@dataclass(frozen=True, slots=True)
class DoclingVlmConfig(DoclingConfig):
    model: str = 'GRANITEDOCLING_TRANSFORMERS'  # attribute name in docling.datamodel.vlm_model_specs


class DoclingVlmParser(DoclingParser):
    """One vision-language model reads page images, replacing layout detection and OCR."""

    name = 'docling-vlm'
    config_cls = DoclingVlmConfig
    config: DoclingVlmConfig

    def _format_option(self) -> PdfFormatOption:
        c = self.config
        options = VlmPipelineOptions(
            accelerator_options=self._accelerator,
            vlm_options=getattr(vlm_model_specs, c.model),
            images_scale=c.images_scale,
            generate_picture_images=c.picture_images,
        )
        return PdfFormatOption(pipeline_cls=VlmPipeline, pipeline_options=options, backend=self._backend)
