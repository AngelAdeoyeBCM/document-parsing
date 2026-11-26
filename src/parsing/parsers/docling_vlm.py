"""Docling's VLM pipeline: one vision-language model reads page images, replacing layout detection and OCR."""

from __future__ import annotations

from dataclasses import dataclass

from docling.datamodel import vlm_model_specs
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.document_converter import PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

from .docling_common import DoclingConfig, DoclingParser


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
