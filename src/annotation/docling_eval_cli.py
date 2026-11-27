"""Run docling-eval's command line (``python -m annotation.docling_eval_cli <docling-eval arguments>``).

docling-eval 1.4.2 imports ``docling_ibm_models.layoutmodel`` and ``docling_ibm_models.reading_order`` when its
command line starts, but docling-ibm-models 4, which Docling 2.153 requires, no longer has them. Only docling-eval's
pixel-layout and reading-order evaluators use them, and the annotation workflow (create-gt, create-eval with the
Docling provider, create-cvat) never runs either, so the missing modules are stubbed: any name imported from them
fails only when it is used.

docling-eval's Docling provider runs OCR, but the pre-annotations must match B0 (``do_ocr=False``): every sampled
document has an embedded text layer. docling-eval has no option for it, so OCR is switched off where Docling's PDF
pipeline is built.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from typing import Any

REMOVED = ('docling_ibm_models.layoutmodel.labels', 'docling_ibm_models.reading_order.reading_order_rb')


def _unsupported(name: str) -> type:
    if name.startswith('__'):
        raise AttributeError(name)  # module internals (__file__, __path__) others inspect

    def fail(*_: object, **__: object) -> None:
        raise RuntimeError(f'{name} needs docling-ibm-models < 4; docling-eval evaluators using it are not supported')

    return type(name, (), {'__init__': fail})


def _stub_removed_modules() -> None:
    for path in REMOVED:
        package = path.rsplit('.', 1)[0]
        if importlib.util.find_spec(package) is not None:
            continue  # an older docling-ibm-models that still has it
        for name in (package, path):
            module = types.ModuleType(name)
            module.__getattr__ = _unsupported  # type: ignore[method-assign]  # PEP 562: any imported name
            sys.modules[name] = module


def _disable_ocr() -> None:
    from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

    init = StandardPdfPipeline.__init__

    def no_ocr_init(self: StandardPdfPipeline, pipeline_options: Any) -> None:
        pipeline_options.do_ocr = False
        init(self, pipeline_options)

    StandardPdfPipeline.__init__ = no_ocr_init  # type: ignore[method-assign]


if __name__ == '__main__':
    _stub_removed_modules()
    _disable_ocr()
    from docling_eval.cli.main import app

    sys.argv[0] = 'docling-eval'
    app()
