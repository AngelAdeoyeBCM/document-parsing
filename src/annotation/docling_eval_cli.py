"""Run docling-eval's command line (``python -m annotation.docling_eval_cli <docling-eval arguments>``).

docling-eval 1.4.2 imports ``docling_ibm_models.layoutmodel`` and ``docling_ibm_models.reading_order`` when its
command line starts, but docling-ibm-models 4, which Docling 2.153 requires, no longer has them. Only docling-eval's
pixel-layout and reading-order evaluators use them, and the annotation workflow (create-gt, create-eval with the
Docling provider, create-cvat) never runs either, so the missing modules are stubbed: any name imported from them
fails only when it is used.
"""

from __future__ import annotations

import importlib.util
import sys
import types

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


if __name__ == '__main__':
    _stub_removed_modules()
    from docling_eval.cli.main import app

    sys.argv[0] = 'docling-eval'
    app()
