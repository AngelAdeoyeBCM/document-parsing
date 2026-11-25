"""Gold-standard annotation with docling-eval and CVAT; see ``cli.py`` for the workflow."""

from .cvat import collect, export_documents, prepare
from .gold import Gold, load_gold
from .workspace import Workspace

__all__ = ['Gold', 'Workspace', 'collect', 'export_documents', 'load_gold', 'prepare']
