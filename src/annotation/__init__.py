"""Gold-standard annotation with docling-eval and CVAT; see ``cli.py`` for the workflow."""

from .cvat import collect, prepare
from .gold import Gold, load_gold
from .workspace import Workspace

__all__ = ['Gold', 'Workspace', 'collect', 'load_gold', 'prepare']
