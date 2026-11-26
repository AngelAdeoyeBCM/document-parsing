from __future__ import annotations

from docling_core.types.doc import PictureItem

# Docling picture classes treated as charts; 'table' is a table drawn as an image.
CHART_CLASSES = frozenset(
    {
        'bar_chart',
        'stacked_bar_chart',
        'line_chart',
        'pie_chart',
        'scatter_plot',
        'scatter_chart',
        'box_plot',
        'heatmap',
        'flow_chart',
        'other_chart',
        'table',
    }
)


def picture_class(picture: PictureItem) -> tuple[str, float] | None:
    """Top class Docling's picture classifier gave this picture, with its confidence; None if not classified."""
    classification = picture.meta.classification if picture.meta else None
    if classification is None or not classification.predictions:
        return None
    best = max(classification.predictions, key=lambda p: p.confidence or 0.0)
    return best.class_name, best.confidence or 0.0
