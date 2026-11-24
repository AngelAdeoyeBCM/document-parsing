from __future__ import annotations

import html
import re
from collections.abc import Sequence

from docling_core.types.doc import TableData

_PALETTE = ('#2563eb', '#dc2626', '#059669', '#d97706', '#7c3aed', '#db2777', '#0891b2', '#4b5563')
_NUMBER = re.compile(r'-?\d+(?:\.\d+)?')
_W, _LEFT, _RIGHT, _TOP, _BOTTOM = 420, 48, 10, 10, 34
_X_LABELS = 6


def chart_svg(data: TableData, kind: str, aspect: float | None = None, y_ticks: Sequence[float] | None = None) -> str:
    """Re-plot a chart table (column 0 = x, others = series) using the printed chart's aspect and y ticks if known."""
    grid = data.grid
    if len(grid) < 2 or len(grid[0]) < 2:
        return ''
    names = [cell.text for cell in grid[0][1:]]
    labels = [row[0].text for row in grid[1:]]
    series = [[number(row[col].text) for row in grid[1:]] for col in range(1, len(grid[0]))]
    values = [v for s in series for v in s if v is not None]
    if not values:
        return ''
    height = min(max(round(_W / aspect), 140), 420) if aspect else 220
    ticks = sorted(set(y_ticks)) if y_ticks and len(set(y_ticks)) >= 2 else _default_ticks(values, 'bar' in kind)
    low, high = ticks[0], ticks[-1]
    plot = _Plot(height, low, high)
    body = _bars(series, plot) if 'bar' in kind else _lines(series, plot)
    legend_h = 14 * ((len(names) + 2) // 3)
    return (
        f'<svg class="replot" viewBox="0 0 {_W} {height + legend_h}" xmlns="http://www.w3.org/2000/svg">'
        f'<defs><clipPath id="c{id(data)}"><rect x="{_LEFT}" y="{_TOP}" width="{plot.width}" height="{plot.height}"/>'
        f'</clipPath></defs>{_axes(labels, ticks, plot)}<g clip-path="url(#c{id(data)})">{body}</g>'
        f'{_legend(names, height)}</svg>'
    )


def number(text: str) -> float | None:
    match = _NUMBER.search(text.replace(',', '').replace('\u2212', '-'))
    return float(match.group()) if match else None


class _Plot:
    """Maps data values and category indices to SVG coordinates."""

    def __init__(self, height: int, low: float, high: float) -> None:
        self.width = _W - _LEFT - _RIGHT
        self.height = height - _TOP - _BOTTOM
        self.bottom = _TOP + self.height
        self.low, self.high = low, high if high > low else low + 1

    def y(self, value: float) -> float:
        return _TOP + self.height * (self.high - value) / (self.high - self.low)

    def x(self, index: int, count: int) -> float:
        return _LEFT + self.width * (index + 0.5) / count


def _default_ticks(values: list[float], from_zero: bool) -> list[float]:
    low, high = min(values), max(values)
    if from_zero:
        low, high = min(low, 0.0), max(high, 0.0)
    pad = (high - low) * 0.05 or 1
    low, high = (low if from_zero and low == 0 else low - pad), high + pad
    return [low, (low + high) / 2, high]


def _lines(series: list[list[float | None]], plot: _Plot) -> str:
    out = []
    for i, values in enumerate(series):
        points = ' '.join(
            f'{plot.x(j, len(values)):.1f},{plot.y(v):.1f}' for j, v in enumerate(values) if v is not None
        )
        out.append(f'<polyline points="{points}" fill="none" stroke="{_PALETTE[i % 8]}" stroke-width="1.5"/>')
    return ''.join(out)


def _bars(series: list[list[float | None]], plot: _Plot) -> str:
    count, groups = len(series[0]), len(series)
    slot = plot.width / count
    width = slot * 0.8 / groups
    base_value = min(max(0.0, plot.low), plot.high)
    out = []
    for i, values in enumerate(series):
        for j, v in enumerate(values):
            if v is None:
                continue
            x = _LEFT + slot * j + slot * 0.1 + width * i
            top, base = min(plot.y(v), plot.y(base_value)), max(plot.y(v), plot.y(base_value))
            out.append(
                f'<rect x="{x:.1f}" y="{top:.1f}" width="{width:.1f}" height="{base - top:.1f}" '
                f'fill="{_PALETTE[i % 8]}"/>'
            )
    return ''.join(out)


def _axes(labels: list[str], ticks: list[float], plot: _Plot) -> str:
    parts = [
        f'<line x1="{_LEFT}" y1="{_TOP}" x2="{_LEFT}" y2="{plot.bottom}" stroke="#9ca3af"/>',
        f'<line x1="{_LEFT}" y1="{plot.bottom}" x2="{_LEFT + plot.width}" y2="{plot.bottom}" stroke="#9ca3af"/>',
    ]
    for value in ticks:
        y = plot.y(value)
        parts.append(f'<line x1="{_LEFT}" y1="{y:.1f}" x2="{_LEFT + plot.width}" y2="{y:.1f}" stroke="#f3f4f6"/>')
        parts.append(f'<text x="{_LEFT - 4}" y="{y + 3:.1f}" text-anchor="end" font-size="9">{value:g}</text>')
    step = max(1, round(len(labels) / _X_LABELS))
    for index in range(0, len(labels), step):
        parts.append(
            f'<text x="{plot.x(index, len(labels)):.1f}" y="{plot.bottom + 12}" text-anchor="middle" font-size="9">'
            f'{html.escape(labels[index][:10])}</text>'
        )
    return ''.join(parts)


def _legend(names: list[str], top: int) -> str:
    parts = []
    for i, name in enumerate(names):
        x, y = 8 + (i % 3) * 140, top + 4 + 14 * (i // 3)
        parts.append(
            f'<rect x="{x}" y="{y}" width="9" height="9" fill="{_PALETTE[i % 8]}"/>'
            f'<text x="{x + 13}" y="{y + 8}" font-size="9">{html.escape(name[:24])}</text>'
        )
    return ''.join(parts)
