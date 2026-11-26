"""Stage 1.1 metrics for one page, by target.

A. Text quality, on everything the parser extracted: garble rate and dropped text against the pypdf text layer
need no gold; ``gold_metrics`` on the 'extracted' serialization gives word loss against the gold full-page text.
B. Layout, on the labels: ``layout_f1`` matches labelled boxes against the gold boxes, and ``label_confusions``
says which labels body text leaked from or was lost to.
C. Serializer, on the body text: ``gold_metrics`` scores loss, leak, separation and order gap against the gold
document serialized the same way.
Words are compared as multisets after the same normalisation, so order only enters order gap, which looks only at
the words output and gold share.
"""

from __future__ import annotations

import math
import re
import statistics
import unicodedata
from collections import Counter
from collections.abc import Mapping

from docling_core.types.doc import BoundingBox, ContentLayer, DocItem, DoclingDocument

from parsing.core import PICTURE_TEXT

_WORD = re.compile(r'[^\W_]+')
_HYPHEN_BREAK = re.compile(r'(?<=[^\W\d_])-\r?\n\s*(?=[^\W\d_])')  # 'eco-\nnomic', not '2024-\n25'
# Replacement character, unmapped glyphs (cid:n), private-use code points, control characters (not whitespace),
# unexpanded ligature characters (U+FB00-FB06, e.g. '\ufb01'), spacing or stray combining diacritics.
_GARBLE = re.compile(
    r'\ufffd|\(cid:\d+\)|[\ue000-\uf8ff\x00-\x08\x0e-\x1f\x7f-\x9f\ufb00-\ufb06\u00a8\u00b4\u00b8\u02c6-\u02dd\u0300-\u036f]'
)
# Targeted confusions: groups of element kinds (``serialize_elements``) that body text gets confused with, in the
# order a word is assigned to them.
CONFUSION_GROUPS = {
    'footnote': ('footnote',),
    'caption': ('caption',),
    'furniture': ('page_header', 'page_footer'),
    'table': ('table',),
    'picture': (PICTURE_TEXT,),
}
IOU_MATCH = 0.5
# Labels scored as another in layout F1: Docling's layout postprocessor relabels every TITLE as SECTION_HEADER, so
# runs never output TITLE, and gold titles would otherwise score 0 for every run.
LAYOUT_ALIASES = {'title': 'section_header'}


def normalise(text: str) -> str:
    """NFC, with words hyphenated across a line break joined."""
    return unicodedata.normalize('NFC', _HYPHEN_BREAK.sub('', text))


def words(text: str) -> list[str]:
    return [word.lower() for word in _WORD.findall(normalise(text))]


def garble_rate(text: str) -> float:
    """Share of whitespace-separated tokens holding a broken character."""
    tokens = normalise(text).split()
    return sum(1 for token in tokens if _GARBLE.search(token)) / len(tokens) if tokens else math.nan


def dropped_text_rate(output: str, reference: str) -> float:
    """Share of the reference words (the pypdf text layer) missing from the output."""
    return 1 - _recall(Counter(words(reference)), Counter(words(output)))


def gold_metrics(expected: str, gold_all: str, output: str) -> dict[str, float]:
    """One page of a run's output against the gold page under the same serialization (``expected``).

    ``gold_all`` is all the gold page's text (its 'extracted' serialization): what it holds beyond ``expected``
    (captions, footers, picture text, tables when the serialization drops them) is what the serialization
    excludes, tracked by ``leak_excluded`` and ``separation``. ``order_gap`` is the share of the shared words that
    are out of order, 1 - LCS / shared words, so lost or leaked words do not count as disorder.
    """
    want_words, produced = words(expected), words(output)
    if not want_words:
        return {}
    want, out = Counter(want_words), Counter(produced)
    excluded = Counter(words(gold_all)) - want
    leaked = out - want
    leaked_excluded = sum((leaked & excluded).values())
    n = len(produced) or 1
    shared = sum((want & out).values())
    return {
        'loss': 1 - _recall(want, out),
        'leak': sum(leaked.values()) / n,
        'leak_excluded': leaked_excluded / n,
        'leak_other': (sum(leaked.values()) - leaked_excluded) / n,  # text the gold page does not hold at all
        'separation': 1 - _recall(excluded, leaked),
        'order_gap': 1 - _lcs(want_words, produced) / shared if shared else math.nan,
    }


def layout_f1(gold: DoclingDocument, run: DoclingDocument, page: int) -> dict[str, float]:
    """Per-label F1 of the page's labelled boxes against the gold boxes (``f1.<label>``), and their mean (``f1``).

    A box matches a gold box of the same label at IoU >= 0.5, greedily from the highest IoU; F1 is
    2 * matches / (gold boxes + run boxes). Items nested in pictures are skipped, as the gold does not box them.
    Labels in ``LAYOUT_ALIASES`` count as their alias (a title as a section header).
    """
    expected, found = _boxes(gold, page), _boxes(run, page)
    scores = {}
    for label in sorted(set(expected) | set(found)):
        want, got = expected.get(label, []), found.get(label, [])
        scores[f'f1.{label}'] = 2 * _matches(want, got) / (len(want) + len(got))
    return scores | {'f1': statistics.fmean(scores.values())} if scores else {}


def label_confusions(
    gold_body: str, gold_groups: Mapping[str, str], run_body: str, run_groups: Mapping[str, str]
) -> dict[str, float]:
    """Which labels the page's body text was confused with, per ``CONFUSION_GROUPS`` group.

    ``leak_from.<group>``: share of the run's body words that are gold words of that group (e.g. a footnote
    labelled as text). ``loss_to.<group>``: share of the gold body words, missing from the run's body, that the run
    put in that group (e.g. body text labelled as a caption). Each word counts towards one group at most.
    """
    want, out = Counter(words(gold_body)), Counter(words(run_body))
    if not want:
        return {}
    scores = {}
    for direction, unmatched, groups, total in (
        ('leak_from', out - want, gold_groups, sum(out.values())),
        ('loss_to', want - out, run_groups, sum(want.values())),
    ):
        for group in CONFUSION_GROUPS:
            hit = unmatched & Counter(words(groups.get(group, '')))
            unmatched -= hit
            scores[f'{direction}.{group}'] = sum(hit.values()) / total if total else math.nan
    return scores


def _boxes(doc: DoclingDocument, page: int) -> dict[str, list[BoundingBox]]:
    """The page's item boxes by label, top-left origin, scaled to a unit page so documents whose page sizes are in
    different units (PDF points, image pixels) still compare; items nested in pictures are not visited."""
    if page not in doc.pages:
        return {}
    size = doc.pages[page].size
    boxes: dict[str, list[BoundingBox]] = {}
    for item, _ in doc.iterate_items(page_no=page, included_content_layers=set(ContentLayer)):
        if isinstance(item, DocItem):
            for prov in item.prov:
                if prov.page_no == page:
                    box = prov.bbox.to_top_left_origin(size.height).normalized(size)
                    label = LAYOUT_ALIASES.get(item.label.value, item.label.value)
                    boxes.setdefault(label, []).append(box)
    return boxes


def _matches(expected: list[BoundingBox], found: list[BoundingBox]) -> int:
    """Greedy one-to-one matches at IoU >= ``IOU_MATCH``, highest IoU first."""
    pairs = sorted(
        ((a.intersection_over_union(b), i, j) for i, a in enumerate(expected) for j, b in enumerate(found)),
        reverse=True,
    )
    taken_expected: set[int] = set()
    taken_found: set[int] = set()
    for iou, i, j in pairs:
        if iou < IOU_MATCH:
            break
        if i not in taken_expected and j not in taken_found:
            taken_expected.add(i)
            taken_found.add(j)
    return len(taken_expected)


def _recall(expected: Counter[str], produced: Counter[str]) -> float:
    total = sum(expected.values())
    return sum((expected & produced).values()) / total if total else math.nan


def _lcs(first: list[str], second: list[str]) -> int:
    """Length of the longest common subsequence: the most words the two texts hold in the same order."""
    previous = [0] * (len(second) + 1)
    for a in first:
        current = [0]
        for j, b in enumerate(second, start=1):
            current.append(previous[j - 1] + 1 if a == b else max(previous[j], current[j - 1]))
        previous = current
    return previous[-1]
