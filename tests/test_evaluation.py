from __future__ import annotations

import math
from pathlib import Path

import pytest
from docling_core.types.doc import (
    BoundingBox,
    ContentLayer,
    DocItemLabel,
    DoclingDocument,
    ProvenanceItem,
    Size,
    TableCell,
    TableData,
)

from evaluation import (
    compare,
    compare_runs,
    dropped_text_rate,
    garble_rate,
    gold_metrics,
    label_confusions,
    layout_f1,
    split,
)
from evaluation.score import EVALUATION_DIR, means, read_csv, write_csv
from parsing.core import SERIALIZATIONS, serialize, serialize_document, serialize_elements


def _prov(page: int = 1) -> ProvenanceItem:
    return ProvenanceItem(page_no=page, bbox=BoundingBox(l=0, t=10, r=10, b=0), charspan=(0, 1))


def test_serializations_are_docling_serializers_filtered_by_label_and_layer() -> None:
    doc = DoclingDocument(name='d')
    for page in (1, 2):
        doc.add_page(page_no=page, size=Size(width=100, height=100))
    doc.add_title('Copper outlook', prov=_prov())
    doc.add_text(DocItemLabel.TEXT, 'Prices rose.', prov=_prov())
    doc.add_text(DocItemLabel.PAGE_FOOTER, 'Page 1', prov=_prov(), content_layer=ContentLayer.FURNITURE)
    caption = doc.add_text(DocItemLabel.CAPTION, 'Figure 1', prov=_prov())
    picture = doc.add_picture(prov=_prov(), caption=caption)
    doc.add_text(DocItemLabel.TEXT, 'axis label', prov=_prov(), parent=picture)
    cell = TableCell(
        text='42', start_row_offset_idx=0, end_row_offset_idx=1, start_col_offset_idx=0, end_col_offset_idx=1
    )
    doc.add_table(data=TableData(num_rows=1, num_cols=1, table_cells=[cell]), prov=_prov())
    doc.add_heading('Second page heading', prov=_prov(2))

    def lines(name: str, page: int = 1) -> list[str]:
        return [line for line in serialize(doc, name)[page].splitlines() if line.strip()]

    assert lines('body') == ['Copper outlook', 'Prices rose.'] and lines('body', 2) == ['Second page heading']
    assert lines('body_tables')[:2] == ['Copper outlook', 'Prices rose.'] and '42' in lines('body_tables')[2]
    assert {'Page 1', 'Figure 1', 'axis label'} <= set(lines('extracted'))  # furniture, captions and picture text
    assert lines('markdown')[0] == '# Copper outlook' and '<!-- image -->' in lines('markdown')
    with pytest.raises(KeyError, match='available'):
        serialize(doc, 'nope')
    whole = serialize_document(doc, 'body')  # one file per document: each page after a marker
    assert whole.startswith('<!-- page 1 -->\n\nCopper outlook') and '<!-- page 2 -->\n\nSecond page heading' in whole
    assert (SERIALIZATIONS['body'].suffix, SERIALIZATIONS['markdown'].suffix) == ('.txt', '.md')

    elements = serialize_elements(doc)  # 'extracted', one element kind at a time
    assert set(elements) == {'caption', 'page_footer', 'picture_text', 'section_header', 'table', 'text', 'title'}
    assert (elements['text'], elements['picture_text'], elements['page_footer']) == (
        {1: 'Prices rose.'},
        {1: 'axis label'},  # inside the picture, so not counted as body 'text'
        {1: 'Page 1'},
    )


def test_text_quality_needs_no_gold() -> None:
    broken = 'T' + chr(0xA8) + ' urkiye (cid:3) fine words'  # a detached diaeresis and an unmapped glyph
    assert garble_rate(broken) == pytest.approx(2 / 5)
    assert garble_rate('ﬁnancial control\x02char Türkiye Ação') == pytest.approx(2 / 4)  # ligature, control
    assert math.isnan(garble_rate(''))
    assert dropped_text_rate('Copper prices rose', 'Copper prices rose sharply') == pytest.approx(1 / 4)
    assert dropped_text_rate('eco-\nnomic', 'economic') == 0.0  # hyphenation across a line break is joined


def test_gold_metrics_body_mode() -> None:
    gold_body, gold_all = 'Copper prices rose in Türkiye', 'Copper prices rose in Türkiye Revenue 42'
    scores = gold_metrics(gold_body, gold_all, 'Copper prices rose in Turkiye Revenue 42 Page 1')
    assert scores['loss'] == pytest.approx(1 / 5)  # Türkiye lost its diacritic
    assert scores['leak'] == pytest.approx(5 / 9)
    assert scores['leak_excluded'] == pytest.approx(2 / 9)  # gold text this mode leaves out: the table
    assert scores['leak_other'] == pytest.approx(3 / 9)  # 'turkiye', 'page', '1' are nowhere in the gold
    assert scores['separation'] == 0.0
    assert scores['order_gap'] == 0.0  # the shared words are in order; loss and leak are not disorder

    perfect = gold_metrics(gold_body, gold_all, gold_body)
    assert (perfect['loss'], perfect['leak'], perfect['separation'], perfect['order_gap']) == (0.0, 0.0, 1.0, 0.0)
    assert gold_metrics('', gold_all, 'anything') == {}  # nothing expected on this page


def test_gold_metrics_tables_mode_expects_tables_and_still_excludes_figures() -> None:
    gold_tables, gold_all = 'Copper prices rose Revenue 42', 'Copper prices rose Revenue 42 USD per tonne'
    scores = gold_metrics(gold_tables, gold_all, 'Copper prices rose Revenue 42 USD')
    assert scores['loss'] == 0.0
    assert scores['leak_excluded'] == pytest.approx(1 / 6)  # 'usd' is figure text
    assert scores['separation'] == pytest.approx(2 / 3)
    assert scores['order_gap'] == 0.0  # tables keep their place in the reading order
    moved = gold_metrics(gold_tables, gold_all, 'Revenue 42 Copper prices rose')  # table read before the prose
    assert moved['order_gap'] == pytest.approx(1 - 3 / 5)  # 3 of the 5 shared words stay in order


def test_order_gap_ignores_lost_and_leaked_words() -> None:
    gold = ' '.join(f'w{i}' for i in range(100))
    leaked = gold + ' ' + ' '.join(f'x{i}' for i in range(50))
    lost = ' '.join(f'w{i}' for i in range(50))
    columns_swapped = ' '.join(f'w{i}' for i in [*range(50, 100), *range(50)])
    assert gold_metrics(gold, gold, leaked)['order_gap'] == gold_metrics(gold, gold, lost)['order_gap'] == 0.0
    assert gold_metrics(gold, gold, columns_swapped)['order_gap'] == pytest.approx(0.5)
    assert math.isnan(gold_metrics(gold, gold, 'nothing shared')['order_gap'])


def _boxed(*items: tuple[DocItemLabel, tuple[float, float, float, float]], scale: float = 1.0) -> DoclingDocument:
    """A one-page document with an item per (label, (left, top, right, bottom)) box, top-left origin, on a
    100 x 100 page; ``scale`` multiplies page and boxes, as when a page is measured in pixels instead of points."""
    doc = DoclingDocument(name='d')
    doc.add_page(page_no=1, size=Size(width=100 * scale, height=100 * scale))
    for label, (left, top, right, bottom) in items:
        box = BoundingBox(l=left * scale, t=top * scale, r=right * scale, b=bottom * scale)
        doc.add_text(label, 'x', prov=ProvenanceItem(page_no=1, bbox=box, charspan=(0, 1)))
    return doc


def test_layout_f1_matches_boxes_of_the_same_label() -> None:
    text, footnote = DocItemLabel.TEXT, DocItemLabel.FOOTNOTE
    gold = _boxed((text, (0, 0, 50, 10)), (text, (0, 20, 50, 30)), (footnote, (0, 90, 50, 100)))
    assert layout_f1(gold, gold, 1)['f1'] == 1.0
    # The second text box is shifted to IoU 0.25, and the footnote is labelled as text.
    run = _boxed((text, (0, 0, 50, 10)), (text, (0, 26, 50, 36)), (text, (0, 90, 50, 100)))
    scores = layout_f1(gold, run, 1)
    assert scores['f1.text'] == pytest.approx(2 * 1 / (2 + 3)) and scores['f1.footnote'] == 0.0
    assert scores['f1'] == pytest.approx(0.2)
    assert layout_f1(gold, run, 2) == {}  # no such page
    in_pixels = _boxed((text, (0, 0, 50, 10)), (text, (0, 20, 50, 30)), (footnote, (0, 90, 50, 100)), scale=2.78)
    assert layout_f1(in_pixels, gold, 1)['f1'] == pytest.approx(1.0)  # same page, other units


def test_layout_f1_scores_a_gold_title_as_a_section_header() -> None:
    gold = _boxed((DocItemLabel.TITLE, (0, 0, 50, 10)))
    run = _boxed((DocItemLabel.SECTION_HEADER, (0, 0, 50, 10)))  # Docling relabels every title
    assert layout_f1(gold, run, 1) == {'f1.section_header': 1.0, 'f1': 1.0}


def test_label_confusions_say_where_body_text_went() -> None:
    gold_body, gold_groups = 'Copper prices rose sharply', {'footnote': 'Source Bloomberg', 'caption': 'Figure 1'}
    run_body, run_groups = 'Copper prices rose Source Bloomberg', {'caption': 'sharply Figure 1'}
    scores = label_confusions(gold_body, gold_groups, run_body, run_groups)
    assert scores['leak_from.footnote'] == pytest.approx(2 / 5)  # the footnote was labelled as body text
    assert scores['loss_to.caption'] == pytest.approx(1 / 4)  # 'sharply' was put in a caption
    assert scores['leak_from.caption'] == scores['loss_to.footnote'] == scores['leak_from.table'] == 0.0
    assert label_confusions('', {}, 'anything', {}) == {}  # no gold body on this page


def test_split_is_fixed_by_content_hash() -> None:
    assert split('0' * 64) == 'holdout' and split('f' * 64) == 'screen'


def test_means_are_overall_and_per_split() -> None:
    rows = [
        {'doc_id': 'a', 'page': 1, 'split': 'screen', 'body.loss': 0.2},
        {'doc_id': 'b', 'page': 1, 'split': 'holdout', 'body.loss': 0.4},
        {'doc_id': 'b', 'page': 2, 'split': 'holdout', 'body.loss': math.nan},  # ignored
    ]
    assert means(rows) == pytest.approx({'body.loss': 0.3, 'screen/body.loss': 0.2, 'holdout/body.loss': 0.4})


def _rows(values: dict[tuple[str, str], str], split_name: str = 'screen') -> list[dict[str, str]]:
    return [{'doc_id': doc, 'page': page, 'split': split_name, 'body.loss': v} for (doc, page), v in values.items()]


def test_compare_pairs_pages_and_resamples_documents() -> None:
    base = _rows({('a', '1'): '0.5', ('a', '2'): '0.5', ('b', '1'): '0.4', ('c', '1'): 'nan'})
    other = _rows({('a', '1'): '0.3', ('a', '2'): '0.3', ('b', '1'): '0.3', ('c', '1'): '0.1'})
    result = compare(base, other, 'body.loss')
    assert (result['documents'], result['pages'], result['lower']) == (2, 3, 3)  # the NaN pair is dropped
    assert result['diff'] == pytest.approx(-0.5 / 3)
    assert result['low'] <= result['diff'] <= result['high'] < 0
    assert compare(base, other, 'missing') is None


def test_compare_runs_writes_into_the_other_runs_evaluation(tmp_path: Path) -> None:
    base, other = tmp_path / 'p0', tmp_path / 'b0'
    write_csv(
        _rows({('a', '1'): '0.5', ('b', '1'): '0.4'}) + _rows({('h', '1'): '0.9'}, 'holdout'),
        base / 'evaluation/pages.csv',
    )
    write_csv(
        _rows({('a', '1'): '0.3', ('b', '1'): '0.3'}) + _rows({('h', '1'): '0.0'}, 'holdout'),
        other / 'evaluation/pages.csv',
    )
    strata = tmp_path / 'strata.csv'
    strata.write_text('doc_id,page,tags\nb,,garbled\n', encoding='utf-8')

    path = compare_runs(base, other, ['body.loss'], split='screen', strata=strata)
    assert path == other / EVALUATION_DIR / 'compare_p0.csv'
    results = {row['stratum']: row for row in read_csv(path)}
    assert results['all']['documents'] == '2'  # the held-out document is left out
    assert float(results['garbled']['diff']) == pytest.approx(-0.1)
