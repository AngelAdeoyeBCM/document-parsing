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
    split,
)
from evaluation.score import EVALUATION_DIR, means, read_csv, write_csv
from parsing.core import serialize, serialize_elements


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
    assert scores['order_gap'] == pytest.approx(8 / 14)

    perfect = gold_metrics(gold_body, gold_all, gold_body)
    assert (perfect['loss'], perfect['leak'], perfect['separation'], perfect['order_gap']) == (0.0, 0.0, 1.0, 0.0)
    assert gold_metrics('', gold_all, 'anything') == {}  # nothing expected on this page


def test_gold_metrics_tables_mode_expects_tables_and_still_excludes_figures() -> None:
    gold_tables, gold_all = 'Copper prices rose Revenue 42', 'Copper prices rose Revenue 42 USD per tonne'
    scores = gold_metrics(gold_tables, gold_all, 'Copper prices rose Revenue 42 USD')
    assert scores['loss'] == 0.0
    assert scores['leak_excluded'] == pytest.approx(1 / 6)  # 'usd' is figure text
    assert scores['separation'] == pytest.approx(2 / 3)
    assert scores['order_gap'] == pytest.approx(10 / 11 - 4 / 5)  # tables keep their place in the reading order


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
