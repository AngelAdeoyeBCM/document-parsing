from inventory import Inventory


def test_inventory_profiles_text_and_image_pages(tmp_path, make_pdf):
    make_pdf('corpus/text.pdf', [
        ['Enough meaningful text on this page to exceed the configured sparse threshold for embedded text'],
        ['More useful text should also be sufficiently dense for the text-rich document classification'],
    ])
    make_pdf('corpus/scan.pdf', [[], []])

    inventory = Inventory(tmp_path / 'corpus', tmp_path / 'inventory.db')
    profiles = inventory.build()

    by_id = {
        profile.profile.relative_path.rsplit('/', 1)[-1].removesuffix('.pdf'): profile
        for profile in profiles
    }
    assert by_id['text'].text_quality == 'embedded_text'
    assert by_id['text'].do_ocr is False
    assert by_id['scan'].text_quality == 'image_only'
    assert by_id['scan'].do_ocr is True

    text_sources = inventory.sources(include_ocr=False)
    assert [source.doc_id for source in text_sources] == ['text']


def test_inventory_hash_deduplicates_duplicate_content(tmp_path, make_pdf):
    make_pdf('corpus/a.pdf', [['Same text']])
    make_pdf('corpus/b.pdf', [['Same text']])

    inventory = Inventory(tmp_path / 'corpus', tmp_path / 'inventory.db')
    profiles = inventory.build()

    assert len(profiles) == 2
    assert profiles[0].profile.hash_id == profiles[1].profile.hash_id


def test_inventory_keeps_invalid_pdf_for_pipeline_failure(tmp_path):
    (tmp_path / 'corpus').mkdir()
    (tmp_path / 'corpus/broken.pdf').write_bytes(b'not a pdf')

    inventory = Inventory(tmp_path / 'corpus', tmp_path / 'inventory.db')
    assert inventory.build() == []
    assert [source.doc_id for source in inventory.sources()] == ['broken']
