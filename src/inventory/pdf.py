"""Applying a document's triage to its PDF: the copy that is parsed or annotated."""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path
from typing import IO

from pypdf import PdfWriter


def truncate_pdf(path: Path, skip: Collection[int], out: Path | IO[bytes]) -> list[int]:
    """Write ``path`` without the trailing pages in ``skip`` to ``out`` (a path or binary stream).

    The whole document is cloned and the skipped pages deleted, rather than the kept pages copied into a new one:
    copying pages leaves the document-level catalog behind, and losing its optional-content settings
    (``/OCProperties``) makes hidden layers render, so a layout model would see content the source hides.

    Returns the page numbers kept, always ``1..k``: tail triage only drops a block ending at the last page, so the
    copy's pages keep their source numbers. Raises ``ValueError`` if ``skip`` is not such a tail.
    """
    writer = PdfWriter(clone_from=path)
    count = len(writer.pages)
    kept = min(skip, default=count + 1) - 1
    if set(skip) != set(range(kept + 1, count + 1)):
        raise ValueError(f'{path}: pages {sorted(skip)} are not a tail of its {count} pages')
    for index in reversed(range(kept, count)):
        writer.remove_page(index, clean=True)  # clean: drop links and outline entries pointing at it
    writer.compress_identical_objects(remove_duplicates=False, remove_unreferenced=True)  # the dropped pages' objects
    writer.write(out)
    return list(range(1, kept + 1))
