"""Read the CVAT ground truth back as DoclingDocuments, keyed by the source PDF's content hash.

Annotated PDFs are tail-truncated copies (``inventory.truncate_pdf``) named ``<hash_id>.pdf``: their pages are the
source's leading pages, so the ground truth's page numbers are already the source's.
"""

from __future__ import annotations

import json
import logging

from docling_core.types.doc import DoclingDocument

from .workspace import Workspace

logger = logging.getLogger(__name__)

Gold = DoclingDocument  # the ground truth of one document, holding its annotated pages


def load_gold(workspace: Workspace) -> dict[str, Gold]:
    """Every annotated document in the workspace by the source PDF's hash_id, numbered like the source.

    docling-eval names a record by file stem (the hash_id) or by the truncated file's content hash depending on the
    stage, so both are matched.
    """
    import pyarrow.parquet as pq  # optional dependency: only reading gold needs it

    documents = workspace.documents()
    by_truncated_hash = {document['truncated_hash']: hash_id for hash_id, document in documents.items()}
    gold: dict[str, Gold] = {}
    for path in sorted(workspace.gold.rglob('*.parquet')):
        names = pq.read_schema(path).names
        wanted = [name for name in ('document_id', 'GroundTruthDocument', 'document_filehash') if name in names]
        table = pq.read_table(path, columns=wanted)  # leaves the page images and PDF bytes unread
        filehashes = table['document_filehash'].to_pylist() if 'document_filehash' in wanted else [None] * len(table)
        for record_id, document, filehash in zip(
            table['document_id'].to_pylist(), table['GroundTruthDocument'].to_pylist(), filehashes, strict=True
        ):
            hash_id = record_id if record_id in documents else None
            hash_id = hash_id or by_truncated_hash.get(record_id) or by_truncated_hash.get(filehash)
            if hash_id is None:
                logger.warning('gold record %r matches no document in %s', record_id, workspace.documents_file)
                continue
            if hash_id in gold:
                raise ValueError(f'{workspace.gold}: document {hash_id!r} appears twice')
            pages = documents[hash_id]['pages']
            if pages != list(range(1, len(pages) + 1)):
                raise ValueError(f'{hash_id}: annotated pages {pages} are not the leading pages; build a new dataset')
            gold[hash_id] = DoclingDocument.model_validate(json.loads(document))
    return gold
