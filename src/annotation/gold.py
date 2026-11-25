"""Read the CVAT ground truth back as DoclingDocuments numbered like the source PDFs."""

from __future__ import annotations

import json
import logging

from docling_core.types.doc import DoclingDocument

from parsing.parsers.model import renumber

from .workspace import Workspace

logger = logging.getLogger(__name__)

Gold = tuple[str, DoclingDocument]  # (content hash of the source PDF, ground truth holding the annotated pages)


def load_gold(workspace: Workspace) -> dict[str, Gold]:
    """Every annotated document in the workspace by doc_id, its pages renumbered to the source PDF's.

    docling-eval names a record by file stem or by content hash depending on the stage, so both are matched.
    """
    import pyarrow.parquet as pq  # optional dependency: only reading gold needs it

    documents = workspace.documents()
    by_hash = {document['subset_hash']: doc_id for doc_id, document in documents.items()}
    gold: dict[str, Gold] = {}
    for path in sorted(workspace.gold.rglob('*.parquet')):
        names = pq.read_schema(path).names
        wanted = [name for name in ('document_id', 'GroundTruthDocument', 'document_filehash') if name in names]
        table = pq.read_table(path, columns=wanted)  # leaves the page images and PDF bytes unread
        hashes = table['document_filehash'].to_pylist() if 'document_filehash' in wanted else [None] * len(table)
        for record_id, document, filehash in zip(
            table['document_id'].to_pylist(), table['GroundTruthDocument'].to_pylist(), hashes, strict=True
        ):
            doc_id = record_id if record_id in documents else by_hash.get(record_id) or by_hash.get(filehash)
            if doc_id is None:
                logger.warning('gold record %r matches no document in %s', record_id, workspace.documents_file)
                continue
            if doc_id in gold:
                raise ValueError(f'{workspace.gold}: document {doc_id!r} appears twice')
            truth = renumber(DoclingDocument.model_validate(json.loads(document)), documents[doc_id]['pages'])
            gold[doc_id] = (documents[doc_id]['hash_id'], truth)
    return gold
