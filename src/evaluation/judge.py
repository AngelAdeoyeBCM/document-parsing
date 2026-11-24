from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import pypdfium2 as pdfium

from parsing.core import ParsedDocument

from .client import VlmClient

PROMPT = """You compare two extractions (A and B) of the attached PDF page, made for a retrieval system.
Judge them against the page image on: completeness of the text, reading order, table values and layout,
heading structure, figure captions, and leftover page headers, footers or disclaimers.
Reply with JSON only: {"winner": "A" | "B" | "tie", "reason": "<one sentence>"}

## Extraction A
<<A>>

## Extraction B
<<B>>
"""


@dataclass(frozen=True, slots=True)
class Verdict:
    doc_id: str
    page: int
    first: str
    second: str
    winner: str  # ``first``, ``second`` or 'tie'
    reasons: tuple[str, str]


class PageImager:
    def __init__(self, scale: float = 2.0) -> None:
        self.scale = scale

    def png(self, path: Path, page: int) -> bytes:
        pdf = pdfium.PdfDocument(path)
        try:
            image = pdf[page - 1].render(scale=self.scale).to_pil()
        finally:
            pdf.close()
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return buffer.getvalue()


class PairwiseJudge:
    """Asks a vision model which of two parses better matches the page, in both orders to cancel position bias."""

    def __init__(self, client: VlmClient, imager: PageImager | None = None) -> None:
        self.client = client
        self.imager = imager or PageImager()

    def judge(self, first: tuple[str, ParsedDocument], second: tuple[str, ParsedDocument], page: int) -> Verdict:
        (first_run, first_doc), (second_run, second_doc) = first, second
        image = self.imager.png(first_doc.source.path, page)
        a, b = self._markdown(first_doc, page), self._markdown(second_doc, page)
        forward, forward_reason = self._ask(a, b, image)
        backward, backward_reason = self._ask(b, a, image)
        outcomes = {'A': first_run, 'B': second_run, 'tie': 'tie'}
        swapped = {'A': second_run, 'B': first_run, 'tie': 'tie'}
        winner = outcomes[forward] if outcomes[forward] == swapped[backward] else 'tie'
        return Verdict(first_doc.source.doc_id, page, first_run, second_run, winner, (forward_reason, backward_reason))

    def _ask(self, a: str, b: str, image: bytes) -> tuple[str, str]:
        reply = self.client.ask(PROMPT.replace('<<A>>', a).replace('<<B>>', b), [image])
        try:
            payload = json.loads(reply[reply.find('{') : reply.rfind('}') + 1])
        except ValueError:
            return 'tie', f'unparseable reply: {reply[:200]}'
        if not isinstance(payload, dict):
            return 'tie', f'unexpected reply: {reply[:200]}'
        winner = payload.get('winner')
        return (winner if winner in ('A', 'B', 'tie') else 'tie'), str(payload.get('reason', ''))

    @staticmethod
    def _markdown(parsed: ParsedDocument, page: int) -> str:
        if page not in parsed.document.pages:
            return '(page skipped by the parser)'
        return parsed.document.export_to_markdown(page_no=page) or '(empty)'
