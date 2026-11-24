from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from pypdf import PdfReader

from inventory import PdfSource


@dataclass(frozen=True, slots=True)
class Reference:
    """The source PDF's text layer per page: the yardstick for metrics that need no gold labels."""

    pages: Mapping[int, str]

    @classmethod
    def from_source(cls, source: PdfSource) -> Reference:
        reader = PdfReader(source.path)
        return cls(MappingProxyType({n: page.extract_text() or '' for n, page in enumerate(reader.pages, start=1)}))

    def text(self, numbers: Iterable[int]) -> str:
        return '\n'.join(self.pages.get(number, '') for number in numbers)
