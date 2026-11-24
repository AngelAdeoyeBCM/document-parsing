from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..core import ParsedDocument


@dataclass(frozen=True, slots=True)
class ProcessResult:
    document: ParsedDocument
    notes: dict[str, object] = field(default_factory=dict)  # what changed, recorded in run.jsonl


class DocumentProcessor(ABC):
    """Edits a parsed DoclingDocument after parsing, e.g. removing or relinking items."""

    name: str

    @abstractmethod
    def process(self, parsed: ParsedDocument) -> ProcessResult: ...
