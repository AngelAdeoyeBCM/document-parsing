from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

DEFAULT_ROUTE = 'default'


class Route(ABC):
    """A named condition on a document's text layer that sends it to a different parser configuration."""

    def __init__(self, name: str) -> None:
        if name == DEFAULT_ROUTE:
            raise ValueError(f'route name {DEFAULT_ROUTE!r} is reserved for the base parser')
        self.name = name

    @abstractmethod
    def matches(self, page_texts: Sequence[str]) -> bool: ...


class SparseTextRoute(Route):
    """Matches documents whose text layer averages fewer than ``max_chars_per_page`` characters, i.e. image-only."""

    def __init__(self, name: str, max_chars_per_page: float = 50) -> None:
        super().__init__(name)
        self.max_chars_per_page = max_chars_per_page

    def matches(self, page_texts: Sequence[str]) -> bool:
        chars = sum(len(text.strip()) for text in page_texts)
        return chars / max(1, len(page_texts)) < self.max_chars_per_page


class Router:
    """Picks the first matching route, falling back to the base parser."""

    def __init__(self, routes: Sequence[Route]) -> None:
        self.routes = tuple(routes)

    def route(self, page_texts: Sequence[str]) -> str:
        return next((route.name for route in self.routes if route.matches(page_texts)), DEFAULT_ROUTE)
