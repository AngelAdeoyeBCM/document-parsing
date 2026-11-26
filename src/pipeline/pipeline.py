from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from functools import reduce

from inventory import InventoryStore, PdfSource
from parsing.parsers import get_parser

from .config import RunConfig
from .manifest import RunManifest
from .routing import DEFAULT_ROUTE
from .stages import ParseStage, Stage, TextLayerStage, WorkItem, WriteStage

# Disabled stages; restore these imports with them:
# from .components import build_processor, build_triage
# from .routing import Router, SparseTextRoute
# from .stages import PostprocessStage, RouteStage, TriageStage

logger = logging.getLogger(__name__)


class Pipeline:
    """Runs each document through the stages in order and records the outcome in the run manifest."""

    def __init__(self, stages: Sequence[Stage], manifest: RunManifest | None = None) -> None:
        self.stages = tuple(stages)
        self.manifest = manifest

    @classmethod
    def from_config(cls, config: RunConfig, store: InventoryStore | None = None) -> Pipeline:
        """``store``, when given, supplies the text layers the inventory already read."""
        if config.triage or config.routes or config.postprocess:
            raise ValueError(
                f'{config.name}: triage, routes and postprocess stages are disabled (commented out in pipeline/stages.py)'
            )
        stages: list[Stage] = [TextLayerStage(store)]
        # if config.triage:
        #     stages.append(TriageStage(build_triage(config.triage), store))
        base = config.parser
        parsers = {DEFAULT_ROUTE: get_parser(base.name, base.options)}
        # if config.routes:
        #     stages.append(RouteStage(Router([SparseTextRoute(r.name, r.max_chars_per_page) for r in config.routes])))
        #     parsers |= {r.name: get_parser(base.name, {**base.options, **r.options}) for r in config.routes}
        stages.append(ParseStage(parsers))
        # if config.postprocess:
        #     stages.append(PostprocessStage([build_processor(spec) for spec in config.postprocess]))
        stages.append(WriteStage(config.output))
        return cls(stages, manifest=RunManifest(config.output))

    def run(self, sources: Sequence[PdfSource], corpus: Sequence[PdfSource] | None = None) -> Iterator[WorkItem]:
        """Process ``sources``; corpus-level statistics come from ``corpus``, defaulting to ``sources``."""
        for stage in self.stages:
            stage.prepare(corpus if corpus is not None else sources)
        for source in sources:
            # One bad PDF must not stop a 10k-document run; the failure is logged and the run continues.
            try:
                item = reduce(lambda current, stage: stage.run(current), self.stages, WorkItem(source))
            except Exception as error:
                logger.exception('failed: %s', source.path)
                if self.manifest:
                    self.manifest.record_failure(source, error)
                continue
            if self.manifest:
                self.manifest.record_ok(item)
            yield item
