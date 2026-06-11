import time
import logging
from typing import Callable, Optional

import tilellm.analytics as analytics
from tilellm.models import ItemSingle
from tilellm.models.schemas import IndexingResult

logger = logging.getLogger(__name__)


class ScrapeSingleService:
    """Orchestrates a single-document indexing run.

    Receives the two indexing callables as constructor parameters (DIP) so the
    service is testable without a live vector store. Measures wall-clock time and
    publishes an analytics event regardless of success or failure (try/finally).
    """

    def __init__(
        self,
        add_item_fn: Optional[Callable] = None,
        add_item_hybrid_fn: Optional[Callable] = None,
    ) -> None:
        if add_item_fn is None:
            from tilellm.controller.controller import add_item
            add_item_fn = add_item
        if add_item_hybrid_fn is None:
            from tilellm.controller.controller import add_item_hybrid
            add_item_hybrid_fn = add_item_hybrid
        self._add_item = add_item_fn
        self._add_item_hybrid = add_item_hybrid_fn

    async def run(self, item: ItemSingle) -> IndexingResult:
        """Run indexing and publish analytics; re-raises any exception."""
        t0 = time.monotonic()
        error_msg: Optional[str] = None
        result: Optional[IndexingResult] = None
        try:
            if item.hybrid:
                result = await self._add_item_hybrid(item)
            else:
                result = await self._add_item(item)
            return result
        except Exception as exc:
            error_msg = str(exc)
            raise
        finally:
            duration_ms = int((time.monotonic() - t0) * 1000)
            source_type = item.type.value if item.type else None
            event_type, payload = analytics.events.content_indexed(
                kb_id=item.namespace,
                kb_name=item.namespace,
                embedding_model=analytics.events.get_embedding_model_name(item.embedding),
                engine=analytics.events.get_engine_value(item.engine),
                duration_ms=duration_ms,
                success=error_msg is None,
                source_url=item.source,
                source_type=source_type,
                chunks_indexed=(result.chunks or 0) if result is not None else 0,
                error_message=error_msg,
                request_id=item.request_id,
            )
            analytics.publish_nowait(event_type, item.id_project, payload)
