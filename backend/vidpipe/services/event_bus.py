"""In-memory pub/sub event bus for real-time pipeline progress."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid as _uuid
from typing import Any, Union

logger = logging.getLogger(__name__)


class EventBus:
    """Lightweight publish-subscribe bus keyed by project_id.

    Pipeline functions call ``emit(project_id, event_type, **payload)`` to
    broadcast progress events.  WebSocket endpoints ``subscribe`` to receive
    events via an ``asyncio.Queue``.

    ``emit()`` is fire-and-forget (``put_nowait``), never blocks the pipeline.
    If a subscriber's queue is full the event is silently dropped.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, set[asyncio.Queue]] = {}

    @staticmethod
    def _normalize(project_id: Union[str, _uuid.UUID]) -> str:
        """Normalize project_id to hex string (no dashes)."""
        if isinstance(project_id, _uuid.UUID):
            return project_id.hex
        return str(project_id).replace("-", "")

    def subscribe(self, project_id: Union[str, _uuid.UUID]) -> asyncio.Queue:
        key = self._normalize(project_id)
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._subscribers.setdefault(key, set()).add(queue)
        logger.debug("EventBus: subscribed to %s (total=%d)", key, len(self._subscribers[key]))
        return queue

    def unsubscribe(self, project_id: Union[str, _uuid.UUID], queue: asyncio.Queue) -> None:
        key = self._normalize(project_id)
        subs = self._subscribers.get(key)
        if subs:
            subs.discard(queue)
            if not subs:
                del self._subscribers[key]
        logger.debug("EventBus: unsubscribed from %s", key)

    def emit(self, project_id: Union[str, _uuid.UUID], event_type: str, **payload: Any) -> None:
        key = self._normalize(project_id)
        subs = self._subscribers.get(key)
        if not subs:
            return
        event = {"type": event_type, "ts": time.time(), **payload}
        for q in subs:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("EventBus: queue full for %s, dropping %s", key, event_type)


# Module-level singleton
event_bus = EventBus()
