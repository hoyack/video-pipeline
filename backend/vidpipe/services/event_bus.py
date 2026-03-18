"""In-memory pub/sub event bus for real-time pipeline progress."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid as _uuid
from typing import Any, Union

logger = logging.getLogger(__name__)
_TASK_LOG_DETAIL_LIMIT = 12000


class EventBus:
    """Lightweight publish-subscribe bus keyed by scene_id.

    Pipeline functions call ``emit(scene_id, event_type, **payload)`` to
    broadcast progress events.  WebSocket endpoints ``subscribe`` to receive
    events via an ``asyncio.Queue``.

    ``emit()`` is fire-and-forget (``put_nowait``), never blocks the pipeline.
    If a subscriber's queue is full the event is silently dropped.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, set[asyncio.Queue]] = {}

    @staticmethod
    def _normalize(scene_id: Union[str, _uuid.UUID]) -> str:
        """Normalize scene_id to hex string (no dashes)."""
        if isinstance(scene_id, _uuid.UUID):
            return scene_id.hex
        return str(scene_id).replace("-", "")

    def subscribe(self, scene_id: Union[str, _uuid.UUID]) -> asyncio.Queue:
        key = self._normalize(scene_id)
        queue: asyncio.Queue = asyncio.Queue(maxsize=1024)
        self._subscribers.setdefault(key, set()).add(queue)
        logger.debug("EventBus: subscribed to %s (total=%d)", key, len(self._subscribers[key]))
        return queue

    def unsubscribe(self, scene_id: Union[str, _uuid.UUID], queue: asyncio.Queue) -> None:
        key = self._normalize(scene_id)
        subs = self._subscribers.get(key)
        if subs:
            subs.discard(queue)
            if not subs:
                del self._subscribers[key]
        logger.debug("EventBus: unsubscribed from %s", key)

    def emit(self, scene_id: Union[str, _uuid.UUID], event_type: str, **payload: Any) -> None:
        key = self._normalize(scene_id)
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


def _truncate_detail(detail: str | None, limit: int = _TASK_LOG_DETAIL_LIMIT) -> str | None:
    if detail is None or len(detail) <= limit:
        return detail
    omitted = len(detail) - limit
    return f"{detail[:limit]}\n\n...[truncated {omitted} characters]"


def emit_task_log(
    scene_id: Union[str, _uuid.UUID],
    *,
    summary: str,
    detail: str | None = None,
    phase: str | None = None,
    shot_index: int | None = None,
    level: str = "info",
    kind: str | None = None,
    source: str | None = None,
) -> None:
    """Emit a verbose task-log event for the Scene Edit live feed."""
    event_bus.emit(
        scene_id,
        "task_log",
        phase=phase,
        shot_index=shot_index,
        level=level,
        kind=kind,
        source=source,
        summary=summary,
        detail=_truncate_detail(detail),
    )
