"""Compatibility helpers for JSON fields stored as TEXT in older databases."""

from __future__ import annotations

import json
from typing import Any


def normalize_json_value(value: Any) -> Any:
    """Parse stringified JSON values while leaving native Python values alone."""
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def normalize_json_dict(value: Any) -> dict[str, Any] | None:
    """Return a dict when the value represents a JSON object."""
    normalized = normalize_json_value(value)
    if isinstance(normalized, dict):
        return normalized
    return None


def normalize_json_list(value: Any) -> list[Any]:
    """Return a list when the value represents a JSON array."""
    normalized = normalize_json_value(value)
    if isinstance(normalized, list):
        return normalized
    return []
