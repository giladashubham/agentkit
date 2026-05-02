from __future__ import annotations

import json
from typing import Any

__all__ = ["client_with_retries", "safe_json_loads", "timeout_seconds"]


def timeout_seconds(timeout_ms: int | None) -> float | None:
    """Convert public millisecond timeout values to SDK timeout seconds."""
    return None if timeout_ms is None else timeout_ms / 1000


def safe_json_loads(value: str | None) -> dict[str, Any]:
    """Parse provider tool-call arguments, returning an empty object on invalid JSON."""
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def client_with_retries(client: Any, max_retries: int | None) -> Any:
    """Return an SDK client configured with per-request retries when supported."""
    if max_retries is None or not hasattr(client, "with_options"):
        return client
    return client.with_options(max_retries=max_retries)
