from __future__ import annotations

from typing import Any

from .model import Model
from .types import AssistantMessage, Response, Role

__all__ = ["attach_response_metadata"]


def attach_response_metadata(model: Model, response: Response) -> Response:
    """Attach response metadata to assistant messages for context replay/debugging."""
    if response.message.role != Role.ASSISTANT:
        return response

    response_id = _response_id(response.raw)
    response_model = response.model if response.model and response.model != model.id else None
    response.message = AssistantMessage(
        content=response.message.content,
        timestamp=response.message.timestamp,
        provider=str(model.provider),
        api=str(model.api),
        model=model.id,
        response_model=response_model,
        response_id=response_id,
        usage=response.usage.model_dump(mode="json"),
        stop_reason=response.stop_reason.value,
    )
    return response


def _response_id(raw: Any) -> str | None:
    if raw is None:
        return None

    direct = getattr(raw, "id", None) or getattr(raw, "response_id", None)
    if direct:
        return str(direct)

    nested = getattr(raw, "response", None)
    nested_id = getattr(nested, "id", None) if nested is not None else None
    return str(nested_id) if nested_id else None
