from __future__ import annotations

from .generated import iter_builtin_models
from .registry import register_model

__all__ = ["register_builtin_models"]


def register_builtin_models() -> None:
    """Register the built-in model catalog."""
    for model in iter_builtin_models():
        register_model(model)
