from .builtins import register_builtin_models
from .presets import OPENAI_COMPATIBLE_BASE_URLS, openai_compatible_model
from .registry import get_model, get_models, list_model_providers, register_model

__all__ = [
    "get_model",
    "get_models",
    "list_model_providers",
    "register_model",
    "register_builtin_models",
    "OPENAI_COMPATIBLE_BASE_URLS",
    "openai_compatible_model",
]
