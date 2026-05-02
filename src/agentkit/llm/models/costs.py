from __future__ import annotations

from ..model import Model
from ..types import Cost, Usage

__all__ = ["calculate_cost"]

TOKENS_PER_MILLION = 1_000_000


def calculate_cost(model: Model, usage: Usage) -> Cost:
    """Calculate USD cost for usage using model pricing per 1M tokens.

    The returned ``Cost`` is also assigned to ``usage.cost`` for convenience.
    Unknown prices default to zero via ``Model.cost``.
    """
    usage.cost = Cost(
        input=(model.cost.input / TOKENS_PER_MILLION) * usage.input,
        output=(model.cost.output / TOKENS_PER_MILLION) * usage.output,
        cache_read=(model.cost.cache_read / TOKENS_PER_MILLION) * usage.cache_read,
        cache_write=(model.cost.cache_write / TOKENS_PER_MILLION) * usage.cache_write,
    )
    usage.cost.total = (
        usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write
    )
    return usage.cost
