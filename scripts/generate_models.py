#!/usr/bin/env python3
"""Generate AgentKit's built-in model catalog.

This is a maintainer-time tool. It performs network requests and writes a
committed Python module; it is not run during package installation.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
OVERRIDES_PATH = ROOT / "data" / "model_overrides.json"
OUTPUT_PATH = ROOT / "src" / "agentkit" / "llm" / "models" / "generated.py"
MODELS_DEV_URL = "https://models.dev/api.json"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
USER_AGENT = "agentkit-model-generator"


@dataclass(slots=True)
class CatalogModel:
    provider: str
    api: str
    id: str
    name: str | None = None
    base_url: str | None = None
    context_window: int | None = None
    max_tokens: int | None = None
    input_types: tuple[str, ...] = ("text",)
    reasoning: bool = False
    cost: dict[str, float] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str]:
        return (self.provider, self.id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if generated.py is not up to date",
    )
    args = parser.parse_args()

    output = generate_catalog_source()
    if args.check:
        current = OUTPUT_PATH.read_text() if OUTPUT_PATH.exists() else ""
        if current != output:
            print(
                "Generated model catalog is out of date. Run: python scripts/generate_models.py",
                file=sys.stderr,
            )
            raise SystemExit(1)
        return

    OUTPUT_PATH.write_text(output)
    print(f"Generated {OUTPUT_PATH.relative_to(ROOT)}")


def generate_catalog_source() -> str:
    overrides = load_overrides()
    models = collect_models(overrides)
    apply_patches(models, overrides.get("patches", []))
    models.extend(model_from_mapping(item) for item in overrides.get("static_models", []))

    if overrides.get("mirror_google_to_vertex", False):
        models.extend(vertex_clone(model) for model in list(models) if model.provider == "google")

    deduped: dict[tuple[str, str], CatalogModel] = {}
    for model in models:
        deduped.setdefault(model.key, model)

    grouped: dict[str, list[CatalogModel]] = {}
    for model in sorted(deduped.values(), key=lambda m: (m.provider, m.id)):
        grouped.setdefault(model.provider, []).append(model)

    return render_catalog(grouped)


def load_overrides() -> dict[str, Any]:
    return json.loads(OVERRIDES_PATH.read_text())


def collect_models(overrides: dict[str, Any]) -> list[CatalogModel]:
    provider_defaults = overrides["provider_defaults"]
    include_providers = set(overrides["include_providers"])
    models: list[CatalogModel] = []

    models_dev = fetch_json(MODELS_DEV_URL)
    for provider in sorted(include_providers):
        provider_data = models_dev.get(provider)
        defaults = provider_defaults.get(provider)
        if not provider_data or not defaults:
            continue
        for item in provider_data.get("models", {}).values():
            if item.get("tool_call") is not True:
                continue
            models.append(model_from_models_dev(provider, item, defaults))

    defaults = provider_defaults.get("openrouter")
    if defaults:
        models.extend(fetch_openrouter_models(defaults))

    return models


def fetch_json(url: str) -> Any:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=30) as response:  # noqa: S310 - maintainer script URL allowlist
        return json.load(response)


def model_from_models_dev(
    provider: str,
    item: dict[str, Any],
    defaults: dict[str, Any],
) -> CatalogModel:
    limit = item.get("limit", {}) or {}
    cost = item.get("cost", {}) or {}
    input_types = normalize_input_types(item.get("modalities", {}).get("input", []))
    return CatalogModel(
        provider=provider,
        api=defaults["api"],
        id=item["id"],
        name=item.get("name"),
        base_url=defaults.get("base_url"),
        context_window=limit.get("context") or limit.get("input"),
        max_tokens=limit.get("output"),
        input_types=input_types,
        reasoning=bool(item.get("reasoning", False)),
        cost=normalize_cost(cost),
    )


def fetch_openrouter_models(defaults: dict[str, Any]) -> list[CatalogModel]:
    data = fetch_json(OPENROUTER_MODELS_URL)
    models: list[CatalogModel] = []
    for item in data.get("data", []):
        if "tools" not in (item.get("supported_parameters") or []):
            continue
        pricing = item.get("pricing", {}) or {}
        architecture = item.get("architecture", {}) or {}
        top_provider = item.get("top_provider", {}) or {}
        input_types = (
            ("text", "image") if "image" in str(architecture.get("modality", "")) else ("text",)
        )
        models.append(
            CatalogModel(
                provider="openrouter",
                api=defaults["api"],
                id=item["id"],
                name=item.get("name"),
                base_url=defaults.get("base_url"),
                context_window=item.get("context_length"),
                max_tokens=top_provider.get("max_completion_tokens"),
                input_types=input_types,
                reasoning="reasoning" in (item.get("supported_parameters") or []),
                cost={
                    "input": price_per_token_to_million(pricing.get("prompt")),
                    "output": price_per_token_to_million(pricing.get("completion")),
                    "cache_read": price_per_token_to_million(pricing.get("input_cache_read")),
                    "cache_write": price_per_token_to_million(pricing.get("input_cache_write")),
                },
            )
        )
    return models


def model_from_mapping(data: dict[str, Any]) -> CatalogModel:
    return CatalogModel(
        provider=data["provider"],
        api=data["api"],
        id=data["id"],
        name=data.get("name"),
        base_url=data.get("base_url"),
        context_window=data.get("context_window"),
        max_tokens=data.get("max_tokens"),
        input_types=tuple(data.get("input_types", ["text"])),
        reasoning=bool(data.get("reasoning", False)),
        cost=normalize_cost(data.get("cost", {})),
        config=data.get("config", {}),
    )


def apply_patches(models: list[CatalogModel], patches: list[dict[str, Any]]) -> None:
    index = {model.key: model for model in models}
    for patch in patches:
        model = index.get((patch["provider"], patch["id"]))
        if model is None:
            continue
        for field_name, value in patch.get("set", {}).items():
            if field_name == "input_types":
                value = tuple(value)
            elif field_name == "cost":
                value = normalize_cost(value)
            setattr(model, field_name, value)


def vertex_clone(model: CatalogModel) -> CatalogModel:
    return CatalogModel(
        provider="google-vertex",
        api="google-vertex",
        id=model.id,
        name=f"{model.name or model.id} on Vertex AI",
        base_url="https://{location}-aiplatform.googleapis.com",
        context_window=model.context_window,
        max_tokens=model.max_tokens,
        input_types=model.input_types,
        reasoning=model.reasoning,
        cost=dict(model.cost),
    )


def normalize_input_types(values: list[str]) -> tuple[str, ...]:
    result = ["text"]
    if "image" in values:
        result.append("image")
    return tuple(result)


def normalize_cost(cost: dict[str, Any]) -> dict[str, float]:
    return {
        "input": to_float(cost.get("input")),
        "output": to_float(cost.get("output")),
        "cache_read": to_float(cost.get("cache_read")),
        "cache_write": to_float(cost.get("cache_write")),
    }


def price_per_token_to_million(value: Any) -> float:
    return to_float(value) * 1_000_000


def to_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return round(float(value), 12)
    except (TypeError, ValueError):
        return 0.0


def render_catalog(grouped: dict[str, list[CatalogModel]]) -> str:
    lines = [
        "from __future__ import annotations",
        "",
        "from ..model import Model, ModelCost",
        "",
        "__all__ = [\"BUILTIN_MODELS\", \"iter_builtin_models\"]",
        "",
        "# Generated by scripts/generate_models.py. Do not edit manually.",
        "# Pricing is USD per 1M tokens.",
        "BUILTIN_MODELS: dict[str, tuple[Model, ...]] = {",
    ]
    for provider, models in grouped.items():
        lines.append(f"    {provider!r}: (")
        for model in models:
            lines.extend(render_model(model))
        lines.append("    ),")
    lines.extend([
        "}",
        "",
        "",
        "def iter_builtin_models() -> tuple[Model, ...]:",
        "    \"\"\"Return all built-in catalog models in provider order.\"\"\"",
        "    return tuple(model for models in BUILTIN_MODELS.values() for model in models)",
        "",
    ])
    return "\n".join(lines)


def render_model(model: CatalogModel) -> list[str]:
    lines = ["        Model("]
    fields: list[tuple[str, Any]] = [
        ("provider", model.provider),
        ("api", model.api),
        ("id", model.id),
        ("name", model.name),
        ("base_url", model.base_url),
        ("context_window", model.context_window),
        ("max_tokens", model.max_tokens),
        ("input_types", model.input_types),
        ("reasoning", model.reasoning),
    ]
    for name, value in fields:
        if value is None:
            continue
        if name == "reasoning" and value is False:
            continue
        lines.append(f"            {name}={python_literal(value)},")
    if any(model.cost.values()):
        lines.extend([
            "            cost=ModelCost(",
            f"                input={model.cost['input']!r},",
            f"                output={model.cost['output']!r},",
            f"                cache_read={model.cost['cache_read']!r},",
            f"                cache_write={model.cost['cache_write']!r},",
            "            ),",
        ])
    if model.config:
        lines.append(f"            config={python_literal(model.config)},")
    lines.append("        ),")
    return lines


def python_literal(value: Any) -> str:
    if isinstance(value, tuple):
        inner = ", ".join(repr(item) for item in value)
        if len(value) == 1:
            inner += ","
        return f"({inner})"
    return repr(value)


if __name__ == "__main__":
    main()
