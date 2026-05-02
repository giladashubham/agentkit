# Pi AI Gap Review and AgentKit Roadmap

This document compares AgentKit against Pi AI's `packages/ai` package and turns the review into an actionable roadmap.

## Verdict

AgentKit is in good shape for a small Python `v0.1` open-source LLM toolkit. The core API is clean, the provider structure is understandable, optional dependencies are handled well, and the codebase is easy to contribute to.

However, AgentKit is not yet at Pi AI's production maturity level. Pi AI has significantly more battle-tested provider compatibility code, model metadata, cost tracking, replay transforms, OAuth support, and edge-case tests.

**Current status:** alpha foundation, not yet Pi-level production abstraction.

## Comparison Summary

| Area | AgentKit | Pi AI | Status |
|---|---|---|---|
| Core API | `complete(model, context)` / `stream(...)` | Similar stream-first API | Good |
| Package structure | Python `src/` layout | Mature TS package layout | Good |
| Provider isolation | Provider packages under `llm/providers` | Strong provider isolation | Good |
| Optional deps | Provider extras | SDK deps bundled | AgentKit advantage |
| Model registry | Curated metadata catalog | Generated model DB | Done |
| Cost tracking | ModelCost + automatic calculation | Full cost calculation | Done |
| Env API keys | Provider-specific env resolver | Provider-specific env resolver | Done |
| Streaming | Basic event support | Robust final/error event contract | Done |
| Tool calls | Validation/coercion added; normalization pending | Partial JSON repair, validation, normalization | Partial |
| Images | Basic converter support | Unsupported-image downgrade and routing | Gap |
| Cross-provider handoff | Not implemented | Strong `transformMessages()` layer | Gap |
| OAuth providers | Not implemented | Anthropic/OpenAI Codex/GitHub Copilot/etc. | Later |
| Tests | Good start | Many provider edge/e2e tests | Gap |
| OSS hygiene | CI/docs added | Mature | Good start |

## High-Priority Roadmap

### 1. Provider environment API key resolver

**Why:** OpenAI-compatible providers like DeepSeek, Groq, and OpenRouter should not require users to manually pass `api_key` every time. Pi AI resolves provider-specific env vars.

**Target API:**

```python
from agentkit.llm import get_env_api_key

key = get_env_api_key("openrouter")
```

**Provider env mapping examples:**

| Provider | Env var |
|---|---|
| `openai` | `OPENAI_API_KEY` |
| `anthropic` | `ANTHROPIC_API_KEY` |
| `google` | `GEMINI_API_KEY` |
| `google-vertex` | `GOOGLE_CLOUD_API_KEY` |
| `deepseek` | `DEEPSEEK_API_KEY` |
| `groq` | `GROQ_API_KEY` |
| `openrouter` | `OPENROUTER_API_KEY` |
| `xai` | `XAI_API_KEY` |
| `fireworks` | `FIREWORKS_API_KEY` |
| `together` | `TOGETHER_API_KEY` |
| `perplexity` | `PERPLEXITY_API_KEY` |
| `cerebras` | `CEREBRAS_API_KEY` |
| `sambanova` | `SAMBANOVA_API_KEY` |
| `nebius` | `NEBIUS_API_KEY` |

**Acceptance criteria:**

- [x] Add `src/agentkit/llm/env.py` or similar.
- [x] Export `get_env_api_key` from `agentkit.llm`.
- [x] Provider factories use env key when `model.api_key is None`.
- [x] Tests cover default and unknown providers.
- [x] README documents env vars.

---

### 2. Robust stream error finalization

**Why:** Pi AI streams always terminate with a final successful or error assistant message. AgentKit currently can emit an error event without a final `Response`, causing `await stream.result()` to return `None` in some cases.

**Decision needed:** choose one contract.

Option A: `StreamResponse.result()` raises on stream error.

Option B: error events carry a final `Response` with `StopReason.ERROR` or `StopReason.ABORTED`.

**Recommended:** Option B for agent loops, because the context can record failed/aborted turns.

**Acceptance criteria:**

- [x] Define error finalization behavior in `streaming.py`.
- [x] Ensure provider error paths yield final error response.
- [x] `StreamResponse.result()` never silently returns `None` after an error event.
- [x] Tests cover provider error event and abort event behavior.
- [x] README documents stream error contract.

---

### 3. Curated model catalog

**Why:** Pi AI stores model metadata in a structured model database. AgentKit now uses a curated Python catalog with model identity, context window, max output tokens, input types, reasoning support, and best-effort pricing.

**Acceptance criteria:**

- [x] Add static model catalog module.
- [x] Register built-ins from catalog data instead of one-off registration calls.
- [x] Include context window, max tokens, input types, reasoning support, and cost metadata.
- [x] Preserve custom `register_model(...)` support.
- [x] Tests cover built-in metadata.

---

### 4. Cost calculation from model metadata

**Why:** `Usage.cost` exists but is not calculated in a unified way. Pi AI calculates cost using model metadata.

**Proposed helper:**

```python
from agentkit.llm import calculate_cost

usage = calculate_cost(model, usage)
```

**Acceptance criteria:**

- [x] Add model cost fields or use existing `Cost` model consistently.
- [x] Add `calculate_cost(model, usage)` helper.
- [x] Apply cost calculation in provider parsers/stream finalization when model metadata is available.
- [x] Tests cover input/output/cache costs.
- [x] README documents cost units.

---

### 4. Assistant response metadata preservation

**Why:** In AgentKit, `Response` has model/usage/stop metadata, but `response.message` loses most of that when appended to `Context`. Pi AI assistant messages carry provider, API, model, usage, stop reason, response ID, etc.

**Use cases:**

- Cross-provider handoff.
- Replay debugging.
- Cost reports.
- Provider response ID continuity.
- Reasoning/thinking replay.

**Acceptance criteria:**

- [x] Decide whether to add optional metadata to `AssistantMessage` or store in `Context.metadata`.
- [x] Preserve at least provider/api/model/usage/stop_reason/response_id when available.
- [x] Keep serialization backward-compatible.
- [x] Tests cover context round-trip with assistant metadata.

---

### 5. Tool argument validation and coercion

**Why:** Pi AI validates and coerces tool arguments against schemas. AgentKit currently creates schemas but does not validate model-produced arguments.

**Proposed API:**

```python
validate_tool_arguments(tool, call.arguments)
```

or as part of execution:

```python
await execute_tool(tool, call.arguments, validate=True)
```

**Acceptance criteria:**

- [x] Store or derive a Pydantic model for decorated tools when possible.
- [x] Validate `ToolCall.arguments` before execution.
- [x] Return clear validation errors.
- [x] Tests cover valid, missing, wrong-type, and coercible arguments.

---

## Medium-Priority Roadmap

### 6. Cross-provider message transform layer

**Why:** Pi AI's `transformMessages()` handles provider replay differences. AgentKit will need this for reliable agent loops and model handoff.

**Features to consider:**

- Unsupported image downgrade to text placeholder.
- Tool-call ID normalization for providers with strict ID rules.
- Orphan tool-call repair with synthetic error tool results.
- Dropping errored/aborted assistant messages before replay.
- Thinking block replay rules.

**Acceptance criteria:**

- [ ] Add `transform_messages(messages, model)` helper.
- [ ] Use it inside provider converters or before conversion.
- [ ] Tests cover cross-provider handoff cases.

---

### 7. Faux/mock provider

**Why:** Pi AI has a faux provider for deterministic tests. AgentKit can use one for future agent-loop tests without live APIs.

**Acceptance criteria:**

- [ ] Add a test-friendly faux provider.
- [ ] Support queued text/tool/error responses.
- [ ] Support streaming deltas.
- [ ] Tests demonstrate use in agent loop scenarios.

---

### 8. Context overflow detection

**Why:** Pi AI has provider-specific overflow detection. AgentKit should eventually expose a helper for agent loops to decide whether to summarize/truncate.

**Acceptance criteria:**

- [ ] Add `is_context_overflow(response, context_window=None)`.
- [ ] Detect common provider error strings.
- [ ] Detect silent overflow when usage exceeds context window.
- [ ] Tests cover common provider messages.

---

### 9. More provider compatibility tests

Add tests for the provider edge cases that Pi AI already covers heavily.

**Candidate tests:**

- [ ] Tool call IDs with long/special characters.
- [ ] Partial/malformed streamed JSON arguments.
- [ ] Tool call without matching tool result.
- [ ] Image tool results per provider.
- [ ] Thinking/reasoning replay.
- [ ] Empty tools list behavior.
- [ ] Provider response IDs.
- [ ] Unicode surrogate sanitization.
- [ ] Cache headers/options if caching is added.

---

## Later / Optional

### OAuth providers

Pi AI supports OAuth-heavy providers such as GitHub Copilot and OpenAI Codex. This is useful, but not necessary for AgentKit's near-term Python core.

**Recommendation:** defer until the basic provider layer and agent loop are stable.

### Generated model database

Pi AI generates model metadata. AgentKit now has a curated Pi-style catalog, but it is not generated from upstream provider data.

**Recommendation:** keep the curated catalog for now. Consider generation only if maintaining model metadata manually becomes painful.

## Suggested Execution Order

1. Provider env API key resolver. Done.
2. Stream error finalization contract. Done.
3. Curated model catalog and cost calculation. Done.
4. Assistant metadata preservation. Done.
5. Tool argument validation. Done.
6. Faux provider.
7. Cross-provider message transforms.
8. Context overflow helper.
9. Expanded provider edge-case tests.

## Guiding Principle

Do not copy Pi AI feature-for-feature. AgentKit should stay Pythonic, small, and explicit. Pull in Pi-inspired hardening only when it directly improves reliability for Python agent loops.
