# Architecture

This repository keeps the current Express + TypeScript + Postgres shape and uses direct provider routing at the gateway layer.

## Direct Provider Routing

Direct routing is the default path for normal requests:

- The client sends a model name.
- The gateway resolves that model to a configured provider.
- The request is forwarded to that provider without orchestration.
- The response is streamed or returned as-is, while usage and cost data are logged to `model_calls`.

This path is meant to stay simple and predictable. It is the baseline behavior that must not change while Phase 1 hardening is added.

## Hybrid-Flow / `qwen-smart` Routing

`hybrid-flow` and `qwen-smart` are aliases for the same orchestration path.

- The gateway inspects the request for tool results and code-edit intent.
- If the request looks like a code change and there is useful context, the orchestrator delegates the draft step to Qwen local.
- The Qwen draft is validated by the gateway before DeepSeek sees the final request.
- If the request does not look safe for delegation, the flow falls back to a normal DeepSeek request.

This is a hybrid path, not a replacement for direct provider routing.

## Qwen Local Draft Flow

When the orchestrator decides to use Qwen local:

- The gateway gathers exact file context when available.
- A reduced prompt is sent to Qwen local to draft the code change.
- The gateway validates the draft deterministically.
- The Qwen call is logged in `model_calls` with draft quality fields, context source, retry state, and fallback reason.

The current implementation is intentionally conservative. It prefers a short local draft over a broad rewrite.

## DeepSeek Fallback / Verify Flow

DeepSeek remains the final verifier and responder for the hybrid path:

- If Qwen is not eligible, the request goes directly to DeepSeek.
- If Qwen produces a valid draft, DeepSeek verifies and applies it.
- If Qwen fails validation, the gateway still preserves the original request shape and falls back cleanly.

This means the public contract stays stable while the internal routing becomes more selective.

## Current Limitations

- No Redis cache or queue exists yet.
- No pgvector store exists yet.
- No Claude review or fine-tune pipeline exists yet.
- The confidence helper is dry-run only and does not change production routing.
- Metrics are derived only from existing `model_calls` rows, so they reflect what was actually logged and nothing else.
- The gateway still depends on the current Postgres schema and logging cadence.

