# Phase Plan

## Phase 1

Ship hardening for the AI Coding Gateway without changing the core request contract.

### Scope

- Qwen worker
- gateway validator
- short DeepSeek verify
- token and cost logging

### Success Criteria

- Valid rate is at least 60%
- Cost reduction is at least 50%

### Stop Condition

If the real measured cost reduction is below the target, do not continue to Phase 2.

## Out of Scope For Now

- Redis
- pgvector
- Claude review
- fine-tune pipeline

