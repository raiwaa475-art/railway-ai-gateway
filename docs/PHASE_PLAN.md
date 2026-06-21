# AI Coding Gateway Roadmap

This document outlines the detailed roadmap for the AI Coding Gateway implementation across all five phases.

---

## Phase 1 — qwen-agent & Trace Logger

### Scope
- Hardening of `qwen-agent` local worker tool normalizer and JSON translator.
- Safe shell checks for blocked commands (leaks, deletions, API keys).
- Trace logs saved to DB/file with key redaction and size truncation.
- Exposing summary metrics and jsonl traces exports.

### Success Criteria
- Valid tool-calling rate of Qwen Agent is at least 60%.
- API cost reduction compared to DeepSeek-only matches or exceeds 50%.

---

## Phase 2 — Trace-Based Tuning System

### Scope
- Adapter Rules Engine (`qwen_adapter_rules` table) supporting tool and argument aliases.
- Prompt Profile overridden configs (`qwen_prompt_profiles` table).
- Dynamic matching of rules and profile prompts in the worker execution flow.
- Administrative controls for creating, testing, and archiving rules/profiles.

### Success Criteria
- Target tool-calling repairs decrease validation retries by at least 30%.
- System prompt variations are successfully traced and A/B metric-monitored.

---

## Phase 3 — Smart Controller (`qwen-smart-v2`)

### Scope
- Smart routing alias `qwen-smart-v2`.
- supervisor model (DeepSeek/Claude) creates plans and reviews edits.
- Supervisor output validation blocks raw code and patch hunks.
- Worker tool execution loops with supervisor repair corrections.

### Success Criteria
- DeepSeek/Claude API calls are reduced to exactly 2 calls (1 plan, 1 review) in normal runs.
- Supervisor never leaks code writing tasks directly (zero code block violations).

---

## Phase 4 — Background Auto Coding Workflow

### Scope
- Asynchronous jobs queue (`auto_coding_jobs` and events tracking tables).
- File reading, writing, editing, grep, glob, and shell runners mapped against `repo_path`.
- Guardrails prohibiting auto git pushes, publishing packages, or production deploys.
- Summary dashboards for queue metrics and cost saving averages.

### Success Criteria
- Jobs successfully transition statuses from `queued` to `completed`/`failed` completely asynchronously.
- Worker tasks run safely without executing dangerous shell calls.

---

## Phase 5 — SFT Dataset pipeline

### Scope
- Constructing export dataset formats: SFT tool calling, repair cases, final answers, failure evals.
- Endpoints for exporting custom limits, success rates, and validation fail evaluations.
- Evaluation split compilations matching tool categories.

### Success Criteria
- JSONL format files match standard fine-tuning layouts (Axolotl/LLaMA-Factory) without manual edits.
- Sensitive information is 100% verified redacted.
