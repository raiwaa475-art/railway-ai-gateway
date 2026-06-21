# Metrics & Workspace Memory

This document explains the architecture, endpoint schema, database tables, and logic of the Lightweight Workspace Memory and Admin Metrics Console implemented in `railway-ai-gateway`.

---

## 1. Metrics & Dashboard Endpoints

All admin endpoints are protected by `adminAuthMiddleware`. An admin token (`GATEWAY_ADMIN_KEY` or `GATEWAY_API_KEY`) must be sent in the request header as `Authorization: Bearer <key>` or `x-api-key: <key>`.

### Backend Endpoints

#### `GET /admin/metrics/overview?range=[today|7d|30d|all]`
Aggregates gateway-wide requests and model invocation trends:
- **total_requests**: Count of requests registered in `gateway_requests`.
- **total_model_calls**: Count of calls registered in `model_calls`.
- **avg_latency**: Rounded average latency in milliseconds.
- **deepseek_call_percentage**: percentage of model calls delegated to DeepSeek.
- **daily_trend**: Grouped daily requests and average latencies.

#### `GET /admin/metrics/qwen?range=[today|7d|30d|all]`
Provides stats specific to the local Qwen-local model agent:
- **qwen_calls**: Total invocations of `qwen-local`.
- **qwen_success_rate**: Successful runs rate from `qwen_agent_traces`.
- **qwen_failure_rate**: Failed runs rate from `qwen_agent_traces`.
- **retry_rate**: Rate of turns where argument/tool-call recovery retries were used.
- **build_status_breakdown**: Count of jobs grouped by `build_status`.
- **top_failure_reasons**: Log of top error reasons and their occurrence count.

#### `GET /admin/metrics/cost?range=[today|7d|30d|all]`
Provides gateway costs and savings summaries:
- **total_cost_usd** / **total_cost_thb**: Total currency spend on LLMs.
- **estimated_saved_usd** / **estimated_saved_thb**: Saved costs by utilizing Qwen for local tool execution instead of calling expensive Claude/DeepSeek APIs directly.
- **daily_trend**: Grouped daily cost and savings.

#### `GET /admin/metrics/failures?range=[today|7d|30d|all]`
Collects recent errors and build breakdowns:
- **top_failure_reasons**: Grouped failure counts.
- **build_status_breakdown**: Grouped counts.
- **recent_failures**: List of the 20 most recent failed agent traces.

---

## 2. Lightweight Memory

This system allows storing local repository configurations, recent task traces, and failure patterns directly in Postgres. 

### Database Tables

1. **`repo_memories`**
   - `repo_key` (VARCHAR, UNIQUE): Normalized directory name slug.
   - `summary` (TEXT): High-level description of codebase structure and patterns.
   - `important_files` (JSONB): Array of paths to files frequently read or modified.
   - `risk_zones` (JSONB): Critical sections of the codebase that require extreme caution.
   - `tech_stack` (JSONB): Primary frameworks, languages, and tools used.

2. **`task_memories`**
   - `repo_key` (VARCHAR): Linked repository key.
   - `task_summary` (TEXT): Task input description.
   - `touched_files` (JSONB): Files modified during execution.
   - `outcome` (VARCHAR): `success` or `failed`.
   - `model_route` (VARCHAR): Routing flow utilized (`qwen-only`, `qwen-smart-v2`, `hybrid`).
   - `cost_thb` (NUMERIC): Aggregated currency cost of this task.

3. **`failure_patterns`**
   - `repo_key` (VARCHAR): Linked repository.
   - `pattern_type` (VARCHAR): Failure category classification.
   - `failure_reason` (TEXT): Last observed failure reason.
   - `examples` (JSONB): List of the last 5 observed raw failure error messages.
   - `hit_count` (INTEGER): Occurrence frequency.

### Memory REST API

- `GET /admin/memory/repos`: Lists all repository memory entries.
- `GET /admin/memory/repos/:repoKey`: Retrieves memory for a specific key.
- `POST /admin/memory/repos/:repoKey`: Upserts repository memory configuration.
- `GET /admin/memory/tasks?repoKey=`: Fetches task history (filtered by repoKey if provided).
- `POST /admin/memory/tasks`: Manually posts a task memory entry.
- `GET /admin/memory/failures?repoKey=`: Fetches grouped failure patterns.
- `POST /admin/memory/failures/rebuild`: Clears failure patterns and rebuilds them by reading historical failed agent traces.

---

## 3. Failure Classification Rules

When an agent execution fails, its raw failure reason is categorized into one of the following `pattern_type` classes:
- **Compilation Error**: Triggers on typescript, syntax, build, compile, or tsc errors.
- **Test Failure**: Triggers on test, expect, assert, spec, or testing validation fails.
- **Timeout**: Triggers on timed out or execution timeouts.
- **Max Tool Rounds Exceeded**: Triggers on hitting maximum agent round limits.
- **Security / Dangerous Command Blocked**: Triggers on rm -rf or other blacklisted console commands.
- **JSON Parsing Error**: Triggers on invalid JSON format or parsing errors.
- **File Access Error**: Triggers on file/directory not found or ENOENT errors.
- **Execution Error**: Default category for unclassified failures.

---

## 4. Prompt Context Injection

Before calling `qwen-local`, `qwen-smart-v2`, or starting background job planners/reviewers, the system extracts the `repo_key` from request headers or payloads. 

If found:
1. Retrives repository summary, stack, caution files, and risks.
2. Retrieves the 5 most recent task memories for this repository.
3. Retrieves the top 5 failure patterns (sorted by hit count).
4. Formats these into a brief markdown block.
5. Checks the length: if the formatted block exceeds **2,000 characters**, it is safely truncated.
6. Prepends this memory block directly into the LLM system instructions:

```markdown
--- LIGHTWEIGHT WORKSPACE MEMORY ---
[Repository Memory]
Summary: ...
Tech Stack: ...
Key Files: ...
Risk/Caution Zones: ...

[Recent Tasks Memory]
- Task: ... | Route: ... | Outcome: ... (Files: ...)

[Top Failure Patterns]
- Pattern: ... (Seen X times)
  Last reason: ...
------------------------------------
```

---

## 5. Dashboard Console UI

A dashboard is hosted statically at `/dashboard/metrics.html` (accessible via the main landing page at `/dashboard/` by clicking "Metrics & Memory").

### Features
1. **API Key Authentication**: Stores the `gatewayAdminKey` in localStorage for ease of use.
2. **Time Range Filters**: Instantly switches metrics between Today, 7 Days, 30 Days, or All Time.
3. **Dynamic Canvas-less SVG Charts**: Renders crisp vectors for request volume trends and cost-savings stack charts.
4. **Interactive Memory Editor**: Directly edit and save settings, summary, important files, tech stack, and risk zones for any repository key.
5. **Interactive Rebuilds**: Allows rebuilding failure patterns dynamically for a selected repo.
