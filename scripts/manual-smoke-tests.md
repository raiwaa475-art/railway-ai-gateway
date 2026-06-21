# Manual Smoke Testing Instructions

This document provides curl templates to verify that each of the implemented gateway features and endpoints is working as expected.

For local development, the default API key is `local-dev-key`. 

---

## Phase 1 — Qwen Agent & Tracing

### 1. Test Qwen Agent tool calling
Send a task to `qwen-agent` requesting tool use. Note that tools are present, so `stream` will be forced to `false` automatically:
```bash
curl -X POST http://localhost:3000/v1/messages \
  -H "Authorization: Bearer local-dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-agent",
    "messages": [
      { "role": "user", "content": "Read package.json" }
    ],
    "tools": [
      {
        "name": "Read",
        "description": "Read file content",
        "input_schema": {
          "type": "object",
          "properties": {
            "file_path": { "type": "string" }
          },
          "required": ["file_path"]
        }
      }
    ]
  }'
```

### 2. Export Agent Traces
Download the jsonl file containing all compiled agent run traces:
```bash
curl -X GET "http://localhost:3000/admin/qwen-agent/traces/export?format=jsonl" \
  -H "Authorization: Bearer local-dev-key"
```

### 3. Get Traces Summary Metrics
Inspect performance, fake JSON conversion, repair rates, and error logs:
```bash
curl -X GET http://localhost:3000/admin/qwen-agent/traces/summary \
  -H "Authorization: Bearer local-dev-key"
```

---

## Phase 2 — Tuning Adapter Rules & Prompt Profiles

### 1. Create a tool normalization rule
Add a rule matching `open` tool and converting it to canonical `Read`:
```bash
curl -X POST http://localhost:3000/admin/qwen-agent/adapter/rules \
  -H "Authorization: Bearer local-dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "enabled": true,
    "rule_type": "tool_alias",
    "match_pattern": "open",
    "replacement": "Read",
    "description": "Auto-convert open tool alias to Read"
  }'
```

### 2. Create and enable a Prompt Profile
Add a custom worker prompt profile:
```bash
curl -X POST http://localhost:3000/admin/qwen-agent/prompt-profiles \
  -H "Authorization: Bearer local-dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "strict-coder-qwen",
    "enabled": true,
    "system_prompt": "You are a senior TypeScript engineer. Write minimal, clean, robust code.",
    "purpose": "Enforce strict coding style for tsx projects"
  }'
```

### 3. Get adapter tuning insights
Identify top fake JSON structures and recommend rules:
```bash
curl -X GET http://localhost:3000/admin/qwen-agent/tuning/insights \
  -H "Authorization: Bearer local-dev-key"
```

---

## Phase 3 — Smart Controller (`qwen-smart-v2`)

Verify the planner-supervisor execution loop:
```bash
curl -X POST http://localhost:3000/v1/messages \
  -H "Authorization: Bearer local-dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-smart-v2",
    "messages": [
      { "role": "user", "content": "Implement an interface in index.ts file" }
    ],
    "tools": [
      {
        "name": "Write",
        "description": "Write file",
        "input_schema": {
          "type": "object",
          "properties": {
            "file_path": { "type": "string" },
            "content": { "type": "string" }
          },
          "required": ["file_path", "content"]
        }
      }
    ]
  }'
```

---

## Phase 4 — Background Auto Coding Workflow

### 1. Create a background job
Start an asynchronous workflow targeting a workspace folder:
```bash
curl -X POST http://localhost:3000/admin/auto/jobs \
  -H "Authorization: Bearer local-dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "user_task": "Add a summary comments block at the top of package.json",
    "repo_path": "g:/railway-ai-gateway",
    "mode": "smart"
  }'
```

### 2. View background job status and events trace
Query the state and step executions:
```bash
curl -X GET http://localhost:3000/admin/auto/jobs/1 \
  -H "Authorization: Bearer local-dev-key"
```

### 3. Get background jobs metrics summary
Verify durational statistics and review success rates:
```bash
curl -X GET http://localhost:3000/admin/auto/summary \
  -H "Authorization: Bearer local-dev-key"
```

---

## Phase 5 — Dataset Fine-Tuning Pipeline

### 1. Compile SFT dataset
Trigger dataset generation for tool-calling SFT format:
```bash
curl -X POST http://localhost:3000/admin/qwen-agent/datasets/build \
  -H "Authorization: Bearer local-dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "minSuccess": true,
    "includeFailures": false,
    "format": "sft_tool_calling",
    "limit": 500
  }'
```
*Expected Response:* `{ "datasetId": "c88f1b95-ef29-43c7-ad87-21a415ff67d3" }`

### 2. Download the JSONL dataset
Download the compiled SFT dataset file:
```bash
curl -X GET "http://localhost:3000/admin/qwen-agent/datasets/c88f1b95-ef29-43c7-ad87-21a415ff67d3/download" \
  -H "Authorization: Bearer local-dev-key"
```

### 3. Compile an evaluation set
```bash
curl -X POST http://localhost:3000/admin/qwen-agent/datasets/build-eval \
  -H "Authorization: Bearer local-dev-key"
```
