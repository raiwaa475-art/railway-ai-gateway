# Tuning and Dataset Guidelines for Qwen Agent

This document explains how to utilize the traces collected by `qwen-agent` to tune and improve tool-calling performance.

## Collected Dataset Overview
Traces are automatically sanitized and recorded to the database table `qwen_agent_traces` (or falling back to `qwen_agent_traces.jsonl` in the workspace root). 

You can export these traces in a JSONL format suitable for SFT/prompt tuning by requesting:
`GET /admin/qwen-agent/traces/export?format=jsonl`

Each exported row contains:
- `messages`: The conversation messages.
- `expected_tool`: The normalized, validated tool call that should have been produced.
- `tool_result`: The tool execution output.
- `final_answer`: The agent's final text summary of the results.
- `metadata`: Request and run identifiers.

## Tuning Strategy & Dataset Sizes

### 1. Prompt and Adapter Tuning (100–300 Traces)
- **Goal**: Correct common syntax issues, prompt layout misunderstandings, or naming inconsistencies.
- **Dataset Size**: **100 to 300 traces** are usually sufficient.
- **Approach**: Read the failed traces (`success: false` or `toolValidationError: not null`) using the `summary` endpoint to identify repeating invalid tool shapes or commands. Tweak the system instructions in `src/routing/qwen-agent.ts` or add specific examples in the system prompt.

### 2. Supervised Fine-Tuning (SFT) (500–1000+ Traces)
- **Goal**: Fine-tune the base LLM weights to behave natively like Claude Code, learning to call tools in JSON structure directly without needing adapters or regex repairs.
- **Dataset Size**: **500 to 1,000 traces** of high-quality agent tool loops.
- **Approach**: Train the model on the `messages` history and the `expected_tool` call. Ensure you only train the loss on the assistant responses (the tool calls and text answers).

### 3. SFT Training Best Practices
- **Never fine-tune GGUF/AWQ quantized weights directly**: Fine-tuning directly on GGUF files leads to catastrophic failure or bad gradients.
- **Base model target**: Train a LoRA (Low-Rank Adaptation) adapter on the unquantized base model weights (e.g. `Qwen/Qwen2.5-Coder-7B-Instruct` or similar) in 16-bit or 8-bit precision.
- **Merge & Quantize**: Merge the trained LoRA weights back into the base model weights, then re-quantize the final merged model back to GGUF format (e.g. using `llama.cpp` quantize script) for local execution via Ollama or LM Studio.
- **Validation**: Test the newly tuned model against the same tasks and track metrics on the `/admin/qwen-agent/traces/summary` dashboard to verify the success rate increase.
