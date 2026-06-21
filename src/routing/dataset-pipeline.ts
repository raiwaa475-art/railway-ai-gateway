import { pool } from "../utils/db.js";
import fs from "fs";
import path from "path";
import crypto from "crypto";

const DATASETS_DIR = path.join(process.cwd(), "datasets");

// Ensure datasets directory exists
if (!fs.existsSync(DATASETS_DIR)) {
    fs.mkdirSync(DATASETS_DIR, { recursive: true });
}

export interface DatasetBuildInput {
    minSuccess?: boolean;
    includeFailures?: boolean;
    format: "prompt_adapter_debug" | "sft_tool_calling" | "sft_tool_repair" | "sft_final_answer_after_tool_result" | "failure_cases_for_eval";
    limit?: number;
}

export async function buildDataset(input: DatasetBuildInput): Promise<string> {
    if (!pool) {
        throw new Error("Database not connected");
    }

    const minSuccess = input.minSuccess !== false;
    const includeFailures = !!input.includeFailures;
    const format = input.format;
    const limit = input.limit || 1000;

    let query = "SELECT * FROM qwen_agent_traces WHERE 1=1";
    const values: any[] = [];
    let idx = 1;

    if (!includeFailures) {
        query += ` AND success = true`;
    } else if (minSuccess) {
        query += ` AND (success = true OR (success = false AND tool_retry_used = true))`;
    }

    query += ` ORDER BY timestamp DESC LIMIT $${idx++}`;
    values.push(limit);

    const res = await pool.query(query, values);
    const traces = res.rows;

    const formattedLines: string[] = [];
    const uniqueRawOutputs = new Set<string>();

    for (const t of traces) {
        // Quality Filters
        // 1. Redact check (secrets)
        const isSanitized = JSON.stringify(t).includes("[SECRET_REDACTED]") || JSON.stringify(t).includes("[PRIVATE_KEY_REDACTED]");
        if (isSanitized && !includeFailures) {
            // Skips raw secrets if not fully redacted, but DB should already be redacted.
        }

        // 2. Exclude empty qwen raw output
        if (!t.qwen_raw_output || t.qwen_raw_output.trim() === "") {
            continue;
        }

        // 3. Deduplicate by raw output
        if (uniqueRawOutputs.has(t.qwen_raw_output)) {
            continue;
        }
        uniqueRawOutputs.add(t.qwen_raw_output);

        // Convert based on format
        let rowObj: any = null;

        const messages = typeof t.sanitized_messages === "string" ? JSON.parse(t.sanitized_messages) : t.sanitized_messages || [];

        switch (format) {
            case "prompt_adapter_debug":
                rowObj = {
                    requestId: t.request_id,
                    userIntent: t.user_intent,
                    rawOutput: t.qwen_raw_output,
                    requestedTool: t.requested_tool_name,
                    normalizedTool: t.normalized_tool_name,
                    originalArgs: typeof t.original_tool_args === "string" ? JSON.parse(t.original_tool_args) : t.original_tool_args,
                    repairedArgs: typeof t.repaired_tool_args === "string" ? JSON.parse(t.repaired_tool_args) : t.repaired_tool_args,
                    success: t.success,
                    validationError: t.tool_validation_error
                };
                break;

            case "sft_tool_calling":
                if (t.normalized_tool_name) {
                    rowObj = {
                        messages: messages.filter((m: any) => m.role === "system" || m.role === "user"),
                        assistant_tool_call: {
                            name: t.normalized_tool_name,
                            input: typeof t.repaired_tool_args === "string" ? JSON.parse(t.repaired_tool_args) : t.repaired_tool_args || {}
                        },
                        metadata: {
                            source: "qwen_agent_trace",
                            requestId: t.request_id,
                            success: t.success
                        }
                    };
                }
                break;

            case "sft_tool_repair":
                if (t.fake_tool_json_detected && t.normalized_tool_name) {
                    rowObj = {
                        bad_output: t.qwen_raw_output,
                        corrected_tool_call: {
                            name: t.normalized_tool_name,
                            input: typeof t.repaired_tool_args === "string" ? JSON.parse(t.repaired_tool_args) : t.repaired_tool_args || {}
                        },
                        metadata: {
                            requestId: t.request_id,
                            repair_type: t.tool_args_repaired ? "tool_alias+arg_alias" : "tool_alias"
                        }
                    };
                }
                break;

            case "sft_final_answer_after_tool_result":
                if (!t.normalized_tool_name && t.success) {
                    rowObj = {
                        messages: messages.slice(0, -1),
                        tool_result: t.tool_result_preview || "",
                        assistant_final_answer: t.final_answer_preview || ""
                    };
                }
                break;

            case "failure_cases_for_eval":
                if (!t.success) {
                    rowObj = {
                        requestId: t.request_id,
                        userIntent: t.user_intent,
                        validationError: t.tool_validation_error,
                        failureReason: t.failure_reason,
                        rawOutput: t.qwen_raw_output,
                        requestedToolName: t.requested_tool_name,
                        originalToolArgs: typeof t.original_tool_args === "string" ? JSON.parse(t.original_tool_args) : t.original_tool_args
                    };
                }
                break;
        }

        if (rowObj) {
            formattedLines.push(JSON.stringify(rowObj));
        }
    }

    const datasetId = crypto.randomUUID();
    const datasetFilePath = path.join(DATASETS_DIR, `${datasetId}.jsonl`);
    fs.writeFileSync(datasetFilePath, formattedLines.join("\n") + "\n", "utf-8");

    return datasetId;
}

export async function buildEvalSet(): Promise<string> {
    if (!pool) {
        throw new Error("Database not connected");
    }

    // Build categories: read file, edit file, write file, grep, glob, bash, repair, final
    const categories = ["Read", "Edit", "Write", "Grep", "Glob", "Bash"];
    const formattedLines: string[] = [];

    for (const cat of categories) {
        const res = await pool.query(
            "SELECT * FROM qwen_agent_traces WHERE normalized_tool_name = $1 AND success = true LIMIT 10",
            [cat]
        );
        for (const t of res.rows) {
            const messages = typeof t.sanitized_messages === "string" ? JSON.parse(t.sanitized_messages) : t.sanitized_messages || [];
            formattedLines.push(JSON.stringify({
                category: `tool_${cat.toLowerCase()}`,
                requestId: t.request_id,
                userIntent: t.user_intent,
                expected_tool: {
                    name: t.normalized_tool_name,
                    arguments: typeof t.repaired_tool_args === "string" ? JSON.parse(t.repaired_tool_args) : t.repaired_tool_args || {}
                },
                messages: messages.filter((m: any) => m.role === "user" || m.role === "system")
            }));
        }
    }

    // Tool repair evaluation cases
    const repairRes = await pool.query(
        "SELECT * FROM qwen_agent_traces WHERE fake_tool_json_detected = true AND success = true LIMIT 15"
    );
    for (const t of repairRes.rows) {
        formattedLines.push(JSON.stringify({
            category: "tool_repair",
            requestId: t.request_id,
            bad_output: t.qwen_raw_output,
            expected_tool: {
                name: t.normalized_tool_name,
                arguments: typeof t.repaired_tool_args === "string" ? JSON.parse(t.repaired_tool_args) : t.repaired_tool_args || {}
            }
        }));
    }

    const evalId = "eval_set_" + crypto.randomUUID().substring(0, 8);
    const evalFilePath = path.join(DATASETS_DIR, `${evalId}.jsonl`);
    fs.writeFileSync(evalFilePath, formattedLines.join("\n") + "\n", "utf-8");

    return evalId;
}

export function getDatasetFilePath(datasetId: string): string {
    const safeId = datasetId.replace(/[^a-zA-Z0-9_\-]/g, "");
    return path.join(DATASETS_DIR, `${safeId}.jsonl`);
}
