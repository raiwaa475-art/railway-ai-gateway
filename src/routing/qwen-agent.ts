import { Request, Response } from "express";
import crypto from "crypto";
import fs from "fs";
import path from "path";
import { config } from "../config/env.js";
import { providerRegistry } from "./registry.js";
import { QwenLocalProvider } from "../providers/qwen-local.js";
import { pool } from "../utils/db.js";
import { 
    getActivePromptProfile, 
    getEnabledAdapterRules, 
    applyToolAliasRules, 
    applyArgAliasRules, 
    checkBashBlockRules, 
    getRetryHintRule, 
    applySystemPromptHintRules, 
    updateProfileStats, 
    incrementRuleHit 
} from "./tuning.js";
import { normalizeRepoKey, buildMemoryPromptContext, insertTaskMemory, recordFailurePattern, classifyFailure } from "./memory.js";

const TRACE_FILE_PATH = path.join(process.cwd(), "qwen_agent_traces.jsonl");

const QWEN_AGENT_SYSTEM_INSTRUCTION = 
  "You are connected to Claude Code tools.\n" +
  "Use real tool_use calls when reading or editing files.\n" +
  "Do not print JSON tool calls as text.\n" +
  "Use Read before Edit unless you already have exact file content.\n" +
  "Prefer small edits.\n" +
  "After tool_result, continue the task or give final answer.\n" +
  "Do not invent file paths.";

// Helper to normalize tool names
function normalizeToolName(name: string): string {
    const norm = String(name || "").trim().toLowerCase();
    
    // read_file, ReadFile, cat, open, view -> Read
    if (["read_file", "readfile", "cat", "open", "view"].includes(norm)) {
        return "Read";
    }
    // list_files, dir, ls -> LS
    if (["list_files", "dir", "ls"].includes(norm)) {
        return "LS";
    }
    // search, grep_search -> Grep
    if (["search", "grep_search"].includes(norm)) {
        return "Grep";
    }
    // find_files -> Glob
    if (norm === "find_files") {
        return "Glob";
    }
    // write_file -> Write
    if (norm === "write_file") {
        return "Write";
    }
    // edit_file, replace -> Edit
    if (["edit_file", "replace"].includes(norm)) {
        return "Edit";
    }
    // multi_edit -> MultiEdit
    if (norm === "multi_edit") {
        return "MultiEdit";
    }
    // run_command, shell, cmd -> Bash
    if (["run_command", "shell", "cmd"].includes(norm)) {
        return "Bash";
    }
    // todowrite -> TodoWrite
    if (["todowrite", "todo_write"].includes(norm)) {
        return "TodoWrite";
    }
    
    // Return capitalized canonical name if it matches case-insensitively
    const canonicals = ["Read", "LS", "Grep", "Glob", "Write", "Edit", "MultiEdit", "Bash", "TodoWrite"];
    const found = canonicals.find(c => c.toLowerCase() === norm);
    if (found) return found;

    return name;
}

interface FakeToolCall {
    name: string;
    input: any;
}

// Extract fake JSON tool use from text response block
function parseFakeToolJson(text: string): FakeToolCall | null {
    if (!text || typeof text !== "string") return null;
    const trimmed = text.trim();
    if (!trimmed) return null;

    let cleaned = trimmed;
    // Strip markdown code block fences if present
    if (cleaned.startsWith("```")) {
        const lines = cleaned.split("\n");
        if (lines[0].startsWith("```")) {
            lines.shift();
        }
        if (lines.length > 0 && lines[lines.length - 1].startsWith("```")) {
            lines.pop();
        }
        cleaned = lines.join("\n").trim();
    }

    // Direct JSON check
    if (cleaned.startsWith("{") && cleaned.endsWith("}")) {
        try {
            const parsed = JSON.parse(cleaned);
            if (parsed && typeof parsed === "object") {
                const name = parsed.name || parsed.tool || parsed.tool_name || parsed.tool_use;
                const input = parsed.arguments || parsed.input || parsed.args || parsed.parameters;
                if (name && typeof name === "string" && input !== undefined) {
                    let parsedInput = input;
                    if (typeof input === "string") {
                        try {
                            parsedInput = JSON.parse(input);
                        } catch {}
                    }
                    return { name, input: parsedInput };
                }
            }
        } catch {}
    }

    // Try finding outer bounds of a JSON object
    const firstBrace = cleaned.indexOf("{");
    const lastBrace = cleaned.lastIndexOf("}");
    if (firstBrace !== -1 && lastBrace !== -1 && lastBrace > firstBrace) {
        const jsonStr = cleaned.slice(firstBrace, lastBrace + 1);
        try {
            const parsed = JSON.parse(jsonStr);
            if (parsed && typeof parsed === "object") {
                const name = parsed.name || parsed.tool || parsed.tool_name || parsed.tool_use;
                const input = parsed.arguments || parsed.input || parsed.args || parsed.parameters;
                if (name && typeof name === "string" && input !== undefined) {
                    let parsedInput = input;
                    if (typeof input === "string") {
                        try {
                            parsedInput = JSON.parse(input);
                        } catch {}
                    }
                    return { name, input: parsedInput };
                }
            }
        } catch {}
    }

    return null;
}

// Repair tool argument naming mismatches
function repairToolArgs(toolName: string, input: any): { repairedInput: any; repaired: boolean } {
    if (!input || typeof input !== "object" || Array.isArray(input)) {
        return { repairedInput: input, repaired: false };
    }

    const repairedInput = { ...input };
    let repaired = false;

    // file/path/filename -> file_path
    const filePathKeys = ["file", "path", "filename"];
    for (const key of filePathKeys) {
        if (key in repairedInput && !("file_path" in repairedInput)) {
            repairedInput.file_path = repairedInput[key];
            delete repairedInput[key];
            repaired = true;
        }
    }

    // content_text/text -> content
    const contentKeys = ["content_text", "text"];
    for (const key of contentKeys) {
        if (key in repairedInput && !("content" in repairedInput)) {
            repairedInput.content = repairedInput[key];
            delete repairedInput[key];
            repaired = true;
        }
    }

    // old/find/oldText -> old_string
    const oldStringKeys = ["old", "find", "oldText"];
    for (const key of oldStringKeys) {
        if (key in repairedInput && !("old_string" in repairedInput)) {
            repairedInput.old_string = repairedInput[key];
            delete repairedInput[key];
            repaired = true;
        }
    }

    // new/replace/newText -> new_string
    const newStringKeys = ["new", "replace", "newText"];
    for (const key of newStringKeys) {
        if (key in repairedInput && !("new_string" in repairedInput)) {
            repairedInput.new_string = repairedInput[key];
            delete repairedInput[key];
            repaired = true;
        }
    }

    // command/cmd -> command
    const commandKeys = ["cmd"];
    for (const key of commandKeys) {
        if (key in repairedInput && !("command" in repairedInput)) {
            repairedInput.command = repairedInput[key];
            delete repairedInput[key];
            repaired = true;
        }
    }

    return { repairedInput, repaired };
}

// Filter for dangerous command blocks
function isCommandDangerous(command: string): string | null {
    if (!command || typeof command !== "string") return null;
    const cmd = command.trim().toLowerCase();

    if (cmd.includes("rm -rf")) return "rm -rf";
    if (cmd.includes("del /s")) return "del /s";
    
    if (/\bformat\b/i.test(cmd)) return "format";
    if (/\bshutdown\b/i.test(cmd)) return "shutdown";
    if (/\breboot\b/i.test(cmd)) return "reboot";
    
    if (/curl\s*.*\|\s*(bash|sh|zsh)/i.test(cmd) || /curl\s*.*\b(bash|sh|zsh)\b/i.test(cmd)) return "curl | bash";
    if (/wget\s*.*\|\s*(bash|sh|zsh)/i.test(cmd) || /wget\s*.*\b(bash|sh|zsh)\b/i.test(cmd)) return "wget | bash";
    
    if (cmd.includes("npm publish")) return "npm publish";
    if (cmd.includes("git push")) return "git push";
    if (cmd.includes("railway up")) return "railway up";
    if (cmd.includes("vercel --prod")) return "vercel --prod";
    
    if (/(?:>|>>|tee|out-file)\s*[^>]*\.env/i.test(cmd) || cmd.includes(".env")) return "writing .env";
    if (/(?:>|>>|tee|out-file)\s*[^>]*secrets/i.test(cmd) || cmd.includes("secrets")) return "writing secrets";
    
    if (cmd.includes("private key") || cmd.includes("private_key") || cmd.includes("private keys")) return "private keys";
    if (cmd.includes("api key") || cmd.includes("api_key") || cmd.includes("apikey") || cmd.includes("api keys")) return "API keys";

    return null;
}

// Scheme validator for canonical tool configurations
function validateToolCall(toolName: string, input: any): string | null {
    if (!toolName) return "Missing tool name";

    switch (toolName) {
        case "Read":
            if (!input || typeof input.file_path !== "string" || input.file_path.trim() === "") {
                return "Read requires file_path";
            }
            break;
        case "Edit":
            if (!input || typeof input.file_path !== "string" || input.file_path.trim() === "") {
                return "Edit requires file_path";
            }
            if (typeof input.old_string !== "string") {
                return "Edit requires old_string";
            }
            if (typeof input.new_string !== "string") {
                return "Edit requires new_string";
            }
            break;
        case "Write":
            if (!input || typeof input.file_path !== "string" || input.file_path.trim() === "") {
                return "Write requires file_path";
            }
            if (typeof input.content !== "string") {
                return "Write requires content";
            }
            break;
        case "Grep":
            if (!input || typeof input.pattern !== "string" || input.pattern.trim() === "") {
                return "Grep requires pattern";
            }
            break;
        case "Glob":
            if (!input || typeof input.pattern !== "string" || input.pattern.trim() === "") {
                return "Glob requires pattern";
            }
            break;
        case "Bash":
            if (!input || typeof input.command !== "string" || input.command.trim() === "") {
                return "Bash requires command";
            }
            const dangerousReason = isCommandDangerous(input.command);
            if (dangerousReason) {
                return `Dangerous command blocked: ${dangerousReason}`;
            }
            break;
        default:
            break;
    }

    return null;
}

// Counts prev tool round executions
function getToolRoundCount(messages: any[]): number {
    let count = 0;
    if (!Array.isArray(messages)) return 0;
    for (const msg of messages) {
        if (msg.role === "assistant" && Array.isArray(msg.content)) {
            const hasToolUse = msg.content.some((b: any) => b?.type === "tool_use");
            if (hasToolUse) count++;
        }
    }
    return count;
}

// Extracts user intent instruction
function getLatestUserIntent(messages: any[]): string {
    if (!Array.isArray(messages)) return "";
    for (const msg of messages.slice().reverse()) {
        if (msg.role === "user") {
            if (typeof msg.content === "string") return msg.content;
            if (Array.isArray(msg.content)) {
                return msg.content
                    .filter((b: any) => b?.type === "text")
                    .map((b: any) => b.text || "")
                    .join("\n");
            }
        }
    }
    return "";
}

// Extract files updated in response
function extractEditedFiles(content: any[]): string[] {
    const files = new Set<string>();
    if (!Array.isArray(content)) return [];
    for (const block of content) {
        if (block?.type === "tool_use") {
            const input = block.input || {};
            const file = input.file_path;
            if (typeof file === "string" && file.trim()) {
                files.add(file.trim());
            }
        }
    }
    return Array.from(files);
}

// Helper to grab latest test/build results and preview
function extractBuildStatusAndResult(messages: any[]): { buildStatus: string; toolResultPreview: string } {
    let buildStatus = "not_run";
    let toolResultPreview = "";

    if (!Array.isArray(messages)) return { buildStatus, toolResultPreview };

    for (const msg of messages.slice().reverse()) {
        if (msg.role === "user" && Array.isArray(msg.content)) {
            const toolResultBlocks = msg.content.filter((b: any) => b?.type === "tool_result");
            if (toolResultBlocks.length > 0) {
                const lastResultBlock = toolResultBlocks[0];
                let contentText = "";
                if (typeof lastResultBlock.content === "string") {
                    contentText = lastResultBlock.content;
                } else if (Array.isArray(lastResultBlock.content)) {
                    contentText = lastResultBlock.content.map((b: any) => b.text || "").join("\n");
                }
                toolResultPreview = contentText;

                const toolUseId = lastResultBlock.tool_use_id;
                if (toolUseId) {
                    for (const assistantMsg of messages.slice().reverse()) {
                        if (assistantMsg.role === "assistant" && Array.isArray(assistantMsg.content)) {
                            const toolUseBlock = assistantMsg.content.find((b: any) => b?.type === "tool_use" && b.id === toolUseId);
                            if (toolUseBlock && toolUseBlock.name === "Bash") {
                                const cmd = String(toolUseBlock.input?.command || "").toLowerCase();
                                if (
                                    cmd.includes("build") ||
                                    cmd.includes("test") ||
                                    cmd.includes("vitest") ||
                                    cmd.includes("jest") ||
                                    cmd.includes("tsc") ||
                                    cmd.includes("npm run")
                                ) {
                                    const output = contentText.toLowerCase();
                                    const hasFailure = 
                                        output.includes("failed") || 
                                        output.includes("failure") || 
                                        output.includes("error ") || 
                                        output.includes("err:") ||
                                        lastResultBlock.is_error === true;
                                    
                                    buildStatus = hasFailure ? "failed" : "passed";
                                }
                            }
                        }
                    }
                }
                break;
            }
        }
    }

    return { buildStatus, toolResultPreview };
}

// Redact credential secrets from logging
function sanitizeText(text: string): string {
    if (!text || typeof text !== "string") return text;
    let sanitized = text;
    
    // Redact private keys
    sanitized = sanitized.replace(/-----BEGIN[\s\S]+?-----END[^\n\r]+/g, "[PRIVATE_KEY_REDACTED]");
    
    // Redact database URLs
    sanitized = sanitized.replace(/(mongodb(?:\+srv)?|postgres(?:ql)?|mysql|sqlite):\/\/[a-zA-Z0-9_]+:[^@\s]+@[a-zA-Z0-9\.-]+(?::\d+)?\/[a-zA-Z0-9_-]*/gi, "$1://[REDACTED_USER]:[REDACTED_PASSWORD]@$3");
    
    // Redact JWT tokens
    sanitized = sanitized.replace(/eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}/g, "[JWT_REDACTED]");
    
    // Redact common API keys/tokens pattern
    sanitized = sanitized.replace(/\bsk-[a-zA-Z0-9]{20,}\b/g, "[OPENAI_KEY_REDACTED]");
    sanitized = sanitized.replace(/\bghp_[a-zA-Z0-9]{30,}\b/g, "[GITHUB_TOKEN_REDACTED]");
    
    // Bearer token value redaction
    sanitized = sanitized.replace(/(Authorization\s*:\s*Bearer\s+)[a-zA-Z0-9\-\._~\+\/]+/gi, "$1[BEARER_REDACTED]");
    
    // Generic keys/passwords
    sanitized = sanitized.replace(/((?:api[-_]?key|password|pass|secret|token|bearer|private[-_]?key|auth[-_]?token|jwt|session[-_]?id)\s*[:=]\s*['"]?)[a-zA-Z0-9\-_~\.\+\/]{8,}(['"]?)/gi, "$1[SECRET_REDACTED]$2");
    
    return sanitized;
}

// Recursive object property sanitizer
function sanitizeObject(obj: any): any {
    if (obj === null || obj === undefined) return obj;
    if (typeof obj === 'string') {
        if (obj.includes("PORT=") && (obj.includes("KEY=") || obj.includes("API=") || obj.includes("PASSWORD=") || obj.includes("SECRET="))) {
            return "[ENV_FILE_REDACTED]";
        }
        return sanitizeText(obj);
    }
    if (Array.isArray(obj)) return obj.map(sanitizeObject);
    if (typeof obj === 'object') {
        const result: any = {};
        for (const [key, val] of Object.entries(obj)) {
            const lowerKey = key.toLowerCase();
            if (
                (lowerKey.includes("key") && lowerKey !== "repo_key" && lowerKey !== "repokey") ||
                lowerKey.includes("password") ||
                lowerKey.includes("secret") ||
                lowerKey.includes("token") ||
                lowerKey.includes("auth") ||
                lowerKey.includes("credential")
            ) {
                if (typeof val === 'string') {
                    result[key] = "[SECRET_REDACTED]";
                } else {
                    result[key] = sanitizeObject(val);
                }
            } else {
                result[key] = sanitizeObject(val);
            }
        }
        return result;
    }
    return obj;
}

// Truncates large elements in traces to prevent size overflow
function truncateTraceSize(trace: any, maxChars: number): any {
    let traceStr = JSON.stringify(trace);
    if (traceStr.length <= maxChars) {
        return trace;
    }

    const t = { ...trace };

    if (t.qwenRawOutput && t.qwenRawOutput.length > 2000) {
        t.qwenRawOutput = t.qwenRawOutput.slice(0, 2000) + "\n... [TRUNCATED RAW OUTPUT]";
    }
    if (t.toolResultPreview && t.toolResultPreview.length > 1000) {
        t.toolResultPreview = t.toolResultPreview.slice(0, 1000) + "\n... [TRUNCATED TOOL RESULT]";
    }
    if (t.finalAnswerPreview && t.finalAnswerPreview.length > 1000) {
        t.finalAnswerPreview = t.finalAnswerPreview.slice(0, 1000) + "\n... [TRUNCATED FINAL ANSWER]";
    }

    traceStr = JSON.stringify(t);
    if (traceStr.length <= maxChars) {
        return t;
    }

    if (Array.isArray(t.sanitizedMessages)) {
        t.sanitizedMessages = t.sanitizedMessages.map((msg: any, idx: number) => {
            if (idx === 0 || idx === t.sanitizedMessages.length - 1) {
                return msg;
            }
            const newMsg = { ...msg };
            if (typeof newMsg.content === 'string' && newMsg.content.length > 400) {
                newMsg.content = newMsg.content.slice(0, 400) + "\n... [TRUNCATED MESSAGE]";
            } else if (Array.isArray(newMsg.content)) {
                newMsg.content = newMsg.content.map((block: any) => {
                    const newBlock = { ...block };
                    if (newBlock.type === 'text' && typeof newBlock.text === 'string' && newBlock.text.length > 400) {
                        newBlock.text = newBlock.text.slice(0, 400) + "\n... [TRUNCATED BLOCK]";
                    }
                    if (newBlock.type === 'tool_result' && typeof newBlock.content === 'string' && newBlock.content.length > 400) {
                        newBlock.content = newBlock.content.slice(0, 400) + "\n... [TRUNCATED RESULT]";
                    }
                    return newBlock;
                });
            }
            return newMsg;
        });
    }

    traceStr = JSON.stringify(t);
    if (traceStr.length <= maxChars) {
        return t;
    }

    if (Array.isArray(t.sanitizedMessages) && t.sanitizedMessages.length > 2) {
        t.sanitizedMessages = [
            t.sanitizedMessages[0],
            { role: "system", content: "... [INTERMEDIATE MESSAGES OMITTED FOR SIZE LIMIT] ..." },
            ...t.sanitizedMessages.slice(-2)
        ];
    }

    return t;
}

// Stores trace dataset to local DB or file backup
async function saveQwenAgentTrace(traceData: any) {
    if (!config.qwenAgentTraceEnabled) return;

    try {
        let sanitized = sanitizeObject(traceData);
        const maxChars = config.qwenAgentTraceMaxChars || 20000;
        sanitized = truncateTraceSize(sanitized, maxChars);

        if (pool) {
            try {
                await pool.query(
                    `INSERT INTO qwen_agent_traces 
                    (request_id, timestamp, mode, user_intent, sanitized_messages, available_tool_names, qwen_raw_output, fake_tool_json_detected, fake_tool_json_converted, requested_tool_name, normalized_tool_name, original_tool_args, repaired_tool_args, tool_args_repaired, tool_validation_error, tool_retry_used, tool_round_count, tool_result_preview, final_answer_preview, edited_files, build_status, success, failure_reason, human_verdict, prompt_profile_name, prompt_profile_version, controller_plan, controller_review, qwen_worker_trace_ids, final_result, accepted, repo_key) 
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32)
                    ON CONFLICT (request_id) DO UPDATE SET
                    timestamp = EXCLUDED.timestamp,
                    mode = EXCLUDED.mode,
                    user_intent = EXCLUDED.user_intent,
                    sanitized_messages = EXCLUDED.sanitized_messages,
                    available_tool_names = EXCLUDED.available_tool_names,
                    qwen_raw_output = EXCLUDED.qwen_raw_output,
                    fake_tool_json_detected = EXCLUDED.fake_tool_json_detected,
                    fake_tool_json_converted = EXCLUDED.fake_tool_json_converted,
                    requested_tool_name = EXCLUDED.requested_tool_name,
                    normalized_tool_name = EXCLUDED.normalized_tool_name,
                    original_tool_args = EXCLUDED.original_tool_args,
                    repaired_tool_args = EXCLUDED.repaired_tool_args,
                    tool_args_repaired = EXCLUDED.tool_args_repaired,
                    tool_validation_error = EXCLUDED.tool_validation_error,
                    tool_retry_used = EXCLUDED.tool_retry_used,
                    tool_round_count = EXCLUDED.tool_round_count,
                    tool_result_preview = EXCLUDED.tool_result_preview,
                    final_answer_preview = EXCLUDED.final_answer_preview,
                    edited_files = EXCLUDED.edited_files,
                    build_status = EXCLUDED.build_status,
                    success = EXCLUDED.success,
                    failure_reason = EXCLUDED.failure_reason,
                    human_verdict = EXCLUDED.human_verdict,
                    prompt_profile_name = EXCLUDED.prompt_profile_name,
                    prompt_profile_version = EXCLUDED.prompt_profile_version,
                    controller_plan = EXCLUDED.controller_plan,
                    controller_review = EXCLUDED.controller_review,
                    qwen_worker_trace_ids = EXCLUDED.qwen_worker_trace_ids,
                    final_result = EXCLUDED.final_result,
                    accepted = EXCLUDED.accepted,
                    repo_key = EXCLUDED.repo_key`,
                    [
                        sanitized.requestId,
                        new Date(sanitized.timestamp || Date.now()),
                        sanitized.mode,
                        sanitized.userIntent,
                        JSON.stringify(sanitized.sanitizedMessages),
                        sanitized.availableToolNames,
                        sanitized.qwenRawOutput,
                        sanitized.fakeToolJsonDetected,
                        sanitized.fakeToolJsonConverted,
                        sanitized.requestedToolName,
                        sanitized.normalizedToolName,
                        JSON.stringify(sanitized.originalToolArgs),
                        JSON.stringify(sanitized.repairedToolArgs),
                        sanitized.toolArgsRepaired,
                        sanitized.toolValidationError,
                        sanitized.toolRetryUsed,
                        sanitized.toolRoundCount,
                        sanitized.toolResultPreview,
                        sanitized.finalAnswerPreview,
                        sanitized.editedFiles,
                        sanitized.buildStatus,
                        sanitized.success,
                        sanitized.failureReason,
                        sanitized.humanVerdict || 'unknown',
                        sanitized.promptProfileName || null,
                        sanitized.promptProfileVersion || null,
                        sanitized.controllerPlan || null,
                        sanitized.controllerReview || null,
                        sanitized.qwenWorkerTraceIds || null,
                        sanitized.finalResult || null,
                        sanitized.accepted !== undefined ? sanitized.accepted : null,
                        sanitized.repo_key || sanitized.repoKey || null
                    ]
                );

                // Auto record task memory and failure pattern
                const rKey = sanitized.repo_key || sanitized.repoKey;
                if (rKey) {
                    const isFinalRound = sanitized.finalAnswerPreview && (!sanitized.requestedToolName);
                    const isFailed = !sanitized.success;
                    if (isFinalRound || isFailed) {
                        let costThb = 0;
                        try {
                            const costRes = await pool.query("SELECT COALESCE(SUM(cost_thb), 0) as total FROM model_calls WHERE request_id = $1", [sanitized.requestId]);
                            costThb = Number(costRes.rows[0]?.total || 0);
                        } catch {}

                        await insertTaskMemory({
                            repo_key: rKey,
                            task_summary: sanitized.userIntent || "Qwen agent task",
                            touched_files: sanitized.editedFiles || [],
                            outcome: sanitized.success ? "success" : "failed",
                            model_route: sanitized.mode === "qwen-smart-v2" ? "qwen-smart-v2" : "qwen-only",
                            cost_thb: costThb
                        });

                        if (!sanitized.success) {
                            const failReason = sanitized.failureReason || sanitized.toolValidationError || "Unknown failure";
                            const pType = classifyFailure(failReason);
                            await recordFailurePattern(rKey, pType, failReason);
                        }
                    }
                }
                return;
            } catch (dbErr) {
                console.error("Failed to save trace to DB, falling back to JSONL:", dbErr);
            }
        }

        const traceStr = JSON.stringify(sanitized) + "\n";
        fs.appendFileSync(TRACE_FILE_PATH, traceStr, "utf-8");

    } catch (err) {
        console.error("Failed to save Qwen agent trace:", err);
    }
}

// Reads traces from DB or fallback file
async function readAllTraces(): Promise<any[]> {
    if (pool) {
        try {
            const res = await pool.query("SELECT * FROM qwen_agent_traces ORDER BY timestamp ASC");
            return res.rows.map(row => ({
                id: row.id,
                requestId: row.request_id,
                timestamp: row.timestamp,
                mode: row.mode,
                userIntent: row.user_intent,
                sanitizedMessages: typeof row.sanitized_messages === 'string' ? JSON.parse(row.sanitized_messages) : row.sanitized_messages,
                availableToolNames: row.available_tool_names,
                qwenRawOutput: row.qwen_raw_output,
                fakeToolJsonDetected: row.fake_tool_json_detected,
                fakeToolJsonConverted: row.fake_tool_json_converted,
                requestedToolName: row.requested_tool_name,
                normalizedToolName: row.normalized_tool_name,
                originalToolArgs: typeof row.original_tool_args === 'string' ? JSON.parse(row.original_tool_args) : row.original_tool_args,
                repairedToolArgs: typeof row.repaired_tool_args === 'string' ? JSON.parse(row.repaired_tool_args) : row.repaired_tool_args,
                toolArgsRepaired: row.tool_args_repaired,
                toolValidationError: row.tool_validation_error,
                toolRetryUsed: row.tool_retry_used,
                toolRoundCount: row.tool_round_count,
                toolResultPreview: row.tool_result_preview,
                finalAnswerPreview: row.final_answer_preview,
                editedFiles: row.edited_files,
                buildStatus: row.build_status,
                success: row.success,
                failureReason: row.failure_reason,
                humanVerdict: row.human_verdict,
                promptProfileName: row.prompt_profile_name,
                promptProfileVersion: row.prompt_profile_version,
                controllerPlan: row.controller_plan,
                controllerReview: row.controller_review,
                qwenWorkerTraceIds: row.qwen_worker_trace_ids,
                finalResult: row.final_result,
                accepted: row.accepted
            }));
        } catch (dbErr) {
            console.error("Failed to read traces from DB, reading from JSONL file instead:", dbErr);
        }
    }

    if (!fs.existsSync(TRACE_FILE_PATH)) {
        return [];
    }

    const content = fs.readFileSync(TRACE_FILE_PATH, "utf-8");
    const lines = content.split("\n").filter(line => line.trim() !== "");
    const traces: any[] = [];
    for (const line of lines) {
        try {
            traces.push(JSON.parse(line));
        } catch {}
    }
    return traces;
}

// Processes the request using direct adapter rules
export async function handleQwenAgentRequest(req: Request, res: Response): Promise<void> {
    const requestId = (req as any).requestId || crypto.randomUUID();
    const startTime = Date.now();
    const clientHeaders: Record<string, string> = {
        "user-agent": req.header("user-agent") || "railway-ai-gateway"
    };

    const messages = req.body.messages || [];
    const toolRoundCount = getToolRoundCount(messages);

    const hasTools = Array.isArray(req.body.tools) && req.body.tools.length > 0;
    const isStream = hasTools ? false : !!req.body.stream;

    const rawRepoKey = (req.headers["x-repo-key"] as string) || 
                       (req.headers["x-repo-path"] as string) || 
                       req.body.repo_key || 
                       req.body.repo_path || 
                       (req.query.repoKey as string) || 
                       (req.query.repo_key as string) || 
                       "";
    const repoKey = rawRepoKey ? normalizeRepoKey(rawRepoKey) : "";

    // Build initial trace values
    const traceData: any = {
        requestId,
        timestamp: new Date().toISOString(),
        mode: "qwen-agent",
        userIntent: getLatestUserIntent(messages),
        sanitizedMessages: messages,
        availableToolNames: hasTools ? req.body.tools.map((t: any) => t.name) : [],
        qwenRawOutput: "",
        fakeToolJsonDetected: false,
        fakeToolJsonConverted: false,
        requestedToolName: null,
        normalizedToolName: null,
        originalToolArgs: null,
        repairedToolArgs: null,
        toolArgsRepaired: false,
        toolValidationError: null,
        toolRetryUsed: false,
        toolRoundCount,
        toolResultPreview: "",
        finalAnswerPreview: "",
        editedFiles: [],
        buildStatus: "not_run",
        success: false,
        failureReason: null,
        humanVerdict: "unknown",
        repoKey: repoKey,
        repo_key: repoKey
    };

    // Populate build status / result preview from historical last action
    const histData = extractBuildStatusAndResult(messages);
    traceData.buildStatus = histData.buildStatus;
    traceData.toolResultPreview = histData.toolResultPreview;

    // 1. Max round limit check
    if (toolRoundCount >= config.qwenAgentMaxToolRounds) {
        const stopMsg = "Qwen-agent stopped: max tool rounds reached.";
        const payload = {
            id: "msg_qwen_max_rounds_" + Math.random().toString(36).substring(7),
            type: "message",
            role: "assistant",
            content: [{ type: "text", text: stopMsg }],
            model: "qwen-agent",
            stop_reason: "end_turn",
            stop_sequence: null,
            usage: { input_tokens: 0, output_tokens: 0 }
        };

        traceData.success = false;
        traceData.failureReason = stopMsg;
        traceData.finalAnswerPreview = stopMsg;

        await saveQwenAgentTrace(traceData);

        console.log(JSON.stringify({
            time: new Date().toISOString(),
            requestId,
            mode: "qwen-agent",
            provider: "qwen-local",
            finalProvider: "qwen-local",
            deepseekFallbackUsed: false,
            requestedToolName: null,
            normalizedToolName: null,
            fakeToolJsonDetected: false,
            fakeToolJsonConverted: false,
            toolArgsRepaired: false,
            toolValidationError: "max_tool_rounds_exceeded",
            toolRetryUsed: false,
            toolRoundCount,
            status: 200
        }));

        res.json(payload);
        return;
    }

    // 2. Setup Qwen Provider
    const provider = providerRegistry.getProvider("qwen-local") as QwenLocalProvider | undefined;
    if (!provider) {
        res.status(503).json({
            error: {
                type: "server_error",
                message: "Qwen local provider is not registered"
            }
        });
        return;
    }

    // 3. System instruction injection
    const finalBody = { ...req.body };

    const rules = await getEnabledAdapterRules();
    const activeProfile = await getActivePromptProfile();
    const profileName = activeProfile ? activeProfile.name : "qwen-agent-default";
    const profileVersion = activeProfile ? "1.0.0" : "default";

    traceData.promptProfileName = profileName;
    traceData.promptProfileVersion = profileVersion;

    let systemPrompt = activeProfile ? activeProfile.system_prompt : QWEN_AGENT_SYSTEM_INSTRUCTION;
    systemPrompt = await applySystemPromptHintRules(systemPrompt, traceData.userIntent, rules);

    if (repoKey) {
        const memContext = await buildMemoryPromptContext(repoKey);
        if (memContext) {
            systemPrompt = memContext + "\n\n" + systemPrompt;
        }
    }

    finalBody.system = (finalBody.system ? finalBody.system + "\n\n" : "") + systemPrompt;
    finalBody.stream = isStream;

    const runtimeConfig = await provider.resolveRuntimeConfig();
    const activeModelName = runtimeConfig.modelName;

    let responseBody: any = null;
    let rawText = "";

    try {
        const upstream = await provider.handleRequest(finalBody, clientHeaders);
        if (!upstream.ok) {
            const errorText = await upstream.text();
            res.status(upstream.status).json({
                error: {
                    type: "api_error",
                    message: errorText || "Local AI is offline"
                }
            });
            return;
        }

        // Check if streamed response
        if (isStream && upstream.body) {
            res.status(upstream.status);
            const contentType = upstream.headers.get("content-type");
            if (contentType) res.setHeader("content-type", contentType);
            res.setHeader("cache-control", "no-cache");
            res.setHeader("x-accel-buffering", "no");
            res.setHeader("connection", "keep-alive");

            const reader = upstream.body.getReader();
            const decoder = new TextDecoder();
            let accumulatedText = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                res.write(Buffer.from(value));
                accumulatedText += decoder.decode(value, { stream: true });
            }
            res.end();

            traceData.success = true;
            traceData.finalAnswerPreview = accumulatedText;
            await saveQwenAgentTrace(traceData);
            await updateProfileStats(profileName, true);
            return;
        }

        rawText = await upstream.text();
        responseBody = JSON.parse(rawText);
    } catch (err: any) {
        res.status(503).json({
            error: {
                type: "api_error",
                message: err.message || "Failed calling Qwen local provider"
            }
        });
        return;
    }

    if (!responseBody || typeof responseBody !== "object") {
        res.status(502).json({
            error: {
                type: "api_error",
                message: "Qwen local returned malformed response"
            }
        });
        return;
    }

    let processedResponse = { ...responseBody };
    let hasRetryHappened = false;

    // Inner processing loop for parsing/normalizing/validating tool call
    async function processAgentResponse(body: any): Promise<{ valid: boolean; errorReason: string | null }> {
        const content = body.content;
        if (!Array.isArray(content)) {
            return { valid: true, errorReason: null };
        }

        // Trace raw Qwen output
        const textBlocks = content.filter(b => b?.type === "text").map(b => b.text || "").join("\n");
        traceData.qwenRawOutput = textBlocks;

        // A. Fake JSON Tool parsing
        let hasFakeJson = false;
        let fakeCall: FakeToolCall | null = null;
        let fakeBlockIndex = -1;

        for (let i = 0; i < content.length; i++) {
            const block = content[i];
            if (block?.type === "text") {
                const parsed = parseFakeToolJson(block.text);
                if (parsed) {
                    hasFakeJson = true;
                    fakeCall = parsed;
                    fakeBlockIndex = i;
                    break;
                }
            }
        }

        if (hasFakeJson && fakeCall) {
            traceData.fakeToolJsonDetected = true;
            traceData.fakeToolJsonConverted = true;

            const toolUseBlock = {
                type: "tool_use",
                id: "toolu_qwen_" + crypto.randomUUID().replace(/-/g, "").substring(0, 16),
                name: fakeCall.name,
                input: fakeCall.input
            };

            const newContent = [...content];
            newContent[fakeBlockIndex] = toolUseBlock;
            body.content = newContent;
            body.stop_reason = "tool_use";
        }

        // B. Tool normalizer & repair & schema validation
        let firstValidationError: string | null = null;

        for (const block of body.content) {
            if (block?.type === "tool_use") {
                traceData.requestedToolName = block.name;

                // Apply tool alias rules
                const aliasResult = await applyToolAliasRules(block.name, rules);
                block.name = aliasResult.name;
                if (aliasResult.hitRuleId) {
                    await incrementRuleHit(aliasResult.hitRuleId);
                }

                // Normalize name
                const normName = normalizeToolName(block.name);
                block.name = normName;
                traceData.normalizedToolName = normName;

                // Save original args
                traceData.originalToolArgs = block.input;

                // Apply arg alias rules
                const argAliasResult = await applyArgAliasRules(normName, block.input, rules);
                block.input = argAliasResult.repairedInput;
                if (argAliasResult.hitRuleId) {
                    await incrementRuleHit(argAliasResult.hitRuleId);
                    traceData.toolArgsRepaired = true;
                }

                // Repair args fallback
                const { repairedInput, repaired } = repairToolArgs(normName, block.input);
                block.input = repairedInput;
                traceData.repairedToolArgs = repairedInput;
                if (repaired) {
                    traceData.toolArgsRepaired = true;
                }

                // Apply dangerous bash command rules
                if (normName === "Bash" && block.input?.command) {
                    const blockCheck = await checkBashBlockRules(block.input.command, rules);
                    if (blockCheck.blocked) {
                        firstValidationError = `Dangerous command blocked: ${blockCheck.reason}`;
                        traceData.toolValidationError = firstValidationError;
                        if (blockCheck.hitRuleId) {
                            await incrementRuleHit(blockCheck.hitRuleId);
                        }
                        break;
                    }
                }

                // Scheme validation
                const valError = validateToolCall(normName, block.input);
                if (valError) {
                    firstValidationError = valError;
                    traceData.toolValidationError = valError;
                    break;
                }
            }
        }

        if (firstValidationError) {
            return { valid: false, errorReason: firstValidationError };
        }

        return { valid: true, errorReason: null };
    }

    let validation = await processAgentResponse(processedResponse);

    // 4. Compact retry once if validation fails
    if (!validation.valid && validation.errorReason) {
        hasRetryHappened = true;
        traceData.toolRetryUsed = true;

        try {
            // Apply retry hint rule
            let retryHint = "";
            const hintResult = await getRetryHintRule(validation.errorReason, rules);
            if (hintResult.hint) {
                retryHint = `\nHint: ${hintResult.hint}`;
                if (hintResult.hitRuleId) {
                    await incrementRuleHit(hintResult.hitRuleId);
                }
            }

            // Append previous invalid assistant response + retry instruction user message
            const retryMessages = [
                ...messages,
                {
                    role: "assistant",
                    content: processedResponse.content
                },
                {
                    role: "user",
                    content: `Your previous tool call was invalid: ${validation.errorReason}. Return exactly one valid tool_use. Do not explain.${retryHint}`
                }
            ];

            const retryBody = {
                ...finalBody,
                messages: retryMessages
            };

            const retryUpstream = await provider.handleRequest(retryBody, clientHeaders);
            if (retryUpstream.ok) {
                const retryRawText = await retryUpstream.text();
                const retryResponse = JSON.parse(retryRawText);
                
                // Re-evaluate retry response
                processedResponse = { ...retryResponse };
                validation = await processAgentResponse(processedResponse);
            }
        } catch (err: any) {
            console.error("Retry attempt failed:", err);
        }
    }

    // 5. Final Output Handling
    if (!validation.valid) {
        // Validation failed twice
        const fallbackText = "Qwen-agent failed to produce a valid tool call.";
        const failedResponse = {
            id: "msg_qwen_failed_" + Math.random().toString(36).substring(7),
            type: "message",
            role: "assistant",
            content: [{ type: "text", text: fallbackText }],
            model: "qwen-agent",
            stop_reason: "end_turn",
            stop_sequence: null,
            usage: responseBody?.usage || { input_tokens: 0, output_tokens: 0 }
        };

        traceData.success = false;
        traceData.failureReason = validation.errorReason || "Failed tool validation";
        traceData.finalAnswerPreview = fallbackText;

        await saveQwenAgentTrace(traceData);
        await updateProfileStats(profileName, false);

        console.log(JSON.stringify({
            time: new Date().toISOString(),
            requestId,
            mode: "qwen-agent",
            provider: "qwen-local",
            finalProvider: "qwen-local",
            deepseekFallbackUsed: false,
            requestedToolName: traceData.requestedToolName,
            normalizedToolName: traceData.normalizedToolName,
            fakeToolJsonDetected: traceData.fakeToolJsonDetected,
            fakeToolJsonConverted: traceData.fakeToolJsonConverted,
            toolArgsRepaired: traceData.toolArgsRepaired,
            toolValidationError: traceData.toolValidationError,
            toolRetryUsed: hasRetryHappened,
            toolRoundCount,
            status: 200
        }));

        res.json(failedResponse);
        return;
    }

    // Success response path
    traceData.success = true;
    traceData.editedFiles = extractEditedFiles(processedResponse.content);
    
    const finalBlocks = processedResponse.content || [];
    const answerText = finalBlocks.filter((b: any) => b?.type === "text").map((b: any) => b.text || "").join("\n");
    traceData.finalAnswerPreview = answerText;

    await saveQwenAgentTrace(traceData);
    await updateProfileStats(profileName, true);

    console.log(JSON.stringify({
        time: new Date().toISOString(),
        requestId,
        mode: "qwen-agent",
        provider: "qwen-local",
        finalProvider: "qwen-local",
        deepseekFallbackUsed: false,
        requestedToolName: traceData.requestedToolName,
        normalizedToolName: traceData.normalizedToolName,
        fakeToolJsonDetected: traceData.fakeToolJsonDetected,
        fakeToolJsonConverted: traceData.fakeToolJsonConverted,
        toolArgsRepaired: traceData.toolArgsRepaired,
        toolValidationError: null,
        toolRetryUsed: hasRetryHappened,
        toolRoundCount,
        status: 200
    }));

    res.json(processedResponse);
}

// Streams out formatted tuning dataset
export async function exportQwenAgentTraces(req: Request, res: Response) {
    try {
        const traces = await readAllTraces();
        const formatted = traces.map(t => {
            let expectedTool = null;
            if (t.normalizedToolName) {
                expectedTool = {
                    name: t.normalizedToolName,
                    arguments: t.repairedToolArgs || {}
                };
            }
            return {
                messages: t.sanitizedMessages || [],
                expected_tool: expectedTool,
                tool_result: t.toolResultPreview || "",
                final_answer: t.finalAnswerPreview || "",
                metadata: {
                    requestId: t.requestId,
                    timestamp: t.timestamp,
                    mode: t.mode,
                    success: t.success,
                    failureReason: t.failureReason,
                    toolRoundCount: t.toolRoundCount,
                    toolValidationError: t.toolValidationError,
                    toolArgsRepaired: t.toolArgsRepaired,
                    fakeToolJsonDetected: t.fakeToolJsonDetected,
                    fakeToolJsonConverted: t.fakeToolJsonConverted,
                    requestedToolName: t.requestedToolName,
                    normalizedToolName: t.normalizedToolName,
                    editedFiles: t.editedFiles,
                    buildStatus: t.buildStatus,
                    humanVerdict: t.humanVerdict || "unknown"
                }
            };
        });

        if (req.query.format === "jsonl") {
            res.setHeader("content-type", "application/x-jsonlines");
            res.write(formatted.map(row => JSON.stringify(row)).join("\n") + "\n");
            res.end();
        } else {
            res.setHeader("content-type", "application/json");
            res.json(formatted);
        }
    } catch (err: any) {
        res.status(500).json({ error: err.message });
    }
}

// Provides summary metrics on collected agent runs
export async function getQwenAgentTracesSummary(req: Request, res: Response) {
    try {
        const traces = await readAllTraces();
        const totalTraces = traces.length;

        if (totalTraces === 0) {
            return res.json({
                totalTraces: 0,
                successRate: 0,
                fakeJsonRate: 0,
                toolConversionRate: 0,
                argRepairRate: 0,
                topInvalidTools: {},
                topValidationErrors: {},
                retrySuccessRate: 0,
                maxToolRoundHits: 0,
                mostCommonFailureReasons: {}
            });
        }

        let successCount = 0;
        let fakeJsonCount = 0;
        let toolConversionCount = 0;
        let argRepairCount = 0;
        let retryUsedCount = 0;
        let retrySuccessCount = 0;
        let maxToolRoundHits = 0;

        const invalidToolsMap: Record<string, number> = {};
        const validationErrorsMap: Record<string, number> = {};
        const failureReasonsMap: Record<string, number> = {};

        for (const t of traces) {
            if (t.success) successCount++;
            if (t.fakeToolJsonDetected) fakeJsonCount++;
            if (t.fakeToolJsonConverted) toolConversionCount++;
            if (t.toolArgsRepaired) argRepairCount++;
            if (t.toolRetryUsed) {
                retryUsedCount++;
                if (t.success) retrySuccessCount++;
            }
            if (t.failureReason && t.failureReason.toLowerCase().includes("max tool rounds")) {
                maxToolRoundHits++;
            }

            if (t.toolValidationError) {
                const toolName = t.requestedToolName || "unknown";
                invalidToolsMap[toolName] = (invalidToolsMap[toolName] || 0) + 1;
                
                const errStr = t.toolValidationError;
                validationErrorsMap[errStr] = (validationErrorsMap[errStr] || 0) + 1;
            }

            if (t.failureReason) {
                const reason = t.failureReason;
                failureReasonsMap[reason] = (failureReasonsMap[reason] || 0) + 1;
            }
        }

        const getSortedEntries = (map: Record<string, number>) => {
            return Object.entries(map)
                .sort((a, b) => b[1] - a[1])
                .reduce((acc, [key, val]) => {
                    acc[key] = val;
                    return acc;
                }, {} as Record<string, number>);
        };

        res.json({
            totalTraces,
            successRate: Number((successCount / totalTraces).toFixed(4)),
            fakeJsonRate: Number((fakeJsonCount / totalTraces).toFixed(4)),
            toolConversionRate: Number((toolConversionCount / totalTraces).toFixed(4)),
            argRepairRate: Number((argRepairCount / totalTraces).toFixed(4)),
            topInvalidTools: getSortedEntries(invalidToolsMap),
            topValidationErrors: getSortedEntries(validationErrorsMap),
            retrySuccessRate: retryUsedCount > 0 ? Number((retrySuccessCount / retryUsedCount).toFixed(4)) : 0,
            maxToolRoundHits,
            mostCommonFailureReasons: getSortedEntries(failureReasonsMap)
        });
    } catch (err: any) {
        res.status(500).json({ error: err.message });
    }
}

// Wrappers for exact trace logger function names requested in Phase 1
export async function insertQwenAgentTrace(traceData: any) {
    return saveQwenAgentTrace(traceData);
}

export function sanitizeTracePayload(payload: any): any {
    return sanitizeObject(payload);
}

export function truncateTracePayload(trace: any, maxChars: number): any {
    return truncateTraceSize(trace, maxChars);
}

export async function exportQwenAgentTracesJsonl(req: Request, res: Response) {
    req.query.format = "jsonl";
    return exportQwenAgentTraces(req, res);
}

export async function getQwenAgentTraceSummary(req: Request, res: Response) {
    return getQwenAgentTracesSummary(req, res);
}

