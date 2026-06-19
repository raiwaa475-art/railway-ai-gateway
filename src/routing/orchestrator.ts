import { Request, Response as ExpressResponse } from "express";
import crypto from "crypto";
import { providerRegistry } from "./registry.js";
import { DeepSeekProvider } from "../providers/deepseek.js";
import { QwenLocalProvider } from "../providers/qwen-local.js";
import { config } from "../config/env.js";
import { insertModelCall, updateGatewayRequest } from "../utils/db.js";

function hasToolResults(messages: any[]): boolean {
    return messages.some(msg => {
        if (!Array.isArray(msg.content)) return false;
        return msg.content.some((block: any) => block?.type === "tool_result");
    });
}

function getLastRealUserInstruction(messages: any[]): string {
    for (const msg of messages.slice().reverse()) {
        if (msg.role !== "user") continue;

        if (typeof msg.content === "string") {
            const text = msg.content.trim();
            if (text) return text;
        }

        if (Array.isArray(msg.content)) {
            const text = msg.content
                .filter((block: any) => block?.type === "text")
                .map((block: any) => String(block.text || ""))
                .join("\n")
                .trim();

            if (text) return text;
        }
    }

    return "";
}

function isLikelyCodeEditTask(text: string): boolean {
    const normalized = String(text || "").toLowerCase();
    const explicitReadOnlyPatterns = [
        "ห้ามแก้",
        "ห้ามแก้ไฟล์",
        "ไม่ต้องแก้",
        "อย่าแก้",
        "read only",
        "do not edit",
        "don't edit",
        "no edit"
    ];
    if (explicitReadOnlyPatterns.some(pattern => normalized.includes(pattern))) {
        return false;
    }

    const codeEditKeywords = [
        "แก้",
        "แก้ไข",
        "ปรับ",
        "เปลี่ยน",
        "เพิ่ม",
        "ลบ",
        "ทำให้",
        "เขียน",
        "สร้าง",
        "ใส่",
        "update",
        "edit",
        "fix",
        "change",
        "implement",
        "patch",
        "refactor",
        "bug",
        "error",
        "build",
        "test",
        "css",
        "html",
        "api",
        "route",
        "endpoint",
        "component",
        "function"
    ];

    if (codeEditKeywords.some(keyword => normalized.includes(keyword))) {
        return true;
    }

    return /(^|[\s./\\_-])(tsx?|jsx?)(\b|$)/i.test(normalized);
}

function hasUsefulCodeContext(messages: any[]): boolean {
    const codeContextMarkers = [
        "function",
        "const",
        "import",
        "export",
        "class=",
        "<html",
        "route",
        "endpoint"
    ];

    for (const msg of messages.slice(-10)) {
        if (!Array.isArray(msg.content)) continue;

        for (const block of msg.content) {
            if (block?.type !== "tool_result") continue;

            const content = typeof block.content === "string"
                ? block.content
                : JSON.stringify(block.content);
            const normalized = content.toLowerCase();

            if (content.length > 300 || codeContextMarkers.some(marker => normalized.includes(marker))) {
                return true;
            }
        }
    }

    return false;
}

interface DeterministicRouterDecision {
    delegate_to_qwen: boolean;
    reason: string;
    userIntentPreview: string;
    likelyCodeEdit: boolean;
    usefulCodeContext: boolean;
}

type QwenDraftMode = "find_replace" | "unified_diff" | "replacement_snippet" | "snippet" | "notes" | "insufficient_context" | "empty";

interface ParsedQwenPatch {
    ok: boolean;
    filePath?: string;
    find?: string;
    replace?: string;
    mode: "find_replace" | "unified_diff" | "invalid";
    reason?: string;
}

function detectQwenDraftMode(text: string): QwenDraftMode {
    const t = String(text || "").trim();
    if (!t) return "empty";
    if (t.startsWith("INSUFFICIENT_CONTEXT")) return "insufficient_context";
    if (t.includes("---") && t.includes("+++") && t.includes("@@")) return "unified_diff";
    if (t.includes("FILE:") && t.includes("FIND:") && t.includes("REPLACE:")) return "find_replace";
    if (
        t.includes("function") ||
        t.includes("const ") ||
        t.includes("let ") ||
        t.includes("import ") ||
        t.includes("export ") ||
        t.includes("<") ||
        t.includes("class=") ||
        t.includes("{")
    ) {
        return "snippet";
    }
    return "notes";
}

function isUsableQwenDraft(mode: QwenDraftMode, chars: number): boolean {
    return mode === "unified_diff" ||
        mode === "find_replace" ||
        mode === "replacement_snippet" ||
        (mode === "snippet" && chars >= 200);
}

function shouldRetryQwenDraft(mode: QwenDraftMode, chars: number): boolean {
    return mode === "notes" || mode === "empty" || (mode === "snippet" && chars < 200);
}

function shouldDelegateToQwen(messages: any[]): DeterministicRouterDecision {
    const hasResults = hasToolResults(messages);
    const userIntent = getLastRealUserInstruction(messages);
    const likelyCodeEdit = isLikelyCodeEditTask(userIntent);
    const usefulCodeContext = hasUsefulCodeContext(messages);
    const delegate_to_qwen = hasResults && likelyCodeEdit && usefulCodeContext;

    let reason = "Deterministic router approved Qwen delegation";
    if (!hasResults) {
        reason = "No tool_result context yet";
    } else if (!likelyCodeEdit) {
        reason = "Latest user intent is not a code edit task";
    } else if (!usefulCodeContext) {
        reason = "Tool results do not contain useful code context";
    }

    return {
        delegate_to_qwen,
        reason,
        userIntentPreview: userIntent.slice(0, 120),
        likelyCodeEdit,
        usefulCodeContext
    };
}

function getTargetFileFromRecentToolUse(messages: any[]): string {
    for (const msg of messages.slice(-10).reverse()) {
        if (!Array.isArray(msg.content)) continue;

        for (const block of msg.content) {
            if (block?.type !== "tool_use") continue;
            const input = block.input || {};
            const target = input.file_path || input.path;
            if (typeof target === "string" && target.trim()) {
                return target.trim();
            }
        }
    }

    return "";
}

function getRecentToolResultText(messages: any[], maxChars = 12000): string {
    const parts: string[] = [];

    for (const msg of messages.slice(-10)) {
        if (!Array.isArray(msg.content)) continue;

        for (const block of msg.content) {
            if (block?.type !== "tool_result") continue;
            const content = typeof block.content === "string"
                ? block.content
                : JSON.stringify(block.content);
            parts.push(content);
        }
    }

    return parts.join("\n\n---\n\n").slice(-maxChars);
}

function hasQwenEditToolResult(messages: any[]): boolean {
    return messages.some(msg => {
        if (!Array.isArray(msg.content)) return false;
        return msg.content.some((block: any) =>
            block?.type === "tool_result" &&
            typeof block.tool_use_id === "string" &&
            block.tool_use_id.startsWith("toolu_qwen_edit_")
        );
    });
}

function countOccurrences(haystack: string, needle: string): number {
    if (!needle) return 0;

    let count = 0;
    let index = 0;
    while (true) {
        index = haystack.indexOf(needle, index);
        if (index === -1) break;
        count++;
        index += needle.length;
    }
    return count;
}

function parseQwenFindReplacePatch(text: string): ParsedQwenPatch {
    const t = String(text || "").trim();
    if (!t) {
        return { ok: false, mode: "invalid", reason: "empty_patch" };
    }

    if (t.includes("---") && t.includes("+++") && t.includes("@@")) {
        return { ok: false, mode: "unified_diff", reason: "patch_parse_unsupported" };
    }

    const fileMatch = t.match(/(?:^|\n)FILE:\s*(.+?)(?=\n)/);
    const fileMarkers = t.match(/(?:^|\n)FILE:/g) || [];
    if (fileMarkers.length > 1) {
        return { ok: false, mode: "invalid", reason: "multiple_files_unsupported" };
    }

    const findMarker = "\nFIND:";
    const replaceMarker = "\nREPLACE:";
    const findIndex = t.indexOf(findMarker);
    const replaceIndex = t.indexOf(replaceMarker);

    if (!fileMatch || findIndex === -1 || replaceIndex === -1 || replaceIndex <= findIndex) {
        return { ok: false, mode: "invalid", reason: "missing_file_find_or_replace" };
    }

    const filePath = fileMatch[1].trim();
    const find = t.slice(findIndex + findMarker.length, replaceIndex).trim();
    const replace = t.slice(replaceIndex + replaceMarker.length).trim();

    return {
        ok: true,
        filePath,
        find,
        replace,
        mode: "find_replace"
    };
}

function validateQwenPatch(
    patch: ParsedQwenPatch,
    messages: any[],
    userIntent: string
): ParsedQwenPatch {
    if (!patch.ok) return patch;

    const filePath = patch.filePath || "";
    const find = patch.find || "";
    const replace = patch.replace || "";
    const targetFile = getTargetFileFromRecentToolUse(messages);
    const context = getRecentToolResultText(messages);
    const normalizedIntent = userIntent.toLowerCase();
    const explicitlyRequestedFile = !!filePath && normalizedIntent.includes(filePath.toLowerCase());
    const explicitlyLargeRewrite = /rewrite|large|full|entire|ทั้งไฟล์|เขียนใหม่ทั้งหมด/i.test(userIntent);
    const blockedPathPatterns = [
        ".env",
        "secrets",
        "node_modules",
        "dist",
        "build",
        "package-lock.json"
    ];

    if (!filePath) {
        return { ...patch, ok: false, reason: "missing_file" };
    }
    if (!find) {
        return { ...patch, ok: false, reason: "empty_find" };
    }
    if (!replace) {
        return { ...patch, ok: false, reason: "empty_replace" };
    }
    if (targetFile && filePath !== targetFile) {
        return { ...patch, ok: false, reason: "file_mismatch" };
    }
    if (blockedPathPatterns.some(pattern => filePath.includes(pattern)) && !explicitlyRequestedFile) {
        return { ...patch, ok: false, reason: "blocked_file_path" };
    }
    if (replace.length > find.length * 3 && !explicitlyLargeRewrite) {
        return { ...patch, ok: false, reason: "replace_too_large" };
    }

    const occurrences = countOccurrences(context, find);
    if (occurrences !== 1) {
        return { ...patch, ok: false, reason: occurrences === 0 ? "find_not_in_context" : "find_not_unique" };
    }

    return patch;
}

function extractReducedContext(messages: any[], userIntent = "", maxChars = 8000): string {
    const toolResultParts: string[] = [];
    const fileContentParts: string[] = [];
    const targetFile = getTargetFileFromRecentToolUse(messages);

    for (const msg of messages.slice(-8)) {
        if (!Array.isArray(msg.content)) continue;

        for (const block of msg.content) {
            if (block?.type === "tool_result") {
                const content = typeof block.content === "string"
                    ? block.content
                    : JSON.stringify(block.content);

                const clipped = content.slice(0, 3000);
                toolResultParts.push(clipped);

                if (
                    content.includes("function") ||
                    content.includes("const ") ||
                    content.includes("import ") ||
                    content.includes("export ") ||
                    content.includes("<html")
                ) {
                    fileContentParts.push(clipped);
                }
            }

            if (block?.type === "text") {
                toolResultParts.push(`TEXT:\n${String(block.text).slice(0, 1500)}`);
            }
        }
    }

    const context = [
        `USER_INTENT:\n${userIntent || getLastRealUserInstruction(messages)}`,
        `TARGET_FILE:\n${targetFile || "unknown"}`,
        `RECENT_TOOL_RESULTS:\n${toolResultParts.join("\n\n---\n\n")}`,
        `RELEVANT_FILE_CONTENT:\n${(fileContentParts.length ? fileContentParts : toolResultParts).join("\n\n---\n\n")}`,
        "TASK:\nWrite the actual patch for the requested change."
    ].join("\n\n");

    return context.slice(-maxChars);
}

function extractJsonFromString(str: string): any {
    // Remove think blocks and special tags if any
    let cleaned = str.replace(/<think>[\s\S]*?<\/think>/gi, "");
    cleaned = cleaned.replace(/<｜｜DSML｜｜thought>[\s\S]*?<\/thought>/gi, "");
    cleaned = cleaned.replace(/<｜｜DSML｜｜thought>/g, ""); // strip raw prefix tags if not closed
    cleaned = cleaned.replace(/<\|[\s\S]*?\|>/g, ""); // strip other special tokens

    // Locate the first { and the last }
    const firstBrace = cleaned.indexOf("{");
    const lastBrace = cleaned.lastIndexOf("}");

    if (firstBrace === -1 || lastBrace === -1 || lastBrace < firstBrace) {
        throw new Error("Could not find a valid JSON object block in model response: " + str.slice(0, 100));
    }

    const jsonSub = cleaned.substring(firstBrace, lastBrace + 1);
    return JSON.parse(jsonSub);
}

function extractJsonObject(text: string): any {
    const cleaned = String(text || "")
        .replace(/```json/gi, "```")
        .replace(/```/g, "")
        .trim();

    const start = cleaned.indexOf("{");
    const end = cleaned.lastIndexOf("}");

    if (start === -1 || end === -1 || end <= start) {
        throw new Error("No JSON object found");
    }

    const jsonText = cleaned.slice(start, end + 1);
    return JSON.parse(jsonText);
}

function getTextFromAnthropicResponse(data: any): string {
    if (!data) return "";

    if (typeof data === "string") return data;

    if (Array.isArray(data.content)) {
        return data.content
            .filter((block: any) => block?.type === "text")
            .map((block: any) => block.text || "")
            .join("\n")
            .trim();
    }

    return "";
}

interface RouterDecision {
    delegate_to_qwen: boolean;
    task_type: string;
    reason: string;
    qwen_instruction: string;
}

function fallbackDelegationFromText(text: string): RouterDecision {
    const compact = String(text || "").replace(/\s+/g, "");

    if (compact.includes('"delegate_to_qwen":true')) {
        return {
            delegate_to_qwen: true,
            task_type: "code_edit",
            reason: "Recovered from router text fallback",
            qwen_instruction: "Draft a concise patch or implementation suggestion from the provided reduced context."
        };
    }

    return {
        delegate_to_qwen: false,
        task_type: "unknown",
        reason: "Router parse failed",
        qwen_instruction: ""
    };
}

async function askDeepSeekPatchApproval(
    deepseekProvider: DeepSeekProvider,
    clientHeaders: Record<string, string>,
    requestId: string,
    params: {
        userIntent: string;
        filePath: string;
        patchMode: string;
        findLength: number;
        replaceLength: number;
        riskFlags: string[];
    }
): Promise<{ approved: boolean; reason: string; inputTokens: number; outputTokens: number; latencyMs: number }> {
    const body = {
        model: config.defaultModel,
        system: `You approve or reject a validated code patch.
Return ONLY one valid JSON object.
No markdown.
No explanation.
Shape: {"approved":true,"reason":"short reason"}`,
        messages: [
            {
                role: "user",
                content: JSON.stringify({
                    userIntent: params.userIntent.slice(0, 300),
                    filePath: params.filePath,
                    patchMode: params.patchMode,
                    findLength: params.findLength,
                    replaceLength: params.replaceLength,
                    riskFlags: params.riskFlags
                })
            }
        ],
        stream: false,
        temperature: 0,
        max_tokens: 150
    };

    const startTime = Date.now();
    const approvalRes = await deepseekProvider.handleRequest(body, clientHeaders);
    const latencyMs = Date.now() - startTime;
    if (!approvalRes.ok) {
        return {
            approved: false,
            reason: `approval_http_status_${approvalRes.status}`,
            inputTokens: 0,
            outputTokens: 0,
            latencyMs
        };
    }

    const data = await approvalRes.json();
    const text = getTextFromAnthropicResponse(data);
    let approved = false;
    let reason = "approval_parse_failed";
    try {
        const parsed = extractJsonObject(text);
        approved = parsed?.approved === true;
        reason = typeof parsed?.reason === "string" ? parsed.reason : reason;
    } catch {}

    await insertModelCall({
        requestId,
        provider: "deepseek",
        model: `${config.defaultModel}-patch-approval`,
        inputTokens: data.usage?.input_tokens || 0,
        outputTokens: data.usage?.output_tokens || 0,
        cacheHitInputTokens: data.usage?.cache_read_input_tokens || 0,
        cacheMissInputTokens: (data.usage?.input_tokens || 0) - (data.usage?.cache_read_input_tokens || 0),
        latencyMs
    });

    return {
        approved,
        reason,
        inputTokens: data.usage?.input_tokens || 0,
        outputTokens: data.usage?.output_tokens || 0,
        latencyMs
    };
}

export class OrchestratorService {
    private static async askDeepSeekDelegationRouter(
        messages: any[],
        clientHeaders: Record<string, string>,
        requestId: string
    ): Promise<RouterDecision> {
        const deepseekProvider = providerRegistry.getProvider("deepseek") as DeepSeekProvider;
        if (!deepseekProvider) {
            throw new Error("DeepSeek provider not registered.");
        }

        // Format a summarized text context instead of sending raw tool loop messages
        let contextSummary = "";
        for (const msg of messages.slice(-6)) {
            if (msg.role === "user") {
                if (typeof msg.content === "string") {
                    contextSummary += `User: ${msg.content}\n`;
                } else if (Array.isArray(msg.content)) {
                    for (const block of msg.content) {
                        if (block?.type === "text") {
                            contextSummary += `User Text: ${block.text}\n`;
                        } else if (block?.type === "tool_result") {
                            const resStr = typeof block.content === "string" 
                                ? block.content 
                                : JSON.stringify(block.content);
                            contextSummary += `Tool Result [Use ID: ${block.tool_use_id}]: ${resStr.slice(0, 1000)}\n`;
                        }
                    }
                }
            } else if (msg.role === "assistant") {
                if (typeof msg.content === "string") {
                    contextSummary += `Assistant: ${msg.content}\n`;
                } else if (Array.isArray(msg.content)) {
                    for (const block of msg.content) {
                        if (block?.type === "text") {
                            contextSummary += `Assistant Text: ${block.text}\n`;
                        } else if (block?.type === "tool_use") {
                            contextSummary += `Assistant requested Tool Use: ${block.name} (input: ${JSON.stringify(block.input)})\n`;
                        }
                    }
                }
            }
        }

        const routerPrompt = `You are the delegation router for a coding gateway.
Decide whether the current request should call Qwen Local as an internal code draft generator.

Rules:
- Return ONLY one valid JSON object.
- No markdown.
- No code fence.
- No explanation.
- No text before or after JSON.
- Do not include reasoning.
- Do not include comments.
- delegate_to_qwen=true only when:
  1. the user wants code to be written, edited, fixed, refactored, or generated
  2. there is enough file/tool context for Qwen to draft a useful patch
  3. Qwen does not need to call tools itself
- delegate_to_qwen=false when:
  - user is just chatting
  - user only asks to read/explain/summarize
  - more files must be read first
  - the task is architecture/planning/review only
  - tool context is missing or insufficient

Expected JSON shape:
{
  "delegate_to_qwen": true,
  "task_type": "code_edit",
  "reason": "short reason",
  "qwen_instruction": "short instruction for Qwen"
}`;

        const routerMessages = [
            {
                role: "user",
                content: `Here is the recent conversation state and tool execution context:\n\n${contextSummary}\n\nBased on this context, decide if we should delegate a coding task to Qwen Local.`
            }
        ];

        const body = {
            model: config.defaultModel,
            system: routerPrompt,
            messages: routerMessages,
            stream: false,
            max_tokens: 300,
            temperature: 0
        };

        const startTime = Date.now();
        const res = await deepseekProvider.handleRequest(body, clientHeaders);
        const latencyMs = Date.now() - startTime;

        if (!res.ok) {
            throw new Error(`Delegation router request failed with status ${res.status}`);
        }

        const routerData = await res.json();
        const routerText = getTextFromAnthropicResponse(routerData);

        let decision: RouterDecision;
        try {
            decision = extractJsonObject(routerText);
        } catch (err) {
            console.error("Delegation router parse failed", {
                requestId,
                routerParseFailed: true,
                responsePreview: routerText.slice(0, 160)
            });
            decision = fallbackDelegationFromText(routerText);
        }

        // Log delegation router model call
        const inputTokens = routerData.usage?.input_tokens || 0;
        const outputTokens = routerData.usage?.output_tokens || 0;
        const cacheReadTokens = routerData.usage?.cache_read_input_tokens || 0;

        await insertModelCall({
            requestId,
            provider: "deepseek",
            model: `${config.defaultModel}-router`,
            inputTokens,
            outputTokens,
            cacheHitInputTokens: cacheReadTokens,
            cacheMissInputTokens: inputTokens - cacheReadTokens,
            latencyMs
        });

        return decision;
    }

    private static async forwardToDeepSeek(
        body: any,
        clientHeaders: Record<string, string>,
        res: ExpressResponse,
        isStream: boolean,
        requestId: string,
        qwenSavings?: { inputTokens: number; outputTokens: number }
    ): Promise<void> {
        const deepseekProvider = providerRegistry.getProvider("deepseek") as DeepSeekProvider;
        if (!deepseekProvider) {
            res.status(500).json({
                error: {
                    type: "gateway_error",
                    message: "DeepSeek provider not registered."
                }
            });
            return;
        }

        const callStartTime = Date.now();
        const deepseekRes = await deepseekProvider.handleRequest(body, clientHeaders);
        res.status(deepseekRes.status);
        const contentType = deepseekRes.headers.get("content-type");
        if (contentType) {
            res.setHeader("content-type", contentType);
        }

        let inputTokens = 0;
        let outputTokens = 0;
        let cacheCreationTokens = 0;
        let cacheReadTokens = 0;

        if (isStream && deepseekRes.body) {
            res.setHeader("cache-control", "no-cache");
            res.setHeader("x-accel-buffering", "no");
            res.setHeader("connection", "keep-alive");

            const reader = deepseekRes.body.getReader();
            const decoder = new TextDecoder();
            let streamBuffer = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                res.write(Buffer.from(value));

                streamBuffer += decoder.decode(value, { stream: true });
                const lines = streamBuffer.split("\n");
                streamBuffer = lines.pop() || "";

                for (const line of lines) {
                    const trimmed = line.trim();
                    if (trimmed.startsWith("data: ")) {
                        try {
                            const dataJson = JSON.parse(trimmed.slice(6));
                            if (dataJson.message?.usage) {
                                if (dataJson.message.usage.input_tokens) {
                                    inputTokens = dataJson.message.usage.input_tokens;
                                }
                                if (dataJson.message.usage.cache_creation_input_tokens) {
                                    cacheCreationTokens = dataJson.message.usage.cache_creation_input_tokens;
                                }
                                if (dataJson.message.usage.cache_read_input_tokens) {
                                    cacheReadTokens = dataJson.message.usage.cache_read_input_tokens;
                                }
                            }
                            if (dataJson.usage) {
                                if (dataJson.usage.output_tokens) {
                                    outputTokens = dataJson.usage.output_tokens;
                                }
                                if (dataJson.usage.input_tokens) {
                                    inputTokens = dataJson.usage.input_tokens;
                                }
                            }
                        } catch {}
                    }
                }
            }
            res.end();
        } else {
            const text = await deepseekRes.text();
            try {
                const dataJson = JSON.parse(text);
                if (dataJson.usage) {
                    inputTokens = dataJson.usage.input_tokens || 0;
                    outputTokens = dataJson.usage.output_tokens || 0;
                    cacheReadTokens = dataJson.usage.cache_read_input_tokens || 0;
                }
            } catch {}
            res.send(text);
        }

        const callLatencyMs = Date.now() - callStartTime;

        let savedUsd = 0;
        let savedThb = 0;
        let savedInputUsd = 0;
        let savedInputThb = 0;
        let savedOutputUsd = 0;
        let savedOutputThb = 0;

        if (qwenSavings) {
            const missRate = config.deepseekInputCacheMissUsdPer1M / 1000000;
            const outRate = config.deepseekOutputUsdPer1M / 1000000;
            savedInputUsd = qwenSavings.inputTokens * missRate;
            savedInputThb = savedInputUsd * config.usdThbRate;
            savedOutputUsd = qwenSavings.outputTokens * outRate;
            savedOutputThb = savedOutputUsd * config.usdThbRate;
            savedUsd = savedInputUsd + savedOutputUsd;
            savedThb = savedInputThb + savedOutputThb;
        }

        const upstreamModel = deepseekProvider.resolveUpstreamModel(body.model);

        await insertModelCall({
            requestId,
            provider: "deepseek",
            model: upstreamModel,
            inputTokens,
            outputTokens,
            cacheHitInputTokens: cacheReadTokens,
            cacheMissInputTokens: inputTokens - cacheReadTokens,
            latencyMs: callLatencyMs,
            savedUsd,
            savedThb,
            savedInputUsd,
            savedInputThb,
            savedOutputUsd,
            savedOutputThb
        });

        await updateGatewayRequest(requestId, deepseekRes.status, callLatencyMs);
    }

    static async handleTwinModels(req: Request, res: ExpressResponse): Promise<void> {
        const requestId = (req as any).requestId || crypto.randomUUID();
        const clientHeaders: Record<string, string> = {
            "user-agent": req.header("user-agent") || "railway-ai-gateway"
        };

        const messages = req.body.messages || [];
        const hasResults = hasToolResults(messages);
        const isStream = !!req.body.stream;
        const decision = shouldDelegateToQwen(messages);

        if (hasQwenEditToolResult(messages)) {
            const finalMessages = [
                ...req.body.messages,
                {
                    role: "user",
                    content: [
                        {
                            type: "text",
                            text: "The Qwen-generated Edit tool result is available. Provide a minimal final response: แก้แล้ว plus 1-2 short bullets. Do not explain broadly."
                        }
                    ]
                }
            ];
            console.log(JSON.stringify({
                time: new Date().toISOString(),
                requestId,
                mode: "hybrid-flow",
                hasToolResults: hasResults,
                delegate_to_qwen: false,
                qwenPatchMode: "find_replace",
                qwenPatchValid: true,
                deepseekApprovalUsed: false,
                finalProvider: "deepseek",
                fallbackReason: "qwen_edit_tool_result_present"
            }));
            await this.forwardToDeepSeek({ ...req.body, messages: finalMessages }, clientHeaders, res, isStream, requestId);
            return;
        }

        // 1. If no tool results yet, pass-through directly
        if (!decision.delegate_to_qwen) {
            console.log(JSON.stringify({
                time: new Date().toISOString(),
                requestId,
                mode: "hybrid-flow",
                hasToolResults: hasResults,
                likelyCodeEdit: decision.likelyCodeEdit,
                usefulCodeContext: decision.usefulCodeContext,
                delegate_to_qwen: false,
                qwenDraftUsed: false,
                qwenDraftMode: "empty",
                qwenDraftChars: 0,
                qwenDraftWeak: true,
                qwenRetryUsed: false,
                reducedContextChars: 0,
                qwenLatencyMs: 0,
                reason: decision.reason,
                finalProvider: "deepseek"
            }));
            await this.forwardToDeepSeek(req.body, clientHeaders, res, isStream, requestId);
            return;
        }

        // 2. Qwen internal coder flow
        const qwenProvider = providerRegistry.getProvider("qwen-local") as QwenLocalProvider;
        if (!qwenProvider) {
            console.log(JSON.stringify({
                time: new Date().toISOString(),
                requestId,
                mode: "hybrid-flow",
                hasToolResults: true,
                likelyCodeEdit: decision.likelyCodeEdit,
                usefulCodeContext: decision.usefulCodeContext,
                delegate_to_qwen: true,
                qwenDraftUsed: false,
                qwenDraftMode: "empty",
                qwenDraftChars: 0,
                qwenDraftWeak: true,
                qwenRetryUsed: false,
                qwenErrorType: "not_registered",
                reducedContextChars: 0,
                qwenLatencyMs: 0,
                reason: decision.reason,
                finalProvider: "deepseek"
            }));
            await this.forwardToDeepSeek(req.body, clientHeaders, res, isStream, requestId);
            return;
        }

        const userIntent = getLastRealUserInstruction(messages);
        const reducedContext = extractReducedContext(messages, userIntent);
        const reducedContextChars = reducedContext.length;

        const qwenSystemPrompt = `You are the primary code patch writer.
You do not control tools.
You do not ask to read files.
Use only the provided reduced context.
Your job is to write the actual implementation patch.

Prefer FIND/REPLACE for this version.
Return ONLY one of these formats:

A) FIND/REPLACE, preferred:
FILE: path/to/file
FIND:
<exact old snippet>
REPLACE:
<exact new snippet>

B) Unified diff, only if FIND/REPLACE is not possible:
--- path/to/file
+++ path/to/file
@@
- old code
+ new code

Rules:
- Do not explain.
- Do not write notes.
- Do not write planning text.
- Do not use markdown fences.
- Do not say "you should".
- Actually write the code change.
- Keep the patch minimal.
- Preserve existing style.
- Only modify what the user requested.
- If there is not enough context, return:
  INSUFFICIENT_CONTEXT: <short reason>`;

        let qwenDraftUsed = false;
        let qwenErrorType: string | undefined = undefined;
        let qwenLatencyMs = 0;
        let draftText = "";
        let qwenDraftMode: QwenDraftMode = "empty";
        let qwenDraftChars = 0;
        let qwenDraftWeak = false;
        let qwenRetryUsed = false;
        let qwenPatchValid = false;
        let qwenPatchReason = "";
        let deepseekApprovalUsed = false;
        let deepseekApprovalApproved = false;
        let emittedToolUse: string | undefined;
        let fallbackReason: string | undefined;
        let parsedPatch: ParsedQwenPatch = { ok: false, mode: "invalid", reason: "not_parsed" };
        let qwenInputTokens = 0;
        let qwenOutputTokens = 0;

        const qwenStartTime = Date.now();
        try {
            const callQwen = async (retryInstruction?: string) => {
                const qwenBody = {
                    system: retryInstruction ? `${qwenSystemPrompt}\n\n${retryInstruction}` : qwenSystemPrompt,
                    messages: [
                        {
                            role: "user",
                            content: `Reduced context:\n${reducedContext}\n\nTask: Generate the primary implementation patch for this code edit request.\n\nLatest user intent preview: ${decision.userIntentPreview}`
                        }
                    ],
                    stream: false,
                    max_tokens: 2048,
                    temperature: 0.15
                };
                const qwenRes = await qwenProvider.handleRequest(qwenBody, clientHeaders);
                const qwenData = qwenRes.ok ? await qwenRes.json() : null;
                let text = "";
                if (Array.isArray(qwenData?.content)) {
                    const textBlock = qwenData.content.find((b: any) => b?.type === "text");
                    text = textBlock?.text || "";
                }

                return {
                    ok: qwenRes.ok,
                    status: qwenRes.status,
                    text,
                    inputTokens: qwenData?.usage?.input_tokens || 0,
                    outputTokens: qwenData?.usage?.output_tokens || 0
                };
            };

            let qwenResult = await callQwen();
            qwenLatencyMs = Date.now() - qwenStartTime;

            if (qwenResult.ok) {
                draftText = qwenResult.text;
                qwenDraftMode = detectQwenDraftMode(draftText);
                qwenDraftChars = draftText.length;
                qwenInputTokens = qwenResult.inputTokens;
                qwenOutputTokens = qwenResult.outputTokens;

                if (shouldRetryQwenDraft(qwenDraftMode, qwenDraftChars)) {
                    qwenRetryUsed = true;
                    qwenResult = await callQwen("Your previous answer was not an implementation patch.\nReturn ONLY a unified diff or FIND/REPLACE snippet.\nNo explanation. No notes. Write the actual code now.");
                    qwenLatencyMs = Date.now() - qwenStartTime;
                    if (qwenResult.ok) {
                        draftText = qwenResult.text;
                        qwenDraftMode = detectQwenDraftMode(draftText);
                        qwenDraftChars = draftText.length;
                        qwenInputTokens += qwenResult.inputTokens;
                        qwenOutputTokens += qwenResult.outputTokens;
                    } else {
                        qwenErrorType = `retry_http_status_${qwenResult.status}`;
                    }
                }

                qwenDraftUsed = isUsableQwenDraft(qwenDraftMode, qwenDraftChars);
                qwenDraftWeak = !qwenDraftUsed;

                if (!draftText) {
                    qwenErrorType = "empty_draft";
                }

            } else {
                qwenErrorType = `http_status_${qwenResult.status}`;
                qwenDraftWeak = true;
            }
        } catch (error: any) {
            qwenLatencyMs = Date.now() - qwenStartTime;
            qwenErrorType = error.name === "AbortError" ? "timeout" : "connection_error";
            qwenDraftWeak = true;
        }

        if (draftText) {
            parsedPatch = validateQwenPatch(parseQwenFindReplacePatch(draftText), messages, userIntent);
            qwenPatchValid = parsedPatch.ok;
            qwenPatchReason = parsedPatch.reason || (parsedPatch.ok ? "valid" : "invalid_patch");
        } else {
            qwenPatchReason = qwenErrorType || "empty_draft";
        }

        if (qwenPatchValid && parsedPatch.filePath && parsedPatch.find !== undefined && parsedPatch.replace !== undefined) {
            const deepseekProvider = providerRegistry.getProvider("deepseek") as DeepSeekProvider;
            if (deepseekProvider) {
                deepseekApprovalUsed = true;
                const approval = await askDeepSeekPatchApproval(deepseekProvider, clientHeaders, requestId, {
                    userIntent,
                    filePath: parsedPatch.filePath,
                    patchMode: parsedPatch.mode,
                    findLength: parsedPatch.find.length,
                    replaceLength: parsedPatch.replace.length,
                    riskFlags: []
                });
                deepseekApprovalApproved = approval.approved;
                if (!approval.approved) {
                    fallbackReason = `approval_rejected:${approval.reason}`;
                }
            } else {
                fallbackReason = "deepseek_provider_not_registered";
            }
        } else if (draftText) {
            fallbackReason = qwenPatchReason;
        } else {
            fallbackReason = qwenErrorType || "empty_draft";
        }

        await insertModelCall({
            requestId,
            provider: "qwen-local",
            model: config.qwenLocalModel,
            inputTokens: qwenInputTokens,
            outputTokens: qwenOutputTokens,
            latencyMs: qwenLatencyMs,
            qwenDraftMode,
            qwenDraftChars,
            qwenDraftWeak,
            qwenRetryUsed,
            qwenPatchMode: parsedPatch.mode,
            qwenPatchValid,
            deepseekApprovalApproved: deepseekApprovalUsed ? deepseekApprovalApproved : undefined,
            emittedToolUse: deepseekApprovalApproved ? "Edit" : undefined,
            fallbackReason
        });

        if (qwenPatchValid && deepseekApprovalApproved && parsedPatch.filePath && parsedPatch.find !== undefined && parsedPatch.replace !== undefined) {
            const toolUseId = `toolu_qwen_edit_${crypto.randomUUID().replace(/-/g, "").slice(0, 16)}`;
            emittedToolUse = "Edit";
            console.log(JSON.stringify({
                time: new Date().toISOString(),
                requestId,
                mode: "hybrid-flow",
                delegate_to_qwen: true,
                qwenPatchMode: parsedPatch.mode,
                qwenPatchValid: true,
                qwenPatchReason,
                qwenDraftChars,
                qwenLatencyMs,
                deepseekApprovalUsed,
                deepseekApprovalApproved,
                finalProvider: "deepseek",
                emittedToolUse
            }));
            await updateGatewayRequest(requestId, 200, qwenLatencyMs);
            res.status(200).json({
                id: `msg_qwen_edit_${crypto.randomUUID().replace(/-/g, "").slice(0, 12)}`,
                type: "message",
                role: "assistant",
                content: [
                    {
                        type: "tool_use",
                        id: toolUseId,
                        name: "Edit",
                        input: {
                            file_path: parsedPatch.filePath,
                            old_string: parsedPatch.find,
                            new_string: parsedPatch.replace
                        }
                    }
                ],
                model: req.body.model || "hybrid-flow",
                stop_reason: "tool_use",
                stop_sequence: null,
                usage: {
                    input_tokens: 0,
                    output_tokens: 0
                }
            });
            return;
        }

        let finalBody = req.body;
        if (draftText) {
            const advisoryIntro = qwenPatchValid
                ? `Internal Qwen primary patch draft below.

Qwen is the primary implementation writer, but Gateway could not emit Edit directly.
Fallback reason: ${fallbackReason || "unknown"}.
If the patch is valid, apply it using Claude Code tools.
Do not rewrite the solution from scratch unless the draft is clearly wrong, unsafe, or inconsistent with the file context.`
                : `Internal Qwen patch draft below.

Gateway validation rejected this draft.
Fallback reason: ${fallbackReason || qwenPatchReason || "invalid_patch"}.
Treat this only as advisory context. Verify against the actual file context before using it.`;
            const augmentedMessages = [
                ...req.body.messages,
                {
                    role: "user",
                    content: [
                        {
                            type: "text",
                            text: `${advisoryIntro}

Keep your final response minimal.
Prefer tool_use/Edit over long explanation.
Do not explain broadly.
If applying a valid Qwen patch, use tools directly.
After applying, summarize in 1-3 bullets only.
Do not generate another full implementation unless Qwen draft is wrong.

<QWEN_DRAFT mode="${qwenDraftMode}" chars="${qwenDraftChars}">
${draftText}
</QWEN_DRAFT>`
                        }
                    ]
                }
            ];
            finalBody = {
                ...req.body,
                messages: augmentedMessages
            };
        }

        console.log(JSON.stringify({
            time: new Date().toISOString(),
            requestId,
            mode: "hybrid-flow",
            hasToolResults: true,
            likelyCodeEdit: decision.likelyCodeEdit,
            usefulCodeContext: decision.usefulCodeContext,
            delegate_to_qwen: true,
            qwenDraftUsed,
            qwenDraftMode,
            qwenDraftChars,
            qwenDraftWeak,
            qwenRetryUsed,
            qwenErrorType,
            qwenPatchMode: parsedPatch.mode,
            qwenPatchValid,
            qwenPatchReason,
            deepseekApprovalUsed,
            deepseekApprovalApproved,
            reducedContextChars,
            qwenLatencyMs,
            reason: decision.reason,
            finalProvider: "deepseek",
            emittedToolUse,
            fallbackReason
        }));

        const qwenSavings = qwenDraftUsed ? { inputTokens: qwenInputTokens, outputTokens: qwenOutputTokens } : undefined;
        await this.forwardToDeepSeek(finalBody, clientHeaders, res, isStream, requestId, qwenSavings);
    }
}
