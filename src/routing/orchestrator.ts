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

type QwenDraftMode = "unified_diff" | "snippet" | "notes" | "empty";

function detectQwenDraftMode(text: string): QwenDraftMode {
    const t = text.trim();
    if (!t) return "empty";
    if (t.includes("---") && t.includes("+++") && t.includes("@@")) return "unified_diff";
    if (
        t.includes("function") ||
        t.includes("const ") ||
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

function extractReducedContext(messages: any[], maxChars = 8000): string {
    const parts: string[] = [];

    for (const msg of messages.slice(-8)) {
        if (!Array.isArray(msg.content)) continue;

        for (const block of msg.content) {
            if (block?.type === "tool_result") {
                const content = typeof block.content === "string"
                    ? block.content
                    : JSON.stringify(block.content);

                parts.push(`TOOL_RESULT:\n${content.slice(0, 3000)}`);
            }

            if (block?.type === "text") {
                parts.push(`TEXT:\n${String(block.text).slice(0, 1500)}`);
            }
        }
    }

    return parts.join("\n\n---\n\n").slice(-maxChars);
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

        await insertModelCall({
            requestId,
            provider: "deepseek",
            model: body.model || config.defaultModel,
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
                qwenErrorType: "not_registered",
                reducedContextChars: 0,
                qwenLatencyMs: 0,
                reason: decision.reason,
                finalProvider: "deepseek"
            }));
            await this.forwardToDeepSeek(req.body, clientHeaders, res, isStream, requestId);
            return;
        }

        const reducedContext = extractReducedContext(messages);
        const reducedContextChars = reducedContext.length;

        const qwenBody = {
            system: `You are an internal patch generator.
You do not control tools.
You do not ask to read files.
Use only the provided reduced context.
Return a unified diff if enough context exists.
If unified diff is not possible, return an exact replacement snippet.
Do not explain broadly.
Do not include markdown fences.
Do not include planning text.
Prefer minimal changes.
Preserve existing style.
Only modify what the user requested.`,
            messages: [
                {
                    role: "user",
                    content: `Reduced context:\n${reducedContext}\n\nTask: Generate the primary patch draft for this code edit request.\n\nLatest user intent preview: ${decision.userIntentPreview}`
                }
            ],
            stream: false,
            max_tokens: 2048,
            temperature: 0.15
        };

        let qwenDraftUsed = false;
        let qwenErrorType: string | undefined = undefined;
        let qwenLatencyMs = 0;
        let draftText = "";
        let qwenDraftMode: QwenDraftMode = "empty";
        let qwenDraftChars = 0;
        let qwenDraftWeak = false;
        let qwenInputTokens = 0;
        let qwenOutputTokens = 0;

        const qwenStartTime = Date.now();
        try {
            const qwenRes = await qwenProvider.handleRequest(qwenBody, clientHeaders);
            qwenLatencyMs = Date.now() - qwenStartTime;

            if (qwenRes.ok) {
                const qwenData = await qwenRes.json();
                if (Array.isArray(qwenData.content)) {
                    const textBlock = qwenData.content.find((b: any) => b?.type === "text");
                    draftText = textBlock?.text || "";
                }
                qwenDraftMode = detectQwenDraftMode(draftText);
                qwenDraftChars = draftText.length;
                qwenDraftWeak = qwenDraftMode === "empty" || qwenDraftMode === "notes";
                if (draftText) {
                    qwenDraftUsed = qwenDraftMode === "unified_diff" || qwenDraftMode === "snippet";
                    qwenInputTokens = qwenData.usage?.input_tokens || 0;
                    qwenOutputTokens = qwenData.usage?.output_tokens || 0;

                    // Log Qwen model call
                    await insertModelCall({
                        requestId,
                        provider: "qwen-local",
                        model: config.qwenLocalModel,
                        inputTokens: qwenInputTokens,
                        outputTokens: qwenOutputTokens,
                        latencyMs: qwenLatencyMs,
                        qwenDraftMode,
                        qwenDraftChars
                    });
                } else {
                    qwenErrorType = "empty_draft";
                }
            } else {
                qwenErrorType = `http_status_${qwenRes.status}`;
            }
        } catch (error: any) {
            qwenLatencyMs = Date.now() - qwenStartTime;
            qwenErrorType = error.name === "AbortError" ? "timeout" : "connection_error";
        }

        let finalBody = req.body;
        if (draftText) {
            const augmentedMessages = [
                ...req.body.messages,
                {
                    role: "user",
                    content: [
                        {
                            type: "text",
                            text: `Internal Qwen patch draft below.

Treat this as the primary implementation suggestion.
Use it unless it is clearly wrong, unsafe, incomplete, or conflicts with the actual file context.
If it is usable, apply it with Claude Code tools.
Keep your final response minimal.
Do not explain broadly.
Do not rewrite the whole solution if the patch is already sufficient.
Avoid generating a second long alternative unless Qwen draft is wrong.

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
            qwenErrorType,
            reducedContextChars,
            qwenLatencyMs,
            reason: decision.reason,
            finalProvider: "deepseek"
        }));

        const qwenSavings = qwenDraftUsed ? { inputTokens: qwenInputTokens, outputTokens: qwenOutputTokens } : undefined;
        await this.forwardToDeepSeek(finalBody, clientHeaders, res, isStream, requestId, qwenSavings);
    }
}
