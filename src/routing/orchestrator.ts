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

interface RouterDecision {
    delegate_to_qwen: boolean;
    task_type: string;
    reason: string;
    qwen_instruction: string;
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

        const routerPrompt = `You are the delegation router for a coding gateway.
Decide whether the current request should call Qwen Local as an internal code draft generator.

Rules:
- Return JSON only. Do not wrap in markdown blocks, just return raw JSON string.
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

Output format exactly:
{
  "delegate_to_qwen": boolean,
  "task_type": "chat" | "read" | "explain" | "code_edit" | "debug" | "test" | "review" | "unknown",
  "reason": "short reason",
  "qwen_instruction": "short instruction for Qwen if delegate_to_qwen is true"
}`;

        const body = {
            model: config.defaultModel,
            system: routerPrompt,
            messages: messages,
            stream: false,
            max_tokens: 256,
            temperature: 0.1
        };

        const startTime = Date.now();
        const res = await deepseekProvider.handleRequest(body, clientHeaders);
        const latencyMs = Date.now() - startTime;

        if (!res.ok) {
            throw new Error(`Delegation router request failed with status ${res.status}`);
        }

        const data = await res.json();
        let text = "";
        if (Array.isArray(data.content)) {
            const textBlock = data.content.find((b: any) => b?.type === "text") ||
                              data.content.find((b: any) => b?.type === "thinking");
            text = textBlock?.text || textBlock?.thinking || "";
        }

        const decision: RouterDecision = extractJsonFromString(text);

        // Log delegation router model call
        const inputTokens = data.usage?.input_tokens || 0;
        const outputTokens = data.usage?.output_tokens || 0;
        const cacheReadTokens = data.usage?.cache_read_input_tokens || 0;

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
        if (qwenSavings) {
            const missRate = config.deepseekInputCacheMissUsdPer1M / 1000000;
            const outRate = config.deepseekOutputUsdPer1M / 1000000;
            savedUsd = (qwenSavings.inputTokens * missRate) + (qwenSavings.outputTokens * outRate);
            savedThb = savedUsd * config.usdThbRate;
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
            savedThb
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

        // 1. If no tool results yet, pass-through directly
        if (!hasResults) {
            console.log(JSON.stringify({
                time: new Date().toISOString(),
                requestId,
                mode: "hybrid-flow",
                hasToolResults: false,
                delegate_to_qwen: false,
                finalProvider: "deepseek"
            }));
            await this.forwardToDeepSeek(req.body, clientHeaders, res, isStream, requestId);
            return;
        }

        // 2. Ask DeepSeek Delegation Router
        let decision: RouterDecision = {
            delegate_to_qwen: false,
            task_type: "unknown",
            reason: "fallback",
            qwen_instruction: ""
        };

        try {
            decision = await this.askDeepSeekDelegationRouter(messages, clientHeaders, requestId);
        } catch (err: any) {
            console.error("Delegation router failure, falling back to DeepSeek:", err.message);
        }

        // 3. If router says false, pass-through directly
        if (!decision.delegate_to_qwen) {
            console.log(JSON.stringify({
                time: new Date().toISOString(),
                requestId,
                mode: "hybrid-flow",
                hasToolResults: true,
                delegate_to_qwen: false,
                reason: decision.reason,
                finalProvider: "deepseek"
            }));
            await this.forwardToDeepSeek(req.body, clientHeaders, res, isStream, requestId);
            return;
        }

        // 4. Qwen internal coder flow
        const qwenProvider = providerRegistry.getProvider("qwen-local") as QwenLocalProvider;
        if (!qwenProvider) {
            console.log(JSON.stringify({
                time: new Date().toISOString(),
                requestId,
                mode: "hybrid-flow",
                hasToolResults: true,
                delegate_to_qwen: true,
                qwenDraftUsed: false,
                qwenErrorType: "not_registered",
                finalProvider: "deepseek"
            }));
            await this.forwardToDeepSeek(req.body, clientHeaders, res, isStream, requestId);
            return;
        }

        const reducedContext = extractReducedContext(messages);
        const reducedContextChars = reducedContext.length;

        const qwenBody = {
            system: `You are an internal code draft generator.
You do not control tools.
You do not ask to read files.
Use only the provided context.
Return a concise patch suggestion or implementation notes.
Prefer unified diff if enough file context exists.
Do not explain broadly.`,
            messages: [
                {
                    role: "user",
                    content: `Context:\n${reducedContext}\n\nTask: ${decision.qwen_instruction}`
                }
            ],
            stream: false,
            max_tokens: 3072,
            temperature: 0.2
        };

        let qwenDraftUsed = false;
        let qwenErrorType: string | undefined = undefined;
        let qwenLatencyMs = 0;
        let draftText = "";
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
                if (draftText) {
                    qwenDraftUsed = true;
                    qwenInputTokens = qwenData.usage?.input_tokens || 0;
                    qwenOutputTokens = qwenData.usage?.output_tokens || 0;

                    // Log Qwen model call
                    await insertModelCall({
                        requestId,
                        provider: "qwen-local",
                        model: config.qwenLocalModel,
                        inputTokens: qwenInputTokens,
                        outputTokens: qwenOutputTokens,
                        latencyMs: qwenLatencyMs
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
        if (qwenDraftUsed && draftText) {
            const augmentedMessages = [
                ...req.body.messages,
                {
                    role: "user",
                    content: [
                        {
                            type: "text",
                            text: `Internal Qwen coder draft below. Use it only as a suggestion. Review it carefully. If correct, apply changes using Claude Code tools. If wrong, ignore it.\n\n<QWEN_DRAFT>\n${draftText}\n</QWEN_DRAFT>`
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
            delegate_to_qwen: true,
            qwenDraftUsed,
            qwenErrorType,
            reducedContextChars,
            qwenLatencyMs,
            finalProvider: "deepseek"
        }));

        const qwenSavings = qwenDraftUsed ? { inputTokens: qwenInputTokens, outputTokens: qwenOutputTokens } : undefined;
        await this.forwardToDeepSeek(finalBody, clientHeaders, res, isStream, requestId, qwenSavings);
    }
}
