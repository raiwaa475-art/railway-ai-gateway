import { Request, Response } from "express";
import crypto from "crypto";
import { config } from "../config/env.js";
import { providerRegistry } from "./registry.js";
import { QwenLocalProvider } from "../providers/qwen-local.js";
import { insertModelCall, updateGatewayRequest } from "../utils/db.js";

export type QwenOnlyIntentType = "chat" | "planning" | "explanation" | "read_only" | "code_edit" | "high_risk_code";
export type QwenOnlyAction = "disabled" | "reject" | "call_qwen";

export interface QwenOnlyIntentDecision {
    realUserIntent: string;
    realUserIntentPreview: string;
    intentType: QwenOnlyIntentType;
    hasExactContext: boolean;
    exactContextSource: "tool_result_exact" | "tool_result_partial" | "disk_exact" | "none";
    targetFile: string;
    qwenOnlyRejectedReason: string;
    confidenceRiskLevel: "low" | "medium" | "high";
    buildCheckStatus: "not_run" | "passed" | "failed";
    action: QwenOnlyAction;
}

const QWEN_ONLY_REJECTION_MESSAGE =
    "Qwen-only rejected: missing exact context / high risk. Use qwen-smart or deepseek-v4-flash.";

function stripSystemReminders(text: string): string {
    return String(text || "")
        .replace(/<system-reminder>[\s\S]*?<\/system-reminder>/gi, " ")
        .replace(/<system-reminder\b[^>]*>[\s\S]*?<\/system-reminder>/gi, " ")
        .replace(/<system-reminder\b[^>]*\/>/gi, " ")
        .replace(/\s+/g, " ")
        .trim();
}

function normalizeText(text: string): string {
    return stripSystemReminders(String(text || "")).toLowerCase();
}

function getUserTextFromMessageContent(content: any): string {
    if (typeof content === "string") {
        return stripSystemReminders(content).trim();
    }

    if (!Array.isArray(content)) {
        return "";
    }

    const parts: string[] = [];
    for (const block of content) {
        if (!block || block.type === "system_reminder") continue;
        if (block.type === "text") {
            const text = stripSystemReminders(String(block.text || "")).trim();
            if (text) parts.push(text);
        }
    }

    return parts.join("\n").trim();
}

export function getLatestRealUserInstruction(messages: any[]): string {
    for (const msg of (Array.isArray(messages) ? messages : []).slice().reverse()) {
        if (msg?.role !== "user") continue;
        const text = getUserTextFromMessageContent(msg.content);
        if (text) return text;
    }

    return "";
}

function isReadOnlyIntent(text: string): boolean {
    const normalized = normalizeText(text);
    const readOnlyKeywords = [
        "อ่านไฟล์",
        "read file",
        "show file",
        "cat file",
        "list files",
        "open file",
        "summarize file",
        "inspect file",
        "ดูไฟล์",
        "สรุปไฟล์"
    ];
    return readOnlyKeywords.some(keyword => normalized.includes(keyword));
}

function isPlanningIntent(text: string): boolean {
    const normalized = normalizeText(text);
    return /\b(plan|planning|outline|roadmap|strategy|steps?|approach)\b/i.test(normalized) ||
        normalized.includes("วางแผน");
}

function isExplanationIntent(text: string): boolean {
    const normalized = normalizeText(text);
    return /\b(explain|explanation|summarize|summary|review|what does|why|how does)\b/i.test(normalized) ||
        normalized.includes("อธิบาย") ||
        normalized.includes("สรุป");
}

function isCodeEditIntent(text: string): boolean {
    const normalized = normalizeText(text);
    const codeKeywords = [
        "edit",
        "fix",
        "update",
        "change",
        "implement",
        "patch",
        "refactor",
        "modify",
        "create",
        "add",
        "remove",
        "rename",
        "debug",
        "bug",
        "error",
        "file",
        "code",
        "function",
        "component",
        "route",
        "endpoint",
        "api",
        "test",
        "build"
    ];

    if (codeKeywords.some(keyword => normalized.includes(keyword))) {
        return true;
    }

    return /(^|[\s./\\_-])(tsx?|jsx?|css|html)($|[\s./\\_-])/i.test(normalized);
}

function hasHighRiskKeywords(text: string): boolean {
    const normalized = normalizeText(text);
    return /\b(auth|login|permission|payment|billing|database|migration|schema|security|secret|api key|token|production|deploy|architecture)\b/i.test(normalized) ||
        normalized.includes("multi-file") ||
        normalized.includes("หลายไฟล์");
}

function stripLineNumberPrefixes(text: string): string {
    return String(text || "")
        .split("\n")
        .map(line => line.replace(/^\s*\d+\s*(?:\||:)?\s?/, ""))
        .join("\n");
}

function hasTruncationMarker(text: string): boolean {
    const lowered = String(text || "").toLowerCase();
    return lowered.includes("truncated") ||
        lowered.includes("omitted") ||
        lowered.includes("output clipped") ||
        lowered.includes("content clipped") ||
        lowered.includes("remaining lines") ||
        lowered.includes("more lines");
}

function isRangeOrSearchToolResult(toolName: string, input: any, text: string): boolean {
    const lowerTool = String(toolName || "").toLowerCase();
    if (/(grep|search|find|rg|select-string|head|tail|sed|range|snippet)/i.test(lowerTool)) {
        return true;
    }
    if (
        input?.offset !== undefined ||
        input?.limit !== undefined ||
        input?.start !== undefined ||
        input?.end !== undefined ||
        input?.start_line !== undefined ||
        input?.end_line !== undefined ||
        input?.line_start !== undefined ||
        input?.line_end !== undefined ||
        input?.pattern !== undefined ||
        input?.query !== undefined
    ) {
        return true;
    }
    return String(text || "").toLowerCase().includes("matches") && /\d+[:|]/.test(text);
}

function isFullFileReadToolResult(toolName: string, input: any, text: string): boolean {
    const lowerTool = String(toolName || "").toLowerCase();
    const exactToolPattern = /(^|[_\-\s.])(read|view|get|open|cat|show)([_\-\s.]|$)/i;
    if (!exactToolPattern.test(lowerTool) && !/file.*(content|text)/i.test(lowerTool)) {
        return false;
    }
    if (hasTruncationMarker(text) || isRangeOrSearchToolResult(toolName, input, text)) {
        return false;
    }
    return String(text || "").trim().length > 0;
}

function normalizePatchPath(filePath: string): string {
    let p = String(filePath || "").trim();
    p = p.replace(/\\/g, "/").replace(/\/+/g, "/");
    p = p.replace(/^\.\//, "");
    p = p.replace(/^[ab]\//, "");

    const cwd = process.cwd().replace(/\\/g, "/").replace(/\/+$/g, "");
    const lower = p.toLowerCase();
    const lowerCwd = cwd.toLowerCase();
    if (lower === lowerCwd) return "";
    if (lower.startsWith(`${lowerCwd}/`)) {
        p = p.slice(cwd.length + 1);
    }

    const workspaceName = cwd.split("/").filter(Boolean).pop()?.toLowerCase() || "";
    const parts = p.split("/").filter(Boolean);
    const workspaceIndex = parts.findIndex(part => part.toLowerCase() === workspaceName);
    if (workspaceIndex >= 0 && workspaceIndex < parts.length - 1) {
        p = parts.slice(workspaceIndex + 1).join("/");
    }

    return p.replace(/^\/+/, "").toLowerCase();
}

function getTargetFileFromRecentToolUse(messages: any[]): string {
    for (const msg of (Array.isArray(messages) ? messages : []).slice(-10).reverse()) {
        if (!Array.isArray(msg?.content)) continue;
        for (const block of msg.content) {
            if (block?.type !== "tool_use") continue;
            const input = block.input || {};
            const filePath = input.AbsolutePath || input.file_path || input.path || input.file || "";
            if (typeof filePath === "string" && filePath.trim()) {
                return filePath.trim();
            }
        }
    }

    return "";
}

function getFileContentFromToolResults(messages: any[], targetFilePath: string): { content: string; isExact: boolean; source: QwenOnlyIntentDecision["exactContextSource"] } {
    if (!targetFilePath) return { content: "", isExact: false, source: "none" };
    const normalizedTarget = normalizePatchPath(targetFilePath);
    const targetName = normalizedTarget.split("/").pop() || "";

    for (const msg of (Array.isArray(messages) ? messages : []).slice().reverse()) {
        if (!Array.isArray(msg?.content)) continue;

        for (const block of msg.content) {
            if (block?.type !== "tool_result" || block.content === undefined) continue;
            const toolUseId = block.tool_use_id;
            if (!toolUseId) continue;

            const toolUseMsg = (Array.isArray(messages) ? messages : []).find(m =>
                Array.isArray(m?.content) && m.content.some((b: any) => b?.type === "tool_use" && b.id === toolUseId)
            );
            const toolUseBlock = toolUseMsg?.content?.find((b: any) => b?.type === "tool_use" && b.id === toolUseId);
            const toolName = toolUseBlock?.name || "";
            const input = toolUseBlock?.input || {};
            const inputPath = input.AbsolutePath || input.file_path || input.path || "";
            const normalizedInputPath = typeof inputPath === "string" ? normalizePatchPath(inputPath) : "";
            const pathMatches = normalizedInputPath === normalizedTarget || (targetName && normalizedInputPath.split("/").pop() === targetName);
            if (!pathMatches) continue;

            const rawText = stripLineNumberPrefixes(
                typeof block.content === "string" ? block.content : JSON.stringify(block.content)
            ).trim();
            if (!rawText) continue;

            const isExact = isFullFileReadToolResult(toolName, input, rawText);
            return {
                content: rawText,
                isExact,
                source: isExact ? "tool_result_exact" : "tool_result_partial"
            };
        }
    }

    return { content: "", isExact: false, source: "none" };
}

export function classifyQwenOnlyIntent(body: any): QwenOnlyIntentDecision {
    const messages = Array.isArray(body?.messages) ? body.messages : [];
    const realUserIntent = getLatestRealUserInstruction(messages);
    const realUserIntentPreview = realUserIntent.slice(0, 160);
    const lower = normalizeText(realUserIntent);

    let intentType: QwenOnlyIntentType = "chat";
    if (isReadOnlyIntent(realUserIntent)) {
        intentType = "read_only";
    } else if (isPlanningIntent(realUserIntent)) {
        intentType = "planning";
    } else if (isExplanationIntent(realUserIntent)) {
        intentType = "explanation";
    } else if (isCodeEditIntent(realUserIntent)) {
        intentType = hasHighRiskKeywords(realUserIntent) ? "high_risk_code" : "code_edit";
    }

    const requiresExactContext = intentType === "code_edit" || intentType === "high_risk_code";
    const targetFile = requiresExactContext ? getTargetFileFromRecentToolUse(messages) : "";
    const exactContext = requiresExactContext && targetFile
        ? getFileContentFromToolResults(messages, targetFile)
        : { content: "", isExact: false, source: "none" as const };

    const hasExactContext = !!exactContext.isExact;
    const confidenceRiskLevel: QwenOnlyIntentDecision["confidenceRiskLevel"] =
        intentType === "high_risk_code" ? "high" : requiresExactContext && !hasExactContext ? "high" : "low";

    const action: QwenOnlyAction = requiresExactContext && (!hasExactContext || intentType === "high_risk_code")
        ? "reject"
        : "call_qwen";

    return {
        realUserIntent,
        realUserIntentPreview,
        intentType,
        hasExactContext,
        exactContextSource: exactContext.source,
        targetFile,
        qwenOnlyRejectedReason: action === "reject" ? QWEN_ONLY_REJECTION_MESSAGE : "",
        confidenceRiskLevel,
        buildCheckStatus: requiresExactContext ? (hasExactContext && intentType !== "high_risk_code" ? "passed" : "failed") : "not_run",
        action
    };
}

function buildQwenOnlyRejectedResponse(message: string) {
    return {
        id: "msg_qwen_only_rejected_" + Math.random().toString(36).substring(7),
        type: "message",
        role: "assistant",
        content: [
            {
                type: "text",
                text: message
            }
        ],
        model: "qwen-only-low-risk",
        stop_reason: "end_turn",
        stop_sequence: null,
        usage: {
            input_tokens: 0,
            output_tokens: 0
        }
    };
}

function logEntry(entry: Record<string, unknown>) {
    console.log(JSON.stringify({
        time: new Date().toISOString(),
        ...entry
    }));
}

async function resolveQwenProvider(): Promise<QwenLocalProvider | undefined> {
    const provider = providerRegistry.getProvider("qwen-local") as QwenLocalProvider | undefined;
    return provider;
}

async function recordModelCall(params: {
    requestId: string;
    provider: string;
    model: string;
    inputTokens: number;
    outputTokens: number;
    latencyMs: number;
    intentType: QwenOnlyIntentType;
    exactContextSource: QwenOnlyIntentDecision["exactContextSource"];
    targetFile: string;
    qwenOnlyUsed: boolean;
}) {
    await insertModelCall({
        requestId: params.requestId,
        provider: params.provider,
        model: params.model,
        inputTokens: params.inputTokens,
        outputTokens: params.outputTokens,
        latencyMs: params.latencyMs,
        qwenDraftMode: params.intentType,
        qwenDraftChars: 0,
        qwenDraftWeak: false,
        qwenRetryUsed: false,
        qwenPatchMode: params.intentType,
        qwenPatchValid: true,
        fallbackReason: params.qwenOnlyUsed ? "qwen_only_low_risk" : "qwen_only_rejected",
        fileContextSource: params.exactContextSource,
        qwenDelegationMode: params.intentType === "chat" ? "qwen_only_low_risk_chat" : "qwen_only_low_risk_code_edit",
        directEditEligible: params.intentType !== "chat",
        qwenAnchorCandidateCount: params.targetFile ? 1 : 0
    });
}

function detectFakeToolJson(text: string): boolean {
    const trimmed = (text || "").trim();
    if (!trimmed) return false;

    // Check if the entire text is a JSON code block
    let cleaned = trimmed;
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

    if (cleaned.startsWith("{") && cleaned.endsWith("}")) {
        try {
            const parsed = JSON.parse(cleaned);
            if (parsed && typeof parsed === "object") {
                const name = parsed.name || parsed.tool_use || parsed.tool || parsed.tool_name;
                const args = parsed.arguments || parsed.input || parsed.args || parsed.parameters;
                if (name && typeof name === "string" && args) {
                    return true;
                }
            }
        } catch {
            // ignore JSON parsing failed, try regex
        }
    }

    // Regex check: e.g. {"name": "Write", "arguments": {...}} or similar
    const namePattern = /["']?name["']?\s*:\s*["'][a-zA-Z0-9_\-]+["']/i;
    const argumentsPattern = /["']?arguments["']?\s*:/i;
    
    if (namePattern.test(cleaned) && argumentsPattern.test(cleaned)) {
        return true;
    }

    return false;
}

function isDangerousTool(name: string, input: any): boolean {
    const lowerName = String(name || "").toLowerCase();
    
    const isWrite = lowerName.includes("write");
    const isEdit = lowerName.includes("edit") || lowerName.includes("replace_file_content");
    if (isWrite || isEdit) {
        return true;
    }
    
    const isBash = lowerName.includes("bash") || lowerName.includes("run_command") || lowerName.includes("shell") || lowerName === "cmd";
    if (isBash) {
        const command = String(input?.command || input?.CommandLine || input?.cmd || "").toLowerCase();
        if (command) {
            const dangerousPatterns = [
                /\b(rm|mv|cp|chmod|chown|ln|mkdir|rmdir|touch|dd|tar|zip|unzip|gzip|gunzip)\b/,
                /\bgit\s+(commit|push|reset|revert|checkout|clean|branch\s+-d|branch\s+-D|merge|rebase|pull|add)\b/,
                />{1,2}\s*\S+/,
                /\b(npm|yarn|pnpm|pip|pip3|apt|apt-get|brew|yum|dnf|apk|gem|cargo)\s+(install|uninstall|update|upgrade|add|remove|prune)\b/,
                /\b(nano|vim|vi|emacs|sed|awk|perl)\b/
            ];
            
            if (dangerousPatterns.some(pattern => pattern.test(command))) {
                return true;
            }
        }
    }
    
    return false;
}

function hasDangerousToolCall(content: any[]): boolean {
    if (!Array.isArray(content)) return false;
    for (const block of content) {
        if (block?.type === "tool_use") {
            if (isDangerousTool(block.name, block.input)) {
                return true;
            }
        }
    }
    return false;
}

async function forwardToQwenLocal(
    req: Request,
    res: Response,
    requestId: string,
    startTime: number,
    decision: QwenOnlyIntentDecision
): Promise<void> {
    const provider = await resolveQwenProvider();
    const finalBody = { ...req.body };

    if (decision.intentType === "read_only") {
        const readOnlyInstruction = "When you need to read files, use the available Read/List/Glob/Grep tools through tool_use. Do not print JSON tool calls as text. Do not use Write/Edit for read-only requests.";
        if (finalBody.system) {
            finalBody.system = `${finalBody.system}\n\n${readOnlyInstruction}`;
        } else {
            finalBody.system = readOnlyInstruction;
        }
    } else if (decision.intentType === "chat" || decision.intentType === "planning" || decision.intentType === "explanation") {
        if (finalBody.tools) {
            delete finalBody.tools;
        }
    }

    if (!provider) {
        await updateGatewayRequest(requestId, 503, Date.now() - startTime);
        logEntry({
            type: "response",
            requestId,
            clientModel: finalBody.model || "qwen-only-low-risk",
            qwenOnlyUsed: false,
            qwenOnlyRejectedReason: "Qwen local provider is not registered",
            qwenToolCallValid: true,
            qwenFakeToolJsonDetected: false,
            finalProvider: "none",
            deepseekFallbackUsed: false,
            confidence_risk_level: "high",
            build_check_status: decision.buildCheckStatus,
            realUserIntentPreview: decision.realUserIntentPreview,
            intentType: decision.intentType,
            status: 503
        });
        res.status(503).json({
            error: {
                type: "server_error",
                message: "Qwen local provider is not registered"
            }
        });
        return;
    }

    const resolvedConfig = await provider.resolveRuntimeConfig();
    const activeModelName = resolvedConfig.modelName;
    const clientHeaders: Record<string, string> = {
        "user-agent": req.header("user-agent") || "railway-ai-gateway"
    };

    const qwenStartTime = Date.now();
    const upstream = await provider.handleRequest(finalBody, clientHeaders);
    const qwenLatencyMs = Date.now() - qwenStartTime;
    const isStream = !!finalBody.stream;

    if (!upstream.ok) {
        const errorText = await upstream.text();
        await updateGatewayRequest(requestId, upstream.status, Date.now() - startTime);
        logEntry({
            type: "response",
            requestId,
            clientModel: finalBody.model || "qwen-only-low-risk",
            realUserIntentPreview: decision.realUserIntentPreview,
            intentType: decision.intentType,
            qwenOnlyUsed: false,
            qwenOnlyRejectedReason: `Qwen local provider returned ${upstream.status}`,
            qwenToolCallValid: true,
            qwenFakeToolJsonDetected: false,
            finalProvider: "none",
            deepseekFallbackUsed: false,
            confidence_risk_level: decision.confidenceRiskLevel,
            build_check_status: decision.buildCheckStatus,
            status: upstream.status
        });
        res.status(upstream.status).json({
            error: {
                type: "api_error",
                message: errorText || "Local AI is currently offline"
            }
        });
        return;
    }

    if (isStream && upstream.body) {
        res.status(upstream.status);
        const contentType = upstream.headers.get("content-type");
        if (contentType) res.setHeader("content-type", contentType);
        res.setHeader("cache-control", "no-cache");
        res.setHeader("x-accel-buffering", "no");
        res.setHeader("connection", "keep-alive");

        const reader = upstream.body.getReader();
        const decoder = new TextDecoder();
        let streamBuffer = "";
        let inputTokens = 0;
        let outputTokens = 0;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            res.write(Buffer.from(value));

            streamBuffer += decoder.decode(value, { stream: true });
            const lines = streamBuffer.split("\n");
            streamBuffer = lines.pop() || "";

            for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed.startsWith("data: ")) continue;
                try {
                    const dataJson = JSON.parse(trimmed.slice(6));
                    if (dataJson.message?.usage) {
                        const usage = dataJson.message.usage;
                        if (usage.input_tokens) inputTokens = usage.input_tokens;
                        if (usage.output_tokens) outputTokens = usage.output_tokens;
                    }
                    if (dataJson.usage) {
                        const usage = dataJson.usage;
                        if (usage.input_tokens) inputTokens = usage.input_tokens;
                        if (usage.prompt_tokens) inputTokens = usage.prompt_tokens;
                        if (usage.output_tokens) outputTokens = usage.output_tokens;
                        if (usage.completion_tokens) outputTokens = usage.completion_tokens;
                    }
                } catch {
                    // ignore parse errors while streaming
                }
            }
        }

        res.end();
        await recordModelCall({
            requestId,
            provider: "qwen-local",
            model: activeModelName,
            inputTokens,
            outputTokens,
            latencyMs: qwenLatencyMs,
            intentType: decision.intentType,
            exactContextSource: decision.exactContextSource,
            targetFile: decision.targetFile,
            qwenOnlyUsed: true
        });
        await updateGatewayRequest(requestId, upstream.status, Date.now() - startTime);
        logEntry({
            type: "response",
            requestId,
            clientModel: finalBody.model || "qwen-only-low-risk",
            realUserIntentPreview: decision.realUserIntentPreview,
            intentType: decision.intentType,
            qwenOnlyUsed: true,
            qwenOnlyRejectedReason: "",
            qwenToolCallValid: true,
            qwenFakeToolJsonDetected: false,
            finalProvider: "qwen-local",
            deepseekFallbackUsed: false,
            confidence_risk_level: decision.confidenceRiskLevel,
            build_check_status: decision.buildCheckStatus,
            status: upstream.status
        });
        return;
    }

    const text = await upstream.text();
    let responseBody: any;
    try {
        responseBody = JSON.parse(text);
    } catch {
        responseBody = null;
    }

    const usageInputTokens = responseBody?.usage?.input_tokens || responseBody?.usage?.prompt_tokens || 0;
    const usageOutputTokens = responseBody?.usage?.output_tokens || responseBody?.usage?.completion_tokens || 0;

    let qwenToolCallValid = true;
    let qwenFakeToolJsonDetected = false;
    let qwenOnlyRejectedReason = "";
    let qwenOnlyUsed = true;
    let finalProvider = "qwen-local";

    let validatedBody = responseBody;

    if (responseBody && upstream.status === 200) {
        const content = responseBody.content;
        if (Array.isArray(content)) {
            for (const block of content) {
                if (block?.type === "text" && detectFakeToolJson(block.text)) {
                    qwenToolCallValid = false;
                    qwenFakeToolJsonDetected = true;
                    qwenOnlyRejectedReason = "qwen_fake_tool_json_detected";
                    qwenOnlyUsed = false;
                    finalProvider = "none";
                    break;
                }
            }

            if (qwenToolCallValid && decision.intentType === "read_only") {
                if (hasDangerousToolCall(content)) {
                    qwenToolCallValid = false;
                    qwenOnlyRejectedReason = "qwen_dangerous_tool_rejected";
                    qwenOnlyUsed = false;
                    finalProvider = "none";
                }
            }
        }
    }

    if (!qwenToolCallValid) {
        validatedBody = {
            id: "msg_qwen_only_failed_tool_use_" + Math.random().toString(36).substring(7),
            type: "message",
            role: "assistant",
            content: [
                {
                    type: "text",
                    text: "Qwen-only could not produce a valid tool call. Use qwen-smart."
                }
            ],
            model: "qwen-only-low-risk",
            stop_reason: "end_turn",
            stop_sequence: null,
            usage: {
                input_tokens: usageInputTokens,
                output_tokens: usageOutputTokens
            }
        };
    }

    await recordModelCall({
        requestId,
        provider: "qwen-local",
        model: activeModelName,
        inputTokens: usageInputTokens,
        outputTokens: usageOutputTokens,
        latencyMs: qwenLatencyMs,
        intentType: decision.intentType,
        exactContextSource: decision.exactContextSource,
        targetFile: decision.targetFile,
        qwenOnlyUsed
    });
    await updateGatewayRequest(requestId, upstream.status, Date.now() - startTime);
    logEntry({
        type: "response",
        requestId,
        clientModel: finalBody.model || "qwen-only-low-risk",
        realUserIntentPreview: decision.realUserIntentPreview,
        intentType: decision.intentType,
        qwenOnlyUsed,
        qwenOnlyRejectedReason,
        qwenToolCallValid,
        qwenFakeToolJsonDetected,
        finalProvider,
        deepseekFallbackUsed: false,
        confidence_risk_level: decision.confidenceRiskLevel,
        build_check_status: decision.buildCheckStatus,
        status: qwenToolCallValid ? upstream.status : 200
    });

    if (responseBody && upstream.status === 200) {
        res.status(200).json(validatedBody);
        return;
    }

    if (responseBody) {
        res.status(upstream.status).json(responseBody);
        return;
    }

    res.status(upstream.status).json({
        error: {
            type: "upstream_error",
            message: text
        }
    });
}

export async function handleQwenOnlyLowRiskRequest(req: Request, res: Response): Promise<void> {
    const requestId = (req as any).requestId || crypto.randomUUID();
    const startTime = Date.now();
    const enabled = config.qwenOnlyLowRiskEnabled === true;
    const decision = classifyQwenOnlyIntent(req.body);
    const clientModel = req.body?.model || "qwen-only-low-risk";

    if (!enabled) {
        await updateGatewayRequest(requestId, 403, Date.now() - startTime);
        logEntry({
            type: "response",
            requestId,
            clientModel,
            realUserIntentPreview: decision.realUserIntentPreview,
            intentType: decision.intentType,
            qwenOnlyUsed: false,
            qwenOnlyRejectedReason: "Qwen-only low-risk mode is disabled",
            qwenToolCallValid: true,
            qwenFakeToolJsonDetected: false,
            finalProvider: "none",
            deepseekFallbackUsed: false,
            confidence_risk_level: "high",
            build_check_status: "not_run",
            status: 403
        });
        res.status(403).json({
            error: {
                type: "permission_error",
                message: "Qwen-only low-risk mode is disabled"
            }
        });
        return;
    }

    if (decision.action === "reject") {
        await updateGatewayRequest(requestId, 200, Date.now() - startTime);
        logEntry({
            type: "response",
            requestId,
            clientModel,
            realUserIntentPreview: decision.realUserIntentPreview,
            intentType: decision.intentType,
            qwenOnlyUsed: false,
            qwenOnlyRejectedReason: QWEN_ONLY_REJECTION_MESSAGE,
            qwenToolCallValid: true,
            qwenFakeToolJsonDetected: false,
            finalProvider: "none",
            deepseekFallbackUsed: false,
            confidence_risk_level: decision.confidenceRiskLevel,
            build_check_status: decision.buildCheckStatus,
            status: 200
        });
        res.status(200).json(buildQwenOnlyRejectedResponse(QWEN_ONLY_REJECTION_MESSAGE));
        return;
    }

    logEntry({
        type: "request",
        requestId,
        method: req.method,
        path: req.path,
        clientModel,
        upstreamModel: "qwen-only-low-risk",
        stream: !!req.body?.stream,
        provider: "qwen-only-orchestrator",
        realUserIntentPreview: decision.realUserIntentPreview,
        intentType: decision.intentType,
        qwenOnlyUsed: true,
        qwenOnlyRejectedReason: "",
        qwenToolCallValid: true,
        qwenFakeToolJsonDetected: false,
        finalProvider: "qwen-local",
        deepseekFallbackUsed: false,
        confidence_risk_level: decision.confidenceRiskLevel,
        build_check_status: decision.buildCheckStatus,
        status: 200
    });

    await forwardToQwenLocal(req, res, requestId, startTime, decision);
}

export { buildQwenOnlyRejectedResponse };


