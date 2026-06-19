import { Request, Response as ExpressResponse } from "express";
import crypto from "crypto";
import fs from "fs";
import path from "path";
import { providerRegistry } from "./registry.js";
import { DeepSeekProvider } from "../providers/deepseek.js";
import { QwenLocalProvider } from "../providers/qwen-local.js";
import { config } from "../config/env.js";
import { insertModelCall, updateGatewayRequest } from "../utils/db.js";
import { extractDeepSeekUsage, calculateDeepSeekCost } from "../utils/pricing.js";

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

type QwenDraftMode = "find_replace" | "unified_diff" | "line_range_replace" | "replacement_snippet" | "snippet" | "notes" | "insufficient_context" | "empty";

interface ParsedQwenPatch {
    ok: boolean;
    filePath?: string;
    find?: string;
    replace?: string;
    mode: "find_replace" | "unified_diff" | "line_range_replace" | "invalid";
    reason?: string;
    startLine?: number;
    endLine?: number;
}

function normalizeNewlines(str: string): string {
    return str.replace(/\r\n/g, "\n");
}

function cleanMarkdownFences(text: string): string {
    return text
        .split("\n")
        .filter(line => !line.trim().startsWith("```"))
        .join("\n");
}

function trimOuterBlankLines(str: string): string {
    const lines = str.split("\n");
    let start = 0;
    while (start < lines.length && lines[start].trim() === "") {
        start++;
    }
    let end = lines.length - 1;
    while (end >= start && lines[end].trim() === "") {
        end--;
    }
    return lines.slice(start, end + 1).join("\n");
}

function detectQwenDraftMode(text: string): QwenDraftMode {
    const t = String(text || "").trim();
    if (!t) return "empty";
    if (t.startsWith("INSUFFICIENT_CONTEXT")) return "insufficient_context";
    if (t.includes("FILE:") && t.includes("START_LINE:") && t.includes("END_LINE:") && t.includes("REPLACE:")) return "line_range_replace";
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
        mode === "line_range_replace" ||
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

function parseUnifiedDiffPatch(text: string): ParsedQwenPatch {
    const cleaned = cleanMarkdownFences(normalizeNewlines(text || "")).trim();
    const lines = cleaned.split("\n");

    let filePaths: string[] = [];
    for (const line of lines) {
        if (line.startsWith("+++ ")) {
            let p = line.slice(4).trim();
            if (p.startsWith("b/")) {
                p = p.slice(2);
            }
            if (p && !filePaths.includes(p)) {
                filePaths.push(p);
            }
        } else if (line.startsWith("--- ")) {
            let p = line.slice(4).trim();
            if (p.startsWith("a/")) {
                p = p.slice(2);
            }
            if (p && !filePaths.includes(p)) {
                filePaths.push(p);
            }
        }
    }

    if (filePaths.length === 0) {
        return { ok: false, mode: "unified_diff", reason: "patch_parse_failed" };
    }
    if (filePaths.length > 1) {
        return { ok: false, mode: "unified_diff", reason: "multiple_files_unsupported" };
    }

    const filePath = filePaths[0];

    const findLines: string[] = [];
    const replaceLines: string[] = [];

    for (const line of lines) {
        if (line.startsWith("---") || line.startsWith("+++") || line.startsWith("@@")) {
            continue;
        }
        if (line.startsWith("\\ No newline at end of file")) {
            continue;
        }

        if (line.startsWith("-")) {
            findLines.push(line.slice(1));
        } else if (line.startsWith("+")) {
            replaceLines.push(line.slice(1));
        } else if (line.startsWith(" ")) {
            findLines.push(line.slice(1));
            replaceLines.push(line.slice(1));
        } else if (line === "") {
            findLines.push("");
            replaceLines.push("");
        } else {
            findLines.push(line);
            replaceLines.push(line);
        }
    }

    const find = trimOuterBlankLines(findLines.join("\n"));
    const replace = trimOuterBlankLines(replaceLines.join("\n"));

    return {
        ok: true,
        filePath,
        find,
        replace,
        mode: "unified_diff"
    };
}

function parseQwenLineRangePatch(text: string, rawFileContent: string): ParsedQwenPatch {
    let cleaned = normalizeNewlines(text || "");
    cleaned = cleanMarkdownFences(cleaned);
    const t = cleaned.trim();

    const fileMatch = t.match(/(?:^|\n)FILE:\s*(.+?)(?=\n)/);
    const startMatch = t.match(/(?:^|\n)START_LINE:\s*(\d+)/);
    const endMatch = t.match(/(?:^|\n)END_LINE:\s*(\d+)/);
    const replaceMarker = "\nREPLACE:";
    const replaceIndex = t.indexOf(replaceMarker);

    if (!fileMatch || !startMatch || !endMatch || replaceIndex === -1) {
        return { ok: false, mode: "invalid", reason: "line_range_invalid_numbers" };
    }

    const filePath = fileMatch[1].trim();
    const startLine = parseInt(startMatch[1], 10);
    const endLine = parseInt(endMatch[1], 10);
    let replace = t.slice(replaceIndex + replaceMarker.length);
    replace = trimOuterBlankLines(replace);

    if (!filePath) {
        return { ok: false, mode: "invalid", reason: "missing_file" };
    }
    if (isNaN(startLine) || isNaN(endLine) || startLine < 1 || endLine < startLine) {
        return { ok: false, mode: "invalid", reason: "line_range_invalid_numbers" };
    }

    if (!rawFileContent) {
        return { ok: false, mode: "invalid", reason: "no_exact_file_context_advisory_only" };
    }

    const originalLines = rawFileContent.split("\n");
    if (endLine > originalLines.length) {
        return { ok: false, mode: "invalid", reason: "line_range_out_of_bounds" };
    }

    const find = originalLines.slice(startLine - 1, endLine).join("\n");

    return {
        ok: true,
        mode: "line_range_replace",
        filePath,
        find,
        replace,
        startLine,
        endLine
    };
}

function parseQwenFindReplacePatch(text: string, rawFileContent?: string): ParsedQwenPatch {
    let cleaned = normalizeNewlines(text || "");
    cleaned = cleanMarkdownFences(cleaned);

    const t = cleaned.trim();
    if (!t) {
        return { ok: false, mode: "invalid", reason: "empty_patch" };
    }

    if (t.includes("FILE:") && t.includes("START_LINE:") && t.includes("END_LINE:") && t.includes("REPLACE:")) {
        return parseQwenLineRangePatch(text, rawFileContent || "");
    }

    if (t.includes("---") && t.includes("+++")) {
        const parsed = parseUnifiedDiffPatch(text);
        if (parsed.ok) {
            return parsed;
        }
        return { ok: false, mode: "unified_diff", reason: "patch_parse_failed" };
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
    let find = t.slice(findIndex + findMarker.length, replaceIndex);
    let replace = t.slice(replaceIndex + replaceMarker.length);

    find = trimOuterBlankLines(find);
    replace = trimOuterBlankLines(replace);

    return {
        ok: true,
        filePath,
        find,
        replace,
        mode: "find_replace"
    };
}

function clampNumber(val: number, min: number, max: number): number {
    return Math.min(Math.max(val, min), max);
}

function getFileContentFromToolResults(messages: any[], targetFilePath: string): { content: string; isExact: boolean } {
    if (!targetFilePath) return { content: "", isExact: false };
    const targetName = path.basename(targetFilePath).toLowerCase();

    for (const msg of messages.slice().reverse()) {
        if (!Array.isArray(msg.content)) continue;

        for (const block of msg.content) {
            if (block?.type === "tool_result" && block.content !== undefined) {
                const toolUseId = block.tool_use_id;
                let isMatch = false;
                let toolName = "";

                if (toolUseId) {
                    const toolUseMsg = messages.find(m => 
                        Array.isArray(m.content) && 
                        m.content.some((b: any) => b?.type === "tool_use" && b.id === toolUseId)
                    );
                    if (toolUseMsg) {
                        const toolUseBlock = toolUseMsg.content.find((b: any) => b?.type === "tool_use" && b.id === toolUseId);
                        toolName = toolUseBlock?.name || "";
                        const inputPath = toolUseBlock?.input?.AbsolutePath || toolUseBlock?.input?.file_path || toolUseBlock?.input?.path || "";
                        if (typeof inputPath === "string" && inputPath.toLowerCase().endsWith(targetName)) {
                            isMatch = true;
                        }
                    }
                }

                if (isMatch) {
                    let rawText = "";
                    if (typeof block.content === "string") {
                        rawText = block.content;
                    } else if (Array.isArray(block.content)) {
                        rawText = block.content.map((b: any) => {
                            if (typeof b === "string") return b;
                            return b?.text || b?.content || JSON.stringify(b);
                        }).join("\n");
                    } else if (typeof block.content === "object" && block.content !== null) {
                        rawText = block.content.text || block.content.content || JSON.stringify(block.content);
                    }

                    try {
                        const parsed = JSON.parse(rawText);
                        if (parsed && typeof parsed === "object") {
                            rawText = parsed.content || parsed.text || rawText;
                        }
                    } catch {}

                    rawText = rawText.replace(/\r\n/g, "\n");

                    const lowerTool = toolName.toLowerCase();
                    const isExact = lowerTool.includes("view") || lowerTool.includes("read") || lowerTool.includes("show") || lowerTool.includes("get");

                    return { content: rawText, isExact };
                }
            }
        }
    }
    return { content: "", isExact: false };
}

function validateQwenPatch(
    patch: ParsedQwenPatch,
    messages: any[],
    userIntent: string,
    rawFileContent: string,
    hasExactOriginalFileContent: boolean,
    source?: string
): ParsedQwenPatch {
    if (!patch.ok) return patch;

    const filePath = patch.filePath || "";
    const find = patch.find || "";
    const replace = patch.replace || "";
    const targetFile = getTargetFileFromRecentToolUse(messages);

    const normalizedFind = normalizeNewlines(find);
    const normalizedReplace = normalizeNewlines(replace);

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
    const findLines = normalizedFind.split("\n");
    if (patch.mode !== "line_range_replace" && findLines.length <= 1) {
        return { ...patch, ok: false, reason: "find_block_too_short" };
    }

    const hasAtLeast5Lines = findLines.length >= 5;
    const isTargetFileMatch = !targetFile || filePath === targetFile;
    const canUseExpandedRatio = hasAtLeast5Lines && isTargetFileMatch && hasExactOriginalFileContent;
    const replaceRatioLimit = explicitlyLargeRewrite ? 20 : (canUseExpandedRatio ? 5 : 3);

    if (replace.length > find.length * replaceRatioLimit) {
        return { ...patch, ok: false, reason: "replace_too_large" };
    }

    if (!hasExactOriginalFileContent) {
        return { ...patch, ok: false, reason: source === "tool_result_partial" ? "tool_result_partial_context_advisory_only" : "no_exact_file_context_advisory_only" };
    }

    const normalizedFileContent = normalizeNewlines(rawFileContent);

    const occurrences = countOccurrences(normalizedFileContent, normalizedFind);
    if (occurrences !== 1) {
        if (patch.mode === "line_range_replace") {
            return {
                ...patch,
                ok: false,
                reason: occurrences === 0 ? "line_range_find_not_found" : "line_range_find_matches_multiple"
            };
        }
        return {
            ...patch,
            ok: false,
            reason: occurrences === 0 ? "find_block_not_found_in_context" : "find_block_matches_multiple"
        };
    }

    return {
        ...patch,
        find: normalizedFind,
        replace: normalizedReplace
    };
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

    const usage = extractDeepSeekUsage(data);
    const costDetails = calculateDeepSeekCost(config.defaultModel, usage);

    await insertModelCall({
        requestId,
        provider: "deepseek",
        model: `${config.defaultModel}-patch-approval`,
        inputTokens: usage.inputTokens,
        outputTokens: usage.outputTokens,
        cacheHitInputTokens: usage.cacheHitInputTokens,
        cacheMissInputTokens: usage.cacheMissInputTokens,
        latencyMs,
        ...costDetails
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
        const usage = extractDeepSeekUsage(routerData);
        const costDetails = calculateDeepSeekCost(config.defaultModel, usage);

        await insertModelCall({
            requestId,
            provider: "deepseek",
            model: `${config.defaultModel}-router`,
            inputTokens: usage.inputTokens,
            outputTokens: usage.outputTokens,
            cacheHitInputTokens: usage.cacheHitInputTokens,
            cacheMissInputTokens: usage.cacheMissInputTokens,
            latencyMs,
            ...costDetails
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
                                const msgUsage = dataJson.message.usage;
                                if (msgUsage.input_tokens) inputTokens = msgUsage.input_tokens;
                                if (msgUsage.output_tokens) outputTokens = msgUsage.output_tokens;
                                if (msgUsage.cache_read_input_tokens) cacheReadTokens = msgUsage.cache_read_input_tokens;
                                if (msgUsage.cache_creation_input_tokens) cacheCreationTokens = msgUsage.cache_creation_input_tokens;
                                if (msgUsage.prompt_cache_hit_tokens) cacheReadTokens = msgUsage.prompt_cache_hit_tokens;
                                if (msgUsage.prompt_cache_miss_tokens) cacheCreationTokens = msgUsage.prompt_cache_miss_tokens;
                            }
                            if (dataJson.usage) {
                                const u = dataJson.usage;
                                if (u.input_tokens) inputTokens = u.input_tokens;
                                if (u.prompt_tokens) inputTokens = u.prompt_tokens;
                                if (u.output_tokens) outputTokens = u.output_tokens;
                                if (u.completion_tokens) outputTokens = u.completion_tokens;
                                if (u.prompt_cache_hit_tokens) cacheReadTokens = u.prompt_cache_hit_tokens;
                                if (u.prompt_cache_miss_tokens) cacheCreationTokens = u.prompt_cache_miss_tokens;
                                if (u.cache_read_input_tokens) cacheReadTokens = u.cache_read_input_tokens;
                                if (u.cache_hit_input_tokens) cacheReadTokens = u.cache_hit_input_tokens;
                                if (u.cache_creation_input_tokens) cacheCreationTokens = u.cache_creation_input_tokens;
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
                const extracted = extractDeepSeekUsage(dataJson);
                inputTokens = extracted.inputTokens;
                outputTokens = extracted.outputTokens;
                cacheReadTokens = extracted.cacheHitInputTokens;
                cacheCreationTokens = extracted.cacheMissInputTokens;
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

        const upstreamModel = deepseekProvider.resolveUpstreamModel(body.model);

        if (qwenSavings) {
            const hypotheticalUsage = {
                inputTokens: qwenSavings.inputTokens,
                outputTokens: qwenSavings.outputTokens,
                cacheHitInputTokens: 0,
                cacheMissInputTokens: qwenSavings.inputTokens
            };
            const hypotheticalCost = calculateDeepSeekCost(upstreamModel, hypotheticalUsage);
            savedUsd = hypotheticalCost.totalCostUsd;
            savedThb = hypotheticalCost.totalCostThb;
            savedInputUsd = hypotheticalCost.inputCostUsd;
            savedInputThb = hypotheticalCost.inputCostThb;
            savedOutputUsd = hypotheticalCost.outputCostUsd;
            savedOutputThb = hypotheticalCost.outputCostThb;
        }

        const usage = {
            inputTokens,
            outputTokens,
            cacheHitInputTokens: cacheReadTokens,
            cacheMissInputTokens: cacheCreationTokens || Math.max(0, inputTokens - cacheReadTokens)
        };
        const costDetails = calculateDeepSeekCost(upstreamModel, usage);

        await insertModelCall({
            requestId,
            provider: "deepseek",
            model: upstreamModel,
            inputTokens: usage.inputTokens,
            outputTokens: usage.outputTokens,
            cacheHitInputTokens: usage.cacheHitInputTokens,
            cacheMissInputTokens: usage.cacheMissInputTokens,
            latencyMs: callLatencyMs,
            savedUsd,
            savedThb,
            savedInputUsd,
            savedInputThb,
            savedOutputUsd,
            savedOutputThb,
            ...costDetails
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

        const resolvedConfig = await qwenProvider.resolveRuntimeConfig();
        const activeModelName = resolvedConfig.modelName;
        const qwenMaxTokens = clampNumber(config.qwenLocalMaxTokens ?? 32000, 512, 32000);

        const userIntent = getLastRealUserInstruction(messages);
        const reducedContext = extractReducedContext(messages, userIntent);
        const reducedContextChars = reducedContext.length;

        const targetFile = getTargetFileFromRecentToolUse(messages);
        let rawFileContent = "";
        let source: "disk" | "tool_result_exact" | "tool_result_partial" | "reduced_context" = "reduced_context";

        if (targetFile) {
            if (fs.existsSync(targetFile)) {
                try {
                    rawFileContent = fs.readFileSync(targetFile, "utf-8");
                    source = "disk";
                } catch (e) {
                    console.error("Failed to read raw target file content for Qwen prompt", e);
                }
            }
            if (!rawFileContent) {
                // Try to extract from previous tool results
                const extracted = getFileContentFromToolResults(messages, targetFile);
                if (extracted.content) {
                    rawFileContent = extracted.content;
                    source = extracted.isExact ? "tool_result_exact" : "tool_result_partial";
                }
            }
        }

        if (!rawFileContent) {
            rawFileContent = reducedContext;
            source = "reduced_context";
        }

        const hasExactOriginalFileContent = rawFileContent.length > 0 && (source === "disk" || source === "tool_result_exact");

        let originalContentBlock = "";
        if (hasExactOriginalFileContent) {
            const lines = rawFileContent.split("\n");
            const numberedLines = lines.map((line, idx) => `${idx + 1}| ${line}`).join("\n");
            originalContentBlock = `ORIGINAL_FILE_CONTENT_WITH_LINE_NUMBERS:
\`\`\`
${numberedLines}
\`\`\``;
        } else {
            originalContentBlock = `ORIGINAL_FILE_CONTENT of ${targetFile}:
\`\`\`
${rawFileContent}
\`\`\``;
        }

        const qwenSystemPrompt = hasExactOriginalFileContent
            ? `Use ONLY ORIGINAL_FILE_CONTENT.
Preferred format:
FILE: <target file path>
START_LINE: <first line number to replace>
END_LINE: <last line number to replace>
REPLACE: <replacement code>

Rules:
- START_LINE and END_LINE must refer to the provided ORIGINAL_FILE_CONTENT_WITH_LINE_NUMBERS.
- Replace the smallest safe block.
- Include enough full block/function context.
- Line numbers are for START_LINE/END_LINE only and must NOT be included in REPLACE.
- Do NOT use unified diff unless explicitly requested.
- Do NOT explain.

Fallback format if line numbers are impossible:
FILE: <target file path>
FIND:
<exact original block copied from ORIGINAL_FILE_CONTENT_WITH_LINE_NUMBERS without line numbers>
REPLACE:
<replacement block>`
            : `Use ONLY the TARGET_FILE.
Return exactly one FIND/REPLACE patch.
Do NOT return unified diff.
Do NOT use --- +++ @@.
Do NOT explain.

Format:
FILE: <target file path>
FIND:
<exact original block>
REPLACE:
<replacement block>`;

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
                            content: `TARGET_FILE: ${targetFile}

${originalContentBlock}

Reduced context:
${reducedContext}

Task: Generate the primary implementation patch for this code edit request.

Latest user intent preview: ${decision.userIntentPreview}`
                        }
                    ],
                    stream: false,
                    max_tokens: qwenMaxTokens,
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

                let patchCheck = validateQwenPatch(parseQwenFindReplacePatch(draftText, rawFileContent), messages, userIntent, rawFileContent, hasExactOriginalFileContent, source);

                const failedReasons = [
                    "patch_parse_unsupported",
                    "patch_parse_failed",
                    "find_block_too_short",
                    "replace_too_large",
                    "missing_file_find_or_replace",
                    "empty_patch",
                    "not_parsed",
                    "find_block_not_found_in_context",
                    "line_range_find_not_found",
                    "line_range_find_matches_multiple",
                    "line_range_invalid_numbers",
                    "line_range_out_of_bounds"
                ];

                const needsRetry = shouldRetryQwenDraft(qwenDraftMode, qwenDraftChars) || 
                    (!patchCheck.ok && failedReasons.includes(patchCheck.reason || ""));

                if (needsRetry) {
                    qwenRetryUsed = true;
                    let retryPrompt = "Your previous answer was not an implementation patch.\nReturn ONLY a unified diff or FIND/REPLACE snippet.\nNo explanation. No notes. Write the actual code now.";
                    
                    if (!patchCheck.ok && failedReasons.includes(patchCheck.reason || "")) {
                        if (hasExactOriginalFileContent) {
                            retryPrompt = `Your previous patch could not be applied.
Return ONLY line-range format.

FILE: <target file path>
START_LINE: <number>
END_LINE: <number>
REPLACE: <replacement code>

Rules:
- Use the line numbers from ORIGINAL_FILE_CONTENT_WITH_LINE_NUMBERS.
- Do not include line numbers in REPLACE.
- Do not use FIND/REPLACE.
- Do not use unified diff.
- Do not explain.`;
                        } else {
                            retryPrompt = `Your previous patch could not be applied by the Gateway.
Return ONLY FIND/REPLACE format, not unified diff.

Required format:
FILE: <target file path>
FIND:
<copy exactly 5 to 25 consecutive lines from ORIGINAL_FILE_CONTENT>
REPLACE:
<full replacement for those exact lines>

Rules:
- Do not use --- +++ @@ unified diff.
- FIND must exist exactly once in ORIGINAL_FILE_CONTENT.
- FIND must contain enough surrounding context.
- If changing a large block, include a larger FIND block so REPLACE is not more than 3x FIND.
- Do not explain.`;
                        }
                    }

                    qwenResult = await callQwen(retryPrompt);
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
            qwenErrorType = error.name === "AbortError" ? "qwen_timeout" : "qwen_connection_error";
            qwenDraftWeak = true;
        }

        if (draftText) {
            parsedPatch = validateQwenPatch(parseQwenFindReplacePatch(draftText, rawFileContent), messages, userIntent, rawFileContent, hasExactOriginalFileContent, source);
            qwenPatchValid = parsedPatch.ok;
            qwenPatchReason = parsedPatch.reason || (parsedPatch.ok ? "valid" : "invalid_patch");
        } else {
            qwenPatchReason = qwenErrorType || "qwen_output_empty";
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
                } else {
                    if (qwenRetryUsed) {
                        fallbackReason = parsedPatch.mode === "line_range_replace" ? "retry_line_range_success" : "retry_find_replace_success";
                    } else if (parsedPatch.mode === "line_range_replace") {
                        fallbackReason = "line_range_replace";
                    } else if (parsedPatch.mode === "unified_diff") {
                        fallbackReason = "unified_diff_converted_to_find_replace";
                    } else if (source === "tool_result_exact") {
                        fallbackReason = "tool_result_exact_context_used";
                    }
                }
            } else {
                fallbackReason = "deepseek_provider_not_registered";
            }
        } else if (draftText) {
            fallbackReason = qwenPatchReason;
        } else {
            fallbackReason = qwenErrorType || "qwen_output_empty";
        }

        await insertModelCall({
            requestId,
            provider: "qwen-local",
            model: activeModelName,
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
