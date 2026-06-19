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

type FileContextSource =
    | "tool_result_exact"
    | "tool_result_partial"
    | "disk_exact"
    | "reduced_context"
    | "none";

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

function isContinuationIntent(text: string): boolean {
    const normalized = String(text || "").trim().toLowerCase();
    if (!normalized) return false;

    const continuationPhrases = [
        "ต่อ",
        "ทำต่อ",
        "ต่อเลย",
        "แก้ต่อ",
        "จัด",
        "เอาเลย",
        "ok",
        "okay",
        "continue",
        "go on",
        "proceed",
        "do it",
        "apply it"
    ];

    return continuationPhrases.includes(normalized);
}

function isReadOnlyIntent(text: string): boolean {
    const normalized = String(text || "").toLowerCase();
    const readOnlyPatterns = [
        "explain",
        "summarize",
        "review only",
        "planning only",
        "plan only",
        "do not edit",
        "don't edit",
        "no edit",
        "read only",
        "ไม่ต้องแก้",
        "อย่าแก้",
        "ห้ามแก้"
    ];
    return readOnlyPatterns.some(pattern => normalized.includes(pattern));
}


function getUserTextFromMessage(msg: any): string {
    if (msg?.role !== "user") return "";

    if (typeof msg.content === "string") {
        return msg.content.trim();
    }

    if (Array.isArray(msg.content)) {
        return msg.content
            .filter((block: any) => block?.type === "text")
            .map((block: any) => String(block.text || ""))
            .join("\n")
            .trim();
    }

    return "";
}

function getRecentCodeEditInstruction(messages: any[]): string {
    for (const msg of messages.slice(-12).reverse()) {
        const text = getUserTextFromMessage(msg);
        if (text && isLikelyCodeEditTask(text)) {
            return text;
        }
    }

    return "";
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
    latestUserIntentPreview: string;
    recoveredCodeEditIntentPreview: string;
    inheritedCodeEditIntent: boolean;
    continuationIntent: boolean;
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

function normalizePatchPath(filePath: string): string {
    let p = String(filePath || "").trim();
    p = p.replace(/^["'`]+|["'`]+$/g, "");
    p = p.replace(/\\/g, "/");
    p = p.replace(/\/+/g, "/");
    p = p.replace(/^\.\//, "");
    p = p.replace(/^[ab]\//, "");

    const cwd = process.cwd().replace(/\\/g, "/").replace(/\/+$/g, "");
    const lower = p.toLowerCase();
    const lowerCwd = cwd.toLowerCase();
    if (lower === lowerCwd) {
        return "";
    }
    if (lower.startsWith(`${lowerCwd}/`)) {
        p = p.slice(cwd.length + 1);
    }

    const workspaceName = path.basename(cwd).toLowerCase();
    const parts = p.split("/").filter(Boolean);
    const workspaceIndex = parts.findIndex(part => part.toLowerCase() === workspaceName);
    if (workspaceIndex >= 0 && workspaceIndex < parts.length - 1) {
        p = parts.slice(workspaceIndex + 1).join("/");
    }

    return p.replace(/^\/+/, "").toLowerCase();
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
    const usefulCodeContext = hasUsefulCodeContext(messages);
    const latestIsCodeEdit = isLikelyCodeEditTask(userIntent);
    const continuationIntent = isContinuationIntent(userIntent);
    const readOnlyIntent = isReadOnlyIntent(userIntent) && !continuationIntent;
    const recentCodeEditInstruction = getRecentCodeEditInstruction(messages);
    const canRecoverIntent = hasResults && usefulCodeContext && !!recentCodeEditInstruction && !readOnlyIntent;
    const inheritedCodeEditIntent = canRecoverIntent && !latestIsCodeEdit;
    const likelyCodeEdit = latestIsCodeEdit ||
        (continuationIntent && canRecoverIntent) ||
        canRecoverIntent;
    const delegate_to_qwen = hasResults && likelyCodeEdit && usefulCodeContext;

    let reason = "Deterministic router approved Qwen delegation";
    if (!hasResults) {
        reason = "No tool_result context yet";
    } else if (!usefulCodeContext) {
        reason = "Tool results do not contain useful code context";
    } else if (!likelyCodeEdit) {
        reason = "Latest user intent is not a code edit task";
    } else if (inheritedCodeEditIntent) {
        reason = continuationIntent
            ? "Recovered code edit intent from recent conversation continuation"
            : "Recovered code edit intent from recent conversation with tool context";
    }

    const effectiveUserIntent = latestIsCodeEdit ? userIntent : (recentCodeEditInstruction || userIntent);

    return {
        delegate_to_qwen,
        reason,
        userIntentPreview: effectiveUserIntent.slice(0, 120),
        latestUserIntentPreview: userIntent.slice(0, 120),
        recoveredCodeEditIntentPreview: recentCodeEditInstruction.slice(0, 120),
        inheritedCodeEditIntent,
        continuationIntent,
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

function getRecentToolFilePaths(messages: any[]): string[] {
    const paths: string[] = [];

    for (const msg of messages.slice(-10)) {
        if (!Array.isArray(msg.content)) continue;

        for (const block of msg.content) {
            if (block?.type !== "tool_use") continue;
            const input = block.input || {};
            const filePath = input.AbsolutePath || input.file_path || input.path;
            if (typeof filePath === "string" && filePath.trim()) {
                paths.push(filePath.trim());
            }
        }
    }

    return paths;
}

function isUniqueRecentBasenameMatch(messages: any[], filePath: string, targetFile: string): boolean {
    const normalizedFilePath = normalizePatchPath(filePath);
    const normalizedTargetFile = normalizePatchPath(targetFile);
    const fileBase = path.basename(normalizedFilePath);
    const targetBase = path.basename(normalizedTargetFile);
    if (!fileBase || fileBase !== targetBase) return false;

    const matchingRecentPaths = new Set(
        getRecentToolFilePaths(messages)
            .map(normalizePatchPath)
            .filter(p => path.basename(p) === targetBase)
    );

    return matchingRecentPaths.size === 1;
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

function extractTextFromToolContent(content: any): string {
    if (typeof content === "string") return content;
    if (Array.isArray(content)) {
        return content.map(item => {
            if (typeof item === "string") return item;
            if (item?.type === "text" && typeof item.text === "string") return item.text;
            return extractTextFromToolContent(item);
        }).filter(Boolean).join("\n");
    }
    if (content && typeof content === "object") {
        for (const key of ["content", "text", "output", "stdout", "data"]) {
            if (content[key] !== undefined) {
                const extracted = extractTextFromToolContent(content[key]);
                if (extracted) return extracted;
            }
        }
        return "";
    }
    return "";
}

function unwrapToolResultText(content: any): string {
    let rawText = extractTextFromToolContent(content);
    for (let i = 0; i < 2; i++) {
        const trimmed = rawText.trim();
        if (!trimmed.startsWith("{") && !trimmed.startsWith("[")) break;
        try {
            const parsed = JSON.parse(trimmed);
            const extracted = extractTextFromToolContent(parsed);
            if (!extracted || extracted === rawText) break;
            rawText = extracted;
        } catch {
            break;
        }
    }
    return rawText;
}

function stripLineNumberPrefixes(text: string): string {
    const normalized = normalizeNewlines(text);
    const lines = normalized.split("\n");
    const prefixedLines = lines.filter(line => /^\s*\d+\s*(?:\||:)\s?/.test(line)).length;
    if (prefixedLines === 0 || prefixedLines < Math.max(2, Math.floor(lines.length * 0.5))) {
        return normalized;
    }
    return lines.map(line => line.replace(/^\s*\d+\s*(?:\||:)\s?/, "")).join("\n");
}

function getFileContentFromToolResults(messages: any[], targetFilePath: string): { content: string; isExact: boolean } {
    if (!targetFilePath) return { content: "", isExact: false };
    const normalizedTarget = normalizePatchPath(targetFilePath);
    const targetName = path.basename(normalizedTarget);

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
                        const normalizedInputPath = typeof inputPath === "string" ? normalizePatchPath(inputPath) : "";
                        if (
                            normalizedInputPath === normalizedTarget ||
                            (path.basename(normalizedInputPath) === targetName && isUniqueRecentBasenameMatch(messages, normalizedInputPath, normalizedTarget))
                        ) {
                            isMatch = true;
                        }
                    }
                }

                if (isMatch) {
                    let rawText = stripLineNumberPrefixes(unwrapToolResultText(block.content));

                    const lowerTool = toolName.toLowerCase();
                    const exactToolPattern = /(^|[_\-\s.])(read|view|get|open|cat|show)([_\-\s.]|$)/i;
                    const isExact = exactToolPattern.test(lowerTool) || /file.*(content|text)/i.test(lowerTool);

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
    const normalizedPatchFile = normalizePatchPath(filePath);
    const normalizedTargetFile = normalizePatchPath(targetFile);

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
    const isTargetFileMatch = !targetFile ||
        normalizedPatchFile === normalizedTargetFile ||
        isUniqueRecentBasenameMatch(messages, filePath, targetFile);

    if (targetFile && !isTargetFileMatch) {
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
    const canUseExpandedRatio = hasAtLeast5Lines && isTargetFileMatch && hasExactOriginalFileContent;
    const replaceRatioLimit = explicitlyLargeRewrite ? 20 : (canUseExpandedRatio ? 5 : 3);

    if (replace.length > find.length * replaceRatioLimit) {
        return { ...patch, ok: false, reason: "replace_too_large" };
    }

    if (!hasExactOriginalFileContent) {
        let reason = "no_exact_file_context_advisory_only";
        if (source === "tool_result_partial") {
            reason = "tool_result_partial_context_advisory_only";
        } else if (source === "reduced_context") {
            reason = "reduced_context_advisory_only";
        } else if (source === "none") {
            reason = "no_workspace_file_on_gateway";
        }
        return { ...patch, ok: false, reason };
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
                latestUserIntentPreview: decision.latestUserIntentPreview,
                recoveredCodeEditIntentPreview: decision.recoveredCodeEditIntentPreview,
                inheritedCodeEditIntent: decision.inheritedCodeEditIntent,
                continuationIntent: decision.continuationIntent,
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
                latestUserIntentPreview: decision.latestUserIntentPreview,
                recoveredCodeEditIntentPreview: decision.recoveredCodeEditIntentPreview,
                inheritedCodeEditIntent: decision.inheritedCodeEditIntent,
                continuationIntent: decision.continuationIntent,
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
                latestUserIntentPreview: decision.latestUserIntentPreview,
                recoveredCodeEditIntentPreview: decision.recoveredCodeEditIntentPreview,
                inheritedCodeEditIntent: decision.inheritedCodeEditIntent,
                continuationIntent: decision.continuationIntent,
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

        const latestUserIntent = getLastRealUserInstruction(messages);
        const recoveredCodeEditIntent = getRecentCodeEditInstruction(messages);
        const userIntent = decision.inheritedCodeEditIntent && recoveredCodeEditIntent
            ? recoveredCodeEditIntent
            : latestUserIntent;
        const reducedContext = extractReducedContext(messages, userIntent);
        const reducedContextChars = reducedContext.length;

        const targetFile = getTargetFileFromRecentToolUse(messages);
        let rawFileContent = "";
        let fileContextSource: FileContextSource = "none";

        if (targetFile) {
            // 1. First try extracting exact file content from tool_result / Read result / message context
            const extracted = getFileContentFromToolResults(messages, targetFile);
            if (extracted.content && extracted.isExact) {
                rawFileContent = extracted.content;
                fileContextSource = "tool_result_exact";
            }

            // 2. Only try fs.existsSync/readFileSync as fallback if ALLOW_RAILWAY_DISK_FILE_CONTEXT is enabled
            if (!rawFileContent && config.allowRailwayDiskFileContext) {
                if (fs.existsSync(targetFile)) {
                    try {
                        rawFileContent = fs.readFileSync(targetFile, "utf-8");
                        fileContextSource = "disk_exact";
                    } catch (e) {
                        console.error("Failed to read raw target file content for Qwen prompt", e);
                    }
                }
            }

            // 3. Fallback to partial tool result if exact not found
            if (!rawFileContent && extracted.content) {
                rawFileContent = extracted.content;
                fileContextSource = "tool_result_partial";
            }
        }

        // 4. Fallback to reduced context
        if (!rawFileContent) {
            rawFileContent = reducedContext;
            fileContextSource = reducedContext ? "reduced_context" : "none";
        }

        const hasExactOriginalFileContent = rawFileContent.length > 0 && (fileContextSource === "tool_result_exact" || fileContextSource === "disk_exact");
        const directEditEligible = hasExactOriginalFileContent;
        const qwenDelegationMode = directEditEligible ? "patch_draft" : "advisory_draft";
        const totalLineCount = hasExactOriginalFileContent ? rawFileContent.split("\n").length : 0;

        let originalContentBlock = "";
        if (hasExactOriginalFileContent) {
            const lines = rawFileContent.split("\n");
            const numberedLines = lines.map((line, idx) => `${idx + 1}| ${line}`).join("\n");
            originalContentBlock = `ORIGINAL_FILE_CONTENT_WITH_LINE_NUMBERS:
\`\`\`
${numberedLines}
\`\`\``;
        } else {
            originalContentBlock = `ORIGINAL_FILE_CONTENT of ${targetFile || "unknown"}:
\`\`\`
${rawFileContent}
\`\`\``;
        }

        const qwenSystemPrompt = hasExactOriginalFileContent
            ? `You are given Exact File Content from the user's coding tool context.
Use it as the only source of truth.
Return a patch that can be applied safely.

Return ONLY this format:
FILE: <target file path>
START_LINE: <first line number>
END_LINE: <last line number>
REPLACE: <replacement code without line numbers>

Rules:
- Return ONLY line-range format. Do not return unified diff. Do not use --- +++ @@.
- Choose the smallest safe line range.
- START_LINE and END_LINE must exist in ORIGINAL_FILE_CONTENT_WITH_LINE_NUMBERS.
- Never invent line numbers.
- Do not include markdown.
- Do not include explanation.
- Do not use unified diff.
- Do not use FIND/REPLACE.`
            : `Exact File Content is not available.
Write concise implementation notes only.
Do not claim this is directly applicable.
Do not return a patch.
Do not return line_range_replace.
Do not return unified diff.
Do not use --- +++ @@.
Do not use FILE/START_LINE/END_LINE/REPLACE.
Do not use FIND/REPLACE.
Do not output Edit tool instructions.
Keep the response under 300 tokens.
Tell DeepSeek/Claude Code which exact file content should be read before patching.`;

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
                    max_tokens: hasExactOriginalFileContent ? qwenMaxTokens : Math.min(qwenMaxTokens, 300),
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

                let patchCheck = hasExactOriginalFileContent
                    ? (qwenDraftMode === "unified_diff"
                        ? { ok: false, mode: "unified_diff" as const, reason: "unified_diff" }
                        : validateQwenPatch(parseQwenFindReplacePatch(draftText, rawFileContent), messages, userIntent, rawFileContent, hasExactOriginalFileContent, fileContextSource))
                    : { ok: false, mode: "invalid" as const, reason: "no_exact_context_advisory_only" };

                const failedReasons = [
                    "patch_parse_unsupported",
                    "patch_parse_failed",
                    "find_block_too_short",
                    "replace_too_large",
                    "missing_file_find_or_replace",
                    "empty_patch",
                    "not_parsed",
                    "find_block_not_found_in_context",
                    "find_block_matches_multiple",
                    "line_range_find_not_found",
                    "line_range_find_matches_multiple",
                    "line_range_invalid_numbers",
                    "line_range_out_of_bounds",
                    "file_mismatch"
                ];

                const needsRetry = hasExactOriginalFileContent && (
                    shouldRetryQwenDraft(qwenDraftMode, qwenDraftChars) ||
                    qwenDraftMode === "unified_diff" ||
                    (!patchCheck.ok && failedReasons.includes(patchCheck.reason || ""))
                );

                if (needsRetry) {
                    qwenRetryUsed = true;
                    let retryPrompt = "Your previous answer was not an implementation patch.\nReturn ONLY a unified diff or FIND/REPLACE snippet.\nNo explanation. No notes. Write the actual code now.";
                    
                    if (qwenDraftMode === "unified_diff" && hasExactOriginalFileContent) {
                        retryPrompt = `Your previous answer used unified diff, which is not allowed.
Return ONLY line-range format.

FILE: ${targetFile}
START_LINE: <number between 1 and ${totalLineCount}>
END_LINE: <number between START_LINE and ${totalLineCount}>
REPLACE: <replacement code>

Rules:
- FILE must exactly equal ${targetFile}.
- Use only line numbers that exist in ORIGINAL_FILE_CONTENT_WITH_LINE_NUMBERS.
- Choose the smallest safe line range.
- Do not include line numbers in REPLACE.
- Do not use FIND/REPLACE.
- Do not use unified diff.
- Do not use --- +++ @@.
- Do not include markdown.
- Do not explain.`;
                    } else if (!patchCheck.ok && failedReasons.includes(patchCheck.reason || "")) {
                        if (hasExactOriginalFileContent) {
                            if (patchCheck.reason === "line_range_out_of_bounds") {
                                retryPrompt = `Your previous patch used line numbers outside the file.
The target file has exactly ${totalLineCount} lines.
Return ONLY line-range format.

FILE: ${targetFile}
START_LINE: <number between 1 and ${totalLineCount}>
END_LINE: <number between START_LINE and ${totalLineCount}>
REPLACE: <replacement code>

Rules:
- START_LINE and END_LINE must be between 1 and ${totalLineCount}.
- Use line numbers from ORIGINAL_FILE_CONTENT_WITH_LINE_NUMBERS only.
- Choose the smallest safe line range.
- Do not include line numbers in REPLACE.
- Do not use FIND/REPLACE.
- Do not use unified diff.
- Do not use --- +++ @@.
- Do not include markdown.
- Do not explain.`;
                            } else if (patchCheck.reason === "file_mismatch") {
                                retryPrompt = `Your previous patch used the wrong FILE value.
Return ONLY line-range format.

FILE: ${targetFile}
START_LINE: <number>
END_LINE: <number>
REPLACE: <replacement code>

Rules:
- FILE must exactly equal ${targetFile}.
- Use the line numbers from ORIGINAL_FILE_CONTENT_WITH_LINE_NUMBERS.
- Choose the smallest safe line range.
- Do not include line numbers in REPLACE.
- Do not use FIND/REPLACE.
- Do not use unified diff.
- Do not use --- +++ @@.
- Do not include markdown.
- Do not explain.`;
                            } else if ((patchCheck.reason || "").includes("find_block_not_found_in_context") || (patchCheck.reason || "").includes("find_block_matches_multiple")) {
                                retryPrompt = `Your previous patch used a FIND block that does not match the exact file context.
Return ONLY line-range format.

FILE: ${targetFile}
START_LINE: <number from ORIGINAL_FILE_CONTENT_WITH_LINE_NUMBERS>
END_LINE: <number from ORIGINAL_FILE_CONTENT_WITH_LINE_NUMBERS>
REPLACE: <replacement code>

Rules:
- FILE must exactly equal ${targetFile}.
- START_LINE and END_LINE must be from the numbered file.
- REPLACE must not include line numbers.
- Choose the smallest safe line range.
- Do not use FIND/REPLACE.
- Do not use unified diff.
- Do not use --- +++ @@.
- Do not include markdown.
- Do not explain.`;
                            } else {
                                retryPrompt = `Your previous patch could not be applied.
Return ONLY line-range format.

FILE: ${targetFile}
START_LINE: <number>
END_LINE: <number>
REPLACE: <replacement code>

Rules:
- Use the line numbers from ORIGINAL_FILE_CONTENT_WITH_LINE_NUMBERS.
- Choose the smallest safe line range.
- Do not include line numbers in REPLACE.
- Do not use FIND/REPLACE.
- Do not use unified diff.
- Do not use --- +++ @@.
- Do not include markdown.
- Do not explain.`;
                            }
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
            parsedPatch = hasExactOriginalFileContent
                ? (qwenDraftMode === "unified_diff"
                    ? { ok: false, mode: "invalid", reason: "unified_diff" }
                    : validateQwenPatch(parseQwenFindReplacePatch(draftText, rawFileContent), messages, userIntent, rawFileContent, hasExactOriginalFileContent, fileContextSource))
                : { ok: false, mode: "invalid", reason: "no_exact_context_advisory_only" };
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
                    } else if (fileContextSource === "tool_result_exact") {
                        fallbackReason = "tool_result_exact_context_used";
                    } else if (fileContextSource === "disk_exact") {
                        fallbackReason = "railway_disk_context_used_dev_only";
                    }
                }
            } else {
                fallbackReason = "deepseek_provider_not_registered";
            }
        } else if (draftText && !hasExactOriginalFileContent) {
            fallbackReason = "no_exact_context_advisory_only";
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
            fallbackReason,
            fileContextSource,
            qwenDelegationMode,
            directEditEligible
        });

        if (qwenPatchValid && deepseekApprovalApproved && parsedPatch.filePath && parsedPatch.find !== undefined && parsedPatch.replace !== undefined) {
            const toolUseId = `toolu_qwen_edit_${crypto.randomUUID().replace(/-/g, "").slice(0, 16)}`;
            emittedToolUse = "Edit";
            console.log(JSON.stringify({
                time: new Date().toISOString(),
                requestId,
                mode: "hybrid-flow",
                delegate_to_qwen: true,
                latestUserIntentPreview: decision.latestUserIntentPreview,
                recoveredCodeEditIntentPreview: decision.recoveredCodeEditIntentPreview,
                inheritedCodeEditIntent: decision.inheritedCodeEditIntent,
                continuationIntent: decision.continuationIntent,
                qwenPatchMode: parsedPatch.mode,
                qwenPatchValid: true,
                qwenPatchReason,
                qwenDraftChars,
                qwenLatencyMs,
                deepseekApprovalUsed,
                deepseekApprovalApproved,
                finalProvider: "deepseek",
                emittedToolUse,
                fileContextSource,
                hasExactOriginalFileContent,
                directEditEligible,
                qwenDelegationMode
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
            const validButNotApproved = qwenPatchValid && !deepseekApprovalApproved;
            const forwardedDraftText = validButNotApproved && hasExactOriginalFileContent ? draftText : draftText.slice(0, 800);
            const advisoryIntro = qwenPatchValid
                ? `Internal Qwen primary patch draft below.

Qwen is the primary implementation writer, but Gateway could not emit Edit directly.
Fallback reason: ${fallbackReason || "unknown"}.
If the patch is valid, apply it using Claude Code tools.
Do not rewrite the solution from scratch unless the draft is clearly wrong, unsafe, or inconsistent with the file context.`
                : `Internal Qwen patch draft below.

Gateway validation rejected this draft.
Fallback reason: ${fallbackReason || qwenPatchReason || "invalid_patch"}.
Patch mode: ${qwenDraftMode}.
Only the first 800 chars of the rejected draft are included.
Treat this only as advisory context. Verify against the actual file context before using it.`;
            const exactContextInstruction = hasExactOriginalFileContent
                ? "If applying a valid Qwen patch, use tools directly."
                : "Exact file context is missing. Read exact file content before creating or applying any patch.";
            const augmentedMessages = [
                ...req.body.messages,
                {
                    role: "user",
                    content: [
                        {
                            type: "text",
                            text: `${advisoryIntro}

Keep response short.
Prefer tool_use/Edit only if verified.
max final explanation 3 bullets.
${exactContextInstruction}
Do not generate another full implementation unless Qwen draft is wrong.

<QWEN_DRAFT mode="${qwenDraftMode}" chars="${qwenDraftChars}">
${forwardedDraftText}
</QWEN_DRAFT>`
                        }
                    ]
                }
            ];
            const originalMaxTokens = typeof req.body.max_tokens === "number" ? req.body.max_tokens : undefined;
            finalBody = {
                ...req.body,
                messages: augmentedMessages,
                max_tokens: originalMaxTokens === undefined ? 700 : Math.min(originalMaxTokens, 700)
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
            latestUserIntentPreview: decision.latestUserIntentPreview,
            recoveredCodeEditIntentPreview: decision.recoveredCodeEditIntentPreview,
            inheritedCodeEditIntent: decision.inheritedCodeEditIntent,
            continuationIntent: decision.continuationIntent,
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
            fallbackReason,
            fileContextSource,
            hasExactOriginalFileContent,
            directEditEligible,
            qwenDelegationMode
        }));

        const qwenSavings = qwenDraftUsed ? { inputTokens: qwenInputTokens, outputTokens: qwenOutputTokens } : undefined;
        await this.forwardToDeepSeek(finalBody, clientHeaders, res, isStream, requestId, qwenSavings);
    }
}
