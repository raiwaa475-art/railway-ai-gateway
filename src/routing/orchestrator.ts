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
import { ConfidenceRiskLevel, evaluateConfidence } from "./confidence.js";
import { handleQwenOnlyLowRiskRequest } from "./qwen-only-low-risk.js";
import { handleQwenAgentRequest } from "./qwen-agent.js";

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
        "เน€เธเธเน€เธยเน€เธเธ’เน€เธเธเน€เธยเน€เธยเน€เธย",
        "เน€เธเธเน€เธยเน€เธเธ’เน€เธเธเน€เธยเน€เธยเน€เธยเน€เธยเน€เธยเน€เธเธ…เน€เธย",
        "เน€เธยเน€เธเธเน€เธยเน€เธโ€ขเน€เธยเน€เธเธเน€เธยเน€เธยเน€เธยเน€เธย",
        "เน€เธเธเน€เธเธเน€เธยเน€เธเธ’เน€เธยเน€เธยเน€เธย",
        "read only",
        "do not edit",
        "don't edit",
        "no edit"
    ];
    if (explicitReadOnlyPatterns.some(pattern => normalized.includes(pattern))) {
        return false;
    }

    const codeEditKeywords = [
        "เน€เธยเน€เธยเน€เธย",
        "เน€เธยเน€เธยเน€เธยเน€เธยเน€เธย",
        "เน€เธยเน€เธเธเน€เธเธ‘เน€เธย",
        "เน€เธโฌเน€เธยเน€เธเธ…เน€เธเธ•เน€เธยเน€เธเธเน€เธย",
        "เน€เธโฌเน€เธยเน€เธเธ”เน€เธยเน€เธเธ",
        "เน€เธเธ…เน€เธย",
        "เน€เธโ€”เน€เธเธ“เน€เธยเน€เธเธเน€เธย",
        "เน€เธโฌเน€เธยเน€เธเธ•เน€เธเธเน€เธย",
        "เน€เธเธเน€เธเธเน€เธยเน€เธเธ’เน€เธย",
        "เน€เธยเน€เธเธเน€เธย",
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
        "เน€เธโ€ขเน€เธยเน€เธเธ",
        "เน€เธโ€”เน€เธเธ“เน€เธโ€ขเน€เธยเน€เธเธ",
        "เน€เธโ€ขเน€เธยเน€เธเธเน€เธโฌเน€เธเธ…เน€เธเธ",
        "เน€เธยเน€เธยเน€เธยเน€เธโ€ขเน€เธยเน€เธเธ",
        "เน€เธยเน€เธเธ‘เน€เธโ€",
        "เน€เธโฌเน€เธเธเน€เธเธ’เน€เธโฌเน€เธเธ…เน€เธเธ",
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
        "เน€เธยเน€เธเธเน€เธยเน€เธโ€ขเน€เธยเน€เธเธเน€เธยเน€เธยเน€เธยเน€เธย",
        "เน€เธเธเน€เธเธเน€เธยเน€เธเธ’เน€เธยเน€เธยเน€เธย",
        "เน€เธเธเน€เธยเน€เธเธ’เน€เธเธเน€เธยเน€เธยเน€เธย"
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

type QwenDraftMode = "code_draft" | "find_replace" | "unified_diff" | "line_range_replace" | "replacement_snippet" | "snippet" | "notes" | "insufficient_context" | "empty";

interface ParsedQwenPatch {
    ok: boolean;
    filePath?: string;
    find?: string;
    replace?: string;
    mode: "code_draft" | "find_replace" | "unified_diff" | "line_range_replace" | "invalid";
    reason?: string;
    startLine?: number;
    endLine?: number;
}

interface QwenCodeDraft {
    target_file: string;
    target_symbol: string;
    anchor_id?: string;
    old_anchor: string;
    change_summary: string;
    new_code: string;
}

interface AnchorCandidate {
    id: string;
    text: string;
    score: number;
}

interface ValidatedQwenCodeDraft {
    ok: boolean;
    draft?: QwenCodeDraft;
    reason: string;
    anchorOccurrences?: number;
}

const MAX_QWEN_DRAFT_CODE_CHARS = 5000;
const DEBUG_QWEN_PREVIEW = false;

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
    try {
        const parsed = extractJsonObject(t);
        if (parsed && typeof parsed === "object" && "new_code" in parsed) return "code_draft";
    } catch {}
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

function containsUnifiedDiffMarkers(text: string): boolean {
    const t = String(text || "");
    return (t.includes("---") && t.includes("+++")) || t.includes("@@");
}

function containsMarkdownFence(text: string): boolean {
    return String(text || "").includes("```");
}

function containsToolUseMarkers(text: string): boolean {
    const t = String(text || "").toLowerCase();
    return t.includes("tool_use") || t.includes("<tool_use") || t.includes("\"name\":\"edit\"") || t.includes("\"name\": \"edit\"");
}

function buildAnchorCandidates(focusedContext: string, rawFileContent: string, maxCandidates = 20): AnchorCandidate[] {
    const keywordPattern = /\b(import|export|function|const|let|class|return|if|await|async|router|route|endpoint|handle|validate)\b|app\./i;
    const strongPattern = /\b(function|export|class|const)\b/i;
    const weakLines = new Set(["}", ");", "};", "{", "else {"]);
    const normalizedRaw = normalizeNewlines(rawFileContent);

    const scored = normalizeNewlines(focusedContext)
        .split("\n")
        .map((line, index) => {
            const trimmed = line.trim();
            if (!trimmed || trimmed.length < 8 || weakLines.has(trimmed)) return undefined;

            let score = 0;
            if (keywordPattern.test(line)) score += 20;
            if (strongPattern.test(line)) score += 20;
            if (line.length >= 20 && line.length <= 160) score += 15;
            if (line.length > 220) score -= 20;

            const occurrences = countOccurrences(normalizedRaw, line);
            if (occurrences === 1) score += 35;
            if (occurrences > 1) score -= 20 + occurrences;

            const punctuationChars = (line.match(/[{}()[\];,.:<>+\-*/='"`|&!?\s]/g) || []).length;
            if (punctuationChars / Math.max(1, line.length) > 0.75) score -= 20;

            return { id: "", text: line, score, index };
        })
        .filter((candidate): candidate is AnchorCandidate & { index: number } => !!candidate)
        .sort((a, b) => b.score - a.score || a.index - b.index)
        .slice(0, maxCandidates);

    return scored.map((candidate, index) => ({
        id: `A${String(index + 1).padStart(2, "0")}`,
        text: candidate.text,
        score: candidate.score
    }));
}

function formatAnchorCandidates(candidates: AnchorCandidate[]): string {
    return candidates
        .map(candidate => `${candidate.id}: ${candidate.text.replace(/\r?\n/g, " ")}`)
        .join("\n");
}

function resolveAnchorFromCandidateId(anchorId: string | undefined, candidates: AnchorCandidate[]): string {
    if (!anchorId) return "";
    const found = candidates.find(candidate => candidate.id === anchorId);
    return found?.text || "";
}

function parseQwenCodeDraft(text: string): QwenCodeDraft | undefined {
    const parsed = extractJsonObject(text);
    const draft = {
        target_file: String(parsed?.target_file || ""),
        target_symbol: String(parsed?.target_symbol || ""),
        anchor_id: parsed?.anchor_id ? String(parsed.anchor_id) : undefined,
        old_anchor: String(parsed?.old_anchor || ""),
        change_summary: String(parsed?.change_summary || ""),
        new_code: String(parsed?.new_code || "")
    };
    return draft;
}

function validateQwenCodeDraft(
    text: string,
    targetFile: string,
    rawFileContent: string,
    focusedContext: string,
    anchorCandidates: AnchorCandidate[] = []
): ValidatedQwenCodeDraft {
    let draft: QwenCodeDraft | undefined;
    try {
        draft = parseQwenCodeDraft(text);
    } catch {
        return { ok: false, reason: "qwen_json_parse_failed" };
    }

    if (!draft) {
        return { ok: false, reason: "qwen_json_parse_failed" };
    }

    if (normalizePatchPath(draft.target_file) !== normalizePatchPath(targetFile)) {
        return { ok: false, draft, reason: "qwen_file_mismatch" };
    }

    if (!draft.new_code.trim()) {
        return { ok: false, draft, reason: "qwen_empty_new_code" };
    }
    if (draft.new_code.length > MAX_QWEN_DRAFT_CODE_CHARS) {
        return { ok: false, draft, reason: "qwen_large_draft" };
    }

    const combinedOutput = `${draft.new_code}\n${draft.old_anchor}\n${draft.anchor_id || ""}`;
    if (containsMarkdownFence(combinedOutput)) {
        return { ok: false, draft, reason: "qwen_markdown_violation" };
    }
    if (containsUnifiedDiffMarkers(combinedOutput) || /\n(?:FIND|REPLACE|START_LINE|END_LINE):/i.test(combinedOutput)) {
        return { ok: false, draft, reason: "qwen_format_violation_diff" };
    }
    if (containsToolUseMarkers(combinedOutput)) {
        return { ok: false, draft, reason: "qwen_tool_use_violation" };
    }

    let anchor = "";
    if (draft.anchor_id) {
        anchor = resolveAnchorFromCandidateId(draft.anchor_id, anchorCandidates);
        if (!anchor) {
            return { ok: false, draft, reason: "qwen_anchor_id_invalid", anchorOccurrences: 0 };
        }
    } else {
        anchor = draft.old_anchor.trim();
    }
    if (!anchor) {
        return { ok: false, draft, reason: "qwen_anchor_not_found" };
    }

    const normalizedRaw = normalizeNewlines(rawFileContent);
    const normalizedFocused = normalizeNewlines(focusedContext);
    const normalizedAnchor = normalizeNewlines(anchor);
    const rawOccurrences = countOccurrences(normalizedRaw, normalizedAnchor);
    const focusedOccurrences = countOccurrences(normalizedFocused, normalizedAnchor);

    if (rawOccurrences === 0 && focusedOccurrences === 0) {
        return { ok: false, draft, reason: "qwen_anchor_not_found", anchorOccurrences: 0 };
    }
    if (rawOccurrences > 1) {
        return { ok: false, draft, reason: "qwen_anchor_ambiguous", anchorOccurrences: rawOccurrences };
    }

    return {
        ok: true,
        draft: {
            ...draft,
            old_anchor: anchor,
            new_code: trimOuterBlankLines(normalizeNewlines(draft.new_code))
        },
        reason: "qwen_code_draft_valid",
        anchorOccurrences: rawOccurrences || focusedOccurrences
    };
}

function isUsableQwenDraft(mode: QwenDraftMode, chars: number): boolean {
    return mode === "code_draft" ||
        mode === "unified_diff" ||
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

function looksLikeErrorPayload(text: string): boolean {
    const trimmed = text.trim().toLowerCase();
    if (!trimmed) return true;
    return /^error\b/.test(trimmed) ||
        trimmed.includes("enoent") ||
        trimmed.includes("no such file") ||
        trimmed.includes("permission denied") ||
        trimmed.includes("access is denied") ||
        trimmed.includes("cannot find") ||
        trimmed.includes("not found");
}

function hasTruncationMarker(text: string): boolean {
    const lowered = text.toLowerCase();
    return lowered.includes("truncated") ||
        lowered.includes("omitted") ||
        lowered.includes("output clipped") ||
        lowered.includes("content clipped") ||
        lowered.includes("remaining lines") ||
        lowered.includes("more lines") ||
        lowered.includes("...") && /(?:lines?|chars?|tokens?)\s+(?:omitted|truncated|remaining)/i.test(text);
}

function isRangeOrSearchToolResult(toolName: string, input: any, text: string): boolean {
    const lowerTool = toolName.toLowerCase();
    const loweredText = text.toLowerCase();
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
    return loweredText.includes("matches") && loweredText.includes("line") && /\d+[:|]/.test(text);
}

function isFullFileReadToolResult(toolName: string, input: any, text: string): boolean {
    const lowerTool = toolName.toLowerCase();
    const exactToolPattern = /(^|[_\-\s.])(read|view|get|open|cat|show)([_\-\s.]|$)/i;
    if (!exactToolPattern.test(lowerTool) && !/file.*(content|text)/i.test(lowerTool)) {
        return false;
    }
    if (looksLikeErrorPayload(text) || hasTruncationMarker(text) || isRangeOrSearchToolResult(toolName, input, text)) {
        return false;
    }
    return text.length > 0;
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
                        const input = toolUseBlock?.input || {};
                        const inputPath = input.AbsolutePath || input.file_path || input.path || "";
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
                    const toolUseMsg = toolUseId
                        ? messages.find(m =>
                            Array.isArray(m.content) &&
                            m.content.some((b: any) => b?.type === "tool_use" && b.id === toolUseId)
                        )
                        : undefined;
                    const toolUseBlock = toolUseMsg?.content?.find((b: any) => b?.type === "tool_use" && b.id === toolUseId);
                    const input = toolUseBlock?.input || {};
                    const isExact = isFullFileReadToolResult(toolName, input, rawText);

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
    const explicitlyLargeRewrite = /rewrite|large|full|entire|เน€เธโ€”เน€เธเธ‘เน€เธยเน€เธยเน€เธยเน€เธยเน€เธเธ…เน€เธย|เน€เธโฌเน€เธยเน€เธเธ•เน€เธเธเน€เธยเน€เธยเน€เธเธเน€เธเธเน€เธยเน€เธโ€”เน€เธเธ‘เน€เธยเน€เธยเน€เธเธเน€เธเธเน€เธโ€/i.test(userIntent);
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

function extractFocusedCodeContext(rawFileContent: string, userIntent = "", maxLines = 260): { context: string; lines: number; chars: number; targetSymbol: string } {
    const normalized = normalizeNewlines(rawFileContent);
    const lines = normalized.split("\n");
    if (lines.length === 0) {
        return { context: "", lines: 0, chars: 0, targetSymbol: "unknown" };
    }

    const symbolMatches = Array.from(userIntent.matchAll(/(?:function|component|class|const|let|var|selector|block)\s+([A-Za-z_$][\w$-]*)/gi));
    const quotedMatches = Array.from(userIntent.matchAll(/[`"']([A-Za-z_$][\w$-]{2,})[`"']/g));
    const candidateSymbols = [...symbolMatches.map(match => match[1]), ...quotedMatches.map(match => match[1])];

    let centerIndex = -1;
    let targetSymbol = candidateSymbols[0] || "unknown";
    for (const symbol of candidateSymbols) {
        const found = lines.findIndex(line => line.includes(symbol));
        if (found >= 0) {
            centerIndex = found;
            targetSymbol = symbol;
            break;
        }
    }

    if (centerIndex === -1) {
        const intentWords = userIntent
            .split(/[^A-Za-z0-9_$-]+/)
            .filter(word => word.length >= 4 && !["implement", "change", "update", "refactor", "function", "component"].includes(word.toLowerCase()))
            .slice(0, 12);
        for (const word of intentWords) {
            const found = lines.findIndex(line => line.toLowerCase().includes(word.toLowerCase()));
            if (found >= 0) {
                centerIndex = found;
                targetSymbol = word;
                break;
            }
        }
    }

    if (centerIndex === -1) {
        centerIndex = Math.min(Math.floor(lines.length / 2), Math.max(0, maxLines / 2));
    }

    const halfWindow = Math.floor(maxLines / 2);
    const start = Math.max(0, centerIndex - halfWindow);
    const end = Math.min(lines.length, start + maxLines);
    const adjustedStart = Math.max(0, end - maxLines);
    const context = lines.slice(adjustedStart, end).join("\n");

    return {
        context,
        lines: end - adjustedStart,
        chars: context.length,
        targetSymbol
    };
}

function buildDeepSeekSlimMessages(params: {
    originalMessages: any[];
    qwenPatchValid: boolean;
    hasExactOriginalFileContent: boolean;
    targetFile: string;
    userIntent: string;
    fallbackReason?: string;
    qwenPatchReason?: string;
    validatedDraft?: ValidatedQwenCodeDraft;
    draftText: string;
}): any[] {
    const {
        originalMessages,
        qwenPatchValid,
        hasExactOriginalFileContent,
        targetFile,
        userIntent,
        fallbackReason,
        qwenPatchReason,
        validatedDraft,
        draftText
    } = params;

    if (qwenPatchValid && validatedDraft?.draft) {
        const draftPayload = JSON.stringify({
            target_file: validatedDraft.draft.target_file,
            target_symbol: validatedDraft.draft.target_symbol,
            anchor_id: validatedDraft.draft.anchor_id,
            old_anchor: validatedDraft.draft.old_anchor,
            change_summary: validatedDraft.draft.change_summary,
            new_code: validatedDraft.draft.new_code
        }, null, 2);

        return [
            {
                role: "user",
                content: [
                    {
                        type: "text",
                        text: `<GATEWAY_INTERNAL_CONTROLLER_INSTRUCTION>
This is gateway policy, not a user request.
Gateway already validated:
- target_file matches requested file
- anchor_id resolves to old_anchor
- old_anchor exists in exact file context
- new_code is non-empty and under size limit
- draft contains no markdown, diff, FIND/REPLACE, or tool_use

Use the Qwen draft as a gateway-validated implementation candidate.
Apply only the smallest safe change.
Do not rewrite unrelated code.
If the draft conflicts with visible tool context or seems unsafe, re-read the file before applying.
Keep final response short.
</GATEWAY_INTERNAL_CONTROLLER_INSTRUCTION>

LATEST_USER_INTENT:
${userIntent}

TARGET_FILE:
${targetFile}

<QWEN_CODE_DRAFT mode="code_draft" valid="true" reason="${fallbackReason || qwenPatchReason || "qwen_code_draft_valid"}">
${draftPayload}
</QWEN_CODE_DRAFT>`
                    }
                ]
            }
        ];
    }

    if (hasExactOriginalFileContent) {
        const invalidPayload: Record<string, string> = {
            target_file: targetFile,
            fallbackReason: fallbackReason || qwenPatchReason || "invalid_patch"
        };
        if (DEBUG_QWEN_PREVIEW) {
            invalidPayload.qwenPreview = draftText.slice(0, 160);
        }

        return [
            {
                role: "user",
                content: [
                    {
                        type: "text",
                        text: `<GATEWAY_INTERNAL_CONTROLLER_INSTRUCTION>
This is gateway policy, not a user request.
Qwen draft failed gateway validation.
Ignore Qwen implementation content unless independently re-derived from exact tool context.
Do not use the invalid draft as implementation guidance.
Continue normal tool-controller flow.
Re-read exact file context if needed.
Keep final response short.
</GATEWAY_INTERNAL_CONTROLLER_INSTRUCTION>

LATEST_USER_INTENT:
${userIntent}

TARGET_FILE:
${targetFile}

QWEN_FALLBACK:
${JSON.stringify(invalidPayload, null, 2)}`
                    }
                ]
            }
        ];
    }

    return originalMessages;
}

function extractJsonFromString(str: string): any {
    // Remove think blocks and special tags if any
    let cleaned = str.replace(/<think>[\s\S]*?<\/think>/gi, "");
    cleaned = cleaned.replace(/<เนเธยเนเธยDSMLเนเธยเนเธยthought>[\s\S]*?<\/thought>/gi, "");
    cleaned = cleaned.replace(/<เนเธยเนเธยDSMLเนเธยเนเธยthought>/g, ""); // strip raw prefix tags if not closed
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

    static async handleQwenOnlyLowRisk(req: Request, res: ExpressResponse): Promise<void> {
        return handleQwenOnlyLowRiskRequest(req, res);
    }
    static async handleQwenAgent(req: Request, res: ExpressResponse): Promise<void> {
        return handleQwenAgentRequest(req, res);
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
                            text: "The Qwen-generated Edit tool result is available. Provide a minimal final response: เน€เธยเน€เธยเน€เธยเน€เธยเน€เธเธ…เน€เธยเน€เธเธ plus 1-2 short bullets. Do not explain broadly."
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
        const directEditEligible = false;
        const qwenDelegationMode = "code_writer_draft";
        const focused = hasExactOriginalFileContent
            ? extractFocusedCodeContext(rawFileContent, userIntent)
            : { context: "", lines: 0, chars: 0, targetSymbol: "unknown" };
        const anchorCandidates = hasExactOriginalFileContent
            ? buildAnchorCandidates(focused.context, rawFileContent)
            : [];
        const anchorCandidateText = formatAnchorCandidates(anchorCandidates);

        const qwenSystemPrompt = `You are a code writer only.
Write the smallest code block that satisfies the requested change.
Use existing project style.
Prefer existing functions, native APIs, standard library, and installed dependencies.
Do not add abstractions unless required.
Do not choose line numbers.
Do not output patches.
Do not output unified diff.
Do not output FIND/REPLACE.
Do not output markdown.
Do not output explanations.
Return JSON only with target_file, target_symbol, anchor_id, change_summary, new_code.`;

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
        let parsedPatch: ParsedQwenPatch = { ok: false, mode: "code_draft", reason: "not_parsed" };
        let validatedDraft: ValidatedQwenCodeDraft = { ok: false, reason: hasExactOriginalFileContent ? "not_parsed" : "skipped_no_exact_context" };
        let qwenInputTokens = 0;
        let qwenOutputTokens = 0;

        const qwenStartTime = Date.now();
        if (!hasExactOriginalFileContent) {
            fallbackReason = "skipped_no_exact_context";
            qwenPatchReason = fallbackReason;
            qwenDraftMode = "empty";
            qwenDraftWeak = true;
        } else {
        try {
            const callQwen = async (retryInstruction?: string) => {
                const qwenBody = {
                    system: retryInstruction ? `${qwenSystemPrompt}\n\n${retryInstruction}` : qwenSystemPrompt,
                    messages: [
                        {
                            role: "user",
                            content: `TARGET_FILE: ${targetFile}

FOCUSED_CODE_CONTEXT:
${focused.context}

ANCHOR_CANDIDATES:
${anchorCandidateText || "NONE"}

Task: Write a replacement/new code block for this requested change.

Latest user intent preview: ${decision.userIntentPreview}

Return exactly:
{"target_file":"${targetFile}","target_symbol":"${focused.targetSymbol}","anchor_id":"A01","change_summary":"short summary","new_code":"replacement code only"}

Rules:
- Return JSON only.
- Do not output old_anchor.
- Do not copy anchor text manually.
- anchor_id MUST be one of ANCHOR_CANDIDATES.
- Do not invent anchor_id.
- If ANCHOR_CANDIDATES is NONE, use anchor_id "".
- If no candidate is relevant, choose the nearest safe candidate from ANCHOR_CANDIDATES.
- Do not output markdown, diff, FIND/REPLACE, line numbers, or tool_use.`
                        }
                    ],
                    stream: false,
                    max_tokens: Math.min(qwenMaxTokens, 1000),
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

                validatedDraft = validateQwenCodeDraft(draftText, targetFile, rawFileContent, focused.context, anchorCandidates);

                const retryableValidationReasons = new Set([
                    "qwen_json_parse_failed",
                    "qwen_anchor_not_found",
                    "qwen_anchor_ambiguous",
                    "qwen_anchor_id_invalid",
                    "qwen_file_mismatch"
                ]);

                if (!validatedDraft.ok && retryableValidationReasons.has(validatedDraft.reason)) {
                    qwenRetryUsed = true;
                    qwenResult = await callQwen(`Your previous answer failed gateway validation with reason: ${validatedDraft.reason}.
ANCHOR_CANDIDATES:
${anchorCandidateText || "NONE"}

Return one JSON object only.
Use this exact shape:
{"target_file":"${targetFile}","target_symbol":"${focused.targetSymbol}","anchor_id":"A01","change_summary":"short summary","new_code":"replacement code only"}
anchor_id MUST be copied from ANCHOR_CANDIDATES exactly.
Do not output old_anchor.
Do not output markdown.
Do not output diff.
Do not output FIND/REPLACE.
Do not output tool_use.`);
                    qwenLatencyMs = Date.now() - qwenStartTime;
                    if (qwenResult.ok) {
                        draftText = qwenResult.text;
                        qwenDraftMode = detectQwenDraftMode(draftText);
                        qwenDraftChars = draftText.length;
                        qwenInputTokens += qwenResult.inputTokens;
                        qwenOutputTokens += qwenResult.outputTokens;
                        validatedDraft = validateQwenCodeDraft(draftText, targetFile, rawFileContent, focused.context, anchorCandidates);
                    } else {
                        qwenErrorType = `retry_http_status_${qwenResult.status}`;
                    }
                }

                qwenDraftMode = "code_draft";
                qwenDraftUsed = validatedDraft.ok;
                qwenDraftWeak = !qwenDraftUsed;
                qwenPatchValid = validatedDraft.ok;
                qwenPatchReason = validatedDraft.reason;
                fallbackReason = validatedDraft.reason;

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
        }

        if (!qwenPatchReason) {
            qwenPatchReason = fallbackReason || qwenErrorType || "qwen_output_empty";
        }

        const qwenOnlyLowRiskRequested = req.body.model === "qwen-only-low-risk";
        const qwenOnlyLowRiskEnabled = config.qwenOnlyLowRiskEnabled && qwenOnlyLowRiskRequested;
        const multipleFilesInvolved = (messages.filter((msg: any) => Array.isArray(msg.content) && msg.content.some((block: any) => block?.type === "tool_result")).length > 1);
        const sensitiveKeywordsPresent = /auth|security|payment|database|migration/i.test(`${userIntent}\n${reducedContext}\n${targetFile}`);
        const patchLarge = qwenDraftChars >= 2000;
        const actualBuildCheckStatus = !qwenPatchValid
            ? "failed"
            : (multipleFilesInvolved || sensitiveKeywordsPresent || patchLarge)
                ? "failed"
                : "passed";
        const confidence = evaluateConfidence({
            hasExactOriginalFileContent,
            fileContextSource,
            qwenDraftMode,
            qwenPatchValid,
            qwenDraftWeak,
            qwenRetryUsed,
            fallbackReason,
            directEditEligible,
            anchorCandidateCount: anchorCandidates.length,
            qwenDraftChars,
            qwenInputTokens,
            qwenOutputTokens,
            multipleFilesInvolved,
            sensitiveKeywordsPresent,
            patchLarge,
            buildCheckStatus: actualBuildCheckStatus
        });
        const buildCheckStatus = qwenOnlyLowRiskEnabled ? actualBuildCheckStatus : "not_run";
        const qwenOnlyRejectedReason = !qwenOnlyLowRiskEnabled
            ? "feature_disabled"
            : !qwenPatchValid
                ? "qwen_output_failed_validator"
                : actualBuildCheckStatus === "failed"
                    ? "build_test_verification_failed"
                    : !confidence.canSkipDeepSeekDryRun
                        ? confidence.reasons[0] || "confidence_not_low"
                        : "";
        const qwenOnlyUsed = qwenOnlyLowRiskEnabled && confidence.canSkipDeepSeekDryRun && qwenPatchValid && actualBuildCheckStatus === "passed";
        const finalProvider = qwenOnlyUsed ? "qwen-local" : "deepseek";

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
            qwenPatchMode: "code_draft",
            qwenPatchValid,
            deepseekApprovalApproved: deepseekApprovalUsed ? deepseekApprovalApproved : undefined,
            emittedToolUse: undefined,
            fallbackReason,
            fileContextSource,
            qwenDelegationMode,
            directEditEligible,
            qwenAnchorId: validatedDraft.draft?.anchor_id,
            qwenAnchorCandidateCount: anchorCandidates.length
        });

        const originalMaxTokens = typeof req.body.max_tokens === "number" ? req.body.max_tokens : undefined;
        let deepseekMaxTokenCap = 1400;
        let deepseekContextMode = "full_no_exact_context";
        let deepseekInputReductionMode = "kept_original_messages";
        if (qwenPatchValid) {
            deepseekMaxTokenCap = 700;
            deepseekContextMode = "slim_valid_qwen_draft";
            deepseekInputReductionMode = "removed_original_messages";
        } else if (hasExactOriginalFileContent) {
            deepseekMaxTokenCap = 900;
            deepseekContextMode = "slim_invalid_qwen_draft";
            deepseekInputReductionMode = "compact_recovery";
        }

        let finalBody = req.body;
        if (draftText || fallbackReason) {
            const slimMessages = buildDeepSeekSlimMessages({
                originalMessages: req.body.messages,
                qwenPatchValid,
                hasExactOriginalFileContent,
                targetFile,
                userIntent,
                fallbackReason,
                qwenPatchReason,
                validatedDraft,
                draftText
            });
            finalBody = {
                ...req.body,
                messages: slimMessages,
                max_tokens: originalMaxTokens === undefined
                    ? deepseekMaxTokenCap
                    : Math.min(originalMaxTokens, deepseekMaxTokenCap)
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
            qwenRole: "code_writer",
            qwenPatchMode: "code_draft",
            qwenPatchValid,
            qwenPatchReason,
            qwenAnchorValid: validatedDraft.ok,
            qwenDraftValid: qwenPatchValid,
            qwenValidationReason: fallbackReason || qwenPatchReason,
            qwenAnchorCandidateCount: anchorCandidates.length,
            qwenAnchorId: validatedDraft.draft?.anchor_id,
            deepseekApprovalUsed,
            deepseekApprovalApproved,
            reducedContextChars,
            focusedContextChars: focused.chars,
            focusedContextLines: focused.lines,
            qwenLatencyMs,
            reason: decision.reason,
            finalProvider,
            qwen_only_used: qwenOnlyUsed,
            qwen_only_rejected_reason: qwenOnlyUsed ? "" : qwenOnlyRejectedReason,
            confidence_risk_level: confidence.riskLevel,
            build_check_status: buildCheckStatus,
            emittedToolUse,
            fallbackReason,
            qwenSkippedReason: !hasExactOriginalFileContent ? fallbackReason : undefined,
            deepseekApplyMode: qwenOnlyUsed ? "qwen_only_final" : "verify_and_apply",
            deepseekContextMode,
            deepseekMaxTokenCap,
            deepseekOriginalMessageCount: req.body.messages.length,
            deepseekFinalMessageCount: finalBody.messages.length,
            deepseekInputReductionMode,
            fileContextSource,
            hasExactOriginalFileContent,
            directEditEligible,
            qwenDelegationMode
        }));

        if (qwenOnlyUsed) {
            const qwenOnlyResponse = {
                id: "msg_qwen_only_" + Math.random().toString(36).substring(7),
                type: "message",
                role: "assistant",
                content: draftText ? [{ type: "text", text: draftText }] : [{ type: "text", text: "" }],
                model: activeModelName,
                stop_reason: "end_turn",
                stop_sequence: null,
                usage: {
                    input_tokens: qwenInputTokens,
                    output_tokens: qwenOutputTokens
                }
            };
            const totalLatencyMs = Date.now() - qwenStartTime;
            await updateGatewayRequest(requestId, 200, totalLatencyMs);
            if (isStream) {
                res.setHeader("content-type", "text/event-stream");
                res.setHeader("cache-control", "no-cache");
                res.setHeader("x-accel-buffering", "no");
                res.setHeader("connection", "keep-alive");
                res.write(`data: ${JSON.stringify(qwenOnlyResponse)}\n\n`);
                res.end();
            } else {
                res.json(qwenOnlyResponse);
            }
            return;
        }

        const qwenSavings = qwenDraftUsed ? { inputTokens: qwenInputTokens, outputTokens: qwenOutputTokens } : undefined;
        await this.forwardToDeepSeek(finalBody, clientHeaders, res, isStream, requestId, qwenSavings);
    }
}






