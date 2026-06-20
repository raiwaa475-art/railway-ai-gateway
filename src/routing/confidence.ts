export type ConfidenceRiskLevel = "low" | "medium" | "high";

export interface ConfidenceInput {
    hasExactOriginalFileContent?: boolean;
    fileContextSource?: string;
    qwenDraftMode?: string;
    qwenPatchValid?: boolean;
    qwenDraftWeak?: boolean;
    qwenRetryUsed?: boolean;
    fallbackReason?: string;
    directEditEligible?: boolean;
    anchorCandidateCount?: number;
    qwenDraftChars?: number;
    qwenInputTokens?: number;
    qwenOutputTokens?: number;
    multipleFilesInvolved?: boolean;
    sensitiveKeywordsPresent?: boolean;
    patchLarge?: boolean;
    buildCheckStatus?: string;
}

export function evaluateConfidence(input: ConfidenceInput): {
    riskLevel: ConfidenceRiskLevel;
    canSkipDeepSeekDryRun: boolean;
    reasons: string[];
} {
    const reasons = new Set<string>();
    let score = 0;

    const fileContextSource = String(input.fileContextSource || "none");
    const qwenDraftMode = String(input.qwenDraftMode || "empty");
    const qwenDraftChars = Number(input.qwenDraftChars || 0);
    const buildCheckStatus = String(input.buildCheckStatus || "not_run");

    if (!input.hasExactOriginalFileContent) {
        score += 2;
        reasons.add("no_exact_original_file_content");
    }

    if (fileContextSource === "none" || fileContextSource === "reduced_context") {
        score += 2;
        reasons.add(`file_context_${fileContextSource}`);
    } else if (fileContextSource === "tool_result_partial") {
        score += 1;
        reasons.add("partial_file_context");
    }

    if (!input.qwenPatchValid) {
        score += 3;
        reasons.add("draft_failed_gateway_validation");
    }

    if (input.qwenDraftWeak) {
        score += 2;
        reasons.add("weak_draft");
    }

    if (input.qwenRetryUsed) {
        score += 1;
        reasons.add("retry_used");
    }

    if (typeof input.anchorCandidateCount === "number" && input.anchorCandidateCount === 0) {
        score += 1;
        reasons.add("no_anchor_candidates");
    }

    if (input.multipleFilesInvolved) {
        score += 2;
        reasons.add("multiple_files_involved");
    }

    if (input.sensitiveKeywordsPresent) {
        score += 3;
        reasons.add("sensitive_keywords_present");
    }

    if (input.patchLarge) {
        score += 2;
        reasons.add("patch_large");
    }

    if (qwenDraftMode === "empty" || qwenDraftMode === "notes" || qwenDraftMode === "insufficient_context") {
        score += 2;
        reasons.add(`draft_mode_${qwenDraftMode}`);
    }

    if (qwenDraftMode === "snippet" && qwenDraftChars > 0 && qwenDraftChars < 200) {
        score += 1;
        reasons.add("short_snippet");
    }

    if (input.fallbackReason && input.fallbackReason !== "qwen_code_draft_valid") {
        score += 1;
        reasons.add(`fallback_${input.fallbackReason}`);
    }

    if (input.directEditEligible) {
        score -= 1;
        reasons.add("direct_edit_eligible");
    }

    if (buildCheckStatus === "failed") {
        score += 3;
        reasons.add("build_check_failed");
    } else if (buildCheckStatus === "warning") {
        score += 1;
        reasons.add("build_check_warning");
    }

    const riskLevel: ConfidenceRiskLevel = score <= 1 ? "low" : score <= 4 ? "medium" : "high";
    const canSkipDeepSeekDryRun =
        riskLevel === "low" &&
        !!input.qwenPatchValid &&
        !!input.hasExactOriginalFileContent &&
        !input.qwenDraftWeak &&
        !input.qwenRetryUsed &&
        !input.multipleFilesInvolved &&
        !input.sensitiveKeywordsPresent &&
        !input.patchLarge &&
        buildCheckStatus !== "failed" &&
        fileContextSource !== "none" &&
        fileContextSource !== "reduced_context";

    return {
        riskLevel,
        canSkipDeepSeekDryRun,
        reasons: Array.from(reasons)
    };
}
