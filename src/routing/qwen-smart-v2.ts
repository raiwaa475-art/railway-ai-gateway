import { Request, Response } from "express";
import crypto from "crypto";
import { config } from "../config/env.js";
import { providerRegistry } from "./registry.js";
import { QwenLocalProvider } from "../providers/qwen-local.js";
import { DeepSeekProvider } from "../providers/deepseek.js";
import { pool, insertModelCall } from "../utils/db.js";
import { handleQwenAgentRequest } from "./qwen-agent.js";
import { calculateDeepSeekCost, extractDeepSeekUsage } from "../utils/pricing.js";
import { normalizeRepoKey, buildMemoryPromptContext, insertTaskMemory, recordFailurePattern, classifyFailure } from "./memory.js";

// Helper to check for code or patch patterns in text
function containsCodeOrPatch(text: string): boolean {
    if (!text) return false;
    const lower = text.toLowerCase();
    return (
        text.includes("```") ||
        text.includes("@@ -") ||
        text.includes("<<<<<<<") ||
        text.includes("=======") ||
        text.includes(">>>>>>>") ||
        lower.includes("find/replace") ||
        lower.includes("edit_file") ||
        lower.includes("write_file")
    );
}

// Clean and extract JSON object from text
function parseStructuredJson(text: string): any {
    if (!text) return null;
    let cleaned = text.trim();
    if (cleaned.startsWith("```")) {
        const lines = cleaned.split("\n");
        if (lines[0].startsWith("```")) lines.shift();
        if (lines.length > 0 && lines[lines.length - 1].startsWith("```")) lines.pop();
        cleaned = lines.join("\n").trim();
    }
    
    // Try matching first '{' and last '}'
    const start = cleaned.indexOf("{");
    const end = cleaned.lastIndexOf("}");
    if (start !== -1 && end !== -1 && end > start) {
        cleaned = cleaned.substring(start, end + 1);
    }

    try {
        return JSON.parse(cleaned);
    } catch {
        return null;
    }
}

export async function handleQwenSmartV2Request(req: Request, res: Response): Promise<void> {
    const requestId = (req as any).requestId || crypto.randomUUID();
    const startTime = Date.now();
    const clientHeaders = {
        "user-agent": req.header("user-agent") || "railway-ai-gateway"
    };

    const messages = req.body.messages || [];
    
    const rawRepoKey = (req.headers["x-repo-key"] as string) || 
                       (req.headers["x-repo-path"] as string) || 
                       req.body.repo_key || 
                       req.body.repo_path || 
                       (req.query.repoKey as string) || 
                       (req.query.repo_key as string) || 
                       "";
    const repoKey = rawRepoKey ? normalizeRepoKey(rawRepoKey) : "";
    
    // Count previous tool rounds
    let toolRoundCount = 0;
    for (const msg of messages) {
        if (msg.role === "assistant" && Array.isArray(msg.content)) {
            if (msg.content.some((b: any) => b?.type === "tool_use")) {
                toolRoundCount++;
            }
        }
    }

    // Count review rounds by looking for controller feedback messages in user role
    const reviewRoundCount = messages.filter((m: any) => 
        m.role === "user" && 
        typeof m.content === "string" && 
        m.content.includes("The smart controller reviewed your edits")
    ).length;

    // Check if SMART_CONTROLLER_ENABLED is false. If so, fall back directly to qwen-agent
    if (config.qwenOnlyLowRiskEnabled || process.env.SMART_CONTROLLER_ENABLED === "false") {
        return handleQwenAgentRequest(req, res);
    }

    // Find providers
    const qwenProvider = providerRegistry.getProvider("qwen-local") as QwenLocalProvider | undefined;
    const deepseekProvider = providerRegistry.getProvider("deepseek") as DeepSeekProvider | undefined;

    if (!qwenProvider) {
        res.status(503).json({ error: { type: "server_error", message: "Qwen local provider is not registered" } });
        return;
    }
    if (!deepseekProvider) {
        res.status(503).json({ error: { type: "server_error", message: "DeepSeek provider is not registered" } });
        return;
    }

    // Trace logging metadata
    const traceData: any = {
        requestId,
        timestamp: new Date().toISOString(),
        mode: "qwen-smart-v2",
        userIntent: messages[0]?.content || "",
        sanitizedMessages: messages,
        success: false,
        controllerViolation: false,
        controllerModel: config.defaultModel,
        controllerRole: "planner",
        qwenWorkerUsed: true,
        deepseekWroteCode: false,
        claudeWroteCode: false,
        qwenEditedFiles: [],
        reviewResult: "pass",
        controllerPlan: null,
        controllerReview: null,
        qwenWorkerTraceIds: [requestId],
        finalResult: "",
        accepted: false
    };

    // 1. PLANNING STEP (First turn only)
    let planPromptSuffix = "";
    if (toolRoundCount === 0 && reviewRoundCount === 0) {
        traceData.controllerRole = "planner";
        const systemPrompt = `You are a planner for a coding task. Your goal is to design a high-level plan and steps for a coding worker.
You MUST output ONLY a valid JSON object matching the schema below.
DO NOT include any code blocks, unified diff patches, code replacements, or tool instructions in your output.
Forbidden keywords/structures:
- Markdown code blocks (\`\`\`)
- Unified diff hunks (@@)
- FIND/REPLACE blocks

JSON Schema:
{
  "role": "planner",
  "task_summary": "Summary of the user request",
  "target_files": ["array of paths to files that need editing"],
  "steps_for_qwen": ["detailed step 1", "detailed step 2"],
  "risk_notes": ["points of high risk to check"],
  "acceptance_checks": ["how to verify successful implementation"]
}`;

        let controllerText = "";
        let controllerViolation = false;
        let controllerPayload: any = null;

        let finalSystem = systemPrompt;
        if (repoKey) {
            const memPrompt = await buildMemoryPromptContext(repoKey);
            if (memPrompt) {
                finalSystem = memPrompt + "\n\n" + systemPrompt;
            }
        }

        try {
            // First attempt
            const dsRes = await deepseekProvider.handleRequest({
                model: config.defaultModel,
                system: finalSystem,
                messages: [{ role: "user", content: `Review the task and create a plan: ${JSON.stringify(messages)}` }],
                stream: false,
                temperature: 0.1,
                max_tokens: 1000
            }, clientHeaders);

            if (dsRes.ok) {
                const data = await dsRes.json();
                controllerText = data.content?.[0]?.text || "";
                
                // Track deepseek usage & pricing
                const usage = extractDeepSeekUsage(data);
                const cost = calculateDeepSeekCost(config.defaultModel, usage);
                await insertModelCall({
                    requestId,
                    provider: "deepseek",
                    model: `${config.defaultModel}-planner`,
                    inputTokens: usage.inputTokens,
                    outputTokens: usage.outputTokens,
                    cacheHitInputTokens: usage.cacheHitInputTokens,
                    cacheMissInputTokens: usage.cacheMissInputTokens,
                    latencyMs: 0,
                    ...cost
                });

                // Check for violations
                if (containsCodeOrPatch(controllerText)) {
                    controllerViolation = true;
                    traceData.controllerViolation = true;
                } else {
                    controllerPayload = parseStructuredJson(controllerText);
                }
            }

            // Retry if violation or parse failed
            if (controllerViolation || !controllerPayload) {
                const retryPrompt = `Your previous response was rejected because it violated formatting rules (either it contained markdown code blocks, patches, or was invalid JSON). 
Please rewrite the plan. Output ONLY a valid JSON object. No explanations. No code blocks.`;
                const dsRetryRes = await deepseekProvider.handleRequest({
                    model: config.defaultModel,
                    system: finalSystem,
                    messages: [
                        { role: "user", content: `Review the task and create a plan: ${JSON.stringify(messages)}` },
                        { role: "assistant", content: controllerText },
                        { role: "user", content: retryPrompt }
                    ],
                    stream: false,
                    temperature: 0.05,
                    max_tokens: 1000
                }, clientHeaders);

                if (dsRetryRes.ok) {
                    const retryData = await dsRetryRes.json();
                    controllerText = retryData.content?.[0]?.text || "";
                    controllerPayload = parseStructuredJson(controllerText);
                }
            }
        } catch (err) {
            console.error("DeepSeek planning step failed:", err);
        }

        if (controllerPayload) {
            traceData.controllerPlan = JSON.stringify(controllerPayload);
            planPromptSuffix = `\n\nSmart Controller Plan:\n- Summary: ${controllerPayload.task_summary}\n- Files: ${controllerPayload.target_files?.join(", ")}\n- Steps:\n${controllerPayload.steps_for_qwen?.map((s: string) => `  * ${s}`).join("\n")}`;
        }
    }

    // Call Qwen Local to run the execution
    const qwenBody = { ...req.body };
    if (repoKey) {
        const memPrompt = await buildMemoryPromptContext(repoKey);
        if (memPrompt) {
            qwenBody.system = memPrompt + "\n\n" + (qwenBody.system || "");
        }
    }
    if (planPromptSuffix) {
        qwenBody.system = (qwenBody.system ? qwenBody.system + "\n\n" : "") + planPromptSuffix;
    }

    try {
        const qwenRes = await qwenProvider.handleRequest(qwenBody, clientHeaders);
        if (!qwenRes.ok) {
            const errText = await qwenRes.text();
            res.status(qwenRes.status).json({ error: { type: "api_error", message: errText || "Qwen local failed" } });
            return;
        }

        const qwenData = await qwenRes.json();
        
        // INTERCEPT FINAL ANSWER FOR REVIEW
        const isFinalAnswer = !Array.isArray(qwenData.content) || !qwenData.content.some((b: any) => b?.type === "tool_use");

        if (isFinalAnswer && reviewRoundCount < (Number(process.env.SMART_CONTROLLER_MAX_REVIEW_ROUNDS) || 2)) {
            traceData.controllerRole = "reviewer";
            const reviewPrompt = `You are a code reviewer. Evaluate the worker's changes.
You MUST output ONLY a valid JSON object matching the schema below.
DO NOT include code blocks or unified patches.

JSON Schema:
{
  "role": "reviewer",
  "review_result": "pass" | "needs_fix",
  "instructions_for_qwen": ["instruction 1", "instruction 2"]
}`;

            let reviewText = "";
            let reviewPayload: any = null;

            let finalReviewSystem = reviewPrompt;
            if (repoKey) {
                const memPrompt = await buildMemoryPromptContext(repoKey);
                if (memPrompt) {
                    finalReviewSystem = memPrompt + "\n\n" + reviewPrompt;
                }
            }

            try {
                const dsReviewRes = await deepseekProvider.handleRequest({
                    model: config.defaultModel,
                    system: finalReviewSystem,
                    messages: [{ role: "user", content: `Original request and worker response history:\n${JSON.stringify(messages)}\n\nWorker final response:\n${JSON.stringify(qwenData)}` }],
                    stream: false,
                    temperature: 0.1,
                    max_tokens: 1000
                }, clientHeaders);

                if (dsReviewRes.ok) {
                    const rData = await dsReviewRes.json();
                    reviewText = rData.content?.[0]?.text || "";
                    
                    const usage = extractDeepSeekUsage(rData);
                    const cost = calculateDeepSeekCost(config.defaultModel, usage);
                    await insertModelCall({
                        requestId,
                        provider: "deepseek",
                        model: `${config.defaultModel}-reviewer`,
                        inputTokens: usage.inputTokens,
                        outputTokens: usage.outputTokens,
                        cacheHitInputTokens: usage.cacheHitInputTokens,
                        cacheMissInputTokens: usage.cacheMissInputTokens,
                        latencyMs: 0,
                        ...cost
                    });

                    reviewPayload = parseStructuredJson(reviewText);
                }
            } catch (err) {
                console.error("DeepSeek review step failed:", err);
            }

            if (reviewPayload) {
                traceData.controllerReview = JSON.stringify(reviewPayload);
                traceData.reviewResult = reviewPayload.review_result;

                if (reviewPayload.review_result === "needs_fix") {
                    const fixInstructions = reviewPayload.instructions_for_qwen?.join("\n- ") || "Please repair the implementation.";
                    const fixMessageText = `The smart controller reviewed your edits and found issues. Please fix them according to these instructions:\n- ${fixInstructions}\nMake necessary tool calls to repair the code.`;
                    
                    // Request Qwen again with the fix instructions appended
                    const fixMessages = [
                        ...messages,
                        {
                            role: "assistant",
                            content: qwenData.content
                        },
                        {
                            role: "user",
                            content: fixMessageText
                        }
                    ];

                    const qwenFixBody = {
                        ...qwenBody,
                        messages: fixMessages
                    };

                    const qwenFixRes = await qwenProvider.handleRequest(qwenFixBody, clientHeaders);
                    if (qwenFixRes.ok) {
                        const qwenFixData = await qwenFixRes.json();
                        traceData.success = true;
                        traceData.accepted = false;
                        
                        if (pool) {
                            await pool.query(
                                `INSERT INTO qwen_agent_traces (request_id, timestamp, mode, user_intent, success, controller_plan, controller_review, accepted, repo_key)
                                 VALUES ($1, CURRENT_TIMESTAMP, $2, $3, $4, $5, $6, $7, $8)
                                 ON CONFLICT (request_id) DO UPDATE SET success = EXCLUDED.success, controller_review = EXCLUDED.controller_review, repo_key = EXCLUDED.repo_key`,
                                [requestId, "qwen-smart-v2", messages[0]?.content || "", true, traceData.controllerPlan, traceData.controllerReview, false, repoKey || null]
                            );
                        }

                        if (repoKey) {
                            let costThb = 0;
                            try {
                                if (pool) {
                                    const costRes = await pool.query("SELECT COALESCE(SUM(cost_thb), 0) as total FROM model_calls WHERE request_id = $1", [requestId]);
                                    costThb = Number(costRes.rows[0]?.total || 0);
                                }
                            } catch {}

                            await insertTaskMemory({
                                repo_key: repoKey,
                                task_summary: messages[0]?.content || "Qwen Smart v2 task",
                                touched_files: traceData.qwenEditedFiles || [],
                                outcome: "success",
                                model_route: "qwen-smart-v2",
                                cost_thb: costThb
                            });
                        }

                        console.log(JSON.stringify({
                            time: new Date().toISOString(),
                            requestId,
                            mode: "qwen-smart-v2",
                            provider: "qwen-local",
                            finalProvider: "qwen-local",
                            deepseekFallbackUsed: false,
                            qwenWorkerUsed: true,
                            deepseekWroteCode: false,
                            claudeWroteCode: false,
                            controllerModel: traceData.controllerModel,
                            controllerRole: traceData.controllerRole,
                            controllerViolation: traceData.controllerViolation,
                            controllerWroteCodeViolation: traceData.controllerViolation,
                            qwenEditedFiles: traceData.qwenEditedFiles,
                            reviewResult: traceData.reviewResult,
                            success: traceData.success,
                            failureReason: null
                        }));

                        res.json(qwenFixData);
                        return;
                    }
                } else {
                    traceData.accepted = true;
                }
            }
        }

        traceData.success = true;
        if (pool) {
            await pool.query(
                `INSERT INTO qwen_agent_traces (request_id, timestamp, mode, user_intent, success, controller_plan, controller_review, accepted, repo_key)
                 VALUES ($1, CURRENT_TIMESTAMP, $2, $3, $4, $5, $6, $7, $8)
                 ON CONFLICT (request_id) DO UPDATE SET success = EXCLUDED.success, controller_plan = EXCLUDED.controller_plan, controller_review = EXCLUDED.controller_review, accepted = EXCLUDED.accepted, repo_key = EXCLUDED.repo_key`,
                [requestId, "qwen-smart-v2", messages[0]?.content || "", true, traceData.controllerPlan, traceData.controllerReview, traceData.accepted, repoKey || null]
            );
        }

        if (repoKey) {
            let costThb = 0;
            try {
                if (pool) {
                    const costRes = await pool.query("SELECT COALESCE(SUM(cost_thb), 0) as total FROM model_calls WHERE request_id = $1", [requestId]);
                    costThb = Number(costRes.rows[0]?.total || 0);
                }
            } catch {}

            await insertTaskMemory({
                repo_key: repoKey,
                task_summary: messages[0]?.content || "Qwen Smart v2 task",
                touched_files: traceData.qwenEditedFiles || [],
                outcome: "success",
                model_route: "qwen-smart-v2",
                cost_thb: costThb
            });
        }

        console.log(JSON.stringify({
            time: new Date().toISOString(),
            requestId,
            mode: "qwen-smart-v2",
            provider: "qwen-local",
            finalProvider: "qwen-local",
            deepseekFallbackUsed: false,
            qwenWorkerUsed: true,
            deepseekWroteCode: false,
            claudeWroteCode: false,
            controllerModel: traceData.controllerModel,
            controllerRole: traceData.controllerRole,
            controllerViolation: traceData.controllerViolation,
            controllerWroteCodeViolation: traceData.controllerViolation,
            qwenEditedFiles: traceData.qwenEditedFiles,
            reviewResult: traceData.reviewResult,
            success: traceData.success,
            failureReason: null
        }));

        res.json(qwenData);

    } catch (err: any) {
        if (repoKey) {
            let costThb = 0;
            try {
                if (pool) {
                    const costRes = await pool.query("SELECT COALESCE(SUM(cost_thb), 0) as total FROM model_calls WHERE request_id = $1", [requestId]);
                    costThb = Number(costRes.rows[0]?.total || 0);
                }
            } catch {}

            await insertTaskMemory({
                repo_key: repoKey,
                task_summary: messages[0]?.content || "Qwen Smart v2 task",
                touched_files: traceData.qwenEditedFiles || [],
                outcome: "failed",
                model_route: "qwen-smart-v2",
                cost_thb: costThb
            });

            const failReason = err.message || "Smart controller error";
            const pType = classifyFailure(failReason);
            await recordFailurePattern(repoKey, pType, failReason);
        }

        res.status(500).json({ error: { type: "server_error", message: err.message || "Smart controller error" } });
    }
}
