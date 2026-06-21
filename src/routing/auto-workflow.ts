import { pool } from "../utils/db.js";
import { config } from "../config/env.js";
import { providerRegistry } from "./registry.js";
import { QwenLocalProvider } from "../providers/qwen-local.js";
import { DeepSeekProvider } from "../providers/deepseek.js";
import { normalizeRepoKey, buildMemoryPromptContext, insertTaskMemory, recordFailurePattern, classifyFailure } from "./memory.js";
import fs from "fs";
import path from "path";
import { execSync } from "child_process";
import crypto from "crypto";

export interface AutoJob {
    id: number;
    created_at: Date;
    updated_at: Date;
    status: string;
    user_task: string;
    mode: string;
    repo_path: string;
    branch_name: string | null;
    model_worker: string;
    controller_model: string | null;
    current_step: number;
    max_steps: number;
    success: boolean | null;
    failure_reason: string | null;
    summary: string | null;
}

// Active background jobs store for cancellation checking
const activeJobs = new Map<number, boolean>();

export class AutoJobManager {
    static async createJob(params: {
        user_task: string;
        repo_path: string;
        branch_name?: string;
        mode?: string;
        model_worker?: string;
        controller_model?: string;
    }): Promise<AutoJob> {
        if (!pool) {
            throw new Error("Database not connected");
        }

        const mode = params.mode || "smart";
        const model_worker = params.model_worker || "qwen-agent";
        const controller_model = params.controller_model || config.defaultModel;
        const branch_name = params.branch_name || null;
        const max_steps = Number(process.env.AUTO_CODING_MAX_STEPS || 12);

        const res = await pool.query(
            `INSERT INTO auto_coding_jobs 
             (status, user_task, mode, repo_path, branch_name, model_worker, controller_model, current_step, max_steps) 
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) RETURNING *`,
            ["queued", params.user_task, mode, params.repo_path, branch_name, model_worker, controller_model, 0, max_steps]
        );

        const job = res.rows[0];
        activeJobs.set(job.id, true);

        // Spawn job execution asynchronously in the background
        this.runJobInBackground(job.id).catch(err => {
            console.error(`Background job ${job.id} crashed:`, err);
        });

        return job;
    }

    static async cancelJob(jobId: number): Promise<boolean> {
        if (!pool) return false;
        activeJobs.set(jobId, false);
        const res = await pool.query(
            "UPDATE auto_coding_jobs SET status = 'cancelled', updated_at = CURRENT_TIMESTAMP WHERE id = $1 RETURNING *",
            [jobId]
        );
        await this.logEvent(jobId, 0, "job_cancelled", { reason: "User requested cancellation" });
        return res.rows.length > 0;
    }

    static async getJob(jobId: number): Promise<AutoJob | null> {
        if (!pool) return null;
        const res = await pool.query("SELECT * FROM auto_coding_jobs WHERE id = $1", [jobId]);
        return res.rows[0] || null;
    }

    static async listJobs(): Promise<AutoJob[]> {
        if (!pool) return [];
        const res = await pool.query("SELECT * FROM auto_coding_jobs ORDER BY created_at DESC");
        return res.rows;
    }

    private static async logEvent(jobId: number, step: number, eventType: string, payload: any) {
        if (!pool) return;
        try {
            await pool.query(
                "INSERT INTO auto_coding_job_events (job_id, step, event_type, payload) VALUES ($1, $2, $3, $4)",
                [jobId, step, eventType, JSON.stringify(payload)]
            );
        } catch (err) {
            console.error("Failed to log job event:", err);
        }
    }

    private static async updateJobStatus(jobId: number, update: Partial<AutoJob>) {
        if (!pool) return;
        const keys = Object.keys(update);
        if (keys.length === 0) return;

        const setClause = keys.map((k, i) => `${k} = $${i + 2}`).join(", ");
        const values = [jobId, ...Object.values(update)];

        await pool.query(
            `UPDATE auto_coding_jobs SET ${setClause}, updated_at = CURRENT_TIMESTAMP WHERE id = $1`,
            values
        );
    }

    private static async runJobInBackground(jobId: number) {
        const job = await this.getJob(jobId);
        if (!job) return;

        console.log(`Starting background job ${jobId}: ${job.user_task.slice(0, 50)}`);
        await this.logEvent(jobId, 0, "job_started", { task: job.user_task, repo_path: job.repo_path });

        const qwenProvider = providerRegistry.getProvider("qwen-local") as QwenLocalProvider | undefined;
        const deepseekProvider = providerRegistry.getProvider("deepseek") as DeepSeekProvider | undefined;

        if (!qwenProvider) {
            await this.failJob(jobId, "Qwen local provider not registered");
            return;
        }

        const repoKey = job.repo_path ? normalizeRepoKey(job.repo_path) : "";
        const editedFiles = new Set<string>();

        try {
            let messages: any[] = [{ role: "user", content: job.user_task }];
            let currentStep = 1;
            let finalSummary = "";
            let plan: any = null;

            // 1. Planning Step (if mode is smart)
            if (job.mode === "smart" && deepseekProvider) {
                await this.updateJobStatus(jobId, { status: "planning", current_step: currentStep });
                await this.logEvent(jobId, currentStep, "planning_started", {});

                const planPrompt = `You are a planner for a coding task. Create a simple JSON plan. No code blocks. No explanations.
JSON Schema:
{
  "task_summary": "Summary",
  "target_files": ["files"],
  "steps_for_qwen": ["steps"]
}`;
                let finalPlanPrompt = planPrompt;
                if (repoKey) {
                    const memPrompt = await buildMemoryPromptContext(repoKey);
                    if (memPrompt) {
                        finalPlanPrompt = memPrompt + "\n\n" + planPrompt;
                    }
                }

                const dsRes = await deepseekProvider.handleRequest({
                    model: job.controller_model || config.defaultModel,
                    system: finalPlanPrompt,
                    messages: [{ role: "user", content: `Create plan for: ${job.user_task}` }],
                    stream: false,
                    temperature: 0.1
                }, {});

                if (dsRes.ok) {
                    const data = await dsRes.json();
                    const text = data.content?.[0]?.text || "";
                    try {
                        plan = JSON.parse(text.trim().replace(/^```json|```$/g, ""));
                        await this.logEvent(jobId, currentStep, "planning_completed", plan);
                    } catch {
                        await this.logEvent(jobId, currentStep, "planning_failed", { rawOutput: text });
                    }
                }
                currentStep++;
            }

            // Append plan to system prompt if available
            let systemInstruction = "You are connected to local workspace tools. Use Read, Write, Edit, Grep, Glob, Bash to solve the user task.";
            if (repoKey) {
                const memPrompt = await buildMemoryPromptContext(repoKey);
                if (memPrompt) {
                    systemInstruction = memPrompt + "\n\n" + systemInstruction;
                }
            }
            if (plan) {
                systemInstruction += `\n\nPlan to follow:\n${JSON.stringify(plan, null, 2)}`;
            }

            // 2. Qwen Tool execution loop
            let round = 0;
            let buildStatus = "not_run";
            const maxRounds = 8;
            await this.updateJobStatus(jobId, { status: "qwen_working", current_step: currentStep });

            while (round < maxRounds) {
                if (activeJobs.get(jobId) === false) {
                    return; // Job was cancelled
                }

                await this.logEvent(jobId, currentStep, "qwen_request", { round });

                const response = await qwenProvider.handleRequest({
                    model: "qwen-local",
                    system: systemInstruction,
                    messages,
                    stream: false,
                    tools: [
                        { name: "Read", description: "Read file contents", input_schema: { type: "object", properties: { file_path: { type: "string" } }, required: ["file_path"] } },
                        { name: "Write", description: "Write new file contents", input_schema: { type: "object", properties: { file_path: { type: "string" }, content: { type: "string" } }, required: ["file_path", "content"] } },
                        { name: "Edit", description: "Replace content in file", input_schema: { type: "object", properties: { file_path: { type: "string" }, old_string: { type: "string" }, new_string: { type: "string" } }, required: ["file_path", "old_string", "new_string"] } },
                        { name: "Grep", description: "Search for text patterns", input_schema: { type: "object", properties: { pattern: { type: "string" } }, required: ["pattern"] } },
                        { name: "Glob", description: "Find file patterns", input_schema: { type: "object", properties: { pattern: { type: "string" } }, required: ["pattern"] } },
                        { name: "Bash", description: "Run shell command", input_schema: { type: "object", properties: { command: { type: "string" } }, required: ["command"] } }
                    ]
                }, {});

                if (!response.ok) {
                    throw new Error(`Qwen Local returned HTTP ${response.status}`);
                }

                const responseData = await response.json();
                const content = responseData.content || [];
                const toolCalls = content.filter((b: any) => b?.type === "tool_use");

                messages.push({ role: "assistant", content });

                if (toolCalls.length === 0) {
                    finalSummary = content.filter((b: any) => b?.type === "text").map((b: any) => b.text).join("\n");
                    await this.logEvent(jobId, currentStep, "qwen_text_response", { text: finalSummary });
                    break; // Completed loop
                }

                // Process tool calls
                const toolResults: any[] = [];
                for (const call of toolCalls) {
                    await this.logEvent(jobId, currentStep, "tool_execution_started", { toolName: call.name, input: call.input });
                    
                    let resultText = "";
                    let isError = false;

                    try {
                        const execResult = await this.executeLocalTool(job.repo_path, call.name, call.input);
                        resultText = execResult.output;
                        isError = execResult.isError;
                        
                        if (!isError && (call.name === "Write" || call.name === "Edit")) {
                            if (call.input.file_path) {
                                editedFiles.add(call.input.file_path);
                            }
                        }

                        if (call.name === "Bash" && (call.input.command.includes("test") || call.input.command.includes("build"))) {
                            buildStatus = isError ? "failed" : "passed";
                        }
                    } catch (toolErr: any) {
                        resultText = toolErr.message || "Unknown tool execution error";
                        isError = true;
                    }

                    await this.logEvent(jobId, currentStep, "tool_execution_completed", { toolName: call.name, outputPreview: resultText.slice(0, 200), isError });
                    toolResults.push({
                        type: "tool_result",
                        tool_use_id: call.id,
                        content: resultText,
                        is_error: isError
                    });
                }

                messages.push({ role: "user", content: toolResults });
                round++;
                currentStep++;
                await this.updateJobStatus(jobId, { current_step: currentStep });
            }

            // 3. Optional build verification
            if (buildStatus === "not_run") {
                await this.updateJobStatus(jobId, { status: "build_testing", current_step: currentStep });
                // Check if npm test is configured or exists
                if (fs.existsSync(path.join(job.repo_path, "package.json"))) {
                    try {
                        await this.logEvent(jobId, currentStep, "build_check_started", { command: "npm test" });
                        execSync("npm test", { cwd: job.repo_path, stdio: "pipe", timeout: 30000 });
                        buildStatus = "passed";
                        await this.logEvent(jobId, currentStep, "build_check_passed", {});
                    } catch {
                        buildStatus = "failed";
                        await this.logEvent(jobId, currentStep, "build_check_failed", {});
                    }
                }
                currentStep++;
            }

            // 4. Review Step (if mode is smart)
            if (job.mode === "smart" && deepseekProvider) {
                await this.updateJobStatus(jobId, { status: "reviewing", current_step: currentStep });
                await this.logEvent(jobId, currentStep, "review_started", {});

                const reviewPrompt = `You are a code reviewer. Evaluate the worker's edits and decide whether they pass or fail.
Return ONLY valid JSON.
JSON Schema:
{
  "review_result": "pass" | "needs_fix",
  "instructions_for_qwen": ["instructions"]
}`;
                let finalReviewPrompt = reviewPrompt;
                if (repoKey) {
                    const memPrompt = await buildMemoryPromptContext(repoKey);
                    if (memPrompt) {
                        finalReviewPrompt = memPrompt + "\n\n" + reviewPrompt;
                    }
                }

                const reviewRes = await deepseekProvider.handleRequest({
                    model: job.controller_model || config.defaultModel,
                    system: finalReviewPrompt,
                    messages: [{ role: "user", content: `Task: ${job.user_task}\nEdits review history: ${JSON.stringify(messages.slice(-4))}` }],
                    stream: false,
                    temperature: 0.1
                }, {});

                if (reviewRes.ok) {
                    const data = await reviewRes.json();
                    const text = data.content?.[0]?.text || "";
                    try {
                        const review = JSON.parse(text.trim().replace(/^```json|```$/g, ""));
                        await this.logEvent(jobId, currentStep, "review_completed", review);

                        if (review.review_result === "pass") {
                            await this.completeJob(jobId, finalSummary, Array.from(editedFiles));
                        } else {
                            await this.updateJobStatus(jobId, { status: "needs_human", failure_reason: "Reviewer rejected implementation" });
                            await this.logEvent(jobId, currentStep, "job_needs_human", { instructions: review.instructions_for_qwen });
                            
                            // Log failed review memory
                            if (repoKey) {
                                await insertTaskMemory({
                                    repo_key: repoKey,
                                    task_summary: job.user_task,
                                    touched_files: Array.from(editedFiles),
                                    outcome: "failed",
                                    model_route: "hybrid",
                                    cost_thb: 0
                                });
                                await recordFailurePattern(repoKey, "Reviewer Rejected", "Reviewer rejected implementation");
                            }
                        }
                    } catch {
                        await this.completeJob(jobId, finalSummary, Array.from(editedFiles));
                    }
                } else {
                    await this.completeJob(jobId, finalSummary, Array.from(editedFiles));
                }
            } else {
                await this.completeJob(jobId, finalSummary, Array.from(editedFiles));
            }

        } catch (err: any) {
            await this.failJob(jobId, err.message || "Unknown job failure", Array.from(editedFiles));
        }
    }

    private static async completeJob(jobId: number, summary: string, touchedFiles: string[] = []) {
        const job = await this.getJob(jobId);
        await this.updateJobStatus(jobId, { status: "completed", success: true, summary });
        await this.logEvent(jobId, 0, "job_completed", { summary });
        activeJobs.delete(jobId);

        if (job) {
            const rKey = job.repo_path ? normalizeRepoKey(job.repo_path) : "default";
            await insertTaskMemory({
                repo_key: rKey,
                task_summary: job.user_task,
                touched_files: touchedFiles,
                outcome: "success",
                model_route: job.mode === "smart" ? "hybrid" : "qwen-only",
                cost_thb: 0
            });
        }
    }

    private static async failJob(jobId: number, reason: string, touchedFiles: string[] = []) {
        const job = await this.getJob(jobId);
        await this.updateJobStatus(jobId, { status: "failed", success: false, failure_reason: reason });
        await this.logEvent(jobId, 0, "job_failed", { reason });
        activeJobs.delete(jobId);

        if (job) {
            const rKey = job.repo_path ? normalizeRepoKey(job.repo_path) : "default";
            await insertTaskMemory({
                repo_key: rKey,
                task_summary: job.user_task,
                touched_files: touchedFiles,
                outcome: "failed",
                model_route: job.mode === "smart" ? "hybrid" : "qwen-only",
                cost_thb: 0
            });

            const pType = classifyFailure(reason);
            await recordFailurePattern(rKey, pType, reason);
        }
    }

    // Helper to execute tools on local filesystem
    private static async executeLocalTool(repoPath: string, toolName: string, input: any): Promise<{ output: string; isError: boolean }> {
        const fullPath = input.file_path ? path.join(repoPath, input.file_path) : "";

        switch (toolName) {
            case "Read":
                if (!fs.existsSync(fullPath)) {
                    return { output: `File not found: ${input.file_path}`, isError: true };
                }
                const content = fs.readFileSync(fullPath, "utf-8");
                return { output: content, isError: false };

            case "Write":
                const parentDir = path.dirname(fullPath);
                if (!fs.existsSync(parentDir)) {
                    fs.mkdirSync(parentDir, { recursive: true });
                }
                fs.writeFileSync(fullPath, input.content, "utf-8");
                return { output: `Successfully wrote ${input.file_path}`, isError: false };

            case "Edit":
                if (!fs.existsSync(fullPath)) {
                    return { output: `File not found: ${input.file_path}`, isError: true };
                }
                let fileText = fs.readFileSync(fullPath, "utf-8");
                if (!fileText.includes(input.old_string)) {
                    return { output: `Error: old_string not found in file ${input.file_path}`, isError: true };
                }
                fileText = fileText.replace(input.old_string, input.new_string);
                fs.writeFileSync(fullPath, fileText, "utf-8");
                return { output: `Successfully replaced content in ${input.file_path}`, isError: false };

            case "Grep":
                const searchPattern = input.pattern;
                const matches: string[] = [];
                const searchDir = (dir: string) => {
                    const files = fs.readdirSync(dir);
                    for (const file of files) {
                        const target = path.join(dir, file);
                        if (fs.statSync(target).isDirectory()) {
                            if (file !== "node_modules" && file !== ".git") searchDir(target);
                        } else {
                            const text = fs.readFileSync(target, "utf-8");
                            if (text.includes(searchPattern)) {
                                matches.push(path.relative(repoPath, target));
                            }
                        }
                    }
                };
                searchDir(repoPath);
                return { output: matches.length > 0 ? matches.join("\n") : "No matches found", isError: false };

            case "Glob":
                // Basic glob finder
                const globPattern = input.pattern.replace(/\*/g, ".*");
                const globRegex = new RegExp(`^${globPattern}$`);
                const matchedFiles: string[] = [];
                const scanDir = (dir: string) => {
                    const files = fs.readdirSync(dir);
                    for (const file of files) {
                        const target = path.join(dir, file);
                        const rel = path.relative(repoPath, target);
                        if (fs.statSync(target).isDirectory()) {
                            if (file !== "node_modules" && file !== ".git") scanDir(target);
                        } else {
                            if (globRegex.test(file) || globRegex.test(rel)) {
                                matchedFiles.push(rel);
                            }
                        }
                    }
                };
                scanDir(repoPath);
                return { output: matchedFiles.length > 0 ? matchedFiles.join("\n") : "No files matched", isError: false };

            case "Bash":
                // Run bash command safely inside repo path
                const cmd = input.command;
                
                // Dangerous check
                const dangerousCheck = isCommandDangerous(cmd);
                if (dangerousCheck) {
                    return { output: `Dangerous command blocked: ${dangerousCheck}`, isError: true };
                }

                try {
                    const output = execSync(cmd, { cwd: repoPath, stdio: "pipe", timeout: 45000 });
                    return { output: output.toString("utf-8"), isError: false };
                } catch (cmdErr: any) {
                    const output = cmdErr.stdout ? cmdErr.stdout.toString() : "";
                    const error = cmdErr.stderr ? cmdErr.stderr.toString() : "";
                    return { output: `Exit Code: ${cmdErr.status || 1}\nStdout:\n${output}\nStderr:\n${error}`, isError: true };
                }

            default:
                throw new Error(`Unsupported tool name: ${toolName}`);
        }
    }
}

// Simple dangerous check copy
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
    if (cmd.includes(".env")) return "writing .env";
    if (cmd.includes("secrets")) return "writing secrets";
    if (cmd.includes("private key") || cmd.includes("private keys")) return "private keys";
    if (cmd.includes("api key") || cmd.includes("api keys")) return "API keys";
    return null;
}
