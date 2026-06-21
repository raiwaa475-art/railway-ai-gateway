import { pool } from "../utils/db.js";

export interface RepoMemory {
    id?: number;
    repo_key: string;
    summary: string;
    important_files: string[];
    risk_zones: string[];
    tech_stack: string[];
    updated_at?: Date;
}

export interface TaskMemory {
    id?: number;
    repo_key: string;
    task_summary: string;
    touched_files: string[];
    outcome: string;
    model_route: string;
    cost_thb: number;
    created_at?: Date;
}

export interface FailurePattern {
    id?: number;
    repo_key: string;
    pattern_type: string;
    failure_reason: string;
    examples: string[];
    hit_count: number;
    last_seen_at?: Date;
}

/**
 * Normalizes a repo path or key into a clean string slug.
 */
export function normalizeRepoKey(key: string): string {
    if (!key) return "default";
    let cleaned = key.trim();
    cleaned = cleaned.replace(/\\/g, "/");
    if (cleaned.endsWith("/")) {
        cleaned = cleaned.slice(0, -1);
    }
    const parts = cleaned.split("/");
    const last = parts[parts.length - 1];
    return last || cleaned;
}

/**
 * Categorizes a failure reason into a common failure pattern class.
 */
export function classifyFailure(reason: string): string {
    const r = String(reason || "").toLowerCase();
    if (r.includes("compile") || r.includes("syntax") || r.includes("typescript") || r.includes("build failed") || r.includes("tsc ") || r.includes("compilation")) {
        return "Compilation Error";
    }
    if (r.includes("test") || r.includes("assert") || r.includes("expect") || r.includes("spec") || r.includes("unit test") || r.includes("testing")) {
        return "Test Failure";
    }
    if (r.includes("timeout") || r.includes("timed out") || r.includes("time out")) {
        return "Timeout";
    }
    if (r.includes("max round") || r.includes("max tool round") || r.includes("tool round limit") || r.includes("max_tool_rounds")) {
        return "Max Tool Rounds Exceeded";
    }
    if (r.includes("dangerous command") || r.includes("blocked") || r.includes("rm -rf") || r.includes("dangerous")) {
        return "Security / Dangerous Command Blocked";
    }
    if (r.includes("json") || r.includes("parse error") || r.includes("parsing json") || r.includes("invalid json")) {
        return "JSON Parsing Error";
    }
    if (r.includes("file not found") || r.includes("no such file") || r.includes("directory not found") || r.includes("enoent")) {
        return "File Access Error";
    }
    return "Execution Error";
}

/**
 * Fetches the RepoMemory metadata for a key. Returns null if not found.
 */
export async function getRepoMemory(repoKey: string): Promise<RepoMemory | null> {
    if (!pool) return null;
    const normalized = normalizeRepoKey(repoKey);
    try {
        const res = await pool.query("SELECT * FROM repo_memories WHERE repo_key = $1", [normalized]);
        if (res.rows.length === 0) return null;
        const row = res.rows[0];
        return {
            id: row.id,
            repo_key: row.repo_key,
            summary: row.summary,
            important_files: typeof row.important_files === "string" ? JSON.parse(row.important_files) : (row.important_files || []),
            risk_zones: typeof row.risk_zones === "string" ? JSON.parse(row.risk_zones) : (row.risk_zones || []),
            tech_stack: typeof row.tech_stack === "string" ? JSON.parse(row.tech_stack) : (row.tech_stack || []),
            updated_at: row.updated_at
        };
    } catch (err) {
        console.error(`Failed to get repo memory for ${normalized}:`, err);
        return null;
    }
}

/**
 * Upserts the RepoMemory metadata for a key.
 */
export async function upsertRepoMemory(repoKey: string, data: Partial<RepoMemory>): Promise<RepoMemory | null> {
    if (!pool) return null;
    const normalized = normalizeRepoKey(repoKey);
    const summary = data.summary || "";
    const important_files = JSON.stringify(data.important_files || []);
    const risk_zones = JSON.stringify(data.risk_zones || []);
    const tech_stack = JSON.stringify(data.tech_stack || []);

    try {
        const res = await pool.query(
            `INSERT INTO repo_memories (repo_key, summary, important_files, risk_zones, tech_stack, updated_at)
             VALUES ($1, $2, $3, $4, $5, CURRENT_TIMESTAMP)
             ON CONFLICT (repo_key) DO UPDATE SET
                summary = EXCLUDED.summary,
                important_files = EXCLUDED.important_files,
                risk_zones = EXCLUDED.risk_zones,
                tech_stack = EXCLUDED.tech_stack,
                updated_at = CURRENT_TIMESTAMP
             RETURNING *`,
            [normalized, summary, important_files, risk_zones, tech_stack]
        );
        const row = res.rows[0];
        return {
            id: row.id,
            repo_key: row.repo_key,
            summary: row.summary,
            important_files: typeof row.important_files === "string" ? JSON.parse(row.important_files) : (row.important_files || []),
            risk_zones: typeof row.risk_zones === "string" ? JSON.parse(row.risk_zones) : (row.risk_zones || []),
            tech_stack: typeof row.tech_stack === "string" ? JSON.parse(row.tech_stack) : (row.tech_stack || []),
            updated_at: row.updated_at
        };
    } catch (err) {
        console.error(`Failed to upsert repo memory for ${normalized}:`, err);
        return null;
    }
}

/**
 * Appends a task memory log.
 */
export async function insertTaskMemory(task: TaskMemory): Promise<boolean> {
    if (!pool) return false;
    const normalized = normalizeRepoKey(task.repo_key);
    try {
        await pool.query(
            `INSERT INTO task_memories (repo_key, task_summary, touched_files, outcome, model_route, cost_thb)
             VALUES ($1, $2, $3, $4, $5, $6)`,
            [
                normalized,
                task.task_summary,
                JSON.stringify(task.touched_files || []),
                task.outcome,
                task.model_route,
                task.cost_thb || 0
            ]
        );
        return true;
    } catch (err) {
        console.error("Failed to insert task memory:", err);
        return false;
    }
}

/**
 * Updates or registers a failure pattern.
 */
export async function recordFailurePattern(repoKey: string, patternType: string, reason: string): Promise<boolean> {
    if (!pool) return false;
    const normalized = normalizeRepoKey(repoKey);
    try {
        // Query current pattern
        const currentRes = await pool.query(
            "SELECT * FROM failure_patterns WHERE repo_key = $1 AND pattern_type = $2",
            [normalized, patternType]
        );

        let examples: string[] = [reason];
        let hitCount = 1;

        if (currentRes.rows.length > 0) {
            const row = currentRes.rows[0];
            const oldExamples = typeof row.examples === "string" ? JSON.parse(row.examples) : (row.examples || []);
            // Append only if not already in examples
            examples = Array.from(new Set([...oldExamples, reason])).slice(-5);
            hitCount = (row.hit_count || 0) + 1;
        }

        await pool.query(
            `INSERT INTO failure_patterns (repo_key, pattern_type, failure_reason, examples, hit_count, last_seen_at)
             VALUES ($1, $2, $3, $4, $5, CURRENT_TIMESTAMP)
             ON CONFLICT (repo_key, pattern_type) DO UPDATE SET
                failure_reason = EXCLUDED.failure_reason,
                examples = EXCLUDED.examples,
                hit_count = EXCLUDED.hit_count,
                last_seen_at = CURRENT_TIMESTAMP`,
            [normalized, patternType, reason, JSON.stringify(examples), hitCount]
        );
        return true;
    } catch (err) {
        console.error("Failed to record failure pattern:", err);
        return false;
    }
}

/**
 * Rebuilds failure patterns by scanning history from traces (where success = false).
 */
export async function rebuildFailurePatternsFromTraces(repoKey?: string): Promise<boolean> {
    if (!pool) return false;
    try {
        let query = "SELECT * FROM qwen_agent_traces WHERE success = false";
        const params: any[] = [];
        if (repoKey) {
            query += " AND repo_key = $1";
            params.push(normalizeRepoKey(repoKey));
        }

        const tracesRes = await pool.query(query, params);
        if (tracesRes.rows.length === 0) {
            // Delete all failures if no failed traces
            if (repoKey) {
                await pool.query("DELETE FROM failure_patterns WHERE repo_key = $1", [normalizeRepoKey(repoKey)]);
            } else {
                await pool.query("DELETE FROM failure_patterns");
            }
            return true;
        }

        // Group by (repo_key, pattern_type)
        const groups = new Map<string, { repo_key: string, pattern_type: string, examples: string[], hit_count: number, last_seen_at: Date, last_reason: string }>();

        for (const row of tracesRes.rows) {
            const traceRepoKey = row.repo_key || (repoKey ? normalizeRepoKey(repoKey) : "default");
            const reason = row.failure_reason || row.tool_validation_error || "Unknown error";
            const patternType = classifyFailure(reason);
            const ts = new Date(row.timestamp || Date.now());

            const groupKey = `${traceRepoKey}||${patternType}`;
            let item = groups.get(groupKey);
            if (!item) {
                item = {
                    repo_key: traceRepoKey,
                    pattern_type: patternType,
                    examples: [],
                    hit_count: 0,
                    last_seen_at: ts,
                    last_reason: reason
                };
                groups.set(groupKey, item);
            }

            item.hit_count++;
            if (ts > item.last_seen_at) {
                item.last_seen_at = ts;
                item.last_reason = reason;
            }
            if (!item.examples.includes(reason)) {
                item.examples.push(reason);
            }
        }

        // Reset current failure patterns
        if (repoKey) {
            await pool.query("DELETE FROM failure_patterns WHERE repo_key = $1", [normalizeRepoKey(repoKey)]);
        } else {
            await pool.query("DELETE FROM failure_patterns");
        }

        // Write group summaries
        for (const item of groups.values()) {
            const finalExamples = item.examples.slice(-5);
            await pool.query(
                `INSERT INTO failure_patterns (repo_key, pattern_type, failure_reason, examples, hit_count, last_seen_at)
                 VALUES ($1, $2, $3, $4, $5, $6)`,
                [
                    item.repo_key,
                    item.pattern_type,
                    item.last_reason,
                    JSON.stringify(finalExamples),
                    item.hit_count,
                    item.last_seen_at
                ]
            );
        }

        return true;
    } catch (err) {
        console.error("Failed to rebuild failure patterns:", err);
        return false;
    }
}

/**
 * Builds the brief text context for repository memory injection.
 */
export async function getRepoMemoryContext(repoKey: string): Promise<string> {
    if (!pool) return "";
    const normalized = normalizeRepoKey(repoKey);
    const repoMem = await getRepoMemory(normalized);
    if (!repoMem) return "";

    let context = "";
    if (repoMem.summary) {
        context += `Repository Summary: ${repoMem.summary}\n`;
    }
    if (repoMem.tech_stack && repoMem.tech_stack.length > 0) {
        context += `Tech Stack: ${repoMem.tech_stack.join(", ")}\n`;
    }
    if (repoMem.important_files && repoMem.important_files.length > 0) {
        context += `Important Files: ${repoMem.important_files.join(", ")}\n`;
    }
    if (repoMem.risk_zones && repoMem.risk_zones.length > 0) {
        context += `Risk Zones: ${repoMem.risk_zones.join(", ")}\n`;
    }
    return context.trim();
}

/**
 * Builds the full formatted text to inject into prompts, capped at 2,000 characters.
 */
export async function buildMemoryPromptContext(repoKey: string): Promise<string> {
    if (!pool) return "";
    const normalized = normalizeRepoKey(repoKey);

    try {
        // 1. Get Repo Memory
        const repoMem = await getRepoMemory(normalized);

        // 2. Get Tasks Memory (limit 5)
        const tasksRes = await pool.query(
            "SELECT * FROM task_memories WHERE repo_key = $1 ORDER BY created_at DESC LIMIT 5",
            [normalized]
        );

        // 3. Get Failure Patterns (limit 5)
        const failuresRes = await pool.query(
            "SELECT * FROM failure_patterns WHERE repo_key = $1 ORDER BY hit_count DESC, last_seen_at DESC LIMIT 5",
            [normalized]
        );

        let parts: string[] = [];

        if (repoMem && (repoMem.summary || repoMem.tech_stack.length > 0 || repoMem.important_files.length > 0 || repoMem.risk_zones.length > 0)) {
            let section = "[Repository Memory]\n";
            if (repoMem.summary) section += `Summary: ${repoMem.summary}\n`;
            if (repoMem.tech_stack.length > 0) section += `Tech Stack: ${repoMem.tech_stack.join(", ")}\n`;
            if (repoMem.important_files.length > 0) section += `Key Files: ${repoMem.important_files.join(", ")}\n`;
            if (repoMem.risk_zones.length > 0) section += `Risk/Caution Zones: ${repoMem.risk_zones.join(", ")}\n`;
            parts.push(section.trim());
        }

        if (tasksRes.rows.length > 0) {
            let section = "[Recent Tasks Memory]\n";
            for (const row of tasksRes.rows) {
                const files = typeof row.touched_files === "string" ? JSON.parse(row.touched_files) : (row.touched_files || []);
                const fileStr = files.length > 0 ? ` (Files: ${files.join(", ")})` : "";
                section += `- Task: ${row.task_summary} | Route: ${row.model_route} | Outcome: ${row.outcome}${fileStr}\n`;
            }
            parts.push(section.trim());
        }

        if (failuresRes.rows.length > 0) {
            let section = "[Top Failure Patterns]\n";
            for (const row of failuresRes.rows) {
                section += `- Pattern: ${row.pattern_type} (Seen ${row.hit_count} times)\n  Last reason: ${row.failure_reason || "unknown"}\n`;
            }
            parts.push(section.trim());
        }

        if (parts.length === 0) return "";

        // Combine parts
        let result = `\n--- LIGHTWEIGHT WORKSPACE MEMORY ---\n${parts.join("\n\n")}\n------------------------------------\n`;

        // Cap at 2000 characters
        if (result.length > 2000) {
            result = result.substring(0, 1997) + "...";
        }

        return result;
    } catch (err) {
        console.error("Failed to build memory prompt context:", err);
        return "";
    }
}
