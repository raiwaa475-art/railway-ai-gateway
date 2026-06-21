import { gatewayRouter } from "./gateway.js";
import { config } from "../config/env.js";
import { pool } from "../utils/db.js";
import { 
    normalizeRepoKey, 
    getRepoMemory, 
    upsertRepoMemory, 
    insertTaskMemory, 
    rebuildFailurePatternsFromTraces 
} from "../routing/memory.js";

// Helper to assert admin privilege
function adminAuthMiddleware(req: any, res: any, next: any) {
    const expectedKey = config.gatewayAdminKey || config.gatewayApiKey;

    if (!expectedKey) {
        return res.status(500).json({
            error: {
                type: "server_error",
                message: "GATEWAY_ADMIN_KEY or GATEWAY_API_KEY is not configured"
            }
        });
    }

    const auth = req.header("authorization") || req.header("x-api-key") || "";
    const token = auth.replace(/^Bearer\s+/i, "").trim();

    if (token !== expectedKey) {
        return res.status(401).json({
            error: {
                type: "authentication_error",
                message: "Invalid admin key"
            }
        });
    }

    next();
}

/**
 * Helper to build SQL time filter based on "range" query parameter.
 */
function getTimeFilter(range: string, column: string = "created_at"): string {
    if (range === "today") {
        return ` AND ${column} >= CURRENT_DATE`;
    } else if (range === "7d") {
        return ` AND ${column} >= NOW() - INTERVAL '7 days'`;
    } else if (range === "30d") {
        return ` AND ${column} >= NOW() - INTERVAL '30 days'`;
    }
    return ""; // "all"
}

// -------------------------------------------------------------
// 1) METRICS ENDPOINTS
// -------------------------------------------------------------

// GET /admin/metrics/overview
gatewayRouter.get("/admin/metrics/overview", adminAuthMiddleware, async (req, res) => {
    const range = String(req.query.range || "7d");
    if (!pool) {
        return res.json({
            total_requests: 0,
            total_model_calls: 0,
            avg_latency: 0,
            deepseek_call_percentage: 0,
            daily_trend: []
        });
    }

    try {
        const timeFilter = getTimeFilter(range, "created_at");
        const timeFilterTraces = getTimeFilter(range, "timestamp");

        const requestsRes = await pool.query(`SELECT COUNT(*) as count FROM gateway_requests WHERE 1=1 ${timeFilter}`);
        const callsRes = await pool.query(`SELECT COUNT(*) as count FROM model_calls WHERE 1=1 ${timeFilter}`);
        const dsCallsRes = await pool.query(`SELECT COUNT(*) as count FROM model_calls WHERE provider = 'deepseek' ${timeFilter}`);
        const latencyRes = await pool.query(`SELECT COALESCE(AVG(latency_ms), 0) as avg_lat FROM gateway_requests WHERE 1=1 ${timeFilter}`);

        // Trend grouping by day (default to 30 days if range is 'all')
        const trendFilter = getTimeFilter(range === "all" ? "30d" : range, "created_at");
        const trendRes = await pool.query(`
            SELECT 
                DATE_TRUNC('day', created_at) as day, 
                COUNT(*) as request_count,
                COALESCE(AVG(latency_ms), 0) as avg_latency
            FROM gateway_requests
            WHERE 1=1 ${trendFilter}
            GROUP BY day
            ORDER BY day ASC
        `);

        const totalCalls = Number(callsRes.rows[0]?.count || 0);
        const dsCalls = Number(dsCallsRes.rows[0]?.count || 0);
        const dsPercentage = totalCalls > 0 ? (dsCalls * 100) / totalCalls : 0;

        res.json({
            total_requests: Number(requestsRes.rows[0]?.count || 0),
            total_model_calls: totalCalls,
            avg_latency: Math.round(Number(latencyRes.rows[0]?.avg_lat || 0)),
            deepseek_call_percentage: Number(dsPercentage.toFixed(2)),
            daily_trend: trendRes.rows.map(row => ({
                day: row.day,
                request_count: Number(row.request_count),
                avg_latency: Math.round(Number(row.avg_latency))
            }))
        });
    } catch (err: any) {
        res.status(500).json({ error: err.message });
    }
});

// GET /admin/metrics/qwen
gatewayRouter.get("/admin/metrics/qwen", adminAuthMiddleware, async (req, res) => {
    const range = String(req.query.range || "7d");
    if (!pool) {
        return res.json({
            qwen_calls: 0,
            qwen_success_rate: 0,
            qwen_failure_rate: 0,
            retry_rate: 0,
            build_status_breakdown: [],
            top_failure_reasons: []
        });
    }

    try {
        const timeFilter = getTimeFilter(range, "created_at");
        const timeFilterTraces = getTimeFilter(range, "timestamp");

        // Qwen calls (provider='qwen-local')
        const callsRes = await pool.query(`
            SELECT 
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE qwen_retry_used = true) as retries
            FROM model_calls 
            WHERE provider = 'qwen-local' ${timeFilter}
        `);

        // Success / Failure from traces
        const tracesRes = await pool.query(`
            SELECT 
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE success = true) as success,
                COUNT(*) FILTER (WHERE success = false) as failure
            FROM qwen_agent_traces
            WHERE 1=1 ${timeFilterTraces}
        `);

        // Build status breakdown
        const buildRes = await pool.query(`
            SELECT 
                COALESCE(build_status, 'unknown') as build_status,
                COUNT(*) as count
            FROM qwen_agent_traces
            WHERE 1=1 ${timeFilterTraces}
            GROUP BY COALESCE(build_status, 'unknown')
        `);

        // Top failure reasons
        const failuresRes = await pool.query(`
            SELECT 
                COALESCE(failure_reason, 'unknown') as reason,
                COUNT(*) as count
            FROM qwen_agent_traces
            WHERE success = false ${timeFilterTraces}
            GROUP BY COALESCE(failure_reason, 'unknown')
            ORDER BY count DESC
            LIMIT 10
        `);

        const qwenCalls = Number(callsRes.rows[0]?.total || 0);
        const qwenRetries = Number(callsRes.rows[0]?.retries || 0);
        const retryRate = qwenCalls > 0 ? (qwenRetries * 100) / qwenCalls : 0;

        const totalTraces = Number(tracesRes.rows[0]?.total || 0);
        const successTraces = Number(tracesRes.rows[0]?.success || 0);
        const failureTraces = Number(tracesRes.rows[0]?.failure || 0);

        res.json({
            qwen_calls: qwenCalls,
            qwen_success_rate: totalTraces > 0 ? Number(((successTraces * 100) / totalTraces).toFixed(2)) : 0,
            qwen_failure_rate: totalTraces > 0 ? Number(((failureTraces * 100) / totalTraces).toFixed(2)) : 0,
            retry_rate: Number(retryRate.toFixed(2)),
            build_status_breakdown: buildRes.rows.map(row => ({
                build_status: row.build_status,
                count: Number(row.count)
            })),
            top_failure_reasons: failuresRes.rows.map(row => ({
                reason: row.reason,
                count: Number(row.count)
            }))
        });
    } catch (err: any) {
        res.status(500).json({ error: err.message });
    }
});

// GET /admin/metrics/cost
gatewayRouter.get("/admin/metrics/cost", adminAuthMiddleware, async (req, res) => {
    const range = String(req.query.range || "7d");
    if (!pool) {
        return res.json({
            total_cost_usd: 0,
            total_cost_thb: 0,
            estimated_saved_usd: 0,
            estimated_saved_thb: 0,
            deepseek_call_percentage: 0,
            daily_trend: []
        });
    }

    try {
        const timeFilter = getTimeFilter(range, "created_at");

        const costRes = await pool.query(`
            SELECT 
                COALESCE(SUM(cost_usd), 0) as cost_usd,
                COALESCE(SUM(cost_thb), 0) as cost_thb,
                COALESCE(SUM(saved_usd), 0) as saved_usd,
                COALESCE(SUM(saved_thb), 0) as saved_thb
            FROM model_calls
            WHERE 1=1 ${timeFilter}
        `);

        const callsRes = await pool.query(`SELECT COUNT(*) as count FROM model_calls WHERE 1=1 ${timeFilter}`);
        const dsCallsRes = await pool.query(`SELECT COUNT(*) as count FROM model_calls WHERE provider = 'deepseek' ${timeFilter}`);

        // Trend grouping by day (default to 30 days if range is 'all')
        const trendFilter = getTimeFilter(range === "all" ? "30d" : range, "created_at");
        const trendRes = await pool.query(`
            SELECT 
                DATE_TRUNC('day', created_at) as day, 
                COALESCE(SUM(cost_thb), 0) as cost_thb,
                COALESCE(SUM(saved_thb), 0) as saved_thb
            FROM model_calls
            WHERE 1=1 ${trendFilter}
            GROUP BY day
            ORDER BY day ASC
        `);

        const totalCalls = Number(callsRes.rows[0]?.count || 0);
        const dsCalls = Number(dsCallsRes.rows[0]?.count || 0);
        const dsPercentage = totalCalls > 0 ? (dsCalls * 100) / totalCalls : 0;

        res.json({
            total_cost_usd: Number(Number(costRes.rows[0]?.cost_usd).toFixed(4)),
            total_cost_thb: Number(Number(costRes.rows[0]?.cost_thb).toFixed(2)),
            estimated_saved_usd: Number(Number(costRes.rows[0]?.saved_usd).toFixed(4)),
            estimated_saved_thb: Number(Number(costRes.rows[0]?.saved_thb).toFixed(2)),
            deepseek_call_percentage: Number(dsPercentage.toFixed(2)),
            daily_trend: trendRes.rows.map(row => ({
                day: row.day,
                cost_thb: Number(Number(row.cost_thb).toFixed(2)),
                saved_thb: Number(Number(row.saved_thb).toFixed(2))
            }))
        });
    } catch (err: any) {
        res.status(500).json({ error: err.message });
    }
});

// GET /admin/metrics/failures
gatewayRouter.get("/admin/metrics/failures", adminAuthMiddleware, async (req, res) => {
    const range = String(req.query.range || "7d");
    if (!pool) {
        return res.json({
            top_failure_reasons: [],
            build_status_breakdown: [],
            recent_failures: []
        });
    }

    try {
        const timeFilterTraces = getTimeFilter(range, "timestamp");

        // Top failure reasons
        const failuresRes = await pool.query(`
            SELECT 
                COALESCE(failure_reason, 'unknown') as reason,
                COUNT(*) as count
            FROM qwen_agent_traces
            WHERE success = false ${timeFilterTraces}
            GROUP BY COALESCE(failure_reason, 'unknown')
            ORDER BY count DESC
            LIMIT 10
        `);

        // Build status breakdown
        const buildRes = await pool.query(`
            SELECT 
                COALESCE(build_status, 'unknown') as build_status,
                COUNT(*) as count
            FROM qwen_agent_traces
            WHERE 1=1 ${timeFilterTraces}
            GROUP BY COALESCE(build_status, 'unknown')
        `);

        // Recent failures log list
        const listRes = await pool.query(`
            SELECT 
                id, 
                timestamp, 
                mode, 
                user_intent, 
                failure_reason, 
                build_status, 
                repo_key
            FROM qwen_agent_traces
            WHERE success = false ${timeFilterTraces}
            ORDER BY timestamp DESC
            LIMIT 20
        `);

        res.json({
            top_failure_reasons: failuresRes.rows.map(row => ({
                reason: row.reason,
                count: Number(row.count)
            })),
            build_status_breakdown: buildRes.rows.map(row => ({
                build_status: row.build_status,
                count: Number(row.count)
            })),
            recent_failures: listRes.rows.map(row => ({
                id: row.id,
                timestamp: row.timestamp,
                mode: row.mode,
                user_intent: row.user_intent,
                failure_reason: row.failure_reason,
                build_status: row.build_status,
                repo_key: row.repo_key
            }))
        });
    } catch (err: any) {
        res.status(500).json({ error: err.message });
    }
});

// -------------------------------------------------------------
// 2) MEMORY ENDPOINTS
// -------------------------------------------------------------

// GET /admin/memory/repos
gatewayRouter.get("/admin/memory/repos", adminAuthMiddleware, async (req, res) => {
    if (!pool) return res.json([]);
    try {
        const dbRes = await pool.query("SELECT * FROM repo_memories ORDER BY updated_at DESC");
        res.json(dbRes.rows.map(row => ({
            id: row.id,
            repo_key: row.repo_key,
            summary: row.summary,
            important_files: typeof row.important_files === "string" ? JSON.parse(row.important_files) : (row.important_files || []),
            risk_zones: typeof row.risk_zones === "string" ? JSON.parse(row.risk_zones) : (row.risk_zones || []),
            tech_stack: typeof row.tech_stack === "string" ? JSON.parse(row.tech_stack) : (row.tech_stack || []),
            updated_at: row.updated_at
        })));
    } catch (err: any) {
        res.status(500).json({ error: err.message });
    }
});

// GET /admin/memory/repos/:repoKey
gatewayRouter.get("/admin/memory/repos/:repoKey", adminAuthMiddleware, async (req, res) => {
    const memory = await getRepoMemory(req.params.repoKey);
    if (!memory) {
        return res.status(404).json({ error: `Repo memory not found for key: ${req.params.repoKey}` });
    }
    res.json(memory);
});

// POST /admin/memory/repos/:repoKey
gatewayRouter.post("/admin/memory/repos/:repoKey", adminAuthMiddleware, async (req, res) => {
    const upserted = await upsertRepoMemory(req.params.repoKey, req.body);
    if (!upserted) {
        return res.status(500).json({ error: "Failed to upsert repo memory" });
    }
    res.json(upserted);
});

// GET /admin/memory/tasks
gatewayRouter.get("/admin/memory/tasks", adminAuthMiddleware, async (req, res) => {
    if (!pool) return res.json([]);
    try {
        const repoKey = req.query.repoKey as string;
        let query = "SELECT * FROM task_memories";
        const params: any[] = [];

        if (repoKey) {
            query += " WHERE repo_key = $1";
            params.push(normalizeRepoKey(repoKey));
        }

        query += " ORDER BY created_at DESC LIMIT 50";

        const dbRes = await pool.query(query, params);
        res.json(dbRes.rows.map(row => ({
            id: row.id,
            repo_key: row.repo_key,
            task_summary: row.task_summary,
            touched_files: typeof row.touched_files === "string" ? JSON.parse(row.touched_files) : (row.touched_files || []),
            outcome: row.outcome,
            model_route: row.model_route,
            cost_thb: Number(row.cost_thb),
            created_at: row.created_at
        })));
    } catch (err: any) {
        res.status(500).json({ error: err.message });
    }
});

// POST /admin/memory/tasks
gatewayRouter.post("/admin/memory/tasks", adminAuthMiddleware, async (req, res) => {
    const { repo_key, task_summary, touched_files, outcome, model_route, cost_thb } = req.body;
    if (!repo_key || !task_summary || !outcome || !model_route) {
        return res.status(400).json({ error: "Missing required fields in task memory payload" });
    }

    const ok = await insertTaskMemory({
        repo_key,
        task_summary,
        touched_files: touched_files || [],
        outcome,
        model_route,
        cost_thb: cost_thb || 0
    });

    if (!ok) {
        return res.status(500).json({ error: "Failed to insert task memory" });
    }

    res.json({ success: true });
});

// GET /admin/memory/failures
gatewayRouter.get("/admin/memory/failures", adminAuthMiddleware, async (req, res) => {
    if (!pool) return res.json([]);
    try {
        const repoKey = req.query.repoKey as string;
        let query = "SELECT * FROM failure_patterns";
        const params: any[] = [];

        if (repoKey) {
            query += " WHERE repo_key = $1";
            params.push(normalizeRepoKey(repoKey));
        }

        query += " ORDER BY hit_count DESC, last_seen_at DESC";

        const dbRes = await pool.query(query, params);
        res.json(dbRes.rows.map(row => ({
            id: row.id,
            repo_key: row.repo_key,
            pattern_type: row.pattern_type,
            failure_reason: row.failure_reason,
            examples: typeof row.examples === "string" ? JSON.parse(row.examples) : (row.examples || []),
            hit_count: row.hit_count,
            last_seen_at: row.last_seen_at
        })));
    } catch (err: any) {
        res.status(500).json({ error: err.message });
    }
});

// POST /admin/memory/failures/rebuild
gatewayRouter.post("/admin/memory/failures/rebuild", adminAuthMiddleware, async (req, res) => {
    const repoKey = (req.body.repoKey || req.query.repoKey) as string;
    const ok = await rebuildFailurePatternsFromTraces(repoKey);
    if (!ok) {
        return res.status(500).json({ error: "Failed to rebuild failure patterns" });
    }
    res.json({ success: true });
});
