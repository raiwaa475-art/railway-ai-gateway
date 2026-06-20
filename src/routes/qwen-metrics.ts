import { gatewayRouter } from "./gateway.js";
import { config } from "../config/env.js";
import { pool } from "../utils/db.js";

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

function toCount(value: unknown): number {
    const count = Number(value);
    return Number.isFinite(count) ? count : 0;
}

gatewayRouter.get("/admin/usage/qwen-metrics", adminAuthMiddleware, async (_req, res) => {
    if (!pool) {
        return res.json({
            total_qwen_calls: 0,
            qwen_valid_calls: 0,
            qwen_valid_rate: 0,
            qwen_retry_rate: 0,
            qwen_weak_rate: 0,
            fallback_reason_breakdown: [],
            file_context_source_breakdown: [],
            deepseek_calls: 0,
            estimated_saved_thb: 0
        });
    }

    try {
        const [
            qwenTotalsRes,
            fallbackBreakdownRes,
            fileContextBreakdownRes,
            deepseekRes
        ] = await Promise.all([
            pool.query(`
                SELECT
                    COUNT(*) AS total_qwen_calls,
                    COUNT(*) FILTER (WHERE COALESCE(qwen_patch_valid, false)) AS qwen_valid_calls,
                    COUNT(*) FILTER (WHERE COALESCE(qwen_retry_used, false)) AS qwen_retry_calls,
                    COUNT(*) FILTER (WHERE COALESCE(qwen_draft_weak, false)) AS qwen_weak_calls
                FROM model_calls
                WHERE provider = 'qwen-local'
            `),
            pool.query(`
                SELECT
                    COALESCE(fallback_reason, 'unknown') AS fallback_reason,
                    COUNT(*) AS count
                FROM model_calls
                WHERE provider = 'qwen-local'
                  AND fallback_reason IS NOT NULL
                GROUP BY COALESCE(fallback_reason, 'unknown')
                ORDER BY COUNT(*) DESC, COALESCE(fallback_reason, 'unknown') ASC
            `),
            pool.query(`
                SELECT
                    COALESCE(file_context_source, 'unknown') AS file_context_source,
                    COUNT(*) AS count
                FROM model_calls
                WHERE provider = 'qwen-local'
                  AND file_context_source IS NOT NULL
                GROUP BY COALESCE(file_context_source, 'unknown')
                ORDER BY COUNT(*) DESC, COALESCE(file_context_source, 'unknown') ASC
            `),
            pool.query(`
                SELECT
                    COUNT(*) AS deepseek_calls,
                    COALESCE(SUM(saved_thb), 0) AS estimated_saved_thb
                FROM model_calls
                WHERE provider = 'deepseek'
            `)
        ]);

        const totalQwenCalls = toCount(qwenTotalsRes.rows[0]?.total_qwen_calls);
        const validQwenCalls = toCount(qwenTotalsRes.rows[0]?.qwen_valid_calls);
        const retryQwenCalls = toCount(qwenTotalsRes.rows[0]?.qwen_retry_calls);
        const weakQwenCalls = toCount(qwenTotalsRes.rows[0]?.qwen_weak_calls);
        const deepseekCalls = toCount(deepseekRes.rows[0]?.deepseek_calls);
        const estimatedSavedThb = Number(deepseekRes.rows[0]?.estimated_saved_thb || 0);

        res.json({
            total_qwen_calls: totalQwenCalls,
            qwen_valid_calls: validQwenCalls,
            qwen_valid_rate: totalQwenCalls > 0 ? validQwenCalls / totalQwenCalls : 0,
            qwen_retry_rate: totalQwenCalls > 0 ? retryQwenCalls / totalQwenCalls : 0,
            qwen_weak_rate: totalQwenCalls > 0 ? weakQwenCalls / totalQwenCalls : 0,
            fallback_reason_breakdown: fallbackBreakdownRes.rows.map(row => ({
                fallback_reason: row.fallback_reason,
                count: toCount(row.count)
            })),
            file_context_source_breakdown: fileContextBreakdownRes.rows.map(row => ({
                file_context_source: row.file_context_source,
                count: toCount(row.count)
            })),
            deepseek_calls: deepseekCalls,
            estimated_saved_thb: estimatedSavedThb
        });
    } catch (err: any) {
        res.status(500).json({ error: err.message });
    }
});

