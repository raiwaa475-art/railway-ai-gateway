import { Router } from "express";
import crypto from "crypto";
import { authMiddleware } from "../auth.js";
import { sanitizeAnthropicResponse } from "../providers/deepseek.js";
import { SUPPORTED_MODELS } from "../config/models.js";
import { ModelRouter } from "../routing/router.js";
import { OrchestratorService } from "../routing/orchestrator.js";
import { insertGatewayRequest, updateGatewayRequest, insertModelCall, pool } from "../utils/db.js";

export const gatewayRouter = Router();

function logRequest(info: Record<string, unknown>) {
    console.log(JSON.stringify({
        time: new Date().toISOString(),
        ...info
    }));
}

gatewayRouter.get("/", (_req, res) => {
    res.json({
        ok: true,
        service: "railway-ai-gateway",
        version: "0.2.0",
        provider: "hybrid-gateway"
    });
});

gatewayRouter.get("/health", (_req, res) => {
    res.json({
        ok: true,
        service: "railway-ai-gateway",
        version: "0.2.0",
        provider: "hybrid-gateway"
    });
});

gatewayRouter.get("/v1/models", authMiddleware, (_req, res) => {
    res.json({
        data: SUPPORTED_MODELS.map(m => ({
            id: m.id,
            type: "model",
            display_name: m.displayName,
            provider: m.providerId,
            gateway_role: m.providerId === "deepseek" ? "default" : "local-dev"
        }))
    });
});

gatewayRouter.post("/v1/messages", authMiddleware, async (req, res) => {
    const requestId = crypto.randomUUID();
    const startTime = Date.now();
    const clientModel = req.body?.model || "unknown";
    const isStream = !!req.body?.stream;
    const mode = (clientModel === "hybrid-flow" || clientModel === "qwen-smart") ? "hybrid-flow" : "direct";

    // Insert gateway request to DB
    await insertGatewayRequest(requestId, clientModel, mode, isStream);

    if (clientModel === "hybrid-flow" || clientModel === "qwen-smart") {
        logRequest({
            type: "request",
            requestId,
            method: req.method,
            path: req.path,
            clientModel,
            upstreamModel: "hybrid-orchestration",
            stream: isStream,
            provider: "hybrid-orchestrator"
        });
        try {
            // Pass the requestId to OrchestratorService using custom property or just standard header
            (req as any).requestId = requestId;
            await OrchestratorService.handleTwinModels(req, res);
            logRequest({
                type: "response",
                requestId,
                method: req.method,
                path: req.path,
                clientModel,
                upstreamModel: "hybrid-orchestration",
                status: 200,
                latencyMs: Date.now() - startTime,
                stream: isStream,
                provider: "hybrid-orchestrator"
            });
        } catch (err: any) {
            logRequest({
                type: "response",
                requestId,
                method: req.method,
                path: req.path,
                clientModel,
                upstreamModel: "hybrid-orchestration",
                status: 500,
                latencyMs: Date.now() - startTime,
                stream: isStream,
                errorMessage: err.message,
                provider: "hybrid-orchestrator"
            });
            await updateGatewayRequest(requestId, 500, Date.now() - startTime);
            res.status(500).json({
                error: {
                    type: "gateway_error",
                    message: err.message || "Unknown error inside Hybrid Orchestrator"
                }
            });
        }
        return;
    }

    const provider = ModelRouter.resolve(clientModel);
    const upstreamModel = provider.resolveUpstreamModel(clientModel);

    logRequest({
        type: "request",
        requestId,
        method: req.method,
        path: req.path,
        clientModel,
        upstreamModel,
        stream: isStream,
        provider: provider.id
    });

    try {
        const clientHeaders: Record<string, string> = {
            "user-agent": req.header("user-agent") || "railway-ai-gateway"
        };

        const callStartTime = Date.now();
        const upstream = await provider.handleRequest(req.body, clientHeaders);

        res.status(upstream.status);

        const contentType = upstream.headers.get("content-type");
        if (contentType) {
            res.setHeader("content-type", contentType);
        }

        if (isStream && upstream.body) {
            res.setHeader("cache-control", "no-cache");
            res.setHeader("x-accel-buffering", "no");
            res.setHeader("connection", "keep-alive");

            const reader = upstream.body.getReader();
            const decoder = new TextDecoder();
            let streamBuffer = "";
            let inputTokens = 0;
            let outputTokens = 0;
            let cacheCreationTokens = 0;
            let cacheReadTokens = 0;

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
                                if (dataJson.message.usage.input_tokens) {
                                    inputTokens = dataJson.message.usage.input_tokens;
                                }
                                if (dataJson.message.usage.cache_creation_input_tokens) {
                                    cacheCreationTokens = dataJson.message.usage.cache_creation_input_tokens;
                                }
                                if (dataJson.message.usage.cache_read_input_tokens) {
                                    cacheReadTokens = dataJson.message.usage.cache_read_input_tokens;
                                }
                            }
                            if (dataJson.usage) {
                                if (dataJson.usage.output_tokens) {
                                    outputTokens = dataJson.usage.output_tokens;
                                }
                                if (dataJson.usage.input_tokens) {
                                    inputTokens = dataJson.usage.input_tokens;
                                }
                            }
                        } catch {}
                    }
                }
            }

            res.end();

            const latencyMs = Date.now() - callStartTime;

            // Log model call for direct stream
            await insertModelCall({
                requestId,
                provider: provider.id,
                model: upstreamModel,
                inputTokens,
                outputTokens,
                cacheHitInputTokens: cacheReadTokens,
                cacheMissInputTokens: inputTokens - cacheReadTokens,
                latencyMs
            });

            const totalLatencyMs = Date.now() - startTime;
            await updateGatewayRequest(requestId, upstream.status, totalLatencyMs);

            logRequest({
                type: "response",
                requestId,
                method: req.method,
                path: req.path,
                clientModel,
                upstreamModel,
                status: upstream.status,
                latencyMs: totalLatencyMs,
                stream: true,
                provider: provider.id
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

        const totalLatencyMs = Date.now() - startTime;
        const callLatencyMs = Date.now() - callStartTime;

        if (responseBody && upstream.status === 200) {
            const sanitized = provider.id === "deepseek" ? sanitizeAnthropicResponse(responseBody) : responseBody;
            const inputTokens = responseBody.usage?.input_tokens || 0;
            const outputTokens = responseBody.usage?.output_tokens || 0;
            const cacheReadTokens = responseBody.usage?.cache_read_input_tokens || 0;

            // Log model call for direct non-stream
            await insertModelCall({
                requestId,
                provider: provider.id,
                model: upstreamModel,
                inputTokens,
                outputTokens,
                cacheHitInputTokens: cacheReadTokens,
                cacheMissInputTokens: inputTokens - cacheReadTokens,
                latencyMs: callLatencyMs
            });

            await updateGatewayRequest(requestId, upstream.status, totalLatencyMs);

            logRequest({
                type: "response",
                requestId,
                method: req.method,
                path: req.path,
                clientModel,
                upstreamModel,
                status: upstream.status,
                latencyMs: totalLatencyMs,
                stream: false,
                inputTokens,
                outputTokens,
                provider: provider.id
            });

            return res.json(sanitized);
        } else {
            await updateGatewayRequest(requestId, upstream.status, totalLatencyMs);

            logRequest({
                type: "response",
                requestId,
                method: req.method,
                path: req.path,
                clientModel,
                upstreamModel,
                status: upstream.status,
                latencyMs: totalLatencyMs,
                stream: false,
                errorMessage: responseBody?.error?.message || text,
                provider: provider.id
            });

            if (responseBody) {
                return res.status(upstream.status).json(responseBody);
            } else {
                return res.status(upstream.status).json({
                    error: {
                        type: "upstream_error",
                        message: text
                    }
                });
            }
        }
    } catch (error) {
        const message = error instanceof Error ? error.message : "Unknown error";
        const totalLatencyMs = Date.now() - startTime;

        await updateGatewayRequest(requestId, 500, totalLatencyMs);

        logRequest({
            type: "response",
            requestId,
            method: req.method,
            path: req.path,
            clientModel,
            upstreamModel,
            status: 500,
            latencyMs: totalLatencyMs,
            stream: isStream,
            errorMessage: message,
            provider: provider.id
        });

        return res.status(500).json({
            error: {
                type: "gateway_error",
                message
            }
        });
    }
});

// Admin Usage Summary & Recent Endpoints
gatewayRouter.get("/admin/usage/summary", authMiddleware, async (req, res) => {
    const range = req.query.range;
    let timeFilter = "";
    if (range === "today") {
        timeFilter = "AND created_at >= CURRENT_DATE";
    }

    if (!pool) {
        return res.json({
            total_requests: 0,
            total_cost_usd: 0,
            total_cost_thb: 0,
            total_input_cost_usd: 0,
            total_input_cost_thb: 0,
            total_output_cost_usd: 0,
            total_output_cost_thb: 0,
            total_saved_usd: 0,
            total_saved_thb: 0,
            total_net_cost_usd: 0,
            total_net_cost_thb: 0,
            total_saved_input_usd: 0,
            total_saved_input_thb: 0,
            total_saved_output_usd: 0,
            total_saved_output_thb: 0,
            total_input_tokens: 0,
            total_output_tokens: 0,
            tokens_by_model: []
        });
    }

    try {
        const requestsRes = await pool.query(`SELECT COUNT(*) as count FROM gateway_requests WHERE 1=1 ${timeFilter}`);
        const callsRes = await pool.query(`
            SELECT 
                COALESCE(SUM(cost_usd), 0) as total_cost_usd,
                COALESCE(SUM(cost_thb), 0) as total_cost_thb,
                COALESCE(SUM(input_cost_usd), 0) as total_input_cost_usd,
                COALESCE(SUM(input_cost_thb), 0) as total_input_cost_thb,
                COALESCE(SUM(output_cost_usd), 0) as total_output_cost_usd,
                COALESCE(SUM(output_cost_thb), 0) as total_output_cost_thb,
                COALESCE(SUM(saved_usd), 0) as total_saved_usd,
                COALESCE(SUM(saved_thb), 0) as total_saved_thb,
                COALESCE(SUM(saved_input_usd), 0) as total_saved_input_usd,
                COALESCE(SUM(saved_input_thb), 0) as total_saved_input_thb,
                COALESCE(SUM(saved_output_usd), 0) as total_saved_output_usd,
                COALESCE(SUM(saved_output_thb), 0) as total_saved_output_thb,
                COALESCE(SUM(input_tokens), 0) as total_input_tokens,
                COALESCE(SUM(output_tokens), 0) as total_output_tokens
            FROM model_calls
            WHERE 1=1 ${timeFilter}
        `);
        const modelTokensRes = await pool.query(`
            SELECT
                provider,
                model,
                COALESCE(SUM(input_tokens), 0) as input_tokens,
                COALESCE(SUM(output_tokens), 0) as output_tokens
            FROM model_calls
            WHERE 1=1 ${timeFilter}
            GROUP BY provider, model
            ORDER BY COALESCE(SUM(input_tokens), 0) + COALESCE(SUM(output_tokens), 0) DESC
        `);

        const grossCostUsd = Number(callsRes.rows[0].total_cost_usd);
        const grossCostThb = Number(callsRes.rows[0].total_cost_thb);
        const savedUsd = Number(callsRes.rows[0].total_saved_usd);
        const savedThb = Number(callsRes.rows[0].total_saved_thb);
        const netCostUsd = Math.max(0, grossCostUsd - savedUsd);
        const netCostThb = Math.max(0, grossCostThb - savedThb);

        res.json({
            total_requests: Number(requestsRes.rows[0].count),
            total_cost_usd: grossCostUsd,
            total_cost_thb: grossCostThb,
            total_input_cost_usd: Number(callsRes.rows[0].total_input_cost_usd),
            total_input_cost_thb: Number(callsRes.rows[0].total_input_cost_thb),
            total_output_cost_usd: Number(callsRes.rows[0].total_output_cost_usd),
            total_output_cost_thb: Number(callsRes.rows[0].total_output_cost_thb),
            total_saved_usd: savedUsd,
            total_saved_thb: savedThb,
            total_net_cost_usd: netCostUsd,
            total_net_cost_thb: netCostThb,
            total_saved_input_usd: Number(callsRes.rows[0].total_saved_input_usd),
            total_saved_input_thb: Number(callsRes.rows[0].total_saved_input_thb),
            total_saved_output_usd: Number(callsRes.rows[0].total_saved_output_usd),
            total_saved_output_thb: Number(callsRes.rows[0].total_saved_output_thb),
            total_input_tokens: Number(callsRes.rows[0].total_input_tokens),
            total_output_tokens: Number(callsRes.rows[0].total_output_tokens),
            tokens_by_model: modelTokensRes.rows.map(row => ({
                provider: row.provider,
                model: row.model,
                input_tokens: Number(row.input_tokens),
                output_tokens: Number(row.output_tokens)
            }))
        });
    } catch (err: any) {
        res.status(500).json({ error: err.message });
    }
});

gatewayRouter.get("/admin/usage/recent", authMiddleware, async (req, res) => {
    const limit = Math.min(Number(req.query.limit || 50), 100);
    if (!pool) {
        return res.json([]);
    }
    try {
        const result = await pool.query(
            "SELECT * FROM model_calls ORDER BY created_at DESC LIMIT $1",
            [limit]
        );
        res.json(result.rows);
    } catch (err: any) {
        res.status(500).json({ error: err.message });
    }
});
