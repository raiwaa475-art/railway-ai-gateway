import { Router } from "express";
import crypto from "crypto";
import { authMiddleware } from "../auth.js";
import { sanitizeAnthropicResponse } from "../providers/deepseek.js";
import { SUPPORTED_MODELS } from "../config/models.js";
import { ModelRouter } from "../routing/router.js";
import { OrchestratorService } from "../routing/orchestrator.js";

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
    const clientModel = req.body?.model;

    if (clientModel === "hybrid-flow" || clientModel === "qwen-smart") {
        logRequest({
            type: "request",
            requestId,
            method: req.method,
            path: req.path,
            clientModel,
            upstreamModel: "hybrid-orchestration",
            stream: !!req.body?.stream,
            provider: "hybrid-orchestrator"
        });
        try {
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
                stream: !!req.body?.stream,
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
                stream: !!req.body?.stream,
                errorMessage: err.message,
                provider: "hybrid-orchestrator"
            });
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
    const isStream = !!req.body?.stream;

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

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                res.write(Buffer.from(value));
            }

            res.end();

            const latencyMs = Date.now() - startTime;
            logRequest({
                type: "response",
                requestId,
                method: req.method,
                path: req.path,
                clientModel,
                upstreamModel,
                status: upstream.status,
                latencyMs,
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

        const latencyMs = Date.now() - startTime;

        if (responseBody && upstream.status === 200) {
            // Sanitize only applies for Anthropic/DeepSeek native format responses if needed,
            // but we can pass it through since we sanitized Qwen in the provider.
            const sanitized = provider.id === "deepseek" ? sanitizeAnthropicResponse(responseBody) : responseBody;
            const inputTokens = responseBody.usage?.input_tokens;
            const outputTokens = responseBody.usage?.output_tokens;

            logRequest({
                type: "response",
                requestId,
                method: req.method,
                path: req.path,
                clientModel,
                upstreamModel,
                status: upstream.status,
                latencyMs,
                stream: false,
                inputTokens,
                outputTokens,
                provider: provider.id
            });

            return res.json(sanitized);
        } else {
            logRequest({
                type: "response",
                requestId,
                method: req.method,
                path: req.path,
                clientModel,
                upstreamModel,
                status: upstream.status,
                latencyMs,
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
        const latencyMs = Date.now() - startTime;

        logRequest({
            type: "response",
            requestId,
            method: req.method,
            path: req.path,
            clientModel,
            upstreamModel,
            status: 500,
            latencyMs,
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
