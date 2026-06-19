import { Router } from "express";
import crypto from "crypto";
import { authMiddleware } from "../auth.js";
import { DeepSeekProvider, sanitizeAnthropicResponse } from "../providers/deepseek.js";
import { QwenLocalProvider } from "../providers/qwen-local.js";
import { Provider } from "../providers/base.js";

export const gatewayRouter = Router();

const deepseekProvider = new DeepSeekProvider();
const qwenProvider = new QwenLocalProvider();

function getProvider(model?: string): Provider {
    if (model && model.toLowerCase().includes("qwen")) {
        return qwenProvider;
    }
    return deepseekProvider;
}

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
        data: [
            {
                id: "deepseek-v4-flash",
                type: "model",
                display_name: "DeepSeek V4 Flash",
                provider: "deepseek",
                gateway_role: "default"
            },
            {
                id: "qwen-local",
                type: "model",
                display_name: "Qwen Local (Ollama/Tunnel)",
                provider: "qwen-local",
                gateway_role: "local-dev"
            }
        ]
    });
});

gatewayRouter.post("/v1/messages", authMiddleware, async (req, res) => {
    const requestId = crypto.randomUUID();
    const startTime = Date.now();
    const clientModel = req.body?.model;
    const provider = getProvider(clientModel);
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
