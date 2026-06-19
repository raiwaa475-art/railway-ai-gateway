import express from "express";
import cors from "cors";
import crypto from "crypto";
import { authMiddleware } from "./auth.js";
import {
    forwardToDeepSeekAnthropic,
    resolveUpstreamModel,
    sanitizeAnthropicResponse
} from "./deepseek.js";

const app = express();

app.use(cors());
app.use(express.json({ limit: "20mb" }));

// Error middleware for JSON syntax errors
app.use((err: any, _req: any, res: any, next: any) => {
    if (err instanceof SyntaxError && "body" in err) {
        return res.status(400).json({
            error: {
                type: "invalid_request_error",
                message: "Invalid JSON body"
            }
        });
    }
    next(err);
});

const port = Number(process.env.PORT || 3000);

function logRequest(info: Record<string, unknown>) {
    console.log(JSON.stringify({
        time: new Date().toISOString(),
        ...info
    }));
}

// Routes
app.get("/", (_req, res) => {
    res.json({
        ok: true,
        service: "railway-ai-gateway",
        version: "0.1.1",
        provider: "deepseek-anthropic"
    });
});

app.head("/", (_req, res) => {
    res.status(200).end();
});

app.get("/health", (_req, res) => {
    res.json({
        ok: true,
        service: "railway-ai-gateway",
        version: "0.1.1",
        provider: "deepseek-anthropic"
    });
});

app.get("/v1/models", authMiddleware, (_req, res) => {
    res.json({
        data: [
            {
                id: "deepseek-v4-flash",
                type: "model",
                display_name: "DeepSeek V4 Flash",
                provider: "deepseek",
                gateway_role: "default"
            }
        ]
    });
});

app.post("/v1/messages", authMiddleware, async (req, res) => {
    const requestId = crypto.randomUUID();
    const startTime = Date.now();
    const clientModel = req.body?.model;
    const upstreamModel = resolveUpstreamModel(clientModel);
    const isStream = !!req.body?.stream;

    logRequest({
        type: "request",
        requestId,
        method: req.method,
        path: req.path,
        clientModel,
        upstreamModel,
        stream: isStream
    });

    try {
        const upstream = await forwardToDeepSeekAnthropic(req.body, {
            "user-agent": req.header("user-agent") || "railway-ai-gateway"
        });

        res.status(upstream.status);

        const contentType = upstream.headers.get("content-type");
        if (contentType) {
            res.setHeader("content-type", contentType);
        }

        // Streaming handling
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
                stream: true
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
            const sanitized = sanitizeAnthropicResponse(responseBody);
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
                outputTokens
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
                errorMessage: responseBody?.error?.message || text
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
            errorMessage: message
        });

        return res.status(500).json({
            error: {
                type: "gateway_error",
                message
            }
        });
    }
});

// Fallback 404 middleware
app.use((_req, res) => {
    res.status(404).json({
        error: {
            type: "not_found",
            message: "Route not found"
        }
    });
});

app.listen(port, () => {
    console.log(`Railway AI Gateway v0.1.1 running on port ${port}`);
});