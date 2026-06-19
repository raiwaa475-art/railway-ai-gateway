import express from "express";
import cors from "cors";
import { authMiddleware } from "./auth.js";
import { forwardToDeepSeekAnthropic } from "./deepseek.js";

const app = express();

app.use(cors());
app.use(express.json({ limit: "20mb" }));

const port = Number(process.env.PORT || 3000);

app.get("/health", (_req, res) => {
    res.json({
        ok: true,
        service: "railway-ai-gateway",
        version: "0.1.0",
        provider: "deepseek-anthropic"
    });
});

app.get("/v1/models", authMiddleware, (_req, res) => {
    res.json({
        data: [
            {
                id: "deepseek-v4-flash",
                type: "model",
                display_name: "DeepSeek V4 Flash"
            }
        ]
    });
});

app.post("/v1/messages", authMiddleware, async (req, res) => {
    try {
        const upstream = await forwardToDeepSeekAnthropic(req.body, {
            "user-agent": req.header("user-agent") || "railway-ai-gateway"
        });

        res.status(upstream.status);

        const contentType = upstream.headers.get("content-type");
        if (contentType) {
            res.setHeader("content-type", contentType);
        }

        // รองรับ streaming ถ้า Claude Code ส่ง stream: true
        if (req.body?.stream && upstream.body) {
            res.setHeader("cache-control", "no-cache");
            res.setHeader("connection", "keep-alive");

            const reader = upstream.body.getReader();

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                res.write(Buffer.from(value));
            }

            return res.end();
        }

        const text = await upstream.text();
        return res.send(text);
    } catch (error) {
        const message = error instanceof Error ? error.message : "Unknown error";

        return res.status(500).json({
            error: {
                type: "gateway_error",
                message
            }
        });
    }
});

app.listen(port, () => {
    console.log(`Railway AI Gateway v0.1 running on port ${port}`);
});