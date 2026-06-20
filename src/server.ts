import express from "express";
import cors from "cors";
import path from "path";
import { config } from "./config/env.js";
import { gatewayRouter } from "./routes/gateway.js";
import "./routes/qwen-metrics.js";

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

import { initDb } from "./utils/db.js";

const port = config.port;

// Use Gateway Router
app.use("/dashboard", express.static(path.join(process.cwd(), "public")));
app.use("/", gatewayRouter);
app.use("/v1", gatewayRouter);

// Fallback 404 middleware
app.use((_req, res) => {
    res.status(404).json({
        error: {
            type: "not_found",
            message: "Route not found"
        }
    });
});

initDb().then(() => {
    app.listen(port, () => {
        console.log(`Railway AI Gateway v0.2.0 running on port ${port}`);
    });
});
