import type { Request, Response, NextFunction } from "express";
import { config } from "./config/env.js";

export function authMiddleware(req: Request, res: Response, next: NextFunction) {
    const expectedKey = config.gatewayApiKey;

    if (!expectedKey) {
        return res.status(500).json({
            error: {
                type: "server_error",
                message: "GATEWAY_API_KEY is not configured"
            }
        });
    }

    const auth = req.header("authorization") || req.header("x-api-key") || "";
    const token = auth.replace(/^Bearer\s+/i, "").trim();

    if (token !== expectedKey) {
        return res.status(401).json({
            error: {
                type: "authentication_error",
                message: "Invalid gateway API key"
            }
        });
    }

    next();
}