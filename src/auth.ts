import type { Request, Response, NextFunction } from "express";

export function authMiddleware(req: Request, res: Response, next: NextFunction) {
    const expectedKey = process.env.GATEWAY_API_KEY;

    if (!expectedKey) {
        return res.status(500).json({
            error: {
                type: "server_error",
                message: "GATEWAY_API_KEY is not configured"
            }
        });
    }

    const auth = req.header("authorization") || "";
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