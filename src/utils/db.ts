import pg from "pg";
import { config } from "../config/env.js";

const { Pool } = pg;

export let pool: pg.Pool | null = null;

if (config.databaseUrl) {
    pool = new Pool({
        connectionString: config.databaseUrl,
        ssl: config.databaseUrl.includes("localhost") || config.databaseUrl.includes("127.0.0.1")
            ? false
            : { rejectUnauthorized: false }
    });
} else {
    console.warn("DATABASE_URL is not set. Database logging is disabled.");
}

export async function initDb() {
    if (!pool) return;
    try {
        await pool.query(`
            CREATE TABLE IF NOT EXISTS gateway_requests (
                id VARCHAR(255) PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                client_model VARCHAR(255),
                mode VARCHAR(255),
                stream BOOLEAN,
                status INTEGER,
                latency_ms INTEGER
            );
        `);
        await pool.query(`
            CREATE TABLE IF NOT EXISTS model_calls (
                id SERIAL PRIMARY KEY,
                request_id VARCHAR(255) REFERENCES gateway_requests(id) ON DELETE CASCADE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                provider VARCHAR(255),
                model VARCHAR(255),
                input_tokens INTEGER,
                output_tokens INTEGER,
                cache_hit_input_tokens INTEGER,
                cache_miss_input_tokens INTEGER,
                latency_ms INTEGER,
                cost_usd NUMERIC(12, 6),
                cost_thb NUMERIC(12, 6),
                saved_usd NUMERIC(12, 6) DEFAULT 0,
                saved_thb NUMERIC(12, 6) DEFAULT 0
            );
        `);
        console.log("Database tables initialized successfully.");
    } catch (err) {
        console.error("Failed to initialize database tables:", err);
    }
}

export async function insertGatewayRequest(id: string, clientModel: string, mode: string, stream: boolean) {
    if (!pool) return;
    try {
        await pool.query(
            "INSERT INTO gateway_requests (id, client_model, mode, stream) VALUES ($1, $2, $3, $4)",
            [id, clientModel, mode, stream]
        );
    } catch (err) {
        console.error("Failed to insert gateway request:", err);
    }
}

export async function updateGatewayRequest(id: string, status: number, latencyMs: number) {
    if (!pool) return;
    try {
        await pool.query(
            "UPDATE gateway_requests SET status = $1, latency_ms = $2 WHERE id = $3",
            [status, latencyMs, id]
        );
    } catch (err) {
        console.error("Failed to update gateway request:", err);
    }
}

export async function insertModelCall(params: {
    requestId: string;
    provider: string;
    model: string;
    inputTokens: number;
    outputTokens: number;
    cacheHitInputTokens?: number;
    cacheMissInputTokens?: number;
    latencyMs: number;
    savedUsd?: number;
    savedThb?: number;
}) {
    if (!pool) return;
    try {
        const {
            requestId,
            provider,
            model,
            inputTokens,
            outputTokens,
            latencyMs,
            savedUsd = 0,
            savedThb = 0
        } = params;

        let costUsd = 0;
        let costThb = 0;
        let cacheHit = params.cacheHitInputTokens || 0;
        let cacheMiss = params.cacheMissInputTokens ?? inputTokens;

        if (provider === "deepseek") {
            const hitRate = config.deepseekInputCacheHitUsdPer1M / 1000000;
            const missRate = config.deepseekInputCacheMissUsdPer1M / 1000000;
            const outRate = config.deepseekOutputUsdPer1M / 1000000;

            costUsd = (cacheHit * hitRate) + (cacheMiss * missRate) + (outputTokens * outRate);
            costThb = costUsd * config.usdThbRate;
        }

        await pool.query(
            `INSERT INTO model_calls 
            (request_id, provider, model, input_tokens, output_tokens, cache_hit_input_tokens, cache_miss_input_tokens, latency_ms, cost_usd, cost_thb, saved_usd, saved_thb) 
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)`,
            [
                requestId,
                provider,
                model,
                inputTokens,
                outputTokens,
                cacheHit,
                cacheMiss,
                latencyMs,
                costUsd,
                costThb,
                savedUsd,
                savedThb
            ]
        );
    } catch (err) {
        console.error("Failed to insert model call:", err);
    }
}
