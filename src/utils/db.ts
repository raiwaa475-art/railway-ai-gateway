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
                saved_thb NUMERIC(12, 6) DEFAULT 0,
                input_cost_usd NUMERIC(12, 6) DEFAULT 0,
                input_cost_thb NUMERIC(12, 6) DEFAULT 0,
                output_cost_usd NUMERIC(12, 6) DEFAULT 0,
                output_cost_thb NUMERIC(12, 6) DEFAULT 0,
                saved_input_usd NUMERIC(12, 6) DEFAULT 0,
                saved_input_thb NUMERIC(12, 6) DEFAULT 0,
                saved_output_usd NUMERIC(12, 6) DEFAULT 0,
                saved_output_thb NUMERIC(12, 6) DEFAULT 0,
                qwen_draft_mode VARCHAR(32),
                qwen_draft_chars INTEGER DEFAULT 0,
                qwen_draft_weak BOOLEAN DEFAULT false,
                qwen_retry_used BOOLEAN DEFAULT false,
                qwen_patch_mode VARCHAR(32),
                qwen_patch_valid BOOLEAN,
                emitted_tool_use VARCHAR(64),
                fallback_reason VARCHAR(255),
                file_context_source VARCHAR(50),
                qwen_delegation_mode VARCHAR(50),
                direct_edit_eligible BOOLEAN,
                qwen_anchor_id VARCHAR(32),
                qwen_anchor_candidate_count INTEGER
            );
        `);

        await pool.query(`
            CREATE TABLE IF NOT EXISTS ai_providers (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                type VARCHAR(50) NOT NULL,
                server_url VARCHAR(500) NOT NULL,
                openai_base_url VARCHAR(500),
                native_base_url VARCHAR(500),
                api_key VARCHAR(500),
                default_model VARCHAR(255),
                enabled BOOLEAN DEFAULT true,
                timeout_ms INTEGER DEFAULT 120000,
                stream_enabled BOOLEAN DEFAULT false,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        `);

        await pool.query(`
            CREATE TABLE IF NOT EXISTS ai_models (
                id SERIAL PRIMARY KEY,
                provider_id INTEGER REFERENCES ai_providers(id) ON DELETE CASCADE,
                name VARCHAR(255) NOT NULL,
                size BIGINT,
                modified_at TIMESTAMP,
                raw_metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        `);

        // Safely add columns to existing tables
        await pool.query(`ALTER TABLE model_calls ADD COLUMN IF NOT EXISTS input_cost_usd NUMERIC(12, 6) DEFAULT 0;`);
        await pool.query(`ALTER TABLE model_calls ADD COLUMN IF NOT EXISTS input_cost_thb NUMERIC(12, 6) DEFAULT 0;`);
        await pool.query(`ALTER TABLE model_calls ADD COLUMN IF NOT EXISTS output_cost_usd NUMERIC(12, 6) DEFAULT 0;`);
        await pool.query(`ALTER TABLE model_calls ADD COLUMN IF NOT EXISTS output_cost_thb NUMERIC(12, 6) DEFAULT 0;`);
        await pool.query(`ALTER TABLE model_calls ADD COLUMN IF NOT EXISTS saved_input_usd NUMERIC(12, 6) DEFAULT 0;`);
        await pool.query(`ALTER TABLE model_calls ADD COLUMN IF NOT EXISTS saved_input_thb NUMERIC(12, 6) DEFAULT 0;`);
        await pool.query(`ALTER TABLE model_calls ADD COLUMN IF NOT EXISTS saved_output_usd NUMERIC(12, 6) DEFAULT 0;`);
        await pool.query(`ALTER TABLE model_calls ADD COLUMN IF NOT EXISTS saved_output_thb NUMERIC(12, 6) DEFAULT 0;`);
        await pool.query(`ALTER TABLE model_calls ADD COLUMN IF NOT EXISTS qwen_draft_mode VARCHAR(32);`);
        await pool.query(`ALTER TABLE model_calls ADD COLUMN IF NOT EXISTS qwen_draft_chars INTEGER DEFAULT 0;`);
        await pool.query(`ALTER TABLE model_calls ADD COLUMN IF NOT EXISTS qwen_draft_weak BOOLEAN DEFAULT false;`);
        await pool.query(`ALTER TABLE model_calls ADD COLUMN IF NOT EXISTS qwen_retry_used BOOLEAN DEFAULT false;`);
        await pool.query(`ALTER TABLE model_calls ADD COLUMN IF NOT EXISTS qwen_patch_mode VARCHAR(32);`);
        await pool.query(`ALTER TABLE model_calls ADD COLUMN IF NOT EXISTS qwen_patch_valid BOOLEAN;`);
        await pool.query(`ALTER TABLE model_calls ADD COLUMN IF NOT EXISTS deepseek_approval_approved BOOLEAN;`);
        await pool.query(`ALTER TABLE model_calls ADD COLUMN IF NOT EXISTS emitted_tool_use VARCHAR(64);`);
        await pool.query(`ALTER TABLE model_calls ADD COLUMN IF NOT EXISTS fallback_reason VARCHAR(255);`);
        await pool.query(`ALTER TABLE model_calls ADD COLUMN IF NOT EXISTS input_cache_hit_cost_usd NUMERIC(12, 6);`);
        await pool.query(`ALTER TABLE model_calls ADD COLUMN IF NOT EXISTS pricing_model VARCHAR(255);`);
        await pool.query(`ALTER TABLE model_calls ADD COLUMN IF NOT EXISTS pricing_source VARCHAR(255);`);
        await pool.query(`ALTER TABLE model_calls ADD COLUMN IF NOT EXISTS file_context_source VARCHAR(50);`);
        await pool.query(`ALTER TABLE model_calls ADD COLUMN IF NOT EXISTS qwen_delegation_mode VARCHAR(50);`);
        await pool.query(`ALTER TABLE model_calls ADD COLUMN IF NOT EXISTS direct_edit_eligible BOOLEAN;`);
        await pool.query(`ALTER TABLE model_calls ADD COLUMN IF NOT EXISTS qwen_anchor_id VARCHAR(32);`);
        await pool.query(`ALTER TABLE model_calls ADD COLUMN IF NOT EXISTS qwen_anchor_candidate_count INTEGER;`);

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
    inputCostUsd?: number;
    inputCostThb?: number;
    outputCostUsd?: number;
    outputCostThb?: number;
    savedUsd?: number;
    savedThb?: number;
    savedInputUsd?: number;
    savedInputThb?: number;
    savedOutputUsd?: number;
    savedOutputThb?: number;
    qwenDraftMode?: string;
    qwenDraftChars?: number;
    qwenDraftWeak?: boolean;
    qwenRetryUsed?: boolean;
    qwenPatchMode?: string;
    qwenPatchValid?: boolean;
    deepseekApprovalApproved?: boolean;
    emittedToolUse?: string;
    fallbackReason?: string;
    inputCacheHitCostUsd?: number;
    inputCacheMissCostUsd?: number;
    pricingModel?: string;
    pricingSource?: string;
    fileContextSource?: string;
    qwenDelegationMode?: string;
    directEditEligible?: boolean;
    qwenAnchorId?: string;
    qwenAnchorCandidateCount?: number;
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
            savedThb = 0,
            savedInputUsd = 0,
            savedInputThb = 0,
            savedOutputUsd = 0,
            savedOutputThb = 0,
            qwenDraftMode,
            qwenDraftChars = 0,
            qwenDraftWeak = false,
            qwenRetryUsed = false,
            qwenPatchMode,
            qwenPatchValid,
            deepseekApprovalApproved,
            emittedToolUse,
            fallbackReason,
            inputCacheHitCostUsd,
            inputCacheMissCostUsd,
            pricingModel,
            pricingSource,
            fileContextSource,
            qwenDelegationMode,
            directEditEligible,
            qwenAnchorId,
            qwenAnchorCandidateCount
        } = params;

        let inputCostUsd = params.inputCostUsd || 0;
        let inputCostThb = params.inputCostThb || 0;
        let outputCostUsd = params.outputCostUsd || 0;
        let outputCostThb = params.outputCostThb || 0;
        let cacheHit = params.cacheHitInputTokens || 0;
        let cacheMiss = params.cacheMissInputTokens ?? inputTokens;

        if (provider === "deepseek" && !params.inputCostUsd) {
            // Fallback default calculation if not passed
            const hitRate = config.deepseekInputCacheHitUsdPer1M / 1000000;
            const missRate = config.deepseekInputCacheMissUsdPer1M / 1000000;
            const outRate = config.deepseekOutputUsdPer1M / 1000000;

            inputCostUsd = (cacheHit * hitRate) + (cacheMiss * missRate);
            inputCostThb = inputCostUsd * config.usdThbRate;
            outputCostUsd = outputTokens * outRate;
            outputCostThb = outputCostUsd * config.usdThbRate;
        }

        const costUsd = inputCostUsd + outputCostUsd;
        const costThb = inputCostThb + outputCostThb;

        await pool.query(
            `INSERT INTO model_calls 
            (request_id, provider, model, input_tokens, output_tokens, cache_hit_input_tokens, cache_miss_input_tokens, latency_ms, cost_usd, cost_thb, saved_usd, saved_thb, input_cost_usd, input_cost_thb, output_cost_usd, output_cost_thb, saved_input_usd, saved_input_thb, saved_output_usd, saved_output_thb, qwen_draft_mode, qwen_draft_chars, qwen_draft_weak, qwen_retry_used, qwen_patch_mode, qwen_patch_valid, deepseek_approval_approved, emitted_tool_use, fallback_reason, input_cache_hit_cost_usd, input_cache_miss_cost_usd, pricing_model, pricing_source, file_context_source, qwen_delegation_mode, direct_edit_eligible, qwen_anchor_id, qwen_anchor_candidate_count) 
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38)`,
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
                savedThb,
                inputCostUsd,
                inputCostThb,
                outputCostUsd,
                outputCostThb,
                savedInputUsd,
                savedInputThb,
                savedOutputUsd,
                savedOutputThb,
                qwenDraftMode || null,
                qwenDraftChars,
                qwenDraftWeak,
                qwenRetryUsed,
                qwenPatchMode || null,
                qwenPatchValid ?? null,
                deepseekApprovalApproved ?? null,
                emittedToolUse || null,
                fallbackReason || null,
                inputCacheHitCostUsd !== undefined ? inputCacheHitCostUsd : null,
                inputCacheMissCostUsd !== undefined ? inputCacheMissCostUsd : null,
                pricingModel || null,
                pricingSource || null,
                fileContextSource || null,
                qwenDelegationMode || null,
                directEditEligible ?? null,
                qwenAnchorId || null,
                qwenAnchorCandidateCount ?? null
            ]
        );
    } catch (err) {
        console.error("Failed to insert model call:", err);
    }
}
