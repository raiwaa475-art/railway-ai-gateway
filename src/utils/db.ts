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

        await pool.query(`
            CREATE TABLE IF NOT EXISTS qwen_agent_traces (
                id SERIAL PRIMARY KEY,
                request_id VARCHAR(255) UNIQUE,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                mode VARCHAR(50),
                user_intent TEXT,
                sanitized_messages JSONB,
                available_tool_names TEXT[],
                qwen_raw_output TEXT,
                fake_tool_json_detected BOOLEAN,
                fake_tool_json_converted BOOLEAN,
                requested_tool_name VARCHAR(255),
                normalized_tool_name VARCHAR(255),
                original_tool_args JSONB,
                repaired_tool_args JSONB,
                tool_args_repaired BOOLEAN,
                tool_validation_error TEXT,
                tool_retry_used BOOLEAN,
                tool_round_count INTEGER,
                tool_result_preview TEXT,
                final_answer_preview TEXT,
                edited_files TEXT[],
                build_status VARCHAR(50),
                success BOOLEAN,
                failure_reason TEXT,
                human_verdict VARCHAR(50) DEFAULT 'unknown'
            );
        `);

        // Create adapter tuning tables
        await pool.query(`
            CREATE TABLE IF NOT EXISTS qwen_adapter_rules (
                id SERIAL PRIMARY KEY,
                enabled BOOLEAN DEFAULT true,
                rule_type VARCHAR(50) NOT NULL,
                match_pattern VARCHAR(255) NOT NULL,
                replacement TEXT,
                description TEXT,
                hit_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        `);

        await pool.query(`
            CREATE TABLE IF NOT EXISTS qwen_prompt_profiles (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) UNIQUE NOT NULL,
                enabled BOOLEAN DEFAULT false,
                system_prompt TEXT NOT NULL,
                purpose TEXT,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        `);

        // Create background job tables
        await pool.query(`
            CREATE TABLE IF NOT EXISTS auto_coding_jobs (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status VARCHAR(50) NOT NULL,
                user_task TEXT NOT NULL,
                mode VARCHAR(50) NOT NULL,
                repo_path VARCHAR(500) NOT NULL,
                branch_name VARCHAR(255),
                model_worker VARCHAR(255) DEFAULT 'qwen-agent',
                controller_model VARCHAR(255),
                current_step INTEGER DEFAULT 0,
                max_steps INTEGER DEFAULT 12,
                success BOOLEAN,
                failure_reason TEXT,
                summary TEXT
            );
        `);

        await pool.query(`
            CREATE TABLE IF NOT EXISTS auto_coding_job_events (
                id SERIAL PRIMARY KEY,
                job_id INTEGER REFERENCES auto_coding_jobs(id) ON DELETE CASCADE,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                step INTEGER,
                event_type VARCHAR(100),
                payload JSONB
            );
        `);

        // Safely add columns to existing tables
        await pool.query(`ALTER TABLE qwen_agent_traces ADD COLUMN IF NOT EXISTS prompt_profile_name VARCHAR(255);`);
        await pool.query(`ALTER TABLE qwen_agent_traces ADD COLUMN IF NOT EXISTS prompt_profile_version VARCHAR(50);`);
        await pool.query(`ALTER TABLE qwen_agent_traces ADD COLUMN IF NOT EXISTS controller_plan TEXT;`);
        await pool.query(`ALTER TABLE qwen_agent_traces ADD COLUMN IF NOT EXISTS controller_review TEXT;`);
        await pool.query(`ALTER TABLE qwen_agent_traces ADD COLUMN IF NOT EXISTS qwen_worker_trace_ids VARCHAR(255)[];`);
        await pool.query(`ALTER TABLE qwen_agent_traces ADD COLUMN IF NOT EXISTS final_result TEXT;`);
        await pool.query(`ALTER TABLE qwen_agent_traces ADD COLUMN IF NOT EXISTS accepted BOOLEAN;`);
        await pool.query(`ALTER TABLE qwen_agent_traces ADD COLUMN IF NOT EXISTS duplicate_tool_call_blocked BOOLEAN;`);
        await pool.query(`ALTER TABLE qwen_agent_traces ADD COLUMN IF NOT EXISTS forced_final_after_successful_edit BOOLEAN;`);
        await pool.query(`ALTER TABLE qwen_agent_traces ADD COLUMN IF NOT EXISTS max_tool_rounds_reached BOOLEAN;`);
        await pool.query(`ALTER TABLE qwen_agent_traces ADD COLUMN IF NOT EXISTS intent_mode VARCHAR(50);`);
        await pool.query(`ALTER TABLE qwen_agent_traces ADD COLUMN IF NOT EXISTS allowed_tools TEXT[];`);
        await pool.query(`ALTER TABLE qwen_agent_traces ADD COLUMN IF NOT EXISTS blocked_by_intent_gate BOOLEAN;`);
        await pool.query(`ALTER TABLE qwen_agent_traces ADD COLUMN IF NOT EXISTS blocked_tool_name VARCHAR(255);`);

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

        // Insert default prompt profile if it does not exist
        const defaultPrompt = "You are connected to Claude Code tools.\nUse real tool_use calls when reading or editing files.\nDo not print JSON tool calls as text.\nUse Read before Edit unless you already have exact file content.\nPrefer small edits.\nAfter tool_result, continue the task or give final answer.\nDo not invent file paths.";
        await pool.query(`
            INSERT INTO qwen_prompt_profiles (name, enabled, system_prompt, purpose)
            VALUES ('qwen-agent-default', true, $1, 'Default system instructions for Qwen Agent')
            ON CONFLICT (name) DO NOTHING;
        `, [defaultPrompt]);

        // Memory Feature tables
        await pool.query(`
            CREATE TABLE IF NOT EXISTS repo_memories (
                id SERIAL PRIMARY KEY,
                repo_key VARCHAR(255) UNIQUE NOT NULL,
                summary TEXT,
                important_files JSONB DEFAULT '[]'::jsonb,
                risk_zones JSONB DEFAULT '[]'::jsonb,
                tech_stack JSONB DEFAULT '[]'::jsonb,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        `);

        await pool.query(`
            CREATE TABLE IF NOT EXISTS task_memories (
                id SERIAL PRIMARY KEY,
                repo_key VARCHAR(255) NOT NULL,
                task_summary TEXT,
                touched_files JSONB DEFAULT '[]'::jsonb,
                outcome VARCHAR(50),
                model_route VARCHAR(50),
                cost_thb NUMERIC(12, 6) DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        `);

        await pool.query(`
            CREATE TABLE IF NOT EXISTS failure_patterns (
                id SERIAL PRIMARY KEY,
                repo_key VARCHAR(255) NOT NULL,
                pattern_type VARCHAR(100) NOT NULL,
                failure_reason TEXT,
                examples JSONB DEFAULT '[]'::jsonb,
                hit_count INTEGER DEFAULT 1,
                last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (repo_key, pattern_type)
            );
        `);

        // Safely alter existing tables for memory key tracking
        await pool.query(`ALTER TABLE qwen_agent_traces ADD COLUMN IF NOT EXISTS repo_key VARCHAR(255);`);

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
