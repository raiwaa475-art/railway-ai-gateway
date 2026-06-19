import { config } from "../config/env.js";

export type DeepSeekModelPricing = {
    inputCacheHitUsdPer1M: number;
    inputCacheMissUsdPer1M: number;
    outputUsdPer1M: number;
};

export const DEEPSEEK_PRICING: Record<string, DeepSeekModelPricing> = {
    "deepseek-v4-flash": {
        inputCacheHitUsdPer1M: 0.0028,
        inputCacheMissUsdPer1M: 0.14,
        outputUsdPer1M: 0.28
    },
    "deepseek-v4-pro": {
        inputCacheHitUsdPer1M: 0.003625,
        inputCacheMissUsdPer1M: 0.435,
        outputUsdPer1M: 0.87
    },
    "deepseek-chat": {
        inputCacheHitUsdPer1M: 0.0028,
        inputCacheMissUsdPer1M: 0.14,
        outputUsdPer1M: 0.28
    },
    "deepseek-reasoner": {
        inputCacheHitUsdPer1M: 0.0028,
        inputCacheMissUsdPer1M: 0.14,
        outputUsdPer1M: 0.28
    }
};

export function getDeepSeekPricing(model: string): DeepSeekModelPricing {
    const normalized = String(model || "").toLowerCase();

    // Check env overrides first
    // Note: DEEPSEEK_INPUT_CACHE_HIT_USD_PER_1M etc are global overrides.
    // We also support model-specific overrides:
    // DEEPSEEK_V4_FLASH_...
    // DEEPSEEK_V4_PRO_...
    if (normalized.includes("pro") || normalized.includes("deepseek-v4-pro")) {
        return {
            inputCacheHitUsdPer1M: Number(process.env.DEEPSEEK_V4_PRO_INPUT_CACHE_HIT_USD_PER_1M || process.env.DEEPSEEK_INPUT_CACHE_HIT_USD_PER_1M || DEEPSEEK_PRICING["deepseek-v4-pro"].inputCacheHitUsdPer1M),
            inputCacheMissUsdPer1M: Number(process.env.DEEPSEEK_V4_PRO_INPUT_CACHE_MISS_USD_PER_1M || process.env.DEEPSEEK_INPUT_CACHE_MISS_USD_PER_1M || DEEPSEEK_PRICING["deepseek-v4-pro"].inputCacheMissUsdPer1M),
            outputUsdPer1M: Number(process.env.DEEPSEEK_V4_PRO_OUTPUT_USD_PER_1M || process.env.DEEPSEEK_OUTPUT_USD_PER_1M || DEEPSEEK_PRICING["deepseek-v4-pro"].outputUsdPer1M)
        };
    }

    // Default to flash / chat / reasoner
    return {
        inputCacheHitUsdPer1M: Number(process.env.DEEPSEEK_V4_FLASH_INPUT_CACHE_HIT_USD_PER_1M || process.env.DEEPSEEK_INPUT_CACHE_HIT_USD_PER_1M || DEEPSEEK_PRICING["deepseek-v4-flash"].inputCacheHitUsdPer1M),
        inputCacheMissUsdPer1M: Number(process.env.DEEPSEEK_V4_FLASH_INPUT_CACHE_MISS_USD_PER_1M || process.env.DEEPSEEK_INPUT_CACHE_MISS_USD_PER_1M || DEEPSEEK_PRICING["deepseek-v4-flash"].inputCacheMissUsdPer1M),
        outputUsdPer1M: Number(process.env.DEEPSEEK_V4_FLASH_OUTPUT_USD_PER_1M || process.env.DEEPSEEK_OUTPUT_USD_PER_1M || DEEPSEEK_PRICING["deepseek-v4-flash"].outputUsdPer1M)
    };
}

export function extractDeepSeekUsage(responseBody: any) {
    if (!responseBody) {
        return {
            inputTokens: 0,
            outputTokens: 0,
            cacheHitInputTokens: 0,
            cacheMissInputTokens: 0
        };
    }

    const usage = responseBody.usage || {};
    const inputTokens = usage.input_tokens ?? usage.prompt_tokens ?? 0;
    const outputTokens = usage.output_tokens ?? usage.completion_tokens ?? 0;

    const cacheHitInputTokens = usage.prompt_cache_hit_tokens
        ?? usage.cache_read_input_tokens
        ?? usage.cache_hit_input_tokens
        ?? 0;

    const cacheMissInputTokens = usage.prompt_cache_miss_tokens
        ?? usage.cache_creation_input_tokens
        ?? Math.max(0, inputTokens - cacheHitInputTokens);

    return {
        inputTokens,
        outputTokens,
        cacheHitInputTokens,
        cacheMissInputTokens
    };
}

export function calculateDeepSeekCost(
    model: string,
    usage: { inputTokens: number; outputTokens: number; cacheHitInputTokens: number; cacheMissInputTokens: number }
) {
    const pricing = getDeepSeekPricing(model);
    const inputCacheHitCost = (usage.cacheHitInputTokens / 1000000) * pricing.inputCacheHitUsdPer1M;
    const inputCacheMissCost = (usage.cacheMissInputTokens / 1000000) * pricing.inputCacheMissUsdPer1M;
    const outputCost = (usage.outputTokens / 1000000) * pricing.outputUsdPer1M;
    const totalCost = inputCacheHitCost + inputCacheMissCost + outputCost;
    const totalCostThb = totalCost * config.usdThbRate;

    return {
        pricingModel: model,
        pricingSource: "deepseek_api_docs_2026",
        inputCacheHitCostUsd: inputCacheHitCost,
        inputCacheMissCostUsd: inputCacheMissCost,
        inputCostUsd: inputCacheHitCost + inputCacheMissCost,
        inputCostThb: (inputCacheHitCost + inputCacheMissCost) * config.usdThbRate,
        outputCostUsd: outputCost,
        outputCostThb: outputCost * config.usdThbRate,
        totalCostUsd: totalCost,
        totalCostThb
    };
}
