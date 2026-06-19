import fs from "fs";
import path from "path";

try {
    if (typeof (process as any).loadEnvFile === "function") {
        (process as any).loadEnvFile();
    } else {
        const envPath = path.resolve(process.cwd(), ".env");
        if (fs.existsSync(envPath)) {
            const envContent = fs.readFileSync(envPath, "utf-8");
            for (const line of envContent.split("\n")) {
                const trimmed = line.trim();
                if (!trimmed || trimmed.startsWith("#")) continue;
                const index = trimmed.indexOf("=");
                if (index !== -1) {
                    const key = trimmed.substring(0, index).trim();
                    const val = trimmed.substring(index + 1).replace(/^['"]|['"]$/g, "").trim();
                    process.env[key] = val;
                }
            }
        }
    }
} catch (e) {
    // Ignore error if .env doesn't exist (e.g. on Railway production)
}

export const config = {
    port: Number(process.env.PORT || 3000),
    gatewayApiKey: process.env.GATEWAY_API_KEY || "local-dev-key",
    deepseekApiKey: process.env.DEEPSEEK_API_KEY || "",
    deepseekAnthropicBaseUrl: process.env.DEEPSEEK_ANTHROPIC_BASE_URL || "https://api.deepseek.com/anthropic",
    defaultModel: process.env.DEFAULT_MODEL || "deepseek-v4-flash",
    qwenLocalApiUrl: process.env.QWEN_LOCAL_API_URL || "http://localhost:11434/v1",
    qwenLocalModel: process.env.QWEN_LOCAL_MODEL || "qwen2.5-coder-3b-instruct",
    databaseUrl: process.env.DATABASE_URL || "",
    usdThbRate: Number(process.env.USD_THB_RATE || 35.0),
    deepseekInputCacheHitUsdPer1M: Number(process.env.DEEPSEEK_INPUT_CACHE_HIT_USD_PER_1M || 0.07),
    deepseekInputCacheMissUsdPer1M: Number(process.env.DEEPSEEK_INPUT_CACHE_MISS_USD_PER_1M || 0.27),
    deepseekOutputUsdPer1M: Number(process.env.DEEPSEEK_OUTPUT_USD_PER_1M || 1.10),
    gatewayAdminKey: process.env.GATEWAY_ADMIN_KEY || "",
    allowPrivateProviderUrl: process.env.ALLOW_PRIVATE_PROVIDER_URL === "true",
    defaultProviderId: process.env.DEFAULT_PROVIDER_ID || "",
    qwenLocalMaxTokens: Number(process.env.QWEN_LOCAL_MAX_TOKENS || 32000),
    qwenLocalTimeoutMs: Number(process.env.QWEN_LOCAL_TIMEOUT_MS || 180000)
};
