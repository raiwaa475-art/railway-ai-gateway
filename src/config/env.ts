export const config = {
    port: Number(process.env.PORT || 3000),
    gatewayApiKey: process.env.GATEWAY_API_KEY || "local-dev-key",
    deepseekApiKey: process.env.DEEPSEEK_API_KEY || "",
    deepseekAnthropicBaseUrl: process.env.DEEPSEEK_ANTHROPIC_BASE_URL || "https://api.deepseek.com/anthropic",
    defaultModel: process.env.DEFAULT_MODEL || "deepseek-v4-flash",
    qwenLocalApiUrl: process.env.QWEN_LOCAL_API_URL || "http://localhost:11434/v1"
};
