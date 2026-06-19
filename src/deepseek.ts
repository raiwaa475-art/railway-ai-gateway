const DEEPSEEK_ANTHROPIC_BASE_URL =
    process.env.DEEPSEEK_ANTHROPIC_BASE_URL || "https://api.deepseek.com/anthropic";

const DEFAULT_MODEL = process.env.DEFAULT_MODEL || "deepseek-v4-flash";

export async function forwardToDeepSeekAnthropic(body: any, headers: HeadersInit) {
    const apiKey = process.env.DEEPSEEK_API_KEY;

    if (!apiKey) {
        throw new Error("DEEPSEEK_API_KEY is not configured");
    }

    const requestBody = {
        ...body,

        // สำคัญ: Claude Code อาจส่ง model เป็น claude-xxx
        // v0.1 บังคับให้ใช้ DeepSeek model ก่อน
        model: body.model?.startsWith("deepseek") ? body.model : DEFAULT_MODEL
    };

    return fetch(`${DEEPSEEK_ANTHROPIC_BASE_URL}/v1/messages`, {
        method: "POST",
        headers: {
            "content-type": "application/json",
            "authorization": `Bearer ${apiKey}`,
            "anthropic-version": "2023-06-01",
            ...headers
        },
        body: JSON.stringify(requestBody)
    });
}