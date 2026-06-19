const DEEPSEEK_ANTHROPIC_BASE_URL =
    process.env.DEEPSEEK_ANTHROPIC_BASE_URL || "https://api.deepseek.com/anthropic";

const DEFAULT_MODEL = process.env.DEFAULT_MODEL || "deepseek-v4-flash";

export function resolveUpstreamModel(clientModel?: string): string {
    if (clientModel?.startsWith("deepseek")) {
        return clientModel;
    }
    return DEFAULT_MODEL;
}

export function sanitizeAnthropicResponse(data: any) {
    if (!data || !Array.isArray(data.content)) {
        return data;
    }
    return {
        ...data,
        content: data.content.filter((block: any) => block?.type === "text")
    };
}

export async function forwardToDeepSeekAnthropic(body: any, headers: HeadersInit) {
    const apiKey = process.env.DEEPSEEK_API_KEY;

    if (!apiKey) {
        throw new Error("DEEPSEEK_API_KEY is not configured");
    }

    const resolvedModel = resolveUpstreamModel(body.model);
    const requestBody = {
        ...body,
        model: resolvedModel
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