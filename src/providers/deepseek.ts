import { config } from "../config/env.js";
import { Provider } from "./base.js";

export class DeepSeekProvider implements Provider {
    id = "deepseek";

    resolveUpstreamModel(clientModel?: string): string {
        if (clientModel?.startsWith("deepseek")) {
            return clientModel;
        }
        return config.defaultModel;
    }

    async handleRequest(body: any, headers: Record<string, string>): Promise<Response> {
        const apiKey = config.deepseekApiKey;

        if (!apiKey) {
            throw new Error("DEEPSEEK_API_KEY is not configured");
        }

        const resolvedModel = this.resolveUpstreamModel(body.model);
        const requestBody = {
            ...body,
            model: resolvedModel
        };

        return fetch(`${config.deepseekAnthropicBaseUrl}/v1/messages`, {
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
