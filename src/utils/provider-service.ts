import { config } from "../config/env.js";
import { AiProviderConfig, AiModelConfig } from "./provider-store.js";

export function isPrivateUrl(urlString: string): boolean {
    try {
        const url = new URL(urlString);
        const hostname = url.hostname.toLowerCase();
        
        if (hostname === "localhost" || hostname === "127.0.0.1" || hostname === "::1") {
            return true;
        }
        
        const privateIpRegex = /^(?:10\.\d+\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+|192\.168\.\d+\.\d+)$/;
        if (privateIpRegex.test(hostname)) {
            return true;
        }
    } catch {
        return true;
    }
    return false;
}

export function validateProviderUrl(urlString: string): { ok: boolean; message?: string } {
    if (!urlString.startsWith("http://") && !urlString.startsWith("https://")) {
        return { ok: false, message: "URL must start with http:// or https://" };
    }

    if (!config.allowPrivateProviderUrl && isPrivateUrl(urlString)) {
        return { ok: false, message: "Private or loopback URLs are blocked by default. Enable ALLOW_PRIVATE_PROVIDER_URL=true to allow." };
    }

    return { ok: true };
}

export class ProviderService {
    static async testConnection(params: {
        type: string;
        serverUrl: string;
        apiKey?: string;
    }): Promise<{ ok: boolean; message: string }> {
        const validation = validateProviderUrl(params.serverUrl);
        if (!validation.ok) {
            return { ok: false, message: validation.message || "Invalid URL" };
        }

        const serverUrl = params.serverUrl.trim();
        const type = params.type;
        const apiKey = params.apiKey || "";

        let testUrl = "";
        let headers: Record<string, string> = { "content-type": "application/json" };

        if (type === "ollama") {
            testUrl = `${serverUrl}/api/tags`;
        } else if (type === "lmstudio" || type === "openai_compatible" || type === "deepseek") {
            testUrl = `${serverUrl}/v1/models`;
            if (apiKey) {
                headers["authorization"] = `Bearer ${apiKey}`;
            }
        } else {
            return { ok: false, message: `Unsupported provider type: ${type}` };
        }

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 seconds timeout for test

        try {
            const res = await fetch(testUrl, {
                method: "GET",
                headers,
                signal: controller.signal
            });
            clearTimeout(timeoutId);

            if (res.ok) {
                return { ok: true, message: "Connection successful" };
            } else {
                const text = await res.text();
                return { ok: false, message: `Connection failed with status ${res.status}: ${text.slice(0, 100)}` };
            }
        } catch (e: any) {
            clearTimeout(timeoutId);
            const isTimeout = e.name === "AbortError";
            const errStr = isTimeout ? "Connection timed out (10s)" : (e.message || "Network error");
            return { ok: false, message: `Could not connect to provider: ${errStr}` };
        }
    }

    static async fetchModels(provider: AiProviderConfig): Promise<Omit<AiModelConfig, "providerId">[]> {
        const serverUrl = provider.serverUrl.trim();
        const type = provider.type;
        const apiKey = provider.apiKey || "";

        let fetchUrl = "";
        let headers: Record<string, string> = { "content-type": "application/json" };

        if (type === "ollama") {
            fetchUrl = `${serverUrl}/api/tags`;
        } else {
            fetchUrl = `${provider.openaiBaseUrl || (serverUrl + "/v1")}/models`;
            if (apiKey) {
                headers["authorization"] = `Bearer ${apiKey}`;
            }
        }

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 15000); // 15 seconds timeout

        try {
            const res = await fetch(fetchUrl, {
                method: "GET",
                headers,
                signal: controller.signal
            });
            clearTimeout(timeoutId);

            if (!res.ok) {
                throw new Error(`Failed to fetch models: status ${res.status}`);
            }

            const data = await res.json();
            if (type === "ollama") {
                const models = data.models || [];
                return models.map((m: any) => ({
                    name: m.name,
                    size: m.size ? Number(m.size) : undefined,
                    modifiedAt: m.modified_at ? new Date(m.modified_at) : undefined,
                    rawMetadata: m
                }));
            } else {
                const models = data.data || [];
                return models.map((m: any) => ({
                    name: m.id,
                    rawMetadata: m
                }));
            }
        } catch (e: any) {
            clearTimeout(timeoutId);
            throw new Error(`Model sync failed: ${e.message}`);
        }
    }

    static async pullModel(provider: AiProviderConfig, model: string): Promise<{ ok: boolean; message: string }> {
        if (provider.type !== "ollama") {
            return { ok: false, message: "Pull model is only supported for Ollama" };
        }

        const fetchUrl = `${provider.nativeBaseUrl || (provider.serverUrl + "/api")}/pull`;
        const controller = new AbortController();
        // Allow up to 5 minutes for pull model download
        const timeoutId = setTimeout(() => controller.abort(), 300000);

        try {
            const res = await fetch(fetchUrl, {
                method: "POST",
                headers: { "content-type": "application/json" },
                body: JSON.stringify({
                    model,
                    stream: false
                }),
                signal: controller.signal
            });
            clearTimeout(timeoutId);

            if (!res.ok) {
                const text = await res.text();
                return { ok: false, message: `Ollama pull failed with status ${res.status}: ${text.slice(0, 150)}` };
            }

            return { ok: true, message: `Model ${model} pulled successfully` };
        } catch (e: any) {
            clearTimeout(timeoutId);
            const isTimeout = e.name === "AbortError";
            const errStr = isTimeout ? "Pull request timed out (300s)" : (e.message || "Network error");
            return { ok: false, message: `Pull model error: ${errStr}` };
        }
    }
}
