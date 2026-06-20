export interface ModelDefinition {
    id: string;
    displayName: string;
    providerId: string;
    aliases: string[];
}

export const SUPPORTED_MODELS: ModelDefinition[] = [
    {
        id: "deepseek-v4-flash",
        displayName: "DeepSeek V4 Flash",
        providerId: "deepseek",
        aliases: ["deepseek-v4-flash", "deepseek"]
    },
    {
        id: "qwen-local",
        displayName: "Qwen Local (Ollama/Tunnel)",
        providerId: "qwen-local",
        aliases: ["qwen-local", "qwen"]
    },
    {
        id: "hybrid-flow",
        displayName: "Hybrid Flow (DeepSeek + Qwen Local)",
        providerId: "hybrid",
        aliases: ["hybrid-flow", "qwen-smart"]
    },
    {
        id: "qwen-only-low-risk",
        displayName: "Qwen Only Low Risk",
        providerId: "hybrid",
        aliases: ["qwen-only-low-risk"]
    }
];

export const DEFAULT_PROVIDER_ID = "deepseek";
