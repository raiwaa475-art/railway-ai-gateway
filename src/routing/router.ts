import { Provider } from "../providers/base.js";
import { providerRegistry } from "./registry.js";
import { SUPPORTED_MODELS, DEFAULT_PROVIDER_ID } from "../config/models.js";

export class ModelRouter {
    static resolve(clientModel?: string): Provider {
        const defaultProvider = providerRegistry.getProvider(DEFAULT_PROVIDER_ID);
        if (!defaultProvider) {
            throw new Error(`Default provider '${DEFAULT_PROVIDER_ID}' is not registered in ProviderRegistry.`);
        }

        if (!clientModel) {
            return defaultProvider;
        }

        const modelLower = clientModel.toLowerCase();

        // 1. Try exact match or alias match from configuration
        for (const modelDef of SUPPORTED_MODELS) {
            if (modelDef.id.toLowerCase() === modelLower) {
                const provider = providerRegistry.getProvider(modelDef.providerId);
                if (provider) return provider;
            }
            if (modelDef.aliases.some(alias => alias.toLowerCase() === modelLower)) {
                const provider = providerRegistry.getProvider(modelDef.providerId);
                if (provider) return provider;
            }
        }

        // 2. Try substring match (e.g. if the requested model contains "qwen", resolve to qwen provider)
        for (const modelDef of SUPPORTED_MODELS) {
            if (modelDef.aliases.some(alias => modelLower.includes(alias.toLowerCase()))) {
                const provider = providerRegistry.getProvider(modelDef.providerId);
                if (provider) return provider;
            }
        }

        // 3. Fallback to default
        return defaultProvider;
    }
}
