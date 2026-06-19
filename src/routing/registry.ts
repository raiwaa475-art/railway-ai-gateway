import { Provider } from "../providers/base.js";
import { DeepSeekProvider } from "../providers/deepseek.js";
import { QwenLocalProvider } from "../providers/qwen-local.js";

class ProviderRegistry {
    private providers: Map<string, Provider> = new Map();

    constructor() {
        this.register(new DeepSeekProvider());
        this.register(new QwenLocalProvider());
    }

    register(provider: Provider) {
        this.providers.set(provider.id, provider);
    }

    getProvider(providerId: string): Provider | undefined {
        return this.providers.get(providerId);
    }
}

export const providerRegistry = new ProviderRegistry();
