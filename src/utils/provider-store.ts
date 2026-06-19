import fs from "fs";
import path from "path";
import { pool } from "./db.js";

export interface AiProviderConfig {
    id: number;
    name: string;
    type: "ollama" | "lmstudio" | "deepseek" | "openai_compatible";
    serverUrl: string;
    openaiBaseUrl: string;
    nativeBaseUrl: string | null;
    apiKey?: string;
    defaultModel?: string;
    enabled: boolean;
    timeoutMs: number;
    streamEnabled: boolean;
    createdAt?: Date;
    updatedAt?: Date;
}

export interface AiModelConfig {
    id?: number;
    providerId: number;
    name: string;
    size?: number;
    modifiedAt?: string | Date;
    rawMetadata?: any;
}

const PROVIDERS_FILE = path.join(process.cwd(), "providers_config.json");
const MODELS_FILE = path.join(process.cwd(), "models_config.json");

function readJsonFile(filePath: string): any[] {
    try {
        if (fs.existsSync(filePath)) {
            const data = fs.readFileSync(filePath, "utf-8");
            return JSON.parse(data);
        }
    } catch (e) {
        console.error(`Failed to read file ${filePath}:`, e);
    }
    return [];
}

function writeJsonFile(filePath: string, data: any[]): void {
    try {
        fs.writeFileSync(filePath, JSON.stringify(data, null, 2), "utf-8");
    } catch (e) {
        console.error(`Failed to write file ${filePath}:`, e);
    }
}

export class ProviderStore {
    static async getAllProviders(): Promise<AiProviderConfig[]> {
        if (pool) {
            const res = await pool.query("SELECT * FROM ai_providers ORDER BY id ASC");
            return res.rows.map(row => ({
                id: row.id,
                name: row.name,
                type: row.type,
                serverUrl: row.server_url,
                openaiBaseUrl: row.openai_base_url,
                nativeBaseUrl: row.native_base_url,
                apiKey: row.api_key,
                defaultModel: row.default_model,
                enabled: row.enabled,
                timeoutMs: row.timeout_ms,
                streamEnabled: row.stream_enabled,
                createdAt: row.created_at,
                updatedAt: row.updated_at
            }));
        } else {
            return readJsonFile(PROVIDERS_FILE);
        }
    }

    static async getProviderById(id: number): Promise<AiProviderConfig | null> {
        if (pool) {
            const res = await pool.query("SELECT * FROM ai_providers WHERE id = $1", [id]);
            if (res.rows.length === 0) return null;
            const row = res.rows[0];
            return {
                id: row.id,
                name: row.name,
                type: row.type,
                serverUrl: row.server_url,
                openaiBaseUrl: row.openai_base_url,
                nativeBaseUrl: row.native_base_url,
                apiKey: row.api_key,
                defaultModel: row.default_model,
                enabled: row.enabled,
                timeoutMs: row.timeout_ms,
                streamEnabled: row.stream_enabled,
                createdAt: row.created_at,
                updatedAt: row.updated_at
            };
        } else {
            const list = readJsonFile(PROVIDERS_FILE);
            return list.find(p => p.id === id) || null;
        }
    }

    static async saveProvider(provider: Omit<AiProviderConfig, "id"> & { id?: number }): Promise<AiProviderConfig> {
        let serverUrl = provider.serverUrl.trim();
        if (serverUrl.endsWith("/")) {
            serverUrl = serverUrl.slice(0, -1);
        }
        const type = provider.type;
        let openaiBaseUrl = provider.openaiBaseUrl || "";
        let nativeBaseUrl = provider.nativeBaseUrl || null;

        if (type === "ollama") {
            openaiBaseUrl = `${serverUrl}/v1`;
            nativeBaseUrl = `${serverUrl}/api`;
        } else if (type === "lmstudio") {
            openaiBaseUrl = `${serverUrl}/v1`;
            nativeBaseUrl = serverUrl;
        } else if (type === "openai_compatible") {
            openaiBaseUrl = `${serverUrl}/v1`;
            nativeBaseUrl = serverUrl;
        } else if (type === "deepseek") {
            openaiBaseUrl = serverUrl || "https://api.deepseek.com/v1";
            nativeBaseUrl = serverUrl || "https://api.deepseek.com";
        }

        const timeoutMs = provider.timeoutMs || 120000;
        const streamEnabled = provider.streamEnabled || false;
        const enabled = provider.enabled !== false;

        if (pool) {
            if (provider.id) {
                const res = await pool.query(
                    `UPDATE ai_providers 
                     SET name = $1, type = $2, server_url = $3, openai_base_url = $4, native_base_url = $5, 
                         api_key = COALESCE($6, api_key), default_model = $7, enabled = $8, timeout_ms = $9, 
                         stream_enabled = $10, updated_at = CURRENT_TIMESTAMP
                     WHERE id = $11 RETURNING *`,
                    [
                        provider.name,
                        type,
                        serverUrl,
                        openaiBaseUrl,
                        nativeBaseUrl,
                        provider.apiKey || null,
                        provider.defaultModel || null,
                        enabled,
                        timeoutMs,
                        streamEnabled,
                        provider.id
                    ]
                );
                const row = res.rows[0];
                return {
                    id: row.id,
                    name: row.name,
                    type: row.type,
                    serverUrl: row.server_url,
                    openaiBaseUrl: row.openai_base_url,
                    nativeBaseUrl: row.native_base_url,
                    apiKey: row.api_key,
                    defaultModel: row.default_model,
                    enabled: row.enabled,
                    timeoutMs: row.timeout_ms,
                    streamEnabled: row.stream_enabled
                };
            } else {
                const res = await pool.query(
                    `INSERT INTO ai_providers 
                     (name, type, server_url, openai_base_url, native_base_url, api_key, default_model, enabled, timeout_ms, stream_enabled)
                     VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10) RETURNING *`,
                    [
                        provider.name,
                        type,
                        serverUrl,
                        openaiBaseUrl,
                        nativeBaseUrl,
                        provider.apiKey || null,
                        provider.defaultModel || null,
                        enabled,
                        timeoutMs,
                        streamEnabled
                    ]
                );
                const row = res.rows[0];
                return {
                    id: row.id,
                    name: row.name,
                    type: row.type,
                    serverUrl: row.server_url,
                    openaiBaseUrl: row.openai_base_url,
                    nativeBaseUrl: row.native_base_url,
                    apiKey: row.api_key,
                    defaultModel: row.default_model,
                    enabled: row.enabled,
                    timeoutMs: row.timeout_ms,
                    streamEnabled: row.stream_enabled
                };
            }
        } else {
            const list = readJsonFile(PROVIDERS_FILE);
            if (provider.id) {
                const idx = list.findIndex(p => p.id === provider.id);
                if (idx !== -1) {
                    const existing = list[idx];
                    const updated = {
                        ...existing,
                        name: provider.name,
                        type,
                        serverUrl,
                        openaiBaseUrl,
                        nativeBaseUrl,
                        apiKey: provider.apiKey !== undefined ? provider.apiKey : existing.apiKey,
                        defaultModel: provider.defaultModel || existing.defaultModel,
                        enabled,
                        timeoutMs,
                        streamEnabled,
                        updatedAt: new Date()
                    };
                    list[idx] = updated;
                    writeJsonFile(PROVIDERS_FILE, list);
                    return updated;
                }
            }
            const newId = list.length > 0 ? Math.max(...list.map(p => p.id)) + 1 : 1;
            const newProvider: AiProviderConfig = {
                id: newId,
                name: provider.name,
                type,
                serverUrl,
                openaiBaseUrl,
                nativeBaseUrl,
                apiKey: provider.apiKey || "",
                defaultModel: provider.defaultModel || "",
                enabled,
                timeoutMs,
                streamEnabled,
                createdAt: new Date(),
                updatedAt: new Date()
            };
            list.push(newProvider);
            writeJsonFile(PROVIDERS_FILE, list);
            return newProvider;
        }
    }

    static async deleteProvider(id: number): Promise<boolean> {
        if (pool) {
            const res = await pool.query("DELETE FROM ai_providers WHERE id = $1", [id]);
            return (res.rowCount ?? 0) > 0;
        } else {
            const list = readJsonFile(PROVIDERS_FILE);
            const initialLen = list.length;
            const filtered = list.filter(p => p.id !== id);
            writeJsonFile(PROVIDERS_FILE, filtered);

            const models = readJsonFile(MODELS_FILE);
            const filteredModels = models.filter(m => m.providerId !== id);
            writeJsonFile(MODELS_FILE, filteredModels);

            return filtered.length < initialLen;
        }
    }

    static async syncModels(providerId: number, models: Omit<AiModelConfig, "providerId">[]): Promise<AiModelConfig[]> {
        if (pool) {
            await pool.query("DELETE FROM ai_models WHERE provider_id = $1", [providerId]);
            const savedModels: AiModelConfig[] = [];
            for (const model of models) {
                const res = await pool.query(
                    `INSERT INTO ai_models (provider_id, name, size, modified_at, raw_metadata) 
                     VALUES ($1, $2, $3, $4, $5) RETURNING *`,
                    [
                        providerId,
                        model.name,
                        model.size || null,
                        model.modifiedAt || null,
                        model.rawMetadata ? JSON.stringify(model.rawMetadata) : null
                    ]
                );
                const row = res.rows[0];
                savedModels.push({
                    id: row.id,
                    providerId: row.provider_id,
                    name: row.name,
                    size: row.size ? Number(row.size) : undefined,
                    modifiedAt: row.modified_at,
                    rawMetadata: row.raw_metadata
                });
            }
            return savedModels;
        } else {
            const allModels = readJsonFile(MODELS_FILE);
            const filtered = allModels.filter(m => m.providerId !== providerId);

            const savedModels: AiModelConfig[] = models.map((m, idx) => ({
                id: idx + 1 + (filtered.length > 0 ? Math.max(...filtered.map(x => x.id || 0)) : 0),
                providerId,
                name: m.name,
                size: m.size,
                modifiedAt: m.modifiedAt,
                rawMetadata: m.rawMetadata
            }));

            filtered.push(...savedModels);
            writeJsonFile(MODELS_FILE, filtered);
            return savedModels;
        }
    }

    static async setDefaultModel(providerId: number, modelName: string): Promise<boolean> {
        if (pool) {
            const res = await pool.query(
                "UPDATE ai_providers SET default_model = $1 WHERE id = $2",
                [modelName, providerId]
            );
            return (res.rowCount ?? 0) > 0;
        } else {
            const list = readJsonFile(PROVIDERS_FILE);
            const idx = list.findIndex(p => p.id === providerId);
            if (idx !== -1) {
                list[idx].defaultModel = modelName;
                list[idx].updatedAt = new Date();
                writeJsonFile(PROVIDERS_FILE, list);
                return true;
            }
            return false;
        }
    }

    static async getModelsGroupedByProvider(): Promise<Record<string, AiModelConfig[]>> {
        if (pool) {
            const res = await pool.query(`
                SELECT m.*, p.name as provider_name 
                FROM ai_models m
                JOIN ai_providers p ON m.provider_id = p.id
                ORDER BY p.name ASC, m.name ASC
            `);
            const grouped: Record<string, AiModelConfig[]> = {};
            for (const row of res.rows) {
                const pName = row.provider_name;
                if (!grouped[pName]) grouped[pName] = [];
                grouped[pName].push({
                    id: row.id,
                    providerId: row.provider_id,
                    name: row.name,
                    size: row.size ? Number(row.size) : undefined,
                    modifiedAt: row.modified_at,
                    rawMetadata: row.raw_metadata
                });
            }
            return grouped;
        } else {
            const providers = readJsonFile(PROVIDERS_FILE);
            const models = readJsonFile(MODELS_FILE);
            const grouped: Record<string, AiModelConfig[]> = {};

            for (const model of models) {
                const p = providers.find(x => x.id === model.providerId);
                const pName = p ? p.name : `Provider_${model.providerId}`;
                if (!grouped[pName]) grouped[pName] = [];
                grouped[pName].push(model);
            }
            return grouped;
        }
    }
}
