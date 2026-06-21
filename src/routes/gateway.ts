import { Router } from "express";
import crypto from "crypto";
import path from "path";
import * as fs from "fs";
import { authMiddleware } from "../auth.js";
import { sanitizeAnthropicResponse } from "../providers/deepseek.js";
import { SUPPORTED_MODELS } from "../config/models.js";
import { ModelRouter } from "../routing/router.js";
import { OrchestratorService } from "../routing/orchestrator.js";
import { insertGatewayRequest, updateGatewayRequest, insertModelCall, pool } from "../utils/db.js";
import { config } from "../config/env.js";
import { ProviderStore } from "../utils/provider-store.js";
import { ProviderService } from "../utils/provider-service.js";
import { createOpenAiToAnthropicStream } from "../utils/stream-handler.js";
import { extractDeepSeekUsage, calculateDeepSeekCost } from "../utils/pricing.js";
import { exportQwenAgentTraces, getQwenAgentTracesSummary } from "../routing/qwen-agent.js";
import { qwenTuningRouter } from "./qwen-tuning-routes.js";
import { handleQwenSmartV2Request } from "../routing/qwen-smart-v2.js";
import { AutoJobManager } from "../routing/auto-workflow.js";
import { buildDataset, buildEvalSet, getDatasetFilePath } from "../routing/dataset-pipeline.js";

export const gatewayRouter = Router();

function adminAuthMiddleware(req: any, res: any, next: any) {
    const expectedKey = config.gatewayAdminKey || config.gatewayApiKey;

    if (!expectedKey) {
        return res.status(500).json({
            error: {
                type: "server_error",
                message: "GATEWAY_ADMIN_KEY or GATEWAY_API_KEY is not configured"
            }
        });
    }

    const auth = req.header("authorization") || req.header("x-api-key") || "";
    const token = auth.replace(/^Bearer\s+/i, "").trim();

    if (token !== expectedKey) {
        return res.status(401).json({
            error: {
                type: "authentication_error",
                message: "Invalid admin key"
            }
        });
    }

    next();
}

async function handleConfigurableProviderRequest(
    provider: any,
    modelName: string,
    body: any,
    headers: Record<string, string>,
    req: any,
    res: any,
    requestId: string
): Promise<boolean> {
    const hasTools = Array.isArray(body.tools) && body.tools.length > 0;
    let openaiTools: any[] | undefined = undefined;
    if (hasTools) {
        openaiTools = body.tools.map((t: any) => ({
            type: "function",
            function: {
                name: t.name,
                description: t.description,
                parameters: t.input_schema
            }
        }));
    }

    const openaiMessages: any[] = [];
    if (body.system) {
        openaiMessages.push({
            role: "system",
            content: body.system
        });
    }

    if (Array.isArray(body.messages)) {
        for (const msg of body.messages) {
            if (typeof msg.content === "string") {
                openaiMessages.push({
                    role: msg.role,
                    content: msg.content
                });
            } else if (Array.isArray(msg.content)) {
                if (msg.role === "assistant") {
                    let textContent = "";
                    const toolCalls: any[] = [];
                    for (const block of msg.content) {
                        if (block?.type === "text") {
                            textContent += block.text;
                        } else if (block?.type === "tool_use") {
                            toolCalls.push({
                                id: block.id,
                                type: "function",
                                function: {
                                    name: block.name,
                                    arguments: typeof block.input === "string" ? block.input : JSON.stringify(block.input)
                                }
                            });
                        }
                    }
                    const openAiMsg: any = { role: "assistant" };
                    if (textContent) {
                        openAiMsg.content = textContent;
                    } else {
                        openAiMsg.content = null;
                    }
                    if (toolCalls.length > 0) {
                        openAiMsg.tool_calls = toolCalls;
                    }
                    openaiMessages.push(openAiMsg);
                } else if (msg.role === "user") {
                    const toolResultBlocks = msg.content.filter((b: any) => b?.type === "tool_result");
                    const textBlocks = msg.content.filter((b: any) => b?.type === "text");

                    for (const block of toolResultBlocks) {
                        let resContent = "";
                        if (typeof block.content === "string") {
                            resContent = block.content;
                        } else if (Array.isArray(block.content)) {
                            resContent = block.content.map((cb: any) => cb?.text || "").join("");
                        } else if (block.content !== undefined) {
                            resContent = JSON.stringify(block.content);
                        }
                        openaiMessages.push({
                            role: "tool",
                            tool_call_id: block.tool_use_id,
                            content: resContent
                        });
                    }

                    if (textBlocks.length > 0) {
                        openaiMessages.push({
                            role: "user",
                            content: textBlocks.map((b: any) => b.text || "").join("")
                        });
                    }
                } else {
                    openaiMessages.push({
                        role: msg.role,
                        content: msg.content.map((b: any) => b.text || "").join("")
                    });
                }
            }
        }
    }

    const isStream = !!body.stream && provider.streamEnabled;

    const openAiBody: any = {
        model: modelName,
        messages: openaiMessages,
        temperature: body.temperature ?? 0.7,
        max_tokens: body.max_tokens ?? 2048,
        stream: isStream
    };

    if (hasTools) {
        openAiBody.tools = openaiTools;
        openAiBody.tool_choice = "auto";
    }

    const timeoutMs = Math.min(Math.max(provider.timeoutMs || 120000, 1000), 300000);
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

    try {
        const url = `${provider.openaiBaseUrl}/chat/completions`;
        const authHeader = provider.type === "ollama" ? "Bearer ollama" : `Bearer ${provider.apiKey || ""}`;
        
        const response = await fetch(url, {
            method: "POST",
            headers: {
                "content-type": "application/json",
                "authorization": authHeader
            },
            body: JSON.stringify(openAiBody),
            signal: controller.signal
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
            let errorText = "";
            try {
                errorText = await response.text();
            } catch {}
            
            res.status(503).json({
                error: {
                    type: "api_error",
                    message: `Local AI provider returned error status ${response.status}: ${errorText.slice(0, 150)}`
                }
            });
            return true;
        }

        if (isStream && response.body) {
            res.setHeader("content-type", "text/event-stream");
            res.setHeader("cache-control", "no-cache");
            res.setHeader("x-accel-buffering", "no");
            res.setHeader("connection", "keep-alive");

            const transformedStream = createOpenAiToAnthropicStream(response.body);
            const reader = transformedStream.getReader();
            const decoder = new TextDecoder();
            
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                res.write(value);
            }
            res.end();
            return true;
        }

        const openAiData = await response.json();
        const message = openAiData.choices?.[0]?.message;
        const textContent = message?.content || "";

        const contentBlocks: any[] = [];
        if (textContent) {
            contentBlocks.push({
                type: "text",
                text: textContent
            });
        }

        let stopReason = "end_turn";
        if (Array.isArray(message?.tool_calls) && message.tool_calls.length > 0) {
            stopReason = "tool_use";
            for (const call of message.tool_calls) {
                let parsedInput = {};
                try {
                    parsedInput = typeof call.function.arguments === "string"
                        ? JSON.parse(call.function.arguments)
                        : call.function.arguments;
                } catch {
                    parsedInput = call.function.arguments;
                }
                contentBlocks.push({
                    type: "tool_use",
                    id: call.id,
                    name: call.function.name,
                    input: parsedInput
                });
            }
        }

        const messageId = "msg_local_" + Math.random().toString(36).substring(7);
        const anthropicResponse = {
            id: messageId,
            type: "message",
            role: "assistant",
            content: contentBlocks.length > 0 ? contentBlocks : [{ type: "text", text: "" }],
            model: modelName,
            stop_reason: stopReason,
            stop_sequence: null,
            usage: {
                input_tokens: openAiData.usage?.prompt_tokens || 0,
                output_tokens: openAiData.usage?.completion_tokens || 0
            }
        };

        await insertModelCall({
            requestId,
            provider: provider.type,
            model: modelName,
            inputTokens: openAiData.usage?.prompt_tokens || 0,
            outputTokens: openAiData.usage?.completion_tokens || 0,
            latencyMs: 0
        });

        res.json(anthropicResponse);
        return true;

    } catch (error: any) {
        clearTimeout(timeoutId);
        const isTimeout = error.name === "AbortError";
        let message = "Local AI is currently offline";
        if (isTimeout) {
            message = `Request timed out (timeout: ${timeoutMs}ms)`;
        } else if (error.code === "ECONNREFUSED" || error.message?.includes("fetch failed")) {
            message = `Connection refused: Check if tunnel is offline or Ollama/LM Studio is not running`;
        } else {
            message = error.message || message;
        }

        res.status(503).json({
            error: {
                type: "api_error",
                message
            }
        });
        return true;
    }
}

function logRequest(info: Record<string, unknown>) {
    console.log(JSON.stringify({
        time: new Date().toISOString(),
        ...info
    }));
}

gatewayRouter.get("/", (_req, res) => {
    res.json({
        ok: true,
        service: "railway-ai-gateway",
        version: "0.2.0",
        provider: "hybrid-gateway"
    });
});

gatewayRouter.get("/health", (_req, res) => {
    res.json({
        ok: true,
        service: "railway-ai-gateway",
        version: "0.2.0",
        provider: "hybrid-gateway"
    });
});

gatewayRouter.get("/v1/models", authMiddleware, async (_req, res) => {
    try {
        const dynamicModelsGrouped = await ProviderStore.getModelsGroupedByProvider();
        const allProviders = await ProviderStore.getAllProviders();
        
        const dynamicModelsList: any[] = [];
        for (const [providerName, models] of Object.entries(dynamicModelsGrouped)) {
            const provider = allProviders.find(p => p.name === providerName);
            if (provider && provider.enabled) {
                for (const m of models) {
                    dynamicModelsList.push({
                        id: `${provider.id}/${m.name}`,
                        type: "model",
                        display_name: `${m.name} (${provider.name})`,
                        provider: provider.type,
                        gateway_role: "configurable-provider"
                    });
                }
            }
        }

        const staticModels = SUPPORTED_MODELS.map(m => ({
            id: m.id,
            type: "model",
            display_name: m.displayName,
            provider: m.providerId,
            gateway_role: m.providerId === "deepseek" ? "default" : "local-dev"
        }));

        res.json({
            data: [...staticModels, ...dynamicModelsList]
        });
    } catch (err: any) {
        res.json({
            data: SUPPORTED_MODELS.map(m => ({
                id: m.id,
                type: "model",
                display_name: m.displayName,
                provider: m.providerId,
                gateway_role: m.providerId === "deepseek" ? "default" : "local-dev"
            }))
        });
    }
});

gatewayRouter.post("/v1/messages", authMiddleware, async (req, res) => {
    const requestId = crypto.randomUUID();
    const startTime = Date.now();
    const clientModel = req.body?.model || "unknown";
    const isStream = !!req.body?.stream;
    const mode = (clientModel === "hybrid-flow" || clientModel === "qwen-smart") ? "hybrid-flow" : clientModel === "qwen-only-low-risk" ? "qwen-only-low-risk" : "direct";

    // Insert gateway request to DB
    await insertGatewayRequest(requestId, clientModel, mode, isStream);

    try {
        const allProviders = await ProviderStore.getAllProviders();
        const enabledProviders = allProviders.filter(p => p.enabled);
        const modelsGrouped = await ProviderStore.getModelsGroupedByProvider();

        let matchedProvider: any = null;
        let matchedModelName = "";

        const slashIndex = clientModel.indexOf("/");
        if (slashIndex !== -1) {
            const providerRef = clientModel.substring(0, slashIndex).trim();
            const modelRef = clientModel.substring(slashIndex + 1).trim();

            matchedProvider = enabledProviders.find(p => 
                p.id.toString() === providerRef || 
                p.name.toLowerCase() === providerRef.toLowerCase() ||
                p.type.toLowerCase() === providerRef.toLowerCase()
            );
            if (matchedProvider) {
                matchedModelName = modelRef;
            }
        }

        if (!matchedProvider) {
            for (const [providerName, modelList] of Object.entries(modelsGrouped)) {
                const providerObj = enabledProviders.find(p => p.name === providerName);
                if (providerObj) {
                    const modelObj = modelList.find(m => m.name.toLowerCase() === clientModel.toLowerCase());
                    if (modelObj) {
                        matchedProvider = providerObj;
                        matchedModelName = modelObj.name;
                        break;
                    }
                }
            }
        }

        if (!matchedProvider && (clientModel === "unknown" || clientModel === "" || clientModel === "default")) {
            const defaultProvider = enabledProviders.find(p => p.defaultModel);
            if (defaultProvider && defaultProvider.defaultModel) {
                matchedProvider = defaultProvider;
                matchedModelName = defaultProvider.defaultModel;
            }
        }

        if (matchedProvider && matchedModelName) {
            logRequest({
                type: "request",
                requestId,
                method: req.method,
                path: req.path,
                clientModel,
                upstreamModel: matchedModelName,
                stream: isStream,
                provider: matchedProvider.name
            });

            try {
                const clientHeaders = { "user-agent": req.header("user-agent") || "railway-ai-gateway" };
                const handled = await handleConfigurableProviderRequest(
                    matchedProvider,
                    matchedModelName,
                    req.body,
                    clientHeaders,
                    req,
                    res,
                    requestId
                );
                if (handled) {
                    const totalLatencyMs = Date.now() - startTime;
                    await updateGatewayRequest(requestId, 200, totalLatencyMs);
                    logRequest({
                        type: "response",
                        requestId,
                        method: req.method,
                        path: req.path,
                        clientModel,
                        upstreamModel: matchedModelName,
                        status: 200,
                        latencyMs: totalLatencyMs,
                        stream: isStream,
                        provider: matchedProvider.name
                    });
                    return;
                }
            } catch (err: any) {
                const totalLatencyMs = Date.now() - startTime;
                await updateGatewayRequest(requestId, 500, totalLatencyMs);
                logRequest({
                    type: "response",
                    requestId,
                    method: req.method,
                    path: req.path,
                    clientModel,
                    upstreamModel: matchedModelName,
                    status: 500,
                    latencyMs: totalLatencyMs,
                    stream: isStream,
                    errorMessage: err.message,
                    provider: matchedProvider.name
                });
                return res.status(500).json({
                    error: {
                        type: "gateway_error",
                        message: err.message || "Failed to process request via local provider"
                    }
                });
            }
        }
    } catch (e: any) {
        console.error("Configurable provider routing error:", e);
    }

    if (clientModel === "qwen-agent" || clientModel === "qwen-code") {
        logRequest({
            type: "request",
            requestId,
            method: req.method,
            path: req.path,
            clientModel,
            upstreamModel: "qwen-agent",
            stream: isStream,
            provider: "qwen-local"
        });
        try {
            (req as any).requestId = requestId;
            await OrchestratorService.handleQwenAgent(req, res);
            logRequest({
                type: "response",
                requestId,
                method: req.method,
                path: req.path,
                clientModel,
                upstreamModel: "qwen-agent",
                status: res.statusCode || 200,
                latencyMs: Date.now() - startTime,
                stream: isStream,
                provider: "qwen-local"
            });
        } catch (err: any) {
            logRequest({
                type: "response",
                requestId,
                method: req.method,
                path: req.path,
                clientModel,
                upstreamModel: "qwen-agent",
                status: 500,
                latencyMs: Date.now() - startTime,
                stream: isStream,
                errorMessage: err.message,
                provider: "qwen-local"
            });
            await updateGatewayRequest(requestId, 500, Date.now() - startTime);
            res.status(500).json({
                error: {
                    type: "gateway_error",
                    message: err.message || "Unknown error inside Qwen-agent orchestrator"
                }
            });
        }
        return;
    }

    if (clientModel === "qwen-smart-v2" || clientModel === "smart-qwen") {
        logRequest({
            type: "request",
            requestId,
            method: req.method,
            path: req.path,
            clientModel,
            upstreamModel: "qwen-smart-v2",
            stream: isStream,
            provider: "qwen-local"
        });
        try {
            (req as any).requestId = requestId;
            await handleQwenSmartV2Request(req, res);
            logRequest({
                type: "response",
                requestId,
                method: req.method,
                path: req.path,
                clientModel,
                upstreamModel: "qwen-smart-v2",
                status: res.statusCode || 200,
                latencyMs: Date.now() - startTime,
                stream: isStream,
                provider: "qwen-local"
            });
        } catch (err: any) {
            logRequest({
                type: "response",
                requestId,
                method: req.method,
                path: req.path,
                clientModel,
                upstreamModel: "qwen-smart-v2",
                status: 500,
                latencyMs: Date.now() - startTime,
                stream: isStream,
                errorMessage: err.message,
                provider: "qwen-local"
            });
            await updateGatewayRequest(requestId, 500, Date.now() - startTime);
            res.status(500).json({
                error: {
                    type: "gateway_error",
                    message: err.message || "Unknown error inside Qwen Smart Controller"
                }
            });
        }
        return;
    }

    if (clientModel === "qwen-only-low-risk") {
        logRequest({
            type: "request",
            requestId,
            method: req.method,
            path: req.path,
            clientModel,
            upstreamModel: "qwen-only-low-risk",
            stream: isStream,
            provider: "qwen-only-orchestrator"
        });
        try {
            (req as any).requestId = requestId;
            await OrchestratorService.handleQwenOnlyLowRisk(req, res);
            logRequest({
                type: "response",
                requestId,
                method: req.method,
                path: req.path,
                clientModel,
                upstreamModel: "qwen-only-low-risk",
                status: res.statusCode || 200,
                latencyMs: Date.now() - startTime,
                stream: isStream,
                provider: "qwen-only-orchestrator"
            });
        } catch (err: any) {
            logRequest({
                type: "response",
                requestId,
                method: req.method,
                path: req.path,
                clientModel,
                upstreamModel: "qwen-only-low-risk",
                status: 500,
                latencyMs: Date.now() - startTime,
                stream: isStream,
                errorMessage: err.message,
                provider: "qwen-only-orchestrator"
            });
            await updateGatewayRequest(requestId, 500, Date.now() - startTime);
            res.status(500).json({
                error: {
                    type: "gateway_error",
                    message: err.message || "Unknown error inside Qwen-only low-risk orchestrator"
                }
            });
        }
        return;
    }

    if (clientModel === "hybrid-flow" || clientModel === "qwen-smart") {
        logRequest({
            type: "request",
            requestId,
            method: req.method,
            path: req.path,
            clientModel,
            upstreamModel: "hybrid-orchestration",
            stream: isStream,
            provider: "hybrid-orchestrator"
        });
        try {
            // Pass the requestId to OrchestratorService using custom property or just standard header
            (req as any).requestId = requestId;
            await OrchestratorService.handleTwinModels(req, res);
            logRequest({
                type: "response",
                requestId,
                method: req.method,
                path: req.path,
                clientModel,
                upstreamModel: "hybrid-orchestration",
                status: 200,
                latencyMs: Date.now() - startTime,
                stream: isStream,
                provider: "hybrid-orchestrator"
            });
        } catch (err: any) {
            logRequest({
                type: "response",
                requestId,
                method: req.method,
                path: req.path,
                clientModel,
                upstreamModel: "hybrid-orchestration",
                status: 500,
                latencyMs: Date.now() - startTime,
                stream: isStream,
                errorMessage: err.message,
                provider: "hybrid-orchestrator"
            });
            await updateGatewayRequest(requestId, 500, Date.now() - startTime);
            res.status(500).json({
                error: {
                    type: "gateway_error",
                    message: err.message || "Unknown error inside Hybrid Orchestrator"
                }
            });
        }
        return;
    }

    const provider = ModelRouter.resolve(clientModel);
    const upstreamModel = provider.resolveUpstreamModel(clientModel);

    logRequest({
        type: "request",
        requestId,
        method: req.method,
        path: req.path,
        clientModel,
        upstreamModel,
        stream: isStream,
        provider: provider.id
    });

    try {
        const clientHeaders: Record<string, string> = {
            "user-agent": req.header("user-agent") || "railway-ai-gateway"
        };

        const callStartTime = Date.now();
        const upstream = await provider.handleRequest(req.body, clientHeaders);

        res.status(upstream.status);

        const contentType = upstream.headers.get("content-type");
        if (contentType) {
            res.setHeader("content-type", contentType);
        }

        if (isStream && upstream.body) {
            res.setHeader("cache-control", "no-cache");
            res.setHeader("x-accel-buffering", "no");
            res.setHeader("connection", "keep-alive");

            const reader = upstream.body.getReader();
            const decoder = new TextDecoder();
            let streamBuffer = "";
            let inputTokens = 0;
            let outputTokens = 0;
            let cacheHitInputTokens = 0;
            let cacheMissInputTokens = 0;

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                res.write(Buffer.from(value));

                streamBuffer += decoder.decode(value, { stream: true });
                const lines = streamBuffer.split("\n");
                streamBuffer = lines.pop() || "";

                for (const line of lines) {
                    const trimmed = line.trim();
                    if (trimmed.startsWith("data: ")) {
                        try {
                            const dataJson = JSON.parse(trimmed.slice(6));
                            if (dataJson.message?.usage) {
                                const msgUsage = dataJson.message.usage;
                                if (msgUsage.input_tokens) inputTokens = msgUsage.input_tokens;
                                if (msgUsage.output_tokens) outputTokens = msgUsage.output_tokens;
                                if (msgUsage.cache_read_input_tokens) cacheHitInputTokens = msgUsage.cache_read_input_tokens;
                                if (msgUsage.cache_creation_input_tokens) cacheMissInputTokens = msgUsage.cache_creation_input_tokens;
                                if (msgUsage.prompt_cache_hit_tokens) cacheHitInputTokens = msgUsage.prompt_cache_hit_tokens;
                                if (msgUsage.prompt_cache_miss_tokens) cacheMissInputTokens = msgUsage.prompt_cache_miss_tokens;
                            }
                            if (dataJson.usage) {
                                const u = dataJson.usage;
                                if (u.input_tokens) inputTokens = u.input_tokens;
                                if (u.prompt_tokens) inputTokens = u.prompt_tokens;
                                if (u.output_tokens) outputTokens = u.output_tokens;
                                if (u.completion_tokens) outputTokens = u.completion_tokens;
                                if (u.prompt_cache_hit_tokens) cacheHitInputTokens = u.prompt_cache_hit_tokens;
                                if (u.prompt_cache_miss_tokens) cacheMissInputTokens = u.prompt_cache_miss_tokens;
                                if (u.cache_read_input_tokens) cacheHitInputTokens = u.cache_read_input_tokens;
                                if (u.cache_hit_input_tokens) cacheHitInputTokens = u.cache_hit_input_tokens;
                                if (u.cache_creation_input_tokens) cacheMissInputTokens = u.cache_creation_input_tokens;
                            }
                        } catch {}
                    }
                }
            }

            res.end();

            const latencyMs = Date.now() - callStartTime;

            let usage = {
                inputTokens,
                outputTokens,
                cacheHitInputTokens,
                cacheMissInputTokens: cacheMissInputTokens || Math.max(0, inputTokens - cacheHitInputTokens)
            };
            let costDetails = {};

            if (provider.id === "deepseek") {
                costDetails = calculateDeepSeekCost(upstreamModel, usage);
            }

            // Log model call for direct stream
            await insertModelCall({
                requestId,
                provider: provider.id,
                model: upstreamModel,
                inputTokens: usage.inputTokens,
                outputTokens: usage.outputTokens,
                cacheHitInputTokens: usage.cacheHitInputTokens,
                cacheMissInputTokens: usage.cacheMissInputTokens,
                latencyMs,
                ...costDetails
            });

            const totalLatencyMs = Date.now() - startTime;
            await updateGatewayRequest(requestId, upstream.status, totalLatencyMs);

            logRequest({
                type: "response",
                requestId,
                method: req.method,
                path: req.path,
                clientModel,
                upstreamModel,
                status: upstream.status,
                latencyMs: totalLatencyMs,
                stream: true,
                provider: provider.id
            });
            return;
        }

        const text = await upstream.text();
        let responseBody: any;
        try {
            responseBody = JSON.parse(text);
        } catch {
            responseBody = null;
        }

        const totalLatencyMs = Date.now() - startTime;
        const callLatencyMs = Date.now() - callStartTime;

        if (responseBody && upstream.status === 200) {
            const sanitized = provider.id === "deepseek" ? sanitizeAnthropicResponse(responseBody) : responseBody;
            let usage = { inputTokens: 0, outputTokens: 0, cacheHitInputTokens: 0, cacheMissInputTokens: 0 };
            let costDetails = {};

            if (provider.id === "deepseek") {
                usage = extractDeepSeekUsage(responseBody);
                costDetails = calculateDeepSeekCost(upstreamModel, usage);
            } else {
                usage.inputTokens = responseBody.usage?.input_tokens || responseBody.usage?.prompt_tokens || 0;
                usage.outputTokens = responseBody.usage?.output_tokens || responseBody.usage?.completion_tokens || 0;
                usage.cacheHitInputTokens = responseBody.usage?.cache_read_input_tokens || 0;
                usage.cacheMissInputTokens = usage.inputTokens - usage.cacheHitInputTokens;
            }

            // Log model call for direct non-stream
            await insertModelCall({
                requestId,
                provider: provider.id,
                model: upstreamModel,
                inputTokens: usage.inputTokens,
                outputTokens: usage.outputTokens,
                cacheHitInputTokens: usage.cacheHitInputTokens,
                cacheMissInputTokens: usage.cacheMissInputTokens,
                latencyMs: callLatencyMs,
                ...costDetails
            });

            await updateGatewayRequest(requestId, upstream.status, totalLatencyMs);

            logRequest({
                type: "response",
                requestId,
                method: req.method,
                path: req.path,
                clientModel,
                upstreamModel,
                status: upstream.status,
                latencyMs: totalLatencyMs,
                stream: false,
                inputTokens: usage.inputTokens,
                outputTokens: usage.outputTokens,
                provider: provider.id
            });

            return res.json(sanitized);
        } else {
            await updateGatewayRequest(requestId, upstream.status, totalLatencyMs);

            logRequest({
                type: "response",
                requestId,
                method: req.method,
                path: req.path,
                clientModel,
                upstreamModel,
                status: upstream.status,
                latencyMs: totalLatencyMs,
                stream: false,
                errorMessage: responseBody?.error?.message || text,
                provider: provider.id
            });

            if (responseBody) {
                return res.status(upstream.status).json(responseBody);
            } else {
                return res.status(upstream.status).json({
                    error: {
                        type: "upstream_error",
                        message: text
                    }
                });
            }
        }
    } catch (error) {
        const message = error instanceof Error ? error.message : "Unknown error";
        const totalLatencyMs = Date.now() - startTime;

        await updateGatewayRequest(requestId, 500, totalLatencyMs);

        logRequest({
            type: "response",
            requestId,
            method: req.method,
            path: req.path,
            clientModel,
            upstreamModel,
            status: 500,
            latencyMs: totalLatencyMs,
            stream: isStream,
            errorMessage: message,
            provider: provider.id
        });

        return res.status(500).json({
            error: {
                type: "gateway_error",
                message
            }
        });
    }
});

// Admin Usage Summary & Recent Endpoints
gatewayRouter.get("/admin/usage/summary", adminAuthMiddleware, async (req, res) => {
    const range = req.query.range;
    let timeFilter = "";
    if (range === "today") {
        timeFilter = "AND created_at >= CURRENT_DATE";
    }

    if (!pool) {
        return res.json({
            total_requests: 0,
            total_cost_usd: 0,
            total_cost_thb: 0,
            total_input_cost_usd: 0,
            total_input_cost_thb: 0,
            total_output_cost_usd: 0,
            total_output_cost_thb: 0,
            total_saved_usd: 0,
            total_saved_thb: 0,
            total_net_cost_usd: 0,
            total_net_cost_thb: 0,
            total_saved_input_usd: 0,
            total_saved_input_thb: 0,
            total_saved_output_usd: 0,
            total_saved_output_thb: 0,
            total_input_tokens: 0,
            total_output_tokens: 0,
            tokens_by_model: []
        });
    }

    try {
        const requestsRes = await pool.query(`SELECT COUNT(*) as count FROM gateway_requests WHERE 1=1 ${timeFilter}`);
        const callsRes = await pool.query(`
            SELECT 
                COALESCE(SUM(cost_usd), 0) as total_cost_usd,
                COALESCE(SUM(cost_thb), 0) as total_cost_thb,
                COALESCE(SUM(input_cost_usd), 0) as total_input_cost_usd,
                COALESCE(SUM(input_cost_thb), 0) as total_input_cost_thb,
                COALESCE(SUM(output_cost_usd), 0) as total_output_cost_usd,
                COALESCE(SUM(output_cost_thb), 0) as total_output_cost_thb,
                COALESCE(SUM(saved_usd), 0) as total_saved_usd,
                COALESCE(SUM(saved_thb), 0) as total_saved_thb,
                COALESCE(SUM(saved_input_usd), 0) as total_saved_input_usd,
                COALESCE(SUM(saved_input_thb), 0) as total_saved_input_thb,
                COALESCE(SUM(saved_output_usd), 0) as total_saved_output_usd,
                COALESCE(SUM(saved_output_thb), 0) as total_saved_output_thb,
                COALESCE(SUM(input_tokens), 0) as total_input_tokens,
                COALESCE(SUM(output_tokens), 0) as total_output_tokens
            FROM model_calls
            WHERE 1=1 ${timeFilter}
        `);
        const modelTokensRes = await pool.query(`
            SELECT
                provider,
                model,
                COALESCE(SUM(input_tokens), 0) as input_tokens,
                COALESCE(SUM(output_tokens), 0) as output_tokens
            FROM model_calls
            WHERE 1=1 ${timeFilter}
            GROUP BY provider, model
            ORDER BY COALESCE(SUM(input_tokens), 0) + COALESCE(SUM(output_tokens), 0) DESC
        `);

        const grossCostUsd = Number(callsRes.rows[0].total_cost_usd);
        const grossCostThb = Number(callsRes.rows[0].total_cost_thb);
        const savedUsd = Number(callsRes.rows[0].total_saved_usd);
        const savedThb = Number(callsRes.rows[0].total_saved_thb);
        const netCostUsd = Math.max(0, grossCostUsd - savedUsd);
        const netCostThb = Math.max(0, grossCostThb - savedThb);

        res.json({
            total_requests: Number(requestsRes.rows[0].count),
            total_cost_usd: grossCostUsd,
            total_cost_thb: grossCostThb,
            total_input_cost_usd: Number(callsRes.rows[0].total_input_cost_usd),
            total_input_cost_thb: Number(callsRes.rows[0].total_input_cost_thb),
            total_output_cost_usd: Number(callsRes.rows[0].total_output_cost_usd),
            total_output_cost_thb: Number(callsRes.rows[0].total_output_cost_thb),
            total_saved_usd: savedUsd,
            total_saved_thb: savedThb,
            total_net_cost_usd: netCostUsd,
            total_net_cost_thb: netCostThb,
            total_saved_input_usd: Number(callsRes.rows[0].total_saved_input_usd),
            total_saved_input_thb: Number(callsRes.rows[0].total_saved_input_thb),
            total_saved_output_usd: Number(callsRes.rows[0].total_saved_output_usd),
            total_saved_output_thb: Number(callsRes.rows[0].total_saved_output_thb),
            total_input_tokens: Number(callsRes.rows[0].total_input_tokens),
            total_output_tokens: Number(callsRes.rows[0].total_output_tokens),
            tokens_by_model: modelTokensRes.rows.map(row => ({
                provider: row.provider,
                model: row.model,
                input_tokens: Number(row.input_tokens),
                output_tokens: Number(row.output_tokens)
            }))
        });
    } catch (err: any) {
        res.status(500).json({ error: err.message });
    }
});

gatewayRouter.get("/admin/usage/recent", adminAuthMiddleware, async (req, res) => {
    const limit = Math.min(Number(req.query.limit || 50), 100);
    if (!pool) {
        return res.json([]);
    }
    try {
        const result = await pool.query(
            "SELECT * FROM model_calls ORDER BY created_at DESC LIMIT $1",
            [limit]
        );
        res.json(result.rows);
    } catch (err: any) {
        res.status(500).json({ error: err.message });
    }
});

gatewayRouter.get("/admin/qwen-agent/traces/export", adminAuthMiddleware, exportQwenAgentTraces);
gatewayRouter.get("/admin/qwen-agent/traces/summary", adminAuthMiddleware, getQwenAgentTracesSummary);

// Phase 2 - Adapter rules & prompt profiles endpoints
gatewayRouter.use("/admin/qwen-agent", adminAuthMiddleware, qwenTuningRouter);

// Phase 4 - Auto Coding Jobs endpoints
gatewayRouter.post("/admin/auto/jobs", adminAuthMiddleware, async (req, res) => {
    try {
        const job = await AutoJobManager.createJob(req.body);
        res.status(201).json(job);
    } catch (err: any) {
        res.status(500).json({ error: err.message });
    }
});

gatewayRouter.get("/admin/auto/jobs", adminAuthMiddleware, async (req, res) => {
    try {
        const jobs = await AutoJobManager.listJobs();
        res.json(jobs);
    } catch (err: any) {
        res.status(500).json({ error: err.message });
    }
});

gatewayRouter.get("/admin/auto/jobs/:id", adminAuthMiddleware, async (req, res) => {
    try {
        const job = await AutoJobManager.getJob(Number(req.params.id));
        if (!job) {
            return res.status(404).json({ error: "Job not found" });
        }
        
        let events: any[] = [];
        if (pool) {
            const evRes = await pool.query(
                "SELECT * FROM auto_coding_job_events WHERE job_id = $1 ORDER BY timestamp ASC",
                [job.id]
            );
            events = evRes.rows;
        }

        res.json({ ...job, events });
    } catch (err: any) {
        res.status(500).json({ error: err.message });
    }
});

gatewayRouter.post("/admin/auto/jobs/:id/cancel", adminAuthMiddleware, async (req, res) => {
    try {
        const ok = await AutoJobManager.cancelJob(Number(req.params.id));
        res.json({ ok });
    } catch (err: any) {
        res.status(500).json({ error: err.message });
    }
});

gatewayRouter.post("/admin/auto/jobs/:id/retry", adminAuthMiddleware, async (req, res) => {
    try {
        const job = await AutoJobManager.getJob(Number(req.params.id));
        if (!job) {
            return res.status(404).json({ error: "Job not found" });
        }
        const newJob = await AutoJobManager.createJob({
            user_task: job.user_task,
            repo_path: job.repo_path,
            branch_name: job.branch_name || undefined,
            mode: job.mode,
            model_worker: job.model_worker,
            controller_model: job.controller_model || undefined
        });
        res.status(201).json(newJob);
    } catch (err: any) {
        res.status(500).json({ error: err.message });
    }
});

gatewayRouter.get("/admin/auto/summary", adminAuthMiddleware, async (req, res) => {
    if (!pool) return res.json({});
    try {
        const totalRes = await pool.query("SELECT COUNT(*) FROM auto_coding_jobs");
        const completedRes = await pool.query("SELECT COUNT(*) FROM auto_coding_jobs WHERE status = 'completed'");
        const failedRes = await pool.query("SELECT COUNT(*) FROM auto_coding_jobs WHERE status = 'failed'");
        const humanRes = await pool.query("SELECT COUNT(*) FROM auto_coding_jobs WHERE status = 'needs_human'");
        
        const durationRes = await pool.query(
            "SELECT AVG(EXTRACT(EPOCH FROM (updated_at - created_at))) as avg_dur FROM auto_coding_jobs WHERE status IN ('completed', 'failed')"
        );
        const roundsRes = await pool.query("SELECT AVG(current_step) as avg_rounds FROM auto_coding_jobs WHERE status = 'completed'");
        
        const buildPassRes = await pool.query("SELECT COUNT(*) FROM auto_coding_job_events WHERE event_type = 'build_check_passed'");
        const buildFailRes = await pool.query("SELECT COUNT(*) FROM auto_coding_job_events WHERE event_type = 'build_check_failed'");
        const buildPass = parseInt(buildPassRes.rows[0].count, 10);
        const buildFail = parseInt(buildFailRes.rows[0].count, 10);
        const buildPassRate = (buildPass + buildFail) > 0 ? Number((buildPass / (buildPass + buildFail)).toFixed(4)) : 1;

        const reviewPassRes = await pool.query(
            "SELECT COUNT(*) FROM auto_coding_job_events WHERE event_type = 'review_completed' AND payload->>'review_result' = 'pass'"
        );
        const reviewFailRes = await pool.query(
            "SELECT COUNT(*) FROM auto_coding_job_events WHERE event_type = 'review_completed' AND payload->>'review_result' = 'needs_fix'"
        );
        const reviewPass = parseInt(reviewPassRes.rows[0].count, 10);
        const reviewFail = parseInt(reviewFailRes.rows[0].count, 10);
        const reviewPassRate = (reviewPass + reviewFail) > 0 ? Number((reviewPass / (reviewPass + reviewFail)).toFixed(4)) : 1;

        const savingsRes = await pool.query("SELECT COALESCE(SUM(saved_usd), 0) as saved_usd, COALESCE(SUM(saved_thb), 0) as saved_thb FROM model_calls");

        res.json({
            totalJobs: parseInt(totalRes.rows[0].count, 10),
            completed: parseInt(completedRes.rows[0].count, 10),
            failed: parseInt(failedRes.rows[0].count, 10),
            needsHuman: parseInt(humanRes.rows[0].count, 10),
            averageDurationSeconds: durationRes.rows[0].avg_dur ? Number(Number(durationRes.rows[0].avg_dur).toFixed(2)) : 0,
            averageQwenToolRounds: roundsRes.rows[0].avg_rounds ? Number(Number(roundsRes.rows[0].avg_rounds).toFixed(2)) : 0,
            buildPassRate,
            controllerReviewPassRate: reviewPassRate,
            costEstimateSavedUsd: Number(Number(savingsRes.rows[0].saved_usd).toFixed(4)),
            costEstimateSavedThb: Number(Number(savingsRes.rows[0].saved_thb).toFixed(2))
        });
    } catch (err: any) {
        res.status(500).json({ error: err.message });
    }
});

// Phase 5 - SFT Dataset Pipeline endpoints
gatewayRouter.post("/admin/qwen-agent/datasets/build", adminAuthMiddleware, async (req, res) => {
    try {
        const datasetId = await buildDataset(req.body);
        res.status(201).json({ datasetId });
    } catch (err: any) {
        res.status(500).json({ error: err.message });
    }
});

gatewayRouter.get("/admin/qwen-agent/datasets/:id/download", adminAuthMiddleware, async (req, res) => {
    try {
        const filePath = getDatasetFilePath(req.params.id);
        if (!fs.existsSync(filePath)) {
            return res.status(404).json({ error: "Dataset file not found" });
        }
        res.setHeader("content-type", "application/x-jsonlines");
        res.sendFile(filePath);
    } catch (err: any) {
        res.status(500).json({ error: err.message });
    }
});

gatewayRouter.post("/admin/qwen-agent/datasets/build-eval", adminAuthMiddleware, async (req, res) => {
    try {
        const evalSetId = await buildEvalSet();
        res.status(201).json({ evalSetId });
    } catch (err: any) {
        res.status(500).json({ error: err.message });
    }
});

gatewayRouter.post("/admin/usage/clear", adminAuthMiddleware, async (req, res) => {
    if (!pool) {
        return res.json({ ok: false, error: "Database not connected" });
    }
    try {
        await pool.query("DELETE FROM model_calls");
        await pool.query("DELETE FROM gateway_requests");
        res.json({ ok: true });
    } catch (err: any) {
        res.status(500).json({ ok: false, error: err.message });
    }
});

// Admin Providers page serving
gatewayRouter.get("/admin/ai-providers", (req, res) => {
    res.sendFile(path.join(process.cwd(), "public", "ai-providers.html"));
});

// Admin Providers API Endpoints
gatewayRouter.get("/api/admin/ai-providers", adminAuthMiddleware, async (req, res) => {
    try {
        const list = await ProviderStore.getAllProviders();
        const sanitized = list.map(p => ({
            ...p,
            apiKey: p.apiKey ? "***" : ""
        }));
        res.json(sanitized);
    } catch (err: any) {
        res.status(500).json({ error: err.message });
    }
});

gatewayRouter.post("/api/admin/ai-providers", adminAuthMiddleware, async (req, res) => {
    try {
        const data = req.body;
        if (data.id) {
            const existing = await ProviderStore.getProviderById(Number(data.id));
            if (existing && data.apiKey === "***") {
                data.apiKey = existing.apiKey;
            }
        }
        const saved = await ProviderStore.saveProvider(data);
        res.json({
            ...saved,
            apiKey: saved.apiKey ? "***" : ""
        });
    } catch (err: any) {
        res.status(500).json({ error: err.message });
    }
});

gatewayRouter.delete("/api/admin/ai-providers/:id", adminAuthMiddleware, async (req, res) => {
    try {
        const ok = await ProviderStore.deleteProvider(Number(req.params.id));
        res.json({ ok });
    } catch (err: any) {
        res.status(500).json({ error: err.message });
    }
});

gatewayRouter.post("/api/admin/ai-providers/test", adminAuthMiddleware, async (req, res) => {
    try {
        const { type, serverUrl, apiKey } = req.body;
        const result = await ProviderService.testConnection({ type, serverUrl, apiKey });
        res.json(result);
    } catch (err: any) {
        res.status(500).json({ ok: false, message: err.message });
    }
});

gatewayRouter.post("/api/admin/ai-providers/:id/sync-models", adminAuthMiddleware, async (req, res) => {
    try {
        const provider = await ProviderStore.getProviderById(Number(req.params.id));
        if (!provider) {
            return res.status(404).json({ error: "Provider not found" });
        }
        const fetched = await ProviderService.fetchModels(provider);
        const saved = await ProviderStore.syncModels(provider.id, fetched);
        res.json(saved);
    } catch (err: any) {
        res.status(500).json({ error: err.message });
    }
});

gatewayRouter.post("/api/admin/ai-providers/:id/pull-model", adminAuthMiddleware, async (req, res) => {
    try {
        const provider = await ProviderStore.getProviderById(Number(req.params.id));
        if (!provider) {
            return res.status(404).json({ error: "Provider not found" });
        }
        const { model } = req.body;
        const result = await ProviderService.pullModel(provider, model);
        res.json(result);
    } catch (err: any) {
        res.status(500).json({ error: err.message });
    }
});

gatewayRouter.post("/api/admin/ai-providers/:id/default-model", adminAuthMiddleware, async (req, res) => {
    try {
        const providerId = Number(req.params.id);
        const { model } = req.body;
        const ok = await ProviderStore.setDefaultModel(providerId, model);
        res.json({ ok });
    } catch (err: any) {
        res.status(500).json({ error: err.message });
    }
});

gatewayRouter.get("/api/admin/ai-models", adminAuthMiddleware, async (req, res) => {
    try {
        const grouped = await ProviderStore.getModelsGroupedByProvider();
        res.json(grouped);
    } catch (err: any) {
        res.status(500).json({ error: err.message });
    }
});


