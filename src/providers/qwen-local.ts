import { config } from "../config/env.js";
import { Provider } from "./base.js";
import { createOpenAiToAnthropicStream, cleanText } from "../utils/stream-handler.js";
import { ProviderStore } from "../utils/provider-store.js";

export class QwenLocalProvider implements Provider {
    id = "qwen-local";

    resolveUpstreamModel(clientModel?: string): string {
        return config.qwenLocalModel;
    }

    async handleRequest(body: any, headers: Record<string, string>): Promise<Response> {
        let apiUrl = config.qwenLocalApiUrl;
        let modelName = config.qwenLocalModel;
        let authHeader = "Bearer local-dummy-key";

        try {
            const allProviders = await ProviderStore.getAllProviders();
            const active = allProviders.find(p => p.enabled && (p.type === "ollama" || p.type === "lmstudio" || p.type === "openai_compatible"));
            if (active) {
                apiUrl = active.openaiBaseUrl;
                modelName = active.defaultModel || modelName;
                authHeader = active.type === "ollama" ? "Bearer ollama" : `Bearer ${active.apiKey || ""}`;
            }
        } catch (e) {
            console.error("Failed to query dynamic provider inside QwenLocalProvider:", e);
        }

        // Translate body.tools to OpenAI tools format
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

        // Map messages
        const openaiMessages: any[] = [];

        // Add system message if present
        if (body.system) {
            openaiMessages.push({
                role: "system",
                content: body.system
            });
        }

        // Add user/assistant/tool messages
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

        // Force stream: false when there are tools
        const isStream = hasTools ? false : !!body.stream;

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

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 90000); // 90 seconds timeout

        try {
            const url = `${apiUrl}/chat/completions`;
            const response = await fetch(url, {
                method: "POST",
                headers: {
                    "content-type": "application/json",
                    "authorization": authHeader,
                    ...headers
                },
                body: JSON.stringify(openAiBody),
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                if (response.status === 502 || response.status === 503 || response.status === 504) {
                    return new Response(JSON.stringify({
                        error: {
                            type: "api_error",
                            message: "Local AI is currently offline"
                        }
                    }), {
                        status: 503,
                        headers: { "content-type": "application/json" }
                    });
                }
                return response;
            }

            if (isStream && response.body) {
                const transformedStream = createOpenAiToAnthropicStream(response.body);
                return new Response(transformedStream, {
                    status: 200,
                    headers: {
                        "content-type": "text/event-stream",
                        "cache-control": "no-cache",
                        "connection": "keep-alive"
                    }
                });
            }

            const openAiData = await response.json();
            const message = openAiData.choices?.[0]?.message;
            const textContent = message?.content || "";

            const cleanState = { insideThink: false };
            const cleanedText = cleanText(textContent, cleanState);

            const contentBlocks: any[] = [];
            if (cleanedText) {
                contentBlocks.push({
                    type: "text",
                    text: cleanedText
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
                model: body.model || "qwen-local",
                stop_reason: stopReason,
                stop_sequence: null,
                usage: {
                    input_tokens: openAiData.usage?.prompt_tokens || 0,
                    output_tokens: openAiData.usage?.completion_tokens || 0
                }
            };

            return new Response(JSON.stringify(anthropicResponse), {
                status: 200,
                headers: { "content-type": "application/json" }
            });

        } catch (error: any) {
            clearTimeout(timeoutId);
            const isTimeout = error.name === "AbortError";
            const errorMessage = isTimeout ? "Local AI request timed out" : "Local AI is currently offline";

            console.error("Qwen Local error details:", error);

            return new Response(JSON.stringify({
                error: {
                    type: "api_error",
                    message: errorMessage
                }
            }), {
                status: 503,
                headers: { "content-type": "application/json" }
            });
        }
    }
}
