import { config } from "../config/env.js";
import { Provider } from "./base.js";
import { createOpenAiToAnthropicStream, cleanText } from "../utils/stream-handler.js";

export class QwenLocalProvider implements Provider {
    id = "qwen-local";

    resolveUpstreamModel(clientModel?: string): string {
        return "qwen";
    }

    async handleRequest(body: any, headers: Record<string, string>): Promise<Response> {
        // Map messages
        const openaiMessages: any[] = [];

        // Add system message if present
        if (body.system) {
            openaiMessages.push({
                role: "system",
                content: body.system
            });
        }

        // Add user/assistant messages
        if (Array.isArray(body.messages)) {
            for (const msg of body.messages) {
                let contentStr = "";
                if (typeof msg.content === "string") {
                    contentStr = msg.content;
                } else if (Array.isArray(msg.content)) {
                    contentStr = msg.content
                        .map((block: any) => (block?.type === "text" ? block.text : ""))
                        .join("");
                }
                openaiMessages.push({
                    role: msg.role,
                    content: contentStr
                });
            }
        }

        const openAiBody = {
            model: body.model || "qwen",
            messages: openaiMessages,
            temperature: body.temperature ?? 0.7,
            max_tokens: body.max_tokens ?? 2048,
            stream: !!body.stream
        };

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 90000); // 90 seconds timeout

        try {
            const url = `${config.qwenLocalApiUrl}/chat/completions`;
            const response = await fetch(url, {
                method: "POST",
                headers: {
                    "content-type": "application/json",
                    "authorization": `Bearer local-dummy-key`,
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

            if (body.stream && response.body) {
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
            const textContent = openAiData.choices?.[0]?.message?.content || "";

            const cleanState = { insideThink: false };
            const cleanedText = cleanText(textContent, cleanState);

            const messageId = "msg_local_" + Math.random().toString(36).substring(7);
            const anthropicResponse = {
                id: messageId,
                type: "message",
                role: "assistant",
                content: [
                    {
                        type: "text",
                        text: cleanedText
                    }
                ],
                model: body.model || "qwen",
                stop_reason: "end_turn",
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
