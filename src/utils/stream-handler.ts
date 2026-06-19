// Helper to clean text statefully
export function cleanText(text: string, state: { insideThink: boolean }): string {
    let result = "";
    let i = 0;
    let tempText = text;

    // If we are currently inside a think block, search for ending tag
    if (state.insideThink) {
        const endIdx = tempText.indexOf("</think>");
        if (endIdx !== -1) {
            state.insideThink = false;
            tempText = tempText.substring(endIdx + 8);
        } else {
            // Still inside think, discard all text
            return "";
        }
    }

    // Now look for start tag
    while (tempText.length > 0) {
        const startIdx = tempText.indexOf("<think>");
        if (startIdx !== -1) {
            result += tempText.substring(0, startIdx);
            state.insideThink = true;
            const endIdx = tempText.indexOf("</think>", startIdx + 7);
            if (endIdx !== -1) {
                state.insideThink = false;
                tempText = tempText.substring(endIdx + 8);
            } else {
                // Thinking block goes to the end of this chunk
                break;
            }
        } else {
            result += tempText;
            break;
        }
    }

    // Clean other garbage tokens if present
    result = result.replace(/<\|im_start\|>|<\|im_end\|>|<\|im_sep\|>/g, "");

    return result;
}

// Convert OpenAI stream into Anthropic stream
export function createOpenAiToAnthropicStream(openAiStream: ReadableStream<Uint8Array>): ReadableStream<Uint8Array> {
    const reader = openAiStream.getReader();
    const decoder = new TextDecoder();
    const encoder = new TextEncoder();

    let buffer = "";
    const cleanState = { insideThink: false };
    let messageId = "msg_local_" + Math.random().toString(36).substring(7);

    return new ReadableStream({
        async start(controller) {
            // Anthropic stream start events
            const startEvent = `event: message_start\ndata: ${JSON.stringify({
                type: "message_start",
                message: {
                    id: messageId,
                    type: "message",
                    role: "assistant",
                    content: [],
                    model: "qwen-local",
                    stop_reason: null,
                    stop_sequence: null,
                    usage: { input_tokens: 0, output_tokens: 0 }
                }
            })}\n\n`;
            controller.enqueue(encoder.encode(startEvent));

            const contentStartEvent = `event: content_block_start\ndata: ${JSON.stringify({
                type: "content_block_start",
                index: 0,
                content_block: { type: "text", text: "" }
            })}\n\n`;
            controller.enqueue(encoder.encode(contentStartEvent));
        },

        async pull(controller) {
            try {
                const { done, value } = await reader.read();
                if (done) {
                    // Send message_stop / content_block_stop
                    const contentStop = `event: content_block_stop\ndata: ${JSON.stringify({
                        type: "content_block_stop",
                        index: 0
                    })}\n\n`;
                    controller.enqueue(encoder.encode(contentStop));

                    const messageDelta = `event: message_delta\ndata: ${JSON.stringify({
                        type: "message_delta",
                        delta: { stop_reason: "end_turn", stop_sequence: null },
                        usage: { output_tokens: 0 }
                    })}\n\n`;
                    controller.enqueue(encoder.encode(messageDelta));

                    const messageStop = `event: message_stop\ndata: ${JSON.stringify({
                        type: "message_stop"
                    })}\n\n`;
                    controller.enqueue(encoder.encode(messageStop));

                    controller.close();
                    return;
                }

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n");
                // Keep the last partial line in buffer
                buffer = lines.pop() || "";

                for (const line of lines) {
                    const trimmed = line.trim();
                    if (!trimmed) continue;
                    if (trimmed === "data: [DONE]") continue;

                    if (trimmed.startsWith("data: ")) {
                        try {
                            const dataJson = JSON.parse(trimmed.slice(6));
                            const textDelta = dataJson.choices?.[0]?.delta?.content || "";
                            if (textDelta) {
                                const cleanedText = cleanText(textDelta, cleanState);
                                if (cleanedText) {
                                    const eventStr = `event: content_block_delta\ndata: ${JSON.stringify({
                                        type: "content_block_delta",
                                        index: 0,
                                        delta: { type: "text_delta", text: cleanedText }
                                    })}\n\n`;
                                    controller.enqueue(encoder.encode(eventStr));
                                }
                            }
                        } catch (e) {
                            // JSON parsing error or unrecognized format, skip
                        }
                    }
                }
            } catch (err) {
                controller.error(err);
            }
        },

        cancel() {
            reader.cancel();
        }
    });
}
