import assert from "node:assert/strict";
import { config } from "../src/config/env.js";
import { providerRegistry } from "../src/routing/registry.js";
import { classifyQwenOnlyIntent, handleQwenOnlyLowRiskRequest } from "../src/routing/qwen-only-low-risk.js";

class FakeResponse {
    statusCode = 200;
    headers = new Map<string, string>();
    jsonBody: any = undefined;
    bodyChunks: string[] = [];
    ended = false;

    status(code: number) {
        this.statusCode = code;
        return this;
    }

    setHeader(key: string, value: any) {
        this.headers.set(String(key).toLowerCase(), String(value));
    }

    json(body: any) {
        this.jsonBody = body;
        return this;
    }

    write(chunk: any) {
        this.bodyChunks.push(Buffer.isBuffer(chunk) ? chunk.toString("utf8") : String(chunk));
        return true;
    }

    end() {
        this.ended = true;
        return this;
    }
}

class FakeQwenProvider {
    id = "qwen-local";
    calls = 0;
    lastBody: any = undefined;
    responseHandler?: (body: any) => any = undefined;

    async resolveRuntimeConfig() {
        return {
            apiUrl: "http://fake",
            modelName: "qwen-test-model",
            authHeader: "Bearer fake",
            timeoutMs: 1000
        };
    }

    async handleRequest(body: any) {
        this.calls += 1;
        this.lastBody = body;
        let responseBody: any;
        if (this.responseHandler) {
            responseBody = this.responseHandler(body);
        } else {
            responseBody = {
                id: "msg_fake_qwen",
                type: "message",
                role: "assistant",
                content: [{ type: "text", text: "ok" }],
                model: body.model || "qwen-only-low-risk",
                stop_reason: "end_turn",
                stop_sequence: null,
                usage: {
                    input_tokens: 12,
                    output_tokens: 3
                }
            };
        }

        return new Response(JSON.stringify(responseBody), {
            status: 200,
            headers: {
                "content-type": "application/json"
            }
        });
    }
}

function makeReq(body: any) {
    return {
        body,
        method: "POST",
        path: "/v1/messages",
        requestId: "smoke-request-id",
        header(name: string) {
            return name.toLowerCase() === "user-agent" ? "smoke-test" : "";
        }
    } as any;
}

async function runCase(
    name: string,
    body: any,
    expected: { status: number; calls: number; rejection?: string },
    responseHandler?: (body: any) => any
) {
    const fakeProvider = new FakeQwenProvider();
    if (responseHandler) {
        fakeProvider.responseHandler = responseHandler;
    }
    const originalGetProvider = providerRegistry.getProvider.bind(providerRegistry);
    providerRegistry.getProvider = ((providerId: string) => {
        if (providerId === "qwen-local") return fakeProvider as any;
        return originalGetProvider(providerId);
    }) as any;

    try {
        const req = makeReq(body);
        const res = new FakeResponse();
        await handleQwenOnlyLowRiskRequest(req, res as any);

        assert.equal(res.statusCode, expected.status, `${name}: status mismatch`);
        assert.equal(fakeProvider.calls, expected.calls, `${name}: provider call count mismatch`);
        if (expected.rejection) {
            assert.equal(res.jsonBody?.content?.[0]?.text, expected.rejection, `${name}: rejection text mismatch`);
        }

        const decision = classifyQwenOnlyIntent(body);
        return { decision, jsonBody: res.jsonBody };
    } finally {
        providerRegistry.getProvider = originalGetProvider as any;
    }
}

async function main() {
    const originalEnabled = config.qwenOnlyLowRiskEnabled;
    const originalConsoleError = console.error;
    console.error = () => {};

    config.qwenOnlyLowRiskEnabled = true;
    await runCase("chat_hi", {
        model: "qwen-only-low-risk",
        stream: false,
        messages: [{ role: "user", content: "hi" }]
    }, { status: 200, calls: 1 });

    await runCase("chat_ok", {
        model: "qwen-only-low-risk",
        stream: false,
        messages: [{ role: "user", content: "ตอบแค่ ok" }]
    }, { status: 200, calls: 1 });

    await runCase("read_file_html", {
        model: "qwen-only-low-risk",
        stream: false,
        messages: [{ role: "user", content: "อ่านไฟล์ html" }]
    }, { status: 200, calls: 1 });

    // read_only + fake JSON Write => rejected with unsafe tool message
    await runCase("read_file_fake_write_json", {
        model: "qwen-only-low-risk",
        stream: false,
        messages: [{ role: "user", content: "อ่านไฟล์" }]
    }, {
        status: 200,
        calls: 1,
        rejection: "Qwen-only rejected: unsafe tool for read-only request. Use qwen-smart."
    }, () => {
        return {
            id: "msg_fake_qwen_tool",
            type: "message",
            role: "assistant",
            content: [
                {
                    type: "text",
                    text: '{"name": "Write", "arguments": {"file_path": "/path/to/file.txt", "content": "..."}}'
                }
            ],
            model: "qwen-local",
            stop_reason: "end_turn",
            stop_sequence: null,
            usage: { input_tokens: 10, output_tokens: 10 }
        };
    });

    // read_only + real tool call Write => rejected with unsafe tool message
    await runCase("read_file_dangerous_tool", {
        model: "qwen-only-low-risk",
        stream: false,
        messages: [{ role: "user", content: "อ่านไฟล์" }]
    }, {
        status: 200,
        calls: 1,
        rejection: "Qwen-only rejected: unsafe tool for read-only request. Use qwen-smart."
    }, () => {
        return {
            id: "msg_fake_qwen_tool",
            type: "message",
            role: "assistant",
            content: [
                {
                    type: "tool_use",
                    id: "toolu_123",
                    name: "Write",
                    input: { file_path: "/path/to/file.txt", content: "..." }
                }
            ],
            model: "qwen-local",
            stop_reason: "tool_use",
            stop_sequence: null,
            usage: { input_tokens: 10, output_tokens: 10 }
        };
    });

    // read_only + real tool call Bash => rejected with unsafe tool message
    await runCase("read_file_dangerous_bash_tool", {
        model: "qwen-only-low-risk",
        stream: false,
        messages: [{ role: "user", content: "อ่านไฟล์" }]
    }, {
        status: 200,
        calls: 1,
        rejection: "Qwen-only rejected: unsafe tool for read-only request. Use qwen-smart."
    }, () => {
        return {
            id: "msg_fake_qwen_tool",
            type: "message",
            role: "assistant",
            content: [
                {
                    type: "tool_use",
                    id: "toolu_123",
                    name: "Bash",
                    input: { command: "cat file.txt" }
                }
            ],
            model: "qwen-local",
            stop_reason: "tool_use",
            stop_sequence: null,
            usage: { input_tokens: 10, output_tokens: 10 }
        };
    });

    // read_only + real tool call Read => allowed
    const caseSafe = await runCase("read_file_safe_tool", {
        model: "qwen-only-low-risk",
        stream: false,
        messages: [{ role: "user", content: "อ่านไฟล์" }]
    }, {
        status: 200,
        calls: 1
    }, () => {
        return {
            id: "msg_fake_qwen_tool",
            type: "message",
            role: "assistant",
            content: [
                {
                    type: "tool_use",
                    id: "toolu_123",
                    name: "Read",
                    input: { file_path: "/path/to/file.txt" }
                }
            ],
            model: "qwen-local",
            stop_reason: "tool_use",
            stop_sequence: null,
            usage: { input_tokens: 10, output_tokens: 10 }
        };
    });
    assert.equal(caseSafe.jsonBody?.content?.[0]?.type, "tool_use", "read_file_safe_tool content type mismatch");
    assert.equal(caseSafe.jsonBody?.content?.[0]?.name, "Read", "read_file_safe_tool name mismatch");

    // read_only + fake JSON Read => converted to tool_use
    const caseConvert = await runCase("read_file_fake_read_json_converted", {
        model: "qwen-only-low-risk",
        stream: false,
        messages: [{ role: "user", content: "อ่านไฟล์" }]
    }, {
        status: 200,
        calls: 1
    }, () => {
        return {
            id: "msg_fake_qwen_tool",
            type: "message",
            role: "assistant",
            content: [
                {
                    type: "text",
                    text: '{"name": "Read", "arguments": {"file_path": "/path/to/file.txt"}}'
                }
            ],
            model: "qwen-local",
            stop_reason: "end_turn",
            stop_sequence: null,
            usage: { input_tokens: 10, output_tokens: 10 }
        };
    });
    assert.equal(caseConvert.jsonBody?.content?.[0]?.type, "tool_use", "fake JSON conversion failed to produce tool_use");
    assert.equal(caseConvert.jsonBody?.content?.[0]?.name, "Read", "fake JSON conversion name mismatch");
    assert.deepEqual(caseConvert.jsonBody?.content?.[0]?.input, { file_path: "/path/to/file.txt" }, "fake JSON conversion input mismatch");

    // read_only + normal text answer => accepted
    const caseText = await runCase("read_file_text_only_accepted", {
        model: "qwen-only-low-risk",
        stream: false,
        messages: [{ role: "user", content: "อ่านไฟล์" }]
    }, {
        status: 200,
        calls: 1
    }, () => {
        return {
            id: "msg_fake_qwen_text",
            type: "message",
            role: "assistant",
            content: [
                {
                    type: "text",
                    text: "Here is the summary of the file. No tools needed."
                }
            ],
            model: "qwen-local",
            stop_reason: "end_turn",
            stop_sequence: null,
            usage: { input_tokens: 10, output_tokens: 10 }
        };
    });
    assert.equal(caseText.jsonBody?.content?.[0]?.type, "text");
    assert.equal(caseText.jsonBody?.content?.[0]?.text, "Here is the summary of the file. No tools needed.");

    // invalid tool call fake JSON => rejected
    await runCase("fake_tool_json_text_rejected", {
        model: "qwen-only-low-risk",
        stream: false,
        messages: [{ role: "user", content: "อ่านไฟล์" }]
    }, {
        status: 200,
        calls: 1,
        rejection: "Qwen-only could not produce a valid tool call. Use qwen-smart."
    }, () => {
        return {
            id: "msg_fake_qwen_tool",
            type: "message",
            role: "assistant",
            content: [
                {
                    type: "text",
                    text: 'Here is the tool call:\n```json\n{"name": "InvalidToolName", "arguments": {"file_path": "/path/to/file.txt"}}\n```'
                }
            ],
            model: "qwen-local",
            stop_reason: "end_turn",
            stop_sequence: null,
            usage: { input_tokens: 10, output_tokens: 10 }
        };
    });

    await runCase("code_edit_no_context", {
        model: "qwen-only-low-risk",
        stream: false,
        messages: [{ role: "user", content: "please edit src/routes/gateway.ts to fix the logging" }]
    }, {
        status: 200,
        calls: 0,
        rejection: "Qwen-only rejected: missing exact context / high risk. Use qwen-smart or deepseek-v4-flash."
    });

    await runCase("high_risk_auth", {
        model: "qwen-only-low-risk",
        stream: false,
        messages: [{ role: "user", content: "edit auth login payment database flow" }]
    }, {
        status: 200,
        calls: 0,
        rejection: "Qwen-only rejected: missing exact context / high risk. Use qwen-smart or deepseek-v4-flash."
    });

    config.qwenOnlyLowRiskEnabled = false;
    await runCase("disabled", {
        model: "qwen-only-low-risk",
        stream: false,
        messages: [{ role: "user", content: "hi" }]
    }, { status: 403, calls: 0 });

    config.qwenOnlyLowRiskEnabled = originalEnabled;
    console.error = originalConsoleError;
    console.log("qwen-only-low-risk smoke passed");
}

main().catch(err => {
    console.error(err);
    process.exit(1);
});

