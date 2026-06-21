import assert from "node:assert/strict";
import { config } from "../src/config/env.js";
import { providerRegistry } from "../src/routing/registry.js";
import { handleQwenAgentRequest, exportQwenAgentTraces, getQwenAgentTracesSummary } from "../src/routing/qwen-agent.js";
import fs from "fs";

const TRACE_FILE_PATH = "./qwen_agent_traces.jsonl";

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
                model: body.model || "qwen-agent",
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
        requestId: "qwen-agent-smoke-id-" + Math.random().toString(36).substring(7),
        header(name: string) {
            return name.toLowerCase() === "user-agent" ? "smoke-test" : "";
        },
        query: { format: "jsonl" }
    } as any;
}

async function runCase(
    name: string,
    body: any,
    expected: { status: number; calls: number; textMatch?: string; typeMatch?: string },
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
        await handleQwenAgentRequest(req, res as any);

        assert.equal(res.statusCode, expected.status, `${name}: status mismatch`);
        assert.equal(fakeProvider.calls, expected.calls, `${name}: provider call count mismatch`);
        if (expected.textMatch && res.jsonBody) {
            const firstContent = res.jsonBody.content?.[0];
            const text = firstContent?.text || "";
            assert.ok(text.includes(expected.textMatch), `${name}: text mismatch expected containing '${expected.textMatch}', got '${text}'`);
        }
        if (expected.typeMatch && res.jsonBody) {
            assert.equal(res.jsonBody.content?.[0]?.type, expected.typeMatch, `${name}: block type mismatch`);
        }
        return res.jsonBody;
    } finally {
        providerRegistry.getProvider = originalGetProvider as any;
    }
}

async function main() {
    // Enable tracing for the test
    const origTrace = config.qwenAgentTraceEnabled;
    const origSanitize = config.qwenAgentTraceSanitize;
    config.qwenAgentTraceEnabled = true;
    config.qwenAgentTraceSanitize = true;

    // Clean trace file first
    if (fs.existsSync(TRACE_FILE_PATH)) {
        fs.unlinkSync(TRACE_FILE_PATH);
    }

    console.log("Starting qwen-agent smoke tests...");

    // 1. Text only chat
    await runCase("chat_hi", {
        model: "qwen-agent",
        stream: false,
        messages: [{ role: "user", content: "hi" }]
    }, { status: 200, calls: 1, typeMatch: "text", textMatch: "ok" });

    // 2. Normal tool call
    const resTool = await runCase("normal_tool_call", {
        model: "qwen-agent",
        stream: false,
        messages: [{ role: "user", content: "read it" }],
        tools: [{ name: "Read", description: "read file", input_schema: {} }]
    }, { status: 200, calls: 1, typeMatch: "tool_use" }, () => {
        return {
            content: [{
                type: "tool_use",
                id: "toolu_1",
                name: "Read",
                input: { file_path: "package.json" }
            }]
        };
    });
    assert.equal(resTool.content[0].name, "Read");

    // 3. Tool alias normalization
    const resAlias = await runCase("tool_alias", {
        model: "qwen-agent",
        stream: false,
        messages: [{ role: "user", content: "read it" }],
        tools: [{ name: "Read", description: "read file", input_schema: {} }]
    }, { status: 200, calls: 1, typeMatch: "tool_use" }, () => {
        return {
            content: [{
                type: "tool_use",
                id: "toolu_2",
                name: "read_file", // Should normalize to Read
                input: { file_path: "package.json" }
            }]
        };
    });
    assert.equal(resAlias.content[0].name, "Read");

    // 4. Argument repair
    const resRepair = await runCase("arg_repair", {
        model: "qwen-agent",
        stream: false,
        messages: [{ role: "user", content: "read it" }],
        tools: [{ name: "Read", description: "read file", input_schema: {} }]
    }, { status: 200, calls: 1, typeMatch: "tool_use" }, () => {
        return {
            content: [{
                type: "tool_use",
                id: "toolu_3",
                name: "Read",
                input: { file: "package.json" } // Should repair to file_path
            }]
        };
    });
    assert.ok(resRepair.content[0].input.file_path);
    assert.equal(resRepair.content[0].input.file, undefined);

    // 5. Fake JSON Conversion
    const resFakeJson = await runCase("fake_json", {
        model: "qwen-agent",
        stream: false,
        messages: [{ role: "user", content: "read it" }]
    }, { status: 200, calls: 1, typeMatch: "tool_use" }, () => {
        return {
            content: [{
                type: "text",
                text: '{"name": "Read", "arguments": {"file_path": "package.json"}}'
            }]
        };
    });
    assert.equal(resFakeJson.content[0].type, "tool_use");
    assert.equal(resFakeJson.content[0].name, "Read");

    // 6. Validation retry once then success
    let firstCall = true;
    const resRetrySuccess = await runCase("retry_success", {
        model: "qwen-agent",
        stream: false,
        messages: [{ role: "user", content: "read it" }]
    }, { status: 200, calls: 2, typeMatch: "tool_use" }, (body) => {
        if (firstCall) {
            firstCall = false;
            return {
                content: [{
                    type: "tool_use",
                    id: "toolu_retry",
                    name: "Read",
                    input: {} // Invalid: missing file_path
                }]
            };
        } else {
            return {
                content: [{
                    type: "tool_use",
                    id: "toolu_retry",
                    name: "Read",
                    input: { file_path: "package.json" } // Valid
                }]
            };
        }
    });
    assert.equal(resRetrySuccess.content[0].input.file_path, "package.json");

    // 7. Validation retry failure
    firstCall = true;
    await runCase("retry_failed", {
        model: "qwen-agent",
        stream: false,
        messages: [{ role: "user", content: "read it" }]
    }, {
        status: 200,
        calls: 2,
        typeMatch: "text",
        textMatch: "Qwen-agent failed to produce a valid tool call."
    }, () => {
        return {
            content: [{
                type: "tool_use",
                id: "toolu_retry",
                name: "Read",
                input: {} // Invalid: missing file_path
            }]
        };
    });

    // 8. Dangerous bash command blocked
    await runCase("dangerous_command", {
        model: "qwen-agent",
        stream: false,
        messages: [{ role: "user", content: "rm it" }]
    }, {
        status: 200,
        calls: 2, // Retries once, still fails
        typeMatch: "text",
        textMatch: "Qwen-agent failed to produce a valid tool call."
    }, () => {
        return {
            content: [{
                type: "tool_use",
                id: "toolu_rm",
                name: "Bash",
                input: { command: "rm -rf /" }
            }]
        };
    });

    // 9. Max tool rounds guard
    const messagesMax = [];
    for (let i = 0; i < 9; i++) {
        messagesMax.push({
            role: "assistant",
            content: [{ type: "tool_use", id: `t_${i}`, name: "Read", input: { file_path: "x" } }]
        });
        messagesMax.push({
            role: "user",
            content: [{ type: "tool_result", tool_use_id: `t_${i}`, content: "x" }]
        });
    }
    await runCase("max_rounds", {
        model: "qwen-agent",
        stream: false,
        messages: messagesMax
    }, {
        status: 200,
        calls: 0, // Blocked before calling provider
        typeMatch: "text",
        textMatch: "Qwen-agent stopped: max tool rounds reached."
    });

    // 10. Check traces export and summary
    const expRes = new FakeResponse();
    await exportQwenAgentTraces({ query: { format: "jsonl" } } as any, expRes as any);
    assert.equal(expRes.statusCode, 200);
    assert.ok(expRes.bodyChunks.length > 0);
    const firstLine = JSON.parse(expRes.bodyChunks[0].split("\n")[0]);
    assert.ok("messages" in firstLine);
    assert.ok("metadata" in firstLine);

    const sumRes = new FakeResponse();
    await getQwenAgentTracesSummary({} as any, sumRes as any);
    assert.equal(sumRes.statusCode, 200);
    assert.ok(sumRes.jsonBody.totalTraces > 0);
    assert.ok("successRate" in sumRes.jsonBody);

    // Clean up
    if (fs.existsSync(TRACE_FILE_PATH)) {
        fs.unlinkSync(TRACE_FILE_PATH);
    }

    config.qwenAgentTraceEnabled = origTrace;
    config.qwenAgentTraceSanitize = origSanitize;

    console.log("All qwen-agent smoke tests passed successfully!");
}

main().catch(err => {
    console.error(err);
    process.exit(1);
});
