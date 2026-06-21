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

    // 10. Duplicate tool-call detection
    const messagesDup = [
        {
            role: "assistant",
            content: [{ type: "tool_use", id: "t_dup_1", name: "Write", input: { file_path: "test.txt", content: "hello" } }]
        },
        {
            role: "user",
            content: [{ type: "tool_result", tool_use_id: "t_dup_1", content: "success", is_error: false }]
        }
    ];

    const resDup = await runCase("duplicate_detection", {
        model: "qwen-agent",
        stream: false,
        messages: messagesDup
    }, {
        status: 200,
        calls: 1,
        typeMatch: "text",
        textMatch: "The file changes have already been successfully applied."
    }, () => {
        return {
            content: [{
                type: "tool_use",
                id: "t_dup_2",
                name: "Write",
                input: { file_path: "test.txt", content: "hello" }
            }]
        };
    });
    assert.equal(resDup.stop_reason, "end_turn");

    // 11. Post-success stop hint
    const messagesHint = [
        {
            role: "assistant",
            content: [{ type: "tool_use", id: "t_hint_1", name: "Write", input: { file_path: "test.txt", content: "hello" } }]
        },
        {
            role: "user",
            content: [{ type: "tool_result", tool_use_id: "t_hint_1", content: "success", is_error: false }]
        }
    ];

    let lastProviderBody: any = null;
    await runCase("post_success_stop_hint", {
        model: "qwen-agent",
        stream: false,
        messages: messagesHint
    }, {
        status: 200,
        calls: 1,
        typeMatch: "text"
    }, (body) => {
        lastProviderBody = body;
        return {
            content: [{ type: "text", text: "ok" }]
        };
    });
    
    const lastUserMsg = lastProviderBody.messages[lastProviderBody.messages.length - 1];
    const toolResultBlock = lastUserMsg.content.find((b: any) => b?.type === "tool_result");
    assert.ok(toolResultBlock.content.includes("The file operation succeeded. Do not call Write/Edit again"), "Hint not injected!");

    // 12. Simple task early stop
    const messagesEarlyStop = [
        {
            role: "assistant",
            content: [{ type: "tool_use", id: "t_early_1", name: "Edit", input: { file_path: "app.ts", old_string: "a", new_string: "b" } }]
        },
        {
            role: "user",
            content: [{ type: "tool_result", tool_use_id: "t_early_1", content: "success", is_error: false }]
        }
    ];

    const resEarly = await runCase("early_stop_after_successful_edit", {
        model: "qwen-agent",
        stream: false,
        messages: messagesEarlyStop
    }, {
        status: 200,
        calls: 1,
        typeMatch: "text",
        textMatch: "The file changes have been successfully applied."
    }, () => {
        return {
            content: [{
                type: "tool_use",
                id: "t_early_2",
                name: "Read",
                input: { file_path: "app.ts" }
            }]
        };
    });
    assert.equal(resEarly.stop_reason, "end_turn");

    // 14. "hi" uses 0 tools (chat_only)
    let hiToolsOffered: any[] | undefined = undefined;
    await runCase("chat_only_hi", {
        model: "qwen-agent",
        stream: false,
        messages: [{ role: "user", content: "hi" }],
        tools: [{ name: "Read", description: "read file", input_schema: {} }]
    }, {
        status: 200,
        calls: 1,
        typeMatch: "text"
    }, (body) => {
        hiToolsOffered = body.tools;
        return { content: [{ type: "text", text: "hello" }] };
    });
    assert.ok(!hiToolsOffered || hiToolsOffered.length === 0, "Tools should not be offered in chat_only");

    // 15. "ดี" uses 0 tools (chat_only)
    let thToolsOffered: any[] | undefined = undefined;
    await runCase("chat_only_thai", {
        model: "qwen-agent",
        stream: false,
        messages: [{ role: "user", content: "ดี" }],
        tools: [{ name: "Read", description: "read file", input_schema: {} }]
    }, {
        status: 200,
        calls: 1,
        typeMatch: "text"
    }, (body) => {
        thToolsOffered = body.tools;
        return { content: [{ type: "text", text: "ดีครับ" }] };
    });
    assert.ok(!thToolsOffered || thToolsOffered.length === 0, "Tools should not be offered in chat_only");

    // 16. "หา DATABASE_URL" allows Grep/Read, not Write
    let findToolsOffered: any[] = [];
    await runCase("read_only_find", {
        model: "qwen-agent",
        stream: false,
        messages: [{ role: "user", content: "หา DATABASE_URL" }],
        tools: [
            { name: "Read", description: "read file", input_schema: {} },
            { name: "Grep", description: "grep file", input_schema: {} },
            { name: "Write", description: "write file", input_schema: {} }
        ]
    }, {
        status: 200,
        calls: 1
    }, (body) => {
        findToolsOffered = body.tools || [];
        return { content: [{ type: "text", text: "found nothing" }] };
    });
    const findToolNames = findToolsOffered.map((t: any) => t.name);
    assert.ok(findToolNames.includes("Read"), "Should offer Read");
    assert.ok(findToolNames.includes("Grep"), "Should offer Grep");
    assert.ok(!findToolNames.includes("Write"), "Should NOT offer Write");

    // 17. "สร้าง src/test.ts" uses Write (edit_allowed)
    let createToolsOffered: any[] = [];
    await runCase("edit_allowed_create", {
        model: "qwen-agent",
        stream: false,
        messages: [{ role: "user", content: "สร้าง src/test.ts" }],
        tools: [
            { name: "Read", description: "read file", input_schema: {} },
            { name: "Write", description: "write file", input_schema: {} },
            { name: "Bash", description: "bash", input_schema: {} }
        ]
    }, {
        status: 200,
        calls: 1
    }, (body) => {
        createToolsOffered = body.tools || [];
        return {
            content: [{
                type: "tool_use",
                id: "t_create_test",
                name: "Write",
                input: { file_path: "src/test.ts", content: "export const x = 1" }
            }]
        };
    });
    const createToolNames = createToolsOffered.map((t: any) => t.name);
    assert.ok(createToolNames.includes("Write"), "Should offer Write");
    assert.ok(!createToolNames.includes("Bash"), "Should NOT offer Bash");

    // 18. "แก้ src/test.ts" uses Edit/Write (edit_allowed)
    let fixToolsOffered: any[] = [];
    await runCase("edit_allowed_fix", {
        model: "qwen-agent",
        stream: false,
        messages: [{ role: "user", content: "แก้ src/test.ts" }],
        tools: [
            { name: "Edit", description: "edit file", input_schema: {} },
            { name: "Write", description: "write file", input_schema: {} },
            { name: "Bash", description: "bash", input_schema: {} }
        ]
    }, {
        status: 200,
        calls: 1
    }, (body) => {
        fixToolsOffered = body.tools || [];
        return {
            content: [{
                type: "tool_use",
                id: "t_fix_test",
                name: "Edit",
                input: { file_path: "src/test.ts", old_string: "x = 1", new_string: "x = 2" }
            }]
        };
    });
    const fixToolNames = fixToolsOffered.map((t: any) => t.name);
    assert.ok(fixToolNames.includes("Edit"), "Should offer Edit");
    assert.ok(fixToolNames.includes("Write"), "Should offer Write");
    assert.ok(!fixToolNames.includes("Bash"), "Should NOT offer Bash");

    // 19. "npm run build" uses Bash (bash_allowed)
    let buildToolsOffered: any[] = [];
    await runCase("bash_allowed_build", {
        model: "qwen-agent",
        stream: false,
        messages: [{ role: "user", content: "npm run build" }],
        tools: [
            { name: "Write", description: "write file", input_schema: {} },
            { name: "Bash", description: "bash", input_schema: {} }
        ]
    }, {
        status: 200,
        calls: 1
    }, (body) => {
        buildToolsOffered = body.tools || [];
        return { content: [{ type: "text", text: "done" }] };
    });
    const buildToolNames = buildToolsOffered.map((t: any) => t.name);
    assert.ok(buildToolNames.includes("Bash"), "Should offer Bash");
    assert.ok(!buildToolNames.includes("Write"), "Should NOT offer Write");

    // 20. Intent Gate Tool Blocking
    const resBlocked = await runCase("intent_gate_blocking", {
        model: "qwen-agent",
        stream: false,
        messages: [{ role: "user", content: "หา DATABASE_URL" }],
        tools: [
            { name: "Read", description: "read file", input_schema: {} },
            { name: "Write", description: "write file", input_schema: {} }
        ]
    }, {
        status: 200,
        calls: 1,
        typeMatch: "text",
        textMatch: "Tool Write is blocked."
    }, () => {
        return {
            content: [{
                type: "tool_use",
                id: "t_blocked_1",
                name: "Write",
                input: { file_path: "test.txt", content: "hello" }
            }]
        };
    });
    assert.equal(resBlocked.stop_reason, "end_turn");

    // 20b. Edit Enforcement - Success after Retry (สร้างไฟล์)
    let enforceCalls = 0;
    const resEnforceCreate = await runCase("enforce_create_file", {
        model: "qwen-agent",
        stream: false,
        messages: [{ role: "user", content: "สร้างไฟล์ src/test.ts ใส่ export const x = 1" }],
        tools: [
            { name: "Write", description: "write file", input_schema: {} },
            { name: "Edit", description: "edit file", input_schema: {} }
        ]
    }, {
        status: 200,
        calls: 2,
        typeMatch: "tool_use"
    }, (body) => {
        enforceCalls++;
        if (enforceCalls === 1) {
            // First call returns text only
            return { content: [{ type: "text", text: "Here is the file content you asked for." }] };
        } else {
            // Second call returns correct tool use
            return {
                content: [{
                    type: "tool_use",
                    id: "t_create_retry",
                    name: "Write",
                    input: { file_path: "src/test.ts", content: "export const x = 1" }
                }]
            };
        }
    });
    assert.equal(resEnforceCreate.content[0].name, "Write");

    // 20c. Edit Enforcement - Success after Retry (แก้ไฟล์)
    enforceCalls = 0;
    const resEnforceEdit = await runCase("enforce_edit_file", {
        model: "qwen-agent",
        stream: false,
        messages: [{ role: "user", content: "แก้ src/test.ts จาก x = 1 เป็น x = 2" }],
        tools: [
            { name: "Write", description: "write file", input_schema: {} },
            { name: "Edit", description: "edit file", input_schema: {} }
        ]
    }, {
        status: 200,
        calls: 2,
        typeMatch: "tool_use"
    }, (body) => {
        enforceCalls++;
        if (enforceCalls === 1) {
            return { content: [{ type: "text", text: "I will update the file." }] };
        } else {
            return {
                content: [{
                    type: "tool_use",
                    id: "t_edit_retry",
                    name: "Edit",
                    input: { file_path: "src/test.ts", old_string: "x = 1", new_string: "x = 2" }
                }]
            };
        }
    });
    assert.equal(resEnforceEdit.content[0].name, "Edit");

    // 20d. Edit Enforcement - Success after Retry (เพิ่ม endpoint)
    enforceCalls = 0;
    const resEnforceAdd = await runCase("enforce_add_endpoint", {
        model: "qwen-agent",
        stream: false,
        messages: [{ role: "user", content: "เพิ่ม endpoint /healthz ใน src/server.ts" }],
        tools: [
            { name: "Write", description: "write file", input_schema: {} },
            { name: "Edit", description: "edit file", input_schema: {} }
        ]
    }, {
        status: 200,
        calls: 2,
        typeMatch: "tool_use"
    }, (body) => {
        enforceCalls++;
        if (enforceCalls === 1) {
            return { content: [{ type: "text", text: "Adding /healthz." }] };
        } else {
            return {
                content: [{
                    type: "tool_use",
                    id: "t_add_retry",
                    name: "Edit",
                    input: { file_path: "src/server.ts", old_string: "app.listen", new_string: "app.get('/healthz')\napp.listen" }
                }]
            };
        }
    });
    assert.equal(resEnforceAdd.content[0].name, "Edit");

    // 20e. Edit Enforcement - Negative test case (อธิบายไฟล์ - should NOT enforce)
    enforceCalls = 0;
    const resNoEnforceExplain = await runCase("no_enforce_explain", {
        model: "qwen-agent",
        stream: false,
        messages: [{ role: "user", content: "อธิบาย src/server.ts ทำงานยังไง" }],
        tools: [
            { name: "Write", description: "write file", input_schema: {} },
            { name: "Edit", description: "edit file", input_schema: {} }
        ]
    }, {
        status: 200,
        calls: 1, // should succeed immediately, no retry
        typeMatch: "text",
        textMatch: "This file handles server initialization"
    }, (body) => {
        enforceCalls++;
        return { content: [{ type: "text", text: "This file handles server initialization" }] };
    });

    // 20f. Edit Enforcement - Retry failure path (still returns text, returns safe final message)
    enforceCalls = 0;
    const resEnforceFailed = await runCase("enforce_failed_safe_message", {
        model: "qwen-agent",
        stream: false,
        messages: [{ role: "user", content: "สร้างไฟล์ src/test.ts ใส่ export const x = 1" }],
        tools: [
            { name: "Write", description: "write file", input_schema: {} },
            { name: "Edit", description: "edit file", input_schema: {} }
        ]
    }, {
        status: 200,
        calls: 2,
        typeMatch: "text",
        textMatch: "Edit intent was detected, but no edit tool was produced."
    }, (body) => {
        enforceCalls++;
        return { content: [{ type: "text", text: "Here is text only." }] };
    });

    // 20g. Claude Code tool_use compatibility - สร้างไฟล์
    const resClaudeWrite = await runCase("claude_code_write_compatibility", {
        model: "qwen-agent",
        stream: false,
        messages: [{ role: "user", content: "สร้างไฟล์ src/qwen-test.ts ใส่ export const qwenTest = true" }],
        tools: [
            { name: "Write", description: "write file", input_schema: {} },
            { name: "Read", description: "read file", input_schema: {} }
        ]
    }, {
        status: 200,
        calls: 1,
        typeMatch: "tool_use"
    }, (body) => {
        // Return a mock Write tool call
        return {
            content: [{
                type: "tool_use",
                id: "toolu_test_write",
                name: "Write",
                input: {
                    file_path: "src/qwen-test.ts",
                    content: "export const qwenTest = true;\n"
                }
            }],
            stop_reason: "tool_use"
        };
    });
    assert.equal(resClaudeWrite.content[0].type, "tool_use");
    assert.equal(resClaudeWrite.content[0].name, "Write");
    assert.equal(resClaudeWrite.content[0].input.file_path, "src/qwen-test.ts");
    assert.ok(resClaudeWrite.content[0].input.content.includes("qwenTest"));
    assert.equal(resClaudeWrite.stop_reason, "tool_use");

    // 20h. Claude Code tool_result follow-up
    const resClaudeFollowUp = await runCase("claude_code_tool_result_followup", {
        model: "qwen-agent",
        stream: false,
        messages: [
            { role: "user", content: "สร้างไฟล์ src/qwen-test.ts ใส่ export const qwenTest = true" },
            {
                role: "assistant",
                content: [
                    {
                        type: "tool_use",
                        id: "toolu_test_write",
                        name: "Write",
                        input: {
                            file_path: "src/qwen-test.ts",
                            content: "export const qwenTest = true;\n"
                        }
                    }
                ]
            },
            {
                role: "user",
                content: [
                    {
                        type: "tool_result",
                        tool_use_id: "toolu_test_write",
                        content: "File written successfully"
                    }
                ]
            }
        ],
        tools: [
            { name: "Write", description: "write file", input_schema: {} },
            { name: "Read", description: "read file", input_schema: {} }
        ]
    }, {
        status: 200,
        calls: 1,
        typeMatch: "text"
    }, (body) => {
        // Since tool result is returned, the mock Qwen should return a final text summary, not a tool_use block
        return {
            content: [{
                type: "text",
                text: "I have successfully created the file as requested."
            }],
            stop_reason: "end_turn"
        };
    });
    assert.equal(resClaudeFollowUp.content[0].type, "text");
    assert.equal(resClaudeFollowUp.stop_reason, "end_turn");

    // 21. Check traces export and summary
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
