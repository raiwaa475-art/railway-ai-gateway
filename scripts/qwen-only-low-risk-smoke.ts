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
        const responseBody = {
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

async function runCase(name: string, body: any, expected: { status: number; calls: number; rejection?: string }) {
    const fakeProvider = new FakeQwenProvider();
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
        return decision;
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

