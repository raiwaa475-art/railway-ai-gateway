import { Request, Response as ExpressResponse } from "express";
import { providerRegistry } from "./registry.js";
import { DeepSeekProvider } from "../providers/deepseek.js";
import { QwenLocalProvider } from "../providers/qwen-local.js";

export class OrchestratorService {
    static async handleTwinModels(req: Request, res: ExpressResponse): Promise<void> {
        const clientHeaders: Record<string, string> = {
            "user-agent": req.header("user-agent") || "railway-ai-gateway"
        };

        const deepseekProvider = providerRegistry.getProvider("deepseek") as DeepSeekProvider;
        const qwenProvider = providerRegistry.getProvider("qwen-local") as QwenLocalProvider;

        if (!deepseekProvider || !qwenProvider) {
            res.status(500).json({
                error: {
                    type: "gateway_error",
                    message: "Required providers are not registered."
                }
            });
            return;
        }

        // Prompt Engineering Pattern: Software Architect Planner
        const plannerSystemPrompt = "คุณคือสถาปนิกซอฟต์แวร์ จงวิเคราะห์โค้ดและคำสั่งต่อไปนี้ จากนั้นให้สรุป 'แผนงานการแก้ไขทีละสเต็ป' เป็นข้อๆ อย่างกระชับ โดยไม่ต้องเขียนโค้ดเต็มไฟล์ออกมาเด็ดขาด ย้ำ! เน้นเฉพาะตรรกะและวิธีแก้" + 
            (req.body.system ? `\n\n${req.body.system}` : "");

        console.log("[Orchestrator] Requesting planning blueprint from DeepSeek...");

        try {
            // Step 1 & 2: Plan via DeepSeek API (non-streaming, async)
            const deepseekBody = {
                ...req.body,
                system: plannerSystemPrompt,
                stream: false
            };

            const deepseekRes = await deepseekProvider.handleRequest(deepseekBody, clientHeaders);
            if (!deepseekRes.ok) {
                const errText = await deepseekRes.text();
                console.error("[Orchestrator] DeepSeek planning failed:", errText);
                res.status(deepseekRes.status).send(errText);
                return;
            }

            const deepseekData = await deepseekRes.json();
            const planText = deepseekData.content?.[0]?.text || "";
            console.log("[Orchestrator] Plan generated successfully. Length:", planText.length);

            // Step 3: Reconstruct Payload
            let codingInstruction = "";
            const messages = req.body.messages || [];
            for (let i = messages.length - 1; i >= 0; i--) {
                if (messages[i].role === "user") {
                    const content = messages[i].content;
                    if (typeof content === "string") {
                        codingInstruction = content;
                    } else if (Array.isArray(content)) {
                        codingInstruction = content
                            .map((block: any) => (block?.type === "text" ? block.text : ""))
                            .join("");
                    }
                    break;
                }
            }

            const workerSystemPrompt = "คุณคือวิศวกรคอมพิวเตอร์หน้าที่เขียนโค้ด จงนำ 'แผนงานจากสถาปนิก' ที่แนบมานี้ ไปลงมือเขียนโค้ดจริงให้สมบูรณ์แบบที่สุด พร้อมสตรีมผลลัพธ์กลับไป";
            const qwenUserMessageContent = `แผนงานจากสถาปนิก:\n${planText}\n\nคำสั่งสั่งเขียนโค้ด:\n${codingInstruction}`;

            const qwenBody = {
                ...req.body,
                system: workerSystemPrompt,
                messages: [
                    {
                        role: "user",
                        content: qwenUserMessageContent
                    }
                ],
                stream: req.body.stream ?? true
            };

            // Step 4: Execute Qwen Local
            console.log("[Orchestrator] Executing coding task on Qwen Local...");
            let qwenRes: Response;
            try {
                qwenRes = await qwenProvider.handleRequest(qwenBody, clientHeaders);
            } catch (qwenError) {
                console.warn("[Orchestrator] Qwen Local is offline or connection failed. Triggering Failover. Error:", qwenError);
                await this.handleFailover(req, res, clientHeaders);
                return;
            }

            if (!qwenRes.ok || qwenRes.status >= 500) {
                console.warn(`[Orchestrator] Qwen Local failed with status ${qwenRes.status}. Triggering Failover.`);
                await this.handleFailover(req, res, clientHeaders);
                return;
            }

            // Stream response or return text
            res.status(qwenRes.status);
            const contentType = qwenRes.headers.get("content-type");
            if (contentType) {
                res.setHeader("content-type", contentType);
            }

            const isStream = !!req.body.stream;
            if (isStream && qwenRes.body) {
                res.setHeader("cache-control", "no-cache");
                res.setHeader("connection", "keep-alive");

                const reader = qwenRes.body.getReader();
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    res.write(Buffer.from(value));
                }
                res.end();
            } else {
                const text = await qwenRes.text();
                res.send(text);
            }

        } catch (error) {
            console.error("[Orchestrator] General error:", error);
            await this.handleFailover(req, res, clientHeaders);
        }
    }

    private static async handleFailover(req: Request, res: ExpressResponse, clientHeaders: Record<string, string>): Promise<void> {
        console.log("[Orchestrator] Smart Failover: Routing the original request to DeepSeek...");
        try {
            const deepseekProvider = providerRegistry.getProvider("deepseek") as DeepSeekProvider;
            const deepseekRes = await deepseekProvider.handleRequest(req.body, clientHeaders);

            res.status(deepseekRes.status);
            const contentType = deepseekRes.headers.get("content-type");
            if (contentType) {
                res.setHeader("content-type", contentType);
            }

            const isStream = !!req.body.stream;
            if (isStream && deepseekRes.body) {
                res.setHeader("cache-control", "no-cache");
                res.setHeader("connection", "keep-alive");

                const reader = deepseekRes.body.getReader();
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    res.write(Buffer.from(value));
                }
                res.end();
            } else {
                const text = await deepseekRes.text();
                res.send(text);
            }
        } catch (failoverError) {
            console.error("[Orchestrator] Smart Failover also failed:", failoverError);
            res.status(500).json({
                error: {
                    type: "gateway_error",
                    message: "Both Qwen Local and DeepSeek Failover failed"
                }
            });
        }
    }
}
