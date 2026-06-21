import { Router, Request, Response } from "express";
import { pool } from "../utils/db.js";

export const qwenTuningRouter = Router();

// GET /admin/qwen-agent/tuning/insights
qwenTuningRouter.get("/tuning/insights", async (req: Request, res: Response) => {
    if (!pool) {
        return res.json({
            fakeJsonPatterns: [],
            wrongToolAliases: [],
            missingArgs: [],
            invalidEditCalls: [],
            unsafeBashCommands: [],
            loopFailures: 0,
            finalAnswerFailures: [],
            recommendedRules: []
        });
    }

    try {
        // 1. Fake JSON patterns
        const fakeJsonRes = await pool.query(`
            SELECT qwen_raw_output, count(*) as count 
            FROM qwen_agent_traces 
            WHERE fake_tool_json_detected = true 
            GROUP BY qwen_raw_output 
            ORDER BY count DESC LIMIT 10
        `);

        // 2. Wrong tool aliases
        const toolAliasesRes = await pool.query(`
            SELECT requested_tool_name, normalized_tool_name, count(*) as count 
            FROM qwen_agent_traces 
            WHERE requested_tool_name IS NOT NULL AND requested_tool_name != normalized_tool_name 
            GROUP BY requested_tool_name, normalized_tool_name 
            ORDER BY count DESC LIMIT 10
        `);

        // 3. Validation errors / missing args
        const validationErrorsRes = await pool.query(`
            SELECT tool_validation_error, count(*) as count 
            FROM qwen_agent_traces 
            WHERE tool_validation_error IS NOT NULL 
            GROUP BY tool_validation_error 
            ORDER BY count DESC LIMIT 10
        `);

        // 4. Unsafe Bash commands
        const unsafeBashRes = await pool.query(`
            SELECT original_tool_args, tool_validation_error, count(*) as count 
            FROM qwen_agent_traces 
            WHERE tool_validation_error LIKE 'Dangerous command blocked%' 
            GROUP BY original_tool_args, tool_validation_error 
            ORDER BY count DESC LIMIT 10
        `);

        // 5. Loop failures
        const loopFailuresRes = await pool.query(`
            SELECT count(*) as count 
            FROM qwen_agent_traces 
            WHERE failure_reason LIKE '%max tool rounds%'
        `);

        // 6. Final answer failures
        const finalFailuresRes = await pool.query(`
            SELECT failure_reason, count(*) as count 
            FROM qwen_agent_traces 
            WHERE success = false AND failure_reason IS NOT NULL 
            GROUP BY failure_reason 
            ORDER BY count DESC LIMIT 10
        `);

        // 7. Formulate recommended adapter rules based on common failures
        const recommendedRules: any[] = [];
        for (const row of toolAliasesRes.rows) {
            recommendedRules.push({
                rule_type: "tool_alias",
                match_pattern: row.requested_tool_name,
                replacement: row.normalized_tool_name,
                description: `Automatically created to normalize '${row.requested_tool_name}' to '${row.normalized_tool_name}'`
            });
        }
        for (const row of validationErrorsRes.rows) {
            const err = String(row.tool_validation_error);
            if (err.includes("requires")) {
                const parts = err.split(" ");
                const tool = parts[0];
                const arg = parts[parts.length - 1];
                recommendedRules.push({
                    rule_type: "arg_alias",
                    match_pattern: `${tool}:${arg}`,
                    replacement: arg,
                    description: `Hint to repair arg for ${tool} (detected validation error: ${err})`
                });
            }
        }

        res.json({
            fakeJsonPatterns: fakeJsonRes.rows,
            wrongToolAliases: toolAliasesRes.rows,
            missingArgs: validationErrorsRes.rows.filter(r => String(r.tool_validation_error).includes("requires")),
            invalidEditCalls: validationErrorsRes.rows.filter(r => String(r.tool_validation_error).includes("Edit")),
            unsafeBashCommands: unsafeBashRes.rows,
            loopFailures: parseInt(loopFailuresRes.rows[0]?.count || "0", 10),
            finalAnswerFailures: finalFailuresRes.rows,
            recommendedRules
        });

    } catch (err: any) {
        res.status(500).json({ error: err.message });
    }
});

// GET /admin/qwen-agent/adapter/rules
qwenTuningRouter.get("/adapter/rules", async (req: Request, res: Response) => {
    if (!pool) return res.json([]);
    try {
        const result = await pool.query("SELECT * FROM qwen_adapter_rules ORDER BY created_at DESC");
        res.json(result.rows);
    } catch (err: any) {
        res.status(500).json({ error: err.message });
    }
});

// POST /admin/qwen-agent/adapter/rules
qwenTuningRouter.post("/adapter/rules", async (req: Request, res: Response) => {
    if (!pool) return res.status(500).json({ error: "Database not connected" });
    try {
        const { enabled, rule_type, match_pattern, replacement, description } = req.body;
        const result = await pool.query(
            `INSERT INTO qwen_adapter_rules (enabled, rule_type, match_pattern, replacement, description) 
             VALUES ($1, $2, $3, $4, $5) RETURNING *`,
            [enabled !== false, rule_type, match_pattern, replacement || null, description || null]
        );
        res.status(201).json(result.rows[0]);
    } catch (err: any) {
        res.status(500).json({ error: err.message });
    }
});

// PATCH /admin/qwen-agent/adapter/rules/:id
qwenTuningRouter.patch("/adapter/rules/:id", async (req: Request, res: Response) => {
    if (!pool) return res.status(500).json({ error: "Database not connected" });
    try {
        const { id } = req.params;
        const fields: string[] = [];
        const values: any[] = [];
        let idx = 1;

        const allowedFields = ["enabled", "rule_type", "match_pattern", "replacement", "description", "hit_count"];
        for (const f of allowedFields) {
            if (req.body[f] !== undefined) {
                fields.push(`${f} = $${idx++}`);
                values.push(req.body[f]);
            }
        }

        if (fields.length === 0) {
            return res.status(400).json({ error: "No fields to update" });
        }

        values.push(id);
        const query = `UPDATE qwen_adapter_rules SET ${fields.join(", ")}, updated_at = CURRENT_TIMESTAMP WHERE id = $${idx} RETURNING *`;
        const result = await pool.query(query, values);
        if (result.rows.length === 0) {
            return res.status(404).json({ error: "Rule not found" });
        }
        res.json(result.rows[0]);
    } catch (err: any) {
        res.status(500).json({ error: err.message });
    }
});

// DELETE /admin/qwen-agent/adapter/rules/:id
qwenTuningRouter.delete("/adapter/rules/:id", async (req: Request, res: Response) => {
    if (!pool) return res.status(500).json({ error: "Database not connected" });
    try {
        const { id } = req.params;
        const result = await pool.query("DELETE FROM qwen_adapter_rules WHERE id = $1 RETURNING *", [id]);
        if (result.rows.length === 0) {
            return res.status(404).json({ error: "Rule not found" });
        }
        res.json({ ok: true });
    } catch (err: any) {
        res.status(500).json({ error: err.message });
    }
});

// GET /admin/qwen-agent/prompt-profiles
qwenTuningRouter.get("/prompt-profiles", async (req: Request, res: Response) => {
    if (!pool) return res.json([]);
    try {
        const result = await pool.query("SELECT * FROM qwen_prompt_profiles ORDER BY created_at DESC");
        res.json(result.rows);
    } catch (err: any) {
        res.status(500).json({ error: err.message });
    }
});

// POST /admin/qwen-agent/prompt-profiles
qwenTuningRouter.post("/prompt-profiles", async (req: Request, res: Response) => {
    if (!pool) return res.status(500).json({ error: "Database not connected" });
    try {
        const { name, enabled, system_prompt, purpose } = req.body;
        
        // If enabling this profile, disable all other profiles first
        if (enabled === true) {
            await pool.query("UPDATE qwen_prompt_profiles SET enabled = false");
        }

        const result = await pool.query(
            `INSERT INTO qwen_prompt_profiles (name, enabled, system_prompt, purpose) 
             VALUES ($1, $2, $3, $4) ON CONFLICT (name) DO UPDATE SET enabled = EXCLUDED.enabled, system_prompt = EXCLUDED.system_prompt, purpose = EXCLUDED.purpose RETURNING *`,
            [name, enabled === true, system_prompt, purpose || null]
        );
        res.status(201).json(result.rows[0]);
    } catch (err: any) {
        res.status(500).json({ error: err.message });
    }
});

// PATCH /admin/qwen-agent/prompt-profiles/:id
qwenTuningRouter.patch("/prompt-profiles/:id", async (req: Request, res: Response) => {
    if (!pool) return res.status(500).json({ error: "Database not connected" });
    try {
        const { id } = req.params;
        const { enabled, name, system_prompt, purpose } = req.body;

        // If enabling this profile, disable all other profiles first
        if (enabled === true) {
            await pool.query("UPDATE qwen_prompt_profiles SET enabled = false");
        }

        const fields: string[] = [];
        const values: any[] = [];
        let idx = 1;

        if (enabled !== undefined) {
            fields.push(`enabled = $${idx++}`);
            values.push(enabled);
        }
        if (name !== undefined) {
            fields.push(`name = $${idx++}`);
            values.push(name);
        }
        if (system_prompt !== undefined) {
            fields.push(`system_prompt = $${idx++}`);
            values.push(system_prompt);
        }
        if (purpose !== undefined) {
            fields.push(`purpose = $${idx++}`);
            values.push(purpose);
        }

        if (fields.length === 0) {
            return res.status(400).json({ error: "No fields to update" });
        }

        values.push(id);
        const query = `UPDATE qwen_prompt_profiles SET ${fields.join(", ")}, updated_at = CURRENT_TIMESTAMP WHERE id = $${idx} RETURNING *`;
        const result = await pool.query(query, values);
        if (result.rows.length === 0) {
            return res.status(404).json({ error: "Prompt profile not found" });
        }
        res.json(result.rows[0]);
    } catch (err: any) {
        res.status(500).json({ error: err.message });
    }
});
