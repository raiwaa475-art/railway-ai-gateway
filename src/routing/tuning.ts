import { pool } from "../utils/db.js";

export interface AdapterRule {
    id: number;
    enabled: boolean;
    rule_type: "tool_alias" | "arg_alias" | "bash_block" | "retry_hint" | "system_prompt_hint";
    match_pattern: string;
    replacement: string | null;
    description: string | null;
    hit_count: number;
}

export interface PromptProfile {
    id: number;
    name: string;
    enabled: boolean;
    system_prompt: string;
    purpose: string | null;
    success_count: number;
    failure_count: number;
}

// Fetch enabled adapter rules from database
export async function getEnabledAdapterRules(): Promise<AdapterRule[]> {
    if (!pool) return [];
    try {
        const res = await pool.query("SELECT * FROM qwen_adapter_rules WHERE enabled = true");
        return res.rows.map(row => ({
            id: row.id,
            enabled: row.enabled,
            rule_type: row.rule_type,
            match_pattern: row.match_pattern,
            replacement: row.replacement,
            description: row.description,
            hit_count: row.hit_count
        }));
    } catch (err) {
        console.error("Failed to fetch adapter rules:", err);
        return [];
    }
}

// Increment adapter rule hit count in database
export async function incrementRuleHit(ruleId: number): Promise<void> {
    if (!pool) return;
    try {
        await pool.query("UPDATE qwen_adapter_rules SET hit_count = hit_count + 1 WHERE id = $1", [ruleId]);
    } catch (err) {
        console.error("Failed to increment rule hit count:", err);
    }
}

// Fetch active prompt profile from database
export async function getActivePromptProfile(): Promise<PromptProfile | null> {
    if (!pool) return null;
    try {
        const res = await pool.query("SELECT * FROM qwen_prompt_profiles WHERE enabled = true LIMIT 1");
        if (res.rows.length > 0) {
            const row = res.rows[0];
            return {
                id: row.id,
                name: row.name,
                enabled: row.enabled,
                system_prompt: row.system_prompt,
                purpose: row.purpose,
                success_count: row.success_count,
                failure_count: row.failure_count
            };
        }
    } catch (err) {
        console.error("Failed to fetch active prompt profile:", err);
    }
    return null;
}

// Update profile success/failure count
export async function updateProfileStats(name: string, success: boolean): Promise<void> {
    if (!pool) return;
    try {
        const column = success ? "success_count" : "failure_count";
        await pool.query(`UPDATE qwen_prompt_profiles SET ${column} = ${column} + 1 WHERE name = $1`, [name]);
    } catch (err) {
        console.error("Failed to update prompt profile stats:", err);
    }
}

// Helper to check and apply tool alias rules
export async function applyToolAliasRules(toolName: string, rules: AdapterRule[]): Promise<{ name: string; hitRuleId: number | null }> {
    let currentName = toolName;
    let hitRuleId: number | null = null;

    const toolAliasRules = rules.filter(r => r.rule_type === "tool_alias");
    for (const rule of toolAliasRules) {
        let isMatch = false;
        try {
            // Check if match_pattern is regex
            if (rule.match_pattern.startsWith("/") && rule.match_pattern.endsWith("/")) {
                const regex = new RegExp(rule.match_pattern.slice(1, -1), "i");
                isMatch = regex.test(currentName);
            } else {
                isMatch = currentName.toLowerCase() === rule.match_pattern.toLowerCase();
            }
        } catch {
            isMatch = currentName.toLowerCase() === rule.match_pattern.toLowerCase();
        }

        if (isMatch && rule.replacement) {
            currentName = rule.replacement;
            hitRuleId = rule.id;
            break; // Stop at first matching rule
        }
    }

    return { name: currentName, hitRuleId };
}

// Helper to check and apply argument alias rules
export async function applyArgAliasRules(toolName: string, input: any, rules: AdapterRule[]): Promise<{ repairedInput: any; repaired: boolean; hitRuleId: number | null }> {
    if (!input || typeof input !== "object" || Array.isArray(input)) {
        return { repairedInput: input, repaired: false, hitRuleId: null };
    }

    const repairedInput = { ...input };
    let repaired = false;
    let hitRuleId: number | null = null;

    const argAliasRules = rules.filter(r => r.rule_type === "arg_alias");
    for (const rule of argAliasRules) {
        // match_pattern can be "old_arg" or "ToolName:old_arg"
        let targetTool: string | null = null;
        let oldArgKey = rule.match_pattern;

        if (rule.match_pattern.includes(":")) {
            const parts = rule.match_pattern.split(":");
            targetTool = parts[0];
            oldArgKey = parts[1];
        }

        if (targetTool && targetTool.toLowerCase() !== toolName.toLowerCase()) {
            continue;
        }

        // Search for matching argument key in input
        const matchingKey = Object.keys(repairedInput).find(k => k.toLowerCase() === oldArgKey.toLowerCase());
        if (matchingKey && rule.replacement && !(rule.replacement in repairedInput)) {
            repairedInput[rule.replacement] = repairedInput[matchingKey];
            delete repairedInput[matchingKey];
            repaired = true;
            hitRuleId = rule.id;
            break; // Apply one rule at a time
        }
    }

    return { repairedInput, repaired, hitRuleId };
}

// Helper to check bash blocking rules
export async function checkBashBlockRules(command: string, rules: AdapterRule[]): Promise<{ blocked: boolean; reason: string | null; hitRuleId: number | null }> {
    const bashBlockRules = rules.filter(r => r.rule_type === "bash_block");
    for (const rule of bashBlockRules) {
        let isMatch = false;
        try {
            if (rule.match_pattern.startsWith("/") && rule.match_pattern.endsWith("/")) {
                const regex = new RegExp(rule.match_pattern.slice(1, -1), "i");
                isMatch = regex.test(command);
            } else {
                isMatch = command.toLowerCase().includes(rule.match_pattern.toLowerCase());
            }
        } catch {
            isMatch = command.toLowerCase().includes(rule.match_pattern.toLowerCase());
        }

        if (isMatch) {
            return {
                blocked: true,
                reason: rule.description || `Blocked by rule: ${rule.match_pattern}`,
                hitRuleId: rule.id
            };
        }
    }
    return { blocked: false, reason: null, hitRuleId: null };
}

// Helper to check retry hint rules
export async function getRetryHintRule(validationError: string, rules: AdapterRule[]): Promise<{ hint: string | null; hitRuleId: number | null }> {
    const retryHintRules = rules.filter(r => r.rule_type === "retry_hint");
    for (const rule of retryHintRules) {
        let isMatch = false;
        try {
            if (rule.match_pattern.startsWith("/") && rule.match_pattern.endsWith("/")) {
                const regex = new RegExp(rule.match_pattern.slice(1, -1), "i");
                isMatch = regex.test(validationError);
            } else {
                isMatch = validationError.toLowerCase().includes(rule.match_pattern.toLowerCase());
            }
        } catch {
            isMatch = validationError.toLowerCase().includes(rule.match_pattern.toLowerCase());
        }

        if (isMatch && rule.replacement) {
            return { hint: rule.replacement, hitRuleId: rule.id };
        }
    }
    return { hint: null, hitRuleId: null };
}

// Helper to check system prompt hint rules
export async function applySystemPromptHintRules(systemPrompt: string, userIntent: string, rules: AdapterRule[]): Promise<string> {
    let finalPrompt = systemPrompt;
    const hintRules = rules.filter(r => r.rule_type === "system_prompt_hint");
    for (const rule of hintRules) {
        let isMatch = false;
        try {
            if (rule.match_pattern.startsWith("/") && rule.match_pattern.endsWith("/")) {
                const regex = new RegExp(rule.match_pattern.slice(1, -1), "i");
                isMatch = regex.test(userIntent);
            } else {
                isMatch = userIntent.toLowerCase().includes(rule.match_pattern.toLowerCase());
            }
        } catch {
            isMatch = userIntent.toLowerCase().includes(rule.match_pattern.toLowerCase());
        }

        if (isMatch && rule.replacement) {
            finalPrompt += `\n\n${rule.replacement}`;
            await incrementRuleHit(rule.id);
        }
    }
    return finalPrompt;
}
