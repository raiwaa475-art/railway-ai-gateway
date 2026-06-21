# Qwen Agent Integration & Adapter Engine

The `qwen-agent` configuration turns local Qwen models into resilient coding workers that execute commands directly inside the user's environment.

---

## Model Identifiers

- **ID**: `qwen-agent`
- **Display Name**: `Qwen Agent`
- **Provider ID**: `qwen-local`
- **Aliases**: `qwen-agent`, `qwen-code`

---

## Tool Alias Normalization

Qwen agent automatically normalizes varying tool calls outputted by local models to canonical Anthropic tool definitions:

| Input Alias Pattern | Canonical Tool Name |
|---|---|
| `read_file`, `ReadFile`, `cat`, `open`, `view` | `Read` |
| `list_files`, `dir`, `ls` | `LS` |
| `search`, `grep_search` | `Grep` |
| `find_files` | `Glob` |
| `write_file` | `Write` |
| `edit_file`, `replace` | `Edit` |
| `multi_edit` | `MultiEdit` |
| `run_command`, `shell`, `cmd` | `Bash` |
| `todowrite`, `todo_write` | `TodoWrite` |

---

## Fake JSON Translation

If Qwen outputs structural JSON inside a text block instead of a native tool call block (e.g. `{"name": "Read", "arguments": {"file_path": "index.js"}}`), the adapter interceptor parses it, generates a valid `tool_use_id`, and formats the payload structure to match Anthropic message requirements before returning to Claude Code.

---

## Validation & Arguments Repair

The gateway ensures arguments conform to required properties:
- **Read**: requires `file_path`
- **Edit**: requires `file_path`, `old_string`, `new_string`
- **Write**: requires `file_path`, `content`
- **Grep**: requires `pattern`
- **Glob**: requires `pattern`
- **Bash**: requires `command`

If verification fails, the gateway runs a one-time validation repair call giving the specific failure error back to Qwen.

---

## Unsafe Command Shield (Bash)

The `Bash` tool is restricted from executing destructive operations. Any command matching the following patterns is blocked instantly:
- `rm -rf`, `del /s`
- `format`, `shutdown`, `reboot`
- `curl | bash`, `wget | bash`
- `npm publish`, `git push`, `railway up`, `vercel --prod`
- Any writing to `.env` or `secrets` files
- Any commands attempting to read/expose `private keys` or `API keys`
