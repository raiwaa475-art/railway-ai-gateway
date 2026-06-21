# Background Auto Coding Workflow

The Auto Coding system enables executing long-running coding loops completely in the background, utilizing Qwen as the local editor and DeepSeek/Claude as verifiers.

---

## Status Lifecycle

Background jobs transition through the following states:

```
[ queued ] ──> [ planning ] ──> [ qwen_working ] ──> [ build_testing ]
                                                             │
     ┌───────────────────────────────────────────────────────┘
     ▼
[ reviewing ] ──(pass)──> [ completed ]
     │
  (fail)
     ▼
[ needs_human ] / [ failed ]
```

- **queued**: The job is created and waiting to be processed.
- **planning**: The controller is writing the task steps.
- **qwen_working**: Qwen is calling Read/Edit/Write/Grep tools in the workspace.
- **build_testing**: The workspace is executing tests to verify the builds.
- **reviewing**: The verifier is reviewing the diff changes.
- **needs_human**: The verification failed, requiring human intervention.
- **completed**: The changes were successfully implemented and verified.
- **failed**: The job encountered an error or exceeded step counts.
- **cancelled**: The user manually stopped the job.

---

## Security and Safety Constraints

To protect the workspace and ensure code stability, background jobs enforce the following limitations:

- **No auto git push**: Changes are committed locally or left unstaged. Under no circumstances will a background job push code to remote repositories automatically.
- **No production deployment**: Automated triggers for vercel, railway, or similar production deploys are blocked.
- **No npm publish**: Publishing packages to public registries is blocked.
- **No secret modifications**: Writing passwords or API keys to files is blocked.
- **Max job steps limit**: Default is capped at `12` steps.
- **Max controller review rounds**: Default is capped at `2` loops.
- **Max Qwen tool rounds**: Default is capped at `8` rounds per loop.

---

## Endpoints

Administrative controls are exposed via endpoints:

- `POST /admin/auto/jobs` (starts job)
- `GET /admin/auto/jobs` (lists jobs)
- `GET /admin/auto/jobs/:id` (inspects details & event trace logs)
- `POST /admin/auto/jobs/:id/cancel` (aborts execution)
- `POST /admin/auto/jobs/:id/retry` (runs a duplicate job instance)
- `GET /admin/auto/summary` (provides queue efficiency and cost-saving metrics)
