# pi extension API — the subset these modules may use

Curated from the vendored pi 0.86.1 docs (`docs/extensions.md`). Both import
specifiers below are **virtual modules**: pi resolves them itself, so an
extension works from any directory and needs no `node_modules`.

```typescript
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type, type Static } from "typebox";
```

An extension is a default-exported factory. It may be `async`, but must not
start timers, sockets, or watchers from the factory body.

```typescript
export default function (pi: ExtensionAPI) { /* register here */ }
```

## Registering a tool

```typescript
pi.registerTool({
  name: "run_tests",              // the name the model calls
  label: "Run tests",
  description: "…",               // the model reads this; it decides whether to call
  promptSnippet: "…",             // optional one-liner in the system prompt tool list
  promptGuidelines: ["…"],        // optional bullets appended to system prompt guidelines
  parameters: Type.Object({
    path: Type.Optional(Type.String({ description: "…" })),
  }),
  async execute(toolCallId, params, signal, onUpdate, ctx) {
    // params is typed from `parameters`
    return {
      content: [{ type: "text", text: "output the model will see" }],
      details: {},                // free-form, not shown to the model
      isError: false,             // optional
    };
  },
});
```

## Events

Handlers run in extension load order. Returning `undefined` means "no change".

| Event | Payload | Return value |
| --- | --- | --- |
| `tool_call` | `toolName`, `toolCallId`, `input` (mutable) | `{ block?, reason?, terminate? }` |
| `tool_result` | `toolName`, `toolCallId`, `input`, `content`, `details`, `isError`, `usage` | `{ content?, details?, isError?, usage? }` |
| `turn_end` | `turnIndex`, `message`, `toolResults` | `void` |
| `agent_end` | `messages` (whole transcript) | `void` |
| `message_end` | `message` (assistant messages carry `usage`) | `{ message? }` |
| `session_start` | `reason` | `void` |

Key semantics, quoted from the docs:

- `tool_call` fires after `tool_execution_start` and before execution. Mutating
  `event.input` in place **does** affect the actual execution, and later handlers
  see earlier mutations. No re-validation happens after a mutation.
- `terminate: true` only applies to a call you also `block`ed, and the agent
  stops early **only when every finalized tool result in the batch is
  terminating**. Blocking a single call does not end the run.
- In the default parallel tool mode, sibling calls from one assistant message are
  preflighted sequentially and then executed concurrently, so a `tool_call`
  handler does not reliably see sibling results from the same message.
- `tool_result` handlers chain like middleware; returning a partial patch leaves
  the omitted fields unchanged.
- `turn_end` fires once per assistant turn with that turn's `toolResults`.

Type-narrowing helpers are exported from the same module:
`isToolCallEventType("bash", event)`, `isBashToolResult(event)`, and so on.

## Context (`ctx`)

Useful members: `ctx.cwd`, `ctx.hasUI`, `ctx.signal` (AbortSignal for the current
stream), `ctx.abort()`, `ctx.isIdle()`, `ctx.getContextUsage()`,
`ctx.sessionManager`, `ctx.ui.notify(...)`. The context is not a filesystem
handle — read and write files with `node:fs`, which is always available.

## Constraints inside a task container

- The agent runs as a non-root user with `cwd=/app`; the extension is loaded from
  `/tmp/pi-rsi/module/index.ts`.
- There is **no `node_modules`** next to the extension. Only the virtual modules
  above and Node built-ins (`node:fs`, `node:path`, `node:child_process`, …)
  resolve. Any other import fails at load time and forfeits the task.
- Prefer `pi.exec`-free designs; if you shell out, use `node:child_process` and
  always swallow errors — a throwing handler degrades the run.
- The run is capped by the harness (tool calls and tokens). Nothing here can
  raise that cap.
