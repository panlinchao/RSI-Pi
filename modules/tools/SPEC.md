# Module A — Tools and skills

The evolvable module under test in the tools/skills arm of the experiment.

## What a modification may change

A candidate modification edits `index.ts` in this directory and may only:

- register a new tool via `pi.registerTool()` — including its name, description,
  parameter schema, and implementation;
- reword, or attach guidance to, a built-in tool the model already has
  (`promptSnippet`, `promptGuidelines`, and argument-patching on `tool_call`);
- add reusable helper functions or constants that support the above.

The unit of change is the tool surface the model reasons about: what tools
exist, what they are called, and how their descriptions frame when to use them.

## What it must not change

- **Execution strategy.** No retry, backoff, verification gating, or stopping
  logic. Do not call `pi.on("turn_end")` or `pi.on("agent_end")` to alter
  control flow, and do not block a tool call to enforce a policy. That is
  module B's axis, and it is held fixed while this module is under test.
- **Budgets.** The per-task token and tool-call ceilings are set by the harness
  and are not this module's business.
- **The model, decoding settings, or the task set.**

## Boundary check

The runner statically rejects a candidate that mentions any of the execution
hooks (`turn_end`, `agent_end`, `agent_settled`) or that returns a
`block`/`terminate` verdict from a `tool_call` handler. An invalid proposal gets
one more proposal attempt; if that also fails, the run stops for diagnosis.
