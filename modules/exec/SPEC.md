# Module B — Execution strategy

The evolvable module under test in the execution-strategy arm of the experiment.

## What a modification may change

A candidate modification edits `index.ts` in this directory and may only:

- add retry or backoff logic for a failed action, within the fixed per-task
  budget;
- add verification logic — for example requiring that the task's tests be run,
  and pass, before the agent is allowed to finish;
- add stopping logic — for example refusing to repeat an action that has already
  failed the same way, or ending early once the goal is demonstrably met;
- add reusable helper functions or constants that support the above.

The unit of change is the control flow around the model's actions: when to try
again, when to check, and when to stop.

## What it must not change

- **The tool surface.** No `pi.registerTool()`, and no rewriting of tool
  descriptions or tool results to change what a tool *is*. Blocking a specific
  call as part of a retry/verification policy is in scope; adding or redefining
  a tool is not. That is module A's axis, and it is held fixed while this module
  is under test.
- **Budgets.** The per-task token and tool-call ceilings are set by the harness
  and are not this module's business. This module may decide how to *spend* the
  remaining budget, but may not raise it.
- **The model, decoding settings, or the task set.**

## Boundary check

The runner statically rejects a candidate that calls `pi.registerTool()`. An
invalid proposal gets one more proposal attempt; if that also fails, the run
stops for diagnosis.
