/**
 * Module B -- Execution strategy.  Version 0 (baseline).
 *
 * This is the agent's starting point for the execution-strategy arm of the RSI
 * experiment. v0 is deliberately INERT: it registers nothing and hooks nothing,
 * so an agent running with this file behaves exactly like stock pi. Both arms
 * must start from that same original agent, so an inert v0 is a requirement,
 * not an oversight.
 *
 * What this module is allowed to change is written down in SPEC.md next to this
 * file. In short: retry, verification, and stopping logic. Tool descriptions,
 * tool wrappers and helper tools belong to the other arm and must not be
 * touched here.
 *
 * The scaffolding below is commented out on purpose. It documents the API this
 * module may use, so a modification can be a small edit rather than a rewrite
 * of the pi extension API from memory.
 *
 * ---------------------------------------------------------------------------
 * import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
 *
 * export default function (pi: ExtensionAPI) {
 *   // Fires before a tool runs. Return { block, reason, terminate } to stop it.
 *   // `terminate: true` ends the run once the whole tool batch settles.
 *   pi.on("tool_call", async (event, ctx) => {
 *     if (event.toolName === "bash" && /rm -rf/.test(event.input.command ?? "")) {
 *       return { block: true, reason: "refused" };
 *     }
 *   });
 *
 *   // Fires after a tool runs. `event.isError` says whether it failed;
 *   // return { content, details, isError } to rewrite what the model sees.
 *   pi.on("tool_result", async (event) => { /* ... *\/ });
 *
 *   // Fires once per assistant turn; `event.toolResults` is the batch.
 *   pi.on("turn_end", async (event) => { /* ... *\/ });
 *
 *   // Fires when the agent is done. `event.messages` is the whole transcript.
 *   pi.on("agent_end", async (event) => { /* ... *\/ });
 * }
 * ---------------------------------------------------------------------------
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export default function (_pi: ExtensionAPI) {
	// v0: no retry, verification, or stopping logic installed.
}
