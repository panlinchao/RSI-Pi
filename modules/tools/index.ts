/**
 * Module A -- Tools and skills.  Version 0 (baseline).
 *
 * This is the agent's starting point for the tools/skills arm of the RSI
 * experiment. v0 is deliberately INERT: it registers nothing and hooks nothing,
 * so an agent running with this file behaves exactly like stock pi. Both arms
 * must start from that same original agent, so an inert v0 is a requirement,
 * not an oversight.
 *
 * What this module is allowed to change is written down in SPEC.md next to this
 * file. In short: tool descriptions, tool wrappers, and reusable helper tools.
 * Execution strategy (retry / verification / stopping) belongs to the other
 * arm and must not be touched here.
 *
 * The scaffolding below is commented out on purpose. It documents the API this
 * module may use, so a modification can be a small edit rather than a rewrite
 * of the pi extension API from memory.
 *
 * ---------------------------------------------------------------------------
 * import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
 * import { Type } from "typebox";        // both are virtual modules: always importable
 *
 * export default function (pi: ExtensionAPI) {
 *   // 1. Register a new tool the LLM can call.
 *   pi.registerTool({
 *     name: "my_tool",
 *     label: "My tool",
 *     description: "What the tool does, and when to reach for it.",  // <- the model reads this
 *     parameters: Type.Object({ path: Type.String({ description: "..." }) }),
 *     async execute(toolCallId, params, signal, onUpdate, ctx) {
 *       return { content: [{ type: "text", text: "result" }], details: {} };
 *     },
 *   });
 *
 *   // 2. Reword a built-in tool's description for every call, without replacing
 *   //    the tool itself. Mutating `event.input` in place patches arguments.
 *   pi.on("tool_call", async (event) => {
 *     if (event.toolName === "bash") { /* ... *\/ }
 *   });
 * }
 * ---------------------------------------------------------------------------
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export default function (_pi: ExtensionAPI) {
	// v0: no tools registered, no descriptions changed.
}
