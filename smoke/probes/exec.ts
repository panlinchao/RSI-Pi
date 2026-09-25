import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

// Diagnostic extension only. It is never used as a formal candidate.
export default function (pi: ExtensionAPI) {
  console.error("RSI_EXEC_PROBE_LOADED");
  pi.on("tool_call", async (event) => {
    console.error(`RSI_EXEC_PROBE_TOOL_CALL:${event.toolName}`);
  });
}
