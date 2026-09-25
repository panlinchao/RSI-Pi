import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

// Diagnostic extension only. It is never used as a formal candidate.
export default function (pi: ExtensionAPI) {
  console.error("RSI_TOOLS_PROBE_LOADED");
  pi.registerTool({
    name: "rsi_probe",
    label: "RSI probe",
    description: "Diagnostic tool for verifying extension registration.",
    parameters: Type.Object({}),
    async execute() {
      console.error("RSI_TOOLS_PROBE_EXECUTED");
      return { content: [{ type: "text", text: "probe ok" }], details: {} };
    },
  });
  pi.on("tool_call", async (event) => {
    console.error(`RSI_TOOLS_PROBE_TOOL_CALL:${event.toolName}`);
  });
}
