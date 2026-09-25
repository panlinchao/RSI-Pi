/**
 * Module A -- Tools and skills.  Version 1.
 *
 * v0 was deliberately inert. This version makes two additions within the
 * tools/skills remit:
 *
 *   1. A `go_verify` helper tool that runs the whole Go check pipeline
 *      (gofmt -> go vet -> go build -> go test) in one call and stops at the
 *      first failure. The observed agent repeatedly spent 5-9 separate bash
 *      turns per task on those commands.
 *   2. Tool-usage guidelines carried by that tool. They target three concrete
 *      failure patterns seen in the last evaluation:
 *        - a leftover background test server shadowing the grader's fresh
 *          build (kafka-consuming-messages-go-hyena-304),
 *        - no end-to-end coverage of multi-byte UTF-8 text (interpreter-
 *          statements-and-state-go-yak-769),
 *        - no end-to-end coverage of multi-item / ordered collections
 *          (kafka-consuming-messages-go-eel-430).
 *
 * Nothing here touches execution strategy: no retry, no backoff, and no
 * control-flow intervention. `go_verify` only reports what the toolchain says.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { exec } from "node:child_process";
import { existsSync } from "node:fs";
import { join } from "node:path";

const STEP_TIMEOUT_MS = 180_000;
const MAX_OUTPUT_CHARS = 12_000;

function runStep(command: string, cwd: string): Promise<{ code: number; output: string }> {
	return new Promise((resolve) => {
		try {
			exec(
				command,
				{ cwd, timeout: STEP_TIMEOUT_MS, killSignal: "SIGKILL", maxBuffer: 8 * 1024 * 1024 },
				(error, stdout, stderr) => {
					const raw = `${stdout ?? ""}${stderr ?? ""}`.trim();
					let code = 0;
					if (error) {
						const e = error as { code?: number | string; killed?: boolean };
						code = typeof e.code === "number" ? e.code : e.killed ? 124 : 1;
					}
					const output =
						raw.length > MAX_OUTPUT_CHARS ? `${raw.slice(0, MAX_OUTPUT_CHARS)}\n[... output truncated]` : raw;
					resolve({ code, output });
				},
			);
		} catch (err) {
			resolve({ code: 1, output: `step failed to start: ${err instanceof Error ? err.message : String(err)}` });
		}
	});
}

export default function (pi: ExtensionAPI) {
	pi.registerTool({
		name: "go_verify",
		label: "Go verify",
		description:
			"Run the Go toolchain checks for the project in one step: `gofmt -l .`, `go vet ./...`, `go build ./...`, then `go test ./...` (short-circuiting after the first failure). Use this after editing Go files instead of separate gofmt / go build / go vet / go test bash calls, and before finishing a task.",
		promptSnippet: "Run gofmt, go vet, go build, and go test for the Go project in one call",
		promptGuidelines: [
			"Use go_verify after editing Go code to check formatting, vet, build, and tests in a single call; prefer it over separate gofmt, go vet, go build, and go test bash commands.",
			"Before finishing a task that started a network server or any other long-running process for testing, stop it (for example `pkill -f <binary>` or `fuser -k <port>/tcp`) and confirm the port is free; a leftover process can shadow the binary the grader builds and tests.",
			"Batch independent shell commands into a single bash call with `&&` or `;` rather than issuing several separate calls, and give every server test a bounded timeout so it cannot hang.",
			"When using go_verify or running the project tests, make sure the coverage includes at least one multi-byte UTF-8 (non-ASCII) input and at least one collection with several items in a non-trivial order; these are common sources of otherwise-invisible failures.",
		],
		parameters: Type.Object({
			dir: Type.Optional(
				Type.String({ description: "Project directory to check (defaults to the current working directory)." }),
			),
		}),
		async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
			try {
				const cwd =
					params && typeof params.dir === "string" && params.dir.length > 0 ? params.dir : ctx?.cwd ?? process.cwd();
				if (!existsSync(join(cwd, "go.mod"))) {
					return {
						content: [
							{ type: "text", text: `No go.mod found in ${cwd}; go_verify only supports Go modules.` },
						],
						details: {},
					};
				}
				const steps: Array<{ label: string; command: string }> = [
					{ label: "gofmt -l .", command: "gofmt -l ." },
					{ label: "go vet ./...", command: "go vet ./..." },
					{ label: "go build ./...", command: "go build ./..." },
					{ label: "go test ./...", command: "go test ./..." },
				];
				const lines: string[] = [];
				let failed = false;
				for (const step of steps) {
					const result = await runStep(step.command, cwd);
					if (result.code !== 0) {
						failed = true;
						lines.push(`FAIL ${step.label} (exit ${result.code})\n${result.output || "(no output)"}`);
						break;
					}
					lines.push(result.output ? `ok ${step.label}\n${result.output}` : `ok ${step.label}`);
				}
				return {
					content: [{ type: "text", text: lines.join("\n\n") }],
					details: { failed },
				};
			} catch (err) {
				return {
					content: [
						{ type: "text", text: `go_verify error: ${err instanceof Error ? err.message : String(err)}` },
					],
					details: {},
				};
			}
		},
	});
}
