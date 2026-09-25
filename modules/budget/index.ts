/**
 * Fixed budget guard for the RSI experiment. NOT evolvable.
 *
 * The experiment requires the per-task token and tool-call limits to be
 * identical across every condition (original agent, module A, module B). pi has
 * no built-in step or token ceiling, so the ceiling is imposed here, by the
 * harness, where neither evolving module can reach it.
 *
 * This file is mounted read-only into every trial alongside whichever evolvable
 * module is under test, and is loaded before it.
 *
 * Limits arrive as environment variables so the same file serves every run:
 *   PI_RSI_MAX_TOOL_CALLS  - integer, 0 or unset means unlimited
 *   PI_RSI_MAX_TOKENS      - integer, 0 or unset means unlimited
 *   PI_RSI_BUDGET_REPORT   - path to write the JSON accounting record to
 *
 * When a limit is crossed every further tool call is blocked with
 * `terminate: true`, which is pi's mechanism for ending the run once the
 * current tool batch settles. The agent is told why, so the transcript shows a
 * budget stop rather than a mysterious hang.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { writeFileSync } from "node:fs";

function readLimit(name: string): number {
	const raw = Number(process.env[name] ?? "0");
	return Number.isFinite(raw) && raw > 0 ? raw : Number.POSITIVE_INFINITY;
}

export default function (pi: ExtensionAPI) {
	const maxToolCalls = readLimit("PI_RSI_MAX_TOOL_CALLS");
	const maxTokens = readLimit("PI_RSI_MAX_TOKENS");
	const reportPath = process.env.PI_RSI_BUDGET_REPORT ?? "/tmp/pi-rsi/budget.json";

	let toolCalls = 0;
	let inputTokens = 0;
	let outputTokens = 0;
	let cacheReadTokens = 0;
	let cacheWriteTokens = 0;
	let costUsd = 0;
	let assistantMessages = 0;
	// Why we stopped, if we did. null means the agent finished on its own.
	let exhaustedBy: string | null = null;

	const totalTokens = () => inputTokens + outputTokens + cacheReadTokens + cacheWriteTokens;

	function report() {
		try {
			writeFileSync(
				reportPath,
				JSON.stringify(
					{
						tool_calls: toolCalls,
						assistant_messages: assistantMessages,
						input_tokens: inputTokens,
						output_tokens: outputTokens,
						cache_read_tokens: cacheReadTokens,
						cache_write_tokens: cacheWriteTokens,
						total_tokens: totalTokens(),
						cost_usd: costUsd,
						max_tool_calls: Number.isFinite(maxToolCalls) ? maxToolCalls : null,
						max_tokens: Number.isFinite(maxTokens) ? maxTokens : null,
						exhausted_by: exhaustedBy,
					},
					null,
					2,
				),
			);
		} catch {
			// A missing report must not take down the trial; the runner falls
			// back to parsing pi's own JSONL transcript.
		}
	}

	pi.on("tool_call", async () => {
		if (exhaustedBy) {
			return { block: true, terminate: true, reason: stopMessage() };
		}
		toolCalls += 1;
		if (toolCalls > maxToolCalls) {
			exhaustedBy = "tool_calls";
			report();
			return { block: true, terminate: true, reason: stopMessage() };
		}
		report();
	});

	pi.on("message_end", async (event) => {
		const message = event.message as { role?: string; usage?: Record<string, number> };
		if (message.role !== "assistant" || !message.usage) return;
		const usage = message.usage;
		assistantMessages += 1;
		inputTokens += usage.input ?? 0;
		outputTokens += usage.output ?? 0;
		cacheReadTokens += usage.cacheRead ?? 0;
		cacheWriteTokens += usage.cacheWrite ?? 0;
		// `cost` is an object on the real Usage type; tolerate a flat number too.
		const cost = (message.usage as { cost?: number | { total?: number } }).cost;
		costUsd += typeof cost === "number" ? cost : (cost?.total ?? 0);
		if (totalTokens() > maxTokens && !exhaustedBy) {
			exhaustedBy = "tokens";
		}
		report();
	});

	pi.on("agent_end", async () => {
		report();
	});

	function stopMessage(): string {
		return (
			`Budget exhausted (${exhaustedBy === "tokens" ? "token" : "tool-call"} limit reached: ` +
			`${toolCalls} tool calls, ${totalTokens()} tokens). ` +
			"No further tool calls are available. Stop now and summarize what you have."
		);
	}
}
