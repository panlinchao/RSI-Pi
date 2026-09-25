/**
 * Module B -- Execution strategy.  Version 2: pre-finish verification gate plus
 * a late-run checkpoint.
 *
 * v1 added one general policy: when the agent tries to end its run with a
 * final answer and no tool calls, it is given one mandated self-review before
 * the run is allowed to settle. That gate is kept unchanged here.
 *
 * Why v2. The last evaluation's two failures were not wrong answers reached by
 * a finishing agent -- they were runs killed by the harness token ceiling while
 * still issuing tool calls. In both, the run never produced the clean final
 * turn that v1's gate waits for, so the gate could not fire:
 *   - interpreter-functions-go-cattle-399 cleared 61/62 stages and died one
 *     edit short of the return-statement stage;
 *   - kafka-consuming-messages-go-eel-430 cleared 11/12 stages and failed only
 *     the stage whose spec requires topics sorted alphabetically by name.
 * v1 deliberately skipped long runs (`turn > MAX_TURNS_FOR_VERIFICATION`) on
 * the theory that they were near budget and not worth extending. The evidence
 * shows the opposite: the long runs are exactly the ones that never get a
 * conformance pass, and they are the ones that need it most.
 *
 * v2 therefore reuses the same one-time nudge, but when a run is still working
 * past the turn where the final-turn gate gives up, the self-review is injected
 * immediately as a steering message instead of waiting for a final turn that
 * may never come. The checkpoint is explicitly framed as a checkpoint, not the
 * end of the task, so the agent fixes any spec mismatch and then continues with
 * whatever stages remain. The two triggers share one `nudged` flag, so a task
 * still receives at most one forced review.
 *
 * Cost control. The checkpoint only fires on runs that already exceed the turn
 * budget the final-turn gate tolerates; short runs are untouched. It spends
 * budget that such a run was otherwise going to burn on more of the same work,
 * and it aims to convert a budget death into a completed spec check.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

/** Do not extend a run that already ran longer than this; it is near budget. */
const MAX_TURNS_FOR_VERIFICATION = 45;

/**
 * Once a run is still issuing tool calls at this turn, the final-turn gate may
 * never get a chance to fire, so the same review is injected as a checkpoint.
 */
const LATE_RUN_CHECKPOINT_TURN = MAX_TURNS_FOR_VERIFICATION;

/**
 * The single self-review the agent must perform before finishing. It is phrased
 * as required, concrete actions so the model cannot satisfy it by restating that
 * it already tested the happy path.
 */
const VERIFICATION_PROMPT = [
	"Before you finish, run one focused verification pass. Do not rewrite working code unless a check below actually fails.",
	"",
	"1. Re-read the task's README and specification/feature files. List every explicitly stated requirement, paying special attention to prose rules that are easy to miss: required output ordering (e.g. alphabetical/sorted), exact error codes or messages, null-vs-empty fields, and version or size limits.",
	"2. Run the project's real build and test commands (whatever the repository provides) and confirm they pass.",
	"3. Exercise the edge cases the spec implies with actual commands, not just the success path:",
	"   - empty, missing, and unknown inputs;",
	"   - multiple inputs supplied in a different order than the required output order;",
	"   - non-ASCII / multi-byte text in both input and output, checking the exact bytes printed.",
	"4. If any check fails, fix the code and re-run the checks. If everything already passes, leave the code as it is.",
	"",
	"Then give your final summary and state exactly which commands you ran and what they showed.",
].join("\n");

/**
 * Same review, injected mid-run for tasks that have not reached a natural
 * finish. The framing matters: the agent must not mistake this for permission
 * to stop, so it is told to continue afterwards and the checklist steps above
 * still allow a no-op result.
 */
const CHECKPOINT_PROMPT = [
	"Progress checkpoint: this task is taking a long time and a large share of its budget is already spent, but you have not reached a natural finish.",
	"Pause now and run the focused conformance pass below BEFORE spending more budget. This is not the end of the task. After fixing anything that does not match the spec, continue with any remaining required stages and only then give your final summary.",
	"",
	VERIFICATION_PROMPT,
].join("\n");

export default function (pi: ExtensionAPI) {
	// At most one forced review per task, whichever trigger fires first.
	let nudged = false;

	const startTask = () => {
		nudged = false;
	};

	const turnIndexOf = (event: { turnIndex?: number }): number | undefined =>
		typeof event?.turnIndex === "number" ? event.turnIndex : undefined;

	const isCleanFinalTurn = (event: {
		turnIndex?: number;
		message?: { role?: string; stopReason?: string };
		toolResults?: unknown;
	}): boolean => {
		try {
			const results = event?.toolResults;
			// A turn with tool calls is mid-work; only a text-only turn can end the run.
			if (Array.isArray(results) && results.length > 0) return false;

			const message = event?.message;
			if (!message || message.role !== "assistant") return false;
			// Only a normal completion may be extended; never an abort, error, or
			// output-length cut-off.
			if (message.stopReason !== "stop") return false;

			const turn = turnIndexOf(event);
			if (typeof turn === "number" && turn > MAX_TURNS_FOR_VERIFICATION) return false;

			return true;
		} catch {
			return false;
		}
	};

	const isLateRunCheckpoint = (event: {
		turnIndex?: number;
		message?: { role?: string; stopReason?: string };
	}): boolean => {
		try {
			if (nudged) return false;
			// A failed/aborted/length turn is not a stable place to reason from.
			const message = event?.message;
			if (message && message.role === "assistant") {
				const stop = message.stopReason;
				if (stop && stop !== "stop" && stop !== "toolUse" && stop !== "tool_use") {
					return false;
				}
			}
			const turn = turnIndexOf(event);
			return typeof turn === "number" && turn >= LATE_RUN_CHECKPOINT_TURN;
		} catch {
			return false;
		}
	};

	const hasToolCalls = (event: { toolResults?: unknown }): boolean => {
		const results = event?.toolResults;
		return Array.isArray(results) && results.length > 0;
	};

	const requestVerification = (
		prompt: string,
		mode: "steer" | "followUp",
	): void => {
		if (nudged) return;
		nudged = true;
		try {
			pi.sendUserMessage(prompt, { deliverAs: mode });
		} catch {
			// Delivery mode can be rejected depending on run state; fall back to
			// the mode that reliably starts a turn once the agent settles.
			try {
				pi.sendUserMessage(prompt, { deliverAs: "followUp" });
			} catch {
				// Verification is best-effort: never let it crash the run.
			}
		}
	};

	// Fresh session (or a reloaded one) means a fresh task.
	pi.on("session_start", () => {
		try {
			startTask();
		} catch {
			/* swallow */
		}
	});

	// If a session is reused for several tasks, only re-arm on genuine user input,
	// never on the follow-up message this module injects.
	pi.on("input", (event) => {
		try {
			if (event?.source !== "extension") startTask();
		} catch {
			/* swallow */
		}
	});

	// First point at which a text-only final answer is visible and a follow-up can
	// be queued without landing between a tool call and its result. If the run is
	// still working past the turn the final-turn gate tolerates, steer the same
	// review in now rather than waiting for a final turn that may never arrive.
	pi.on("turn_end", (event) => {
		try {
			if (isCleanFinalTurn(event)) {
				requestVerification(VERIFICATION_PROMPT, "followUp");
				return;
			}
			if (isLateRunCheckpoint(event)) {
				requestVerification(
					CHECKPOINT_PROMPT,
					hasToolCalls(event) ? "steer" : "followUp",
				);
			}
		} catch {
			/* swallow */
		}
	});
}
