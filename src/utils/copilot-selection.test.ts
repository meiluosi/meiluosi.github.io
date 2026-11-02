import assert from "node:assert/strict";
import test from "node:test";
import { getCopilotSelection } from "./copilot-selection";

const node = {} as Node;
const inside = {
	contains: (candidate: Node) => candidate === node,
} as HTMLElement;
const outside = { contains: () => false } as unknown as HTMLElement;
function selection(text: string, collapsed = false): Selection {
	return {
		isCollapsed: collapsed,
		rangeCount: 1,
		toString: () => text,
		getRangeAt: () => ({
			commonAncestorContainer: node,
			getBoundingClientRect: () => ({ top: 100, left: 20, width: 40 }),
		}),
	} as unknown as Selection;
}

test("the quick action uses trimmed article text and viewport position", () => {
	assert.deepEqual(getCopilotSelection(selection(" example "), inside, 0, 50), {
		text: "example",
		top: 112,
		left: 16,
	});
});

test("the quick action ignores invalid, collapsed and outside selections", () => {
	for (const value of [
		selection("a"),
		selection("a".repeat(501)),
		selection("text", true),
	]) {
		assert.equal(getCopilotSelection(value, inside, 0, 0), null);
	}
	assert.equal(getCopilotSelection(selection("text"), outside, 0, 0), null);
	assert.equal(getCopilotSelection(null, inside, 0, 0), null);
});
