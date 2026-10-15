export type SelectionRule = "color" | "position" | "shape";
export type CardScene = "familiar" | "color" | "position" | "task";

export interface SelectionCard {
	position: number;
	triangle: boolean;
	colorA: boolean;
	selected: boolean;
	expected: boolean;
	correct: boolean;
}

/** Explicit, deterministic rules; this function does not fit a model. */
export function selectCards(rule: SelectionRule, scene: CardScene): SelectionCard[] {
	const shapes = scene === "position"
		? [false, true, true, false, false, true]
		: [true, false, true, false, true, false];
	return shapes.map((triangle, index) => {
		const colorA = scene === "color" ? !triangle : triangle;
		const selected = rule === "color" ? colorA : rule === "position" ? index % 2 === 0 : triangle;
		const expected = scene === "task" ? !triangle : triangle;
		return { position: index + 1, triangle, colorA, selected, expected, correct: selected === expected };
	});
}

export interface RepairCard {
	id: string;
	clue: number;
	needsRepair: boolean;
}

/** Hand-authored teaching cards, not observations or model predictions. */
export const REPAIR_CARDS: readonly RepairCard[] = [
	{ id: "A", clue: 9, needsRepair: true },
	{ id: "B", clue: 8, needsRepair: true },
	{ id: "C", clue: 7, needsRepair: false },
	{ id: "D", clue: 6, needsRepair: false },
	{ id: "E", clue: 5, needsRepair: true },
	{ id: "F", clue: 4, needsRepair: false },
	{ id: "G", clue: 3, needsRepair: false },
	{ id: "H", clue: 2, needsRepair: false },
];

export interface RepairEvaluation {
	inspected: number;
	extraChecks: number;
	missedRepairs: number;
	cost: number;
}

export function evaluateRepair(threshold: number, missCost: number): RepairEvaluation {
	let inspected = 0;
	let extraChecks = 0;
	let missedRepairs = 0;
	for (const card of REPAIR_CARDS) {
		const selected = card.clue >= threshold;
		if (selected) inspected += 1;
		if (selected && !card.needsRepair) extraChecks += 1;
		if (!selected && card.needsRepair) missedRepairs += 1;
	}
	return { inspected, extraChecks, missedRepairs, cost: extraChecks + missedRepairs * missCost };
}
