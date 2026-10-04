/** Small, deterministic teaching calculations. These do not train a language model. */
export interface BpeMerge {
	left: string;
	right: string;
	count: number;
}

export interface ToyTokenizer {
	alphabet: string[];
	vocabulary: string[];
	merges: BpeMerge[];
	words: string[][];
}

function mergePair(tokens: string[], left: string, right: string): string[] {
	const merged: string[] = [];
	for (let index = 0; index < tokens.length; index++) {
		if (tokens[index] === left && tokens[index + 1] === right) {
			merged.push(left + right);
			index++;
		} else merged.push(tokens[index]);
	}
	return merged;
}

/** Character BPE inside whitespace-delimited words; ties use first occurrence. */
export function trainToyBpe(
	corpus: string[],
	mergeCount: number,
): ToyTokenizer {
	let words = corpus.flatMap((text) =>
		text
			.split(/\s+/)
			.filter(Boolean)
			.map((word) => [...word]),
	);
	const alphabet = [...new Set(words.flat())];
	const vocabulary = ["<unk>", ...alphabet];
	const merges: BpeMerge[] = [];
	for (let step = 0; step < Math.max(0, Math.floor(mergeCount)); step++) {
		const counts = new Map<string, BpeMerge>();
		for (const word of words) {
			for (let index = 0; index < word.length - 1; index++) {
				const left = word[index];
				const right = word[index + 1];
				const key = JSON.stringify([left, right]);
				const pair = counts.get(key);
				if (pair) pair.count++;
				else counts.set(key, { left, right, count: 1 });
			}
		}
		const best = [...counts.values()].sort((a, b) => b.count - a.count)[0];
		if (!best) break;
		merges.push(best);
		vocabulary.push(best.left + best.right);
		words = words.map((word) => mergePair(word, best.left, best.right));
	}
	return { alphabet, vocabulary, merges, words };
}

export function encodeToyBpe(
	text: string,
	tokenizer: ToyTokenizer,
): string[][] {
	return text
		.split(/\s+/)
		.filter(Boolean)
		.map((word) => {
			let tokens = [...word].map((character) =>
				tokenizer.alphabet.includes(character) ? character : "<unk>",
			);
			for (const merge of tokenizer.merges)
				tokens = mergePair(tokens, merge.left, merge.right);
			return tokens;
		});
}

export interface LossSummary {
	count: number;
	total: number;
	mean: number | null;
	perplexity: number | null;
}

/** Natural-log NLL over only the selected target tokens. */
export function maskedNll(
	probabilities: number[],
	mask: boolean[],
): LossSummary {
	if (probabilities.length !== mask.length)
		throw new Error("Probability and mask lengths must match.");
	let count = 0;
	let total = 0;
	for (let index = 0; index < probabilities.length; index++) {
		if (!mask[index]) continue;
		const probability = probabilities[index];
		if (!Number.isFinite(probability) || probability <= 0 || probability > 1)
			throw new Error("Selected probabilities must be in (0, 1].");
		total -= Math.log(probability);
		count++;
	}
	const mean = count ? total / count : null;
	return {
		count,
		total,
		mean,
		perplexity: mean === null ? null : Math.exp(mean),
	};
}

/** Single-pair DPO: differences are sums of response token log-probabilities. */
export function dpoPairLoss(
	policyGap: number,
	referenceGap: number,
	beta: number,
): number {
	if (![policyGap, referenceGap, beta].every(Number.isFinite) || beta <= 0)
		throw new Error("DPO inputs must be finite and beta positive.");
	const negativeMargin = -beta * (policyGap - referenceGap);
	return (
		Math.max(negativeMargin, 0) +
		Math.log1p(Math.exp(-Math.abs(negativeMargin)))
	);
}

export interface TrainingBudget {
	stateGiB: number;
	weightGiB: number;
	tokensPerUpdate: number;
	tokens: number;
}

export function trainingBudget(
	parameters: number,
	context: number,
	microbatch: number,
	accumulation: number,
	updates: number,
): TrainingBudget {
	if (
		![parameters, context, microbatch, accumulation, updates].every(
			(value) => Number.isSafeInteger(value) && value > 0,
		)
	)
		throw new Error("Budget inputs must be positive safe integers.");
	const tokensPerUpdate = context * microbatch * accumulation;
	return {
		stateGiB: (parameters * 16) / 2 ** 30,
		weightGiB: (parameters * 4) / 2 ** 30,
		tokensPerUpdate,
		tokens: tokensPerUpdate * updates,
	};
}
