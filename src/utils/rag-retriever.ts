import type { RagChunk } from "@/pages/api/rag-chunks.json";

export type { RagChunk };

export interface RagSearchResult {
	chunk: RagChunk;
	score: number;
}

let cachedChunks: RagChunk[] | null = null;
let fetchPromise: Promise<RagChunk[]> | null = null;

/**
 * Load and cache all site chunks from the static Astro endpoint.
 */
export async function loadRagChunks(): Promise<RagChunk[]> {
	if (cachedChunks) return cachedChunks;
	if (fetchPromise) return fetchPromise;

	fetchPromise = (async () => {
		try {
			const res = await fetch("/api/rag-chunks.json");
			if (!res.ok) {
				throw new Error(`HTTP ${res.status}`);
			}
			const data: RagChunk[] = await res.json();
			cachedChunks = data;
			return data;
		} catch (err) {
			console.warn("Failed to load /api/rag-chunks.json:", err);
			return [];
		} finally {
			fetchPromise = null;
		}
	})();

	return fetchPromise;
}

/**
 * Tokenize query into English terms and Chinese characters / bi-grams.
 */
export function tokenizeQuery(text: string): string[] {
	const tokens: string[] = [];
	const lower = text.toLowerCase();

	// English & alphanumeric identifiers
	const enTokens = lower.match(/[a-z0-9_-]{2,}/g) || [];
	tokens.push(...enTokens);

	// Chinese characters & bi-grams
	const cjkMatch = lower.match(/[\u4e00-\u9fa5]/g) || [];
	for (let i = 0; i < cjkMatch.length; i++) {
		tokens.push(cjkMatch[i]);
		if (i + 1 < cjkMatch.length) {
			tokens.push(cjkMatch[i] + cjkMatch[i + 1]);
		}
	}
	return tokens;
}

/**
 * Attempt to connect to Pagefind if present in the build.
 */
interface PagefindResultItem {
	data: () => Promise<{
		url: string;
		excerpt: string;
		meta?: { title?: string };
	}>;
}

interface PagefindInstance {
	search: (
		query: string,
	) => Promise<{ results: PagefindResultItem[] } | null | undefined>;
}

async function getPagefindInstance(): Promise<PagefindInstance | null> {
	if (typeof window === "undefined") return null;
	const win = window as unknown as { __pagefind?: PagefindInstance };
	if (win.__pagefind) return win.__pagefind;

	try {
		// Use dynamic URL to avoid vite build errors if pagefind is absent
		const pagefindUrl = "/pagefind/pagefind.js";
		const pf = (await import(
			/* @vite-ignore */ pagefindUrl
		)) as PagefindInstance;
		if (pf && typeof pf.search === "function") {
			win.__pagefind = pf;
			return pf;
		}
	} catch {
		// Pagefind not generated/available in current environment
	}
	return null;
}

/**
 * Retrieve Top-K knowledge chunks relevant to the user query.
 */
export async function retrieveKnowledge(
	query: string,
	options: {
		topK?: number;
		lang?: "zh" | "en" | "all";
		excludePostTitle?: string;
	} = {},
): Promise<RagSearchResult[]> {
	const { topK = 4, lang = "all", excludePostTitle = "" } = options;
	const trimmedQuery = query.trim();
	if (!trimmedQuery) return [];

	const chunks = await loadRagChunks();
	if (!chunks || chunks.length === 0) return [];

	// Filter by lang if specified
	const filteredChunks = chunks.filter((c) => {
		if (lang !== "all" && c.lang !== lang) return false;
		if (excludePostTitle && c.postTitle === excludePostTitle) return false;
		return true;
	});

	// Check Pagefind for external boost
	const pagefindUrls = new Set<string>();
	try {
		const pf = await getPagefindInstance();
		if (pf) {
			const pfSearch = await pf.search(trimmedQuery);
			if (pfSearch && Array.isArray(pfSearch.results)) {
				const topPf = pfSearch.results.slice(0, 5);
				for (const item of topPf) {
					const data = await item.data();
					if (data?.url) {
						pagefindUrls.add(data.url.replace(/\/index\.html$/, "/"));
					}
				}
			}
		}
	} catch (e) {
		console.warn("Pagefind search query error:", e);
	}

	const qTokens = tokenizeQuery(trimmedQuery);
	const qLower = trimmedQuery.toLowerCase();

	const scored = filteredChunks.map((chunk) => {
		let score = 0;
		const titleLower = chunk.postTitle.toLowerCase();
		const headingLower = chunk.heading.toLowerCase();
		const textLower = chunk.text.toLowerCase();
		const tagsLower = chunk.tags.map((t) => t.toLowerCase()).join(" ");

		// Pagefind boost
		if (
			pagefindUrls.has(chunk.url) ||
			pagefindUrls.has(chunk.url.split("#")[0])
		) {
			score += 65;
		}

		// Exact full phrase bonus
		if (qLower.length > 2) {
			if (titleLower.includes(qLower)) score += 80;
			if (headingLower.includes(qLower)) score += 50;
			if (tagsLower.includes(qLower)) score += 60;
			if (textLower.includes(qLower)) score += 30;
		}

		// Token matches
		for (const tok of qTokens) {
			if (tok.length === 1) {
				if (titleLower.includes(tok)) score += 1.5;
				if (headingLower.includes(tok)) score += 1.0;
				if (tagsLower.includes(tok)) score += 1.5;
				if (textLower.includes(tok)) score += 0.2;
			} else {
				if (titleLower.includes(tok)) score += 10;
				if (headingLower.includes(tok)) score += 7;
				if (tagsLower.includes(tok)) score += 8;
				if (textLower.includes(tok)) score += 2.5;
			}
		}

		return { chunk, score };
	});

	scored.sort((a, b) => b.score - a.score);
	return scored.filter((s) => s.score > 0).slice(0, topK);
}

/**
 * Format retrieved chunks into a prompt-friendly context block.
 */
export function formatRagContextForPrompt(
	results: RagSearchResult[],
	lang: "zh" | "en" = "zh",
): string {
	if (results.length === 0) return "";

	const isEn = lang === "en";
	const header = isEn
		? "--- RELEVANT KNOWLEDGE BASE CONTEXT ---"
		: "--- 全站相关知识库参考资料 ---";

	const blocks = results.map((r, i) => {
		const num = i + 1;
		const chunk = r.chunk;
		return `[参考资料 ${num}] 《${chunk.postTitle}》· 章节: ${chunk.heading} (URL: ${chunk.url})
${chunk.text}`;
	});

	return `${header}\n${blocks.join("\n\n")}\n--- END OF CONTEXT ---`;
}
