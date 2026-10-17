/**
 * 基于共同标签的相关阅读。
 *
 * 纯函数：只接收已归一化的文章元信息，不依赖 Astro / content collections，
 * 中文与英文文章页可以共用同一套排序规则。
 */

export interface RelatedInput {
	slug: string;
	title: string;
	tags: string[];
	published: Date;
}

export interface RelatedArticle {
	slug: string;
	title: string;
	sharedTags: string[];
}

/** 未显式传入 limit 时最多返回 3 篇。 */
const DEFAULT_LIMIT = 3;

/** 标签统一 trim，丢弃空串，并按首次出现的顺序去重。 */
function normalizeTags(tags: string[]): string[] {
	const seen = new Set<string>();
	const normalized: string[] = [];
	for (const tag of tags) {
		const trimmed = tag.trim();
		if (trimmed === "" || seen.has(trimmed)) continue;
		seen.add(trimmed);
		normalized.push(trimmed);
	}
	return normalized;
}

/**
 * 选出与 currentSlug 共享至少 1 个标签的文章。
 *
 * 排序：共享标签数降序 → 发布时间降序 → slug 升序（字典序，不受 locale 影响）。
 * 当前文章不在 entries 里（拿不到它的标签）时返回空数组。
 */
export function buildRelatedByTags(
	entries: RelatedInput[],
	currentSlug: string,
	limit?: number,
): RelatedArticle[] {
	const current = entries.find((entry) => entry.slug === currentSlug);
	if (!current) return [];
	const currentTags = new Set(normalizeTags(current.tags));

	const ranked: Array<RelatedArticle & { published: Date }> = [];
	for (const entry of entries) {
		if (entry.slug === currentSlug) continue;
		const sharedTags = normalizeTags(entry.tags).filter((tag) =>
			currentTags.has(tag),
		);
		if (sharedTags.length === 0) continue;
		ranked.push({
			slug: entry.slug,
			title: entry.title,
			sharedTags,
			published: entry.published,
		});
	}

	ranked.sort((a, b) => {
		if (a.sharedTags.length !== b.sharedTags.length) {
			return b.sharedTags.length - a.sharedTags.length;
		}
		const publishedDiff = b.published.getTime() - a.published.getTime();
		if (publishedDiff !== 0) return publishedDiff;
		if (a.slug === b.slug) return 0;
		return a.slug < b.slug ? -1 : 1;
	});

	const max = Math.max(0, limit ?? DEFAULT_LIMIT);
	return ranked.slice(0, max).map(
		(item): RelatedArticle => ({
			slug: item.slug,
			title: item.title,
			sharedTags: item.sharedTags,
		}),
	);
}
