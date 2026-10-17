/**
 * 文章之间的阅读链。
 *
 * 旧实现依赖 `getSortedPosts()` 在构建时往 frontmatter 里写 `prevSlug/nextSlug`，
 * 顺序是纯发布时间：读完一篇强化学习的文章，下一篇可能接到因果推断。
 *
 * 这里改成「同主题优先」：
 * - 先在同一分类内按发布时间排（同一天的多篇用 slug 稳定排序）；
 * - 分类邻居不存在时，退回到全局发布时间的相邻文章，并在 UI 上标明。
 *
 * 纯函数，不依赖 Astro；中文页与英文页各传自己语言的文章列表。
 */

export interface ReadingChainInput {
	slug: string;
	title: string;
	published: Date;
	category?: string | null;
}

export interface ReadingChainLink {
	slug: string;
	title: string;
	/** 是否来自同一分类；false 表示按发布时间兜底的相邻文章。 */
	sameTopic: boolean;
	/** 同分类时的原始分类值，用于界面展示。 */
	topic: string | null;
}

export interface ReadingChain {
	prev: ReadingChainLink | null;
	next: ReadingChainLink | null;
}

const normalizeTopic = (category: string | null | undefined): string =>
	(category ?? "").trim();

/** 发布时间升序；同一天用 slug 做稳定排序，不依赖运行环境 locale。 */
const compareEntries = (a: ReadingChainInput, b: ReadingChainInput): number => {
	const byDate = a.published.getTime() - b.published.getTime();
	if (byDate !== 0) return byDate;
	if (a.slug === b.slug) return 0;
	return a.slug < b.slug ? -1 : 1;
};

const toLink = (
	entry: ReadingChainInput | null,
	sameTopic: boolean,
	topic: string,
): ReadingChainLink | null => {
	if (!entry) return null;
	return {
		slug: entry.slug,
		title: entry.title,
		sameTopic,
		topic: sameTopic ? topic : null,
	};
};

export function buildReadingChain(
	entries: ReadingChainInput[],
): Map<string, ReadingChain> {
	const sorted = [...entries].sort(compareEntries);

	const byTopic = new Map<string, ReadingChainInput[]>();
	for (const entry of sorted) {
		const topic = normalizeTopic(entry.category);
		if (!topic) continue;
		const list = byTopic.get(topic);
		if (list) list.push(entry);
		else byTopic.set(topic, [entry]);
	}

	const globalIndex = new Map<string, number>();
	sorted.forEach((entry, index) => {
		globalIndex.set(entry.slug, index);
	});

	const chain = new Map<string, ReadingChain>();
	for (const entry of sorted) {
		const topic = normalizeTopic(entry.category);
		const peers = topic ? (byTopic.get(topic) ?? []) : [];
		const peerIndex = peers.findIndex((peer) => peer.slug === entry.slug);
		const index = globalIndex.get(entry.slug) ?? 0;

		const previousPeer = peerIndex > 0 ? peers[peerIndex - 1] : null;
		const nextPeer =
			peerIndex >= 0 && peerIndex < peers.length - 1
				? peers[peerIndex + 1]
				: null;
		const previousGlobal = index > 0 ? sorted[index - 1] : null;
		const nextGlobal = index < sorted.length - 1 ? sorted[index + 1] : null;

		chain.set(entry.slug, {
			prev: previousPeer
				? toLink(previousPeer, true, topic)
				: toLink(previousGlobal, false, topic),
			next: nextPeer
				? toLink(nextPeer, true, topic)
				: toLink(nextGlobal, false, topic),
		});
	}

	return chain;
}
