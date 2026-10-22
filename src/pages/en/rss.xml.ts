import { getCollection } from "astro:content";
import rss, { type RSSFeedItem } from "@astrojs/rss";
import { formatDateI18nWithTime } from "@utils/date-utils";
import { removeFileExtension, url } from "@utils/url-utils";
import type { APIContext } from "astro";
import { siteConfig } from "@/config";

/**
 * 英文文章与概览订阅源 (English RSS Feed).
 *
 * 只输出经过审校并与已发布源文章配对的英文概览与长卷，
 * 采用「标题 + description + 链接」的紧凑摘要输出。
 */
const FEED_LIMIT = 20;

export async function GET(context: APIContext): Promise<Response> {
	const [entries, sources] = await Promise.all([
		getCollection("enPosts"),
		getCollection("posts", ({ data }) => !data.draft && !data.password),
	]);

	const publicSources = new Map(
		sources.map((source) => [removeFileExtension(source.id), source]),
	);

	// 仅收录源文章已发布且未加密的英文篇目
	const validEntries = entries
		.filter((entry) =>
			publicSources.has(removeFileExtension(entry.data.sourceId)),
		)
		.sort((a, b) => b.data.published.getTime() - a.data.published.getTime())
		.slice(0, FEED_LIMIT);

	const items: RSSFeedItem[] = validEntries.map((post) => {
		const description = post.data.description?.trim() || post.data.title;
		return {
			title: post.data.title,
			pubDate: post.data.published,
			description,
			link: url(`/en/posts/${post.data.slug}/`),
		};
	});

	return rss({
		title: `${siteConfig.title} - English Overviews & Essays`,
		description:
			"Research notes, essays, and technical overviews on AI, RL, causal inference, and LLMs by Feng Yu.",
		site: context.site ?? "https://meiluosi.github.io",
		customData: `<generator>Astro</generator>
		<language>en</language>
		<lastBuildDate>${formatDateI18nWithTime(new Date())}</lastBuildDate>`,
		items,
	});
}
