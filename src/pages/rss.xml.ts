import rss, { type RSSFeedItem } from "@astrojs/rss";
import { getSortedPosts } from "@utils/content-utils";
import { formatDateI18nWithTime } from "@utils/date-utils";
import { url } from "@utils/url-utils";
import type { APIContext } from "astro";
import { siteConfig } from "@/config";

/**
 * 只输出摘要的订阅源。
 *
 * 此前每篇都渲染完整正文塞进 content:encoded，37 篇合计 4.1MB，
 * 阅读器和抓取端都会因此超时。现在改为「标题 + description + 链接」，
 * 体积从 4.1MB 降到几十 KB；想读全文的读者点链接即可。
 */
const FEED_LIMIT = 20;

export async function GET(context: APIContext): Promise<Response> {
	const blog = (await getSortedPosts())
		.filter((post) => !post.data.password)
		.slice(0, FEED_LIMIT);

	const items: RSSFeedItem[] = blog.map((post) => {
		const description = post.data.description.trim() || post.data.title;
		return {
			title: post.data.title,
			pubDate: post.data.published,
			description,
			link: url(`/posts/${post.id}/`),
		};
	});

	return rss({
		title: siteConfig.title,
		description: siteConfig.subtitle || "No description",
		site: context.site ?? "https://meiluosi.github.io",
		customData: `<generator>Astro</generator>
		<lastBuildDate>${formatDateI18nWithTime(new Date())}</lastBuildDate>`,
		items,
	});
}
