import { getCollection } from "astro:content";
import { markdownToMdast, type MdastNode } from "satteri";
import { siteConfig } from "@/config";

export interface Backlink {
	sourceId: string;
	title: string;
	href: string;
	section: string;
	english: boolean;
}
type BacklinkIndex = Map<string, Backlink[]>;
let cached: Promise<BacklinkIndex> | undefined;

function plainText(node: MdastNode): string {
	if ("value" in node && typeof node.value === "string") return node.value;
	return "children" in node
		? node.children.map((child: MdastNode) => plainText(child)).join("")
		: "";
}

async function buildIndex(): Promise<BacklinkIndex> {
	const [posts, translations] = await Promise.all([
		getCollection("posts", ({ data }) => !data.draft && !data.password),
		getCollection("enPosts"),
	]);
	const records = posts.map((post) => ({
		sourceId: post.id.replace(/\.mdx?$/, ""),
		title: post.data.title,
		href: `/posts/${post.id.replace(/\.mdx?$/, "")}/`,
		body: post.body ?? "",
		english: false,
	}));
	const published = new Set(records.map((post) => post.sourceId));
	for (const post of translations)
		if (published.has(post.data.sourceId))
			records.push({
				sourceId: post.data.sourceId,
				title: post.data.title,
				href: `/en/posts/${post.data.slug}/`,
				body: post.body ?? "",
				english: true,
			});
	const origin = new URL(siteConfig.site_url).origin;
	const aliases = new Map(
		records.map((post) => [post.href.toLowerCase(), post.sourceId]),
	);
	const index: BacklinkIndex = new Map();
	for (const post of records) {
		const tree = markdownToMdast(post.body);
		const definitions = new Map<string, string>();
		const walk = (node: MdastNode, fn: (node: MdastNode) => void): void => {
			fn(node);
			if ("children" in node)
				for (const child of node.children) walk(child as MdastNode, fn);
		};
		walk(tree, (node) => {
			if (node.type === "definition")
				definitions.set(node.identifier.toLowerCase(), node.url);
		});
		let section = "";
		const seen = new Set<string>();
		walk(tree, (node) => {
			if (node.type === "heading") section = plainText(node);
			const destination =
				node.type === "link"
					? node.url
					: node.type === "linkReference"
						? definitions.get(node.identifier.toLowerCase())
						: undefined;
			if (!destination) return;
			try {
				const url = new URL(destination, origin + post.href);
				if (url.origin !== origin) return;
				const pathname =
					`${decodeURIComponent(url.pathname).replace(/\/$/, "")}/`.toLowerCase();
				const target = aliases.get(pathname);
				if (!target || target === post.sourceId || seen.has(target)) return;
				seen.add(target);
				const incoming = index.get(target) ?? [];
				incoming.push({
					sourceId: post.sourceId,
					title: post.title,
					href: post.href,
					section,
					english: post.english,
				});
				index.set(target, incoming);
			} catch {
				/* Invalid URLs do not create a relationship. */
			}
		});
	}
	return index;
}

export async function getBacklinks(
	sourceId: string,
	english = false,
): Promise<Backlink[]> {
	const index = await (import.meta.env.DEV
		? buildIndex()
		: (cached ??= buildIndex()));
	const matches = [...(index.get(sourceId) ?? [])].sort(
		(a, b) => Number(b.english === english) - Number(a.english === english),
	);
	return [
		...new Map(
			matches
				.map((item) => item.sourceId)
				.map((id) => [id, matches.find((item) => item.sourceId === id)!]),
		).values(),
	];
}
