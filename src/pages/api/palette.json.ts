import { getCollection } from "astro:content";
import type { APIRoute } from "astro";
import { removeFileExtension } from "@/utils/url-utils";

type PaletteKind = "post" | "en-post" | "page";

type PaletteItem = {
	id: string;
	title: string;
	titleEn?: string;
	href: string;
	tags: string[];
	category: string;
	description: string;
	kind: PaletteKind;
	lang: "zh" | "en";
};

// Static navigational entries. Descriptions stay short on purpose: the palette
// only needs enough context to rank a hit, not page copy.
const pages: PaletteItem[] = [
	{
		id: "page-notes",
		title: "技术文章与研究笔记",
		titleEn: "Technical articles and research notes",
		href: "/notes/",
		tags: ["文章", "笔记", "目录", "搜索", "writing", "notes"],
		category: "",
		description: "全部文章与笔记目录，可按主题、形式与关键词筛选。",
		kind: "page",
		lang: "zh",
	},
	{
		id: "page-work",
		title: "研究项目与工程实践",
		titleEn: "Research and engineering projects",
		href: "/work/",
		tags: ["项目", "作品", "案例", "projects", "work"],
		category: "",
		description: "项目、案例与研究方向的入口。",
		kind: "page",
		lang: "zh",
	},
	{
		id: "page-about",
		title: "关于枫语",
		titleEn: "About Feng Yu",
		href: "/about/",
		tags: ["关于", "作者", "简介", "about"],
		category: "",
		description: "个人简介、经历与研究兴趣。",
		kind: "page",
		lang: "zh",
	},
	{
		id: "page-contact",
		title: "联系与交流",
		titleEn: "Contact",
		href: "/contact/",
		tags: ["联系", "邮箱", "交流", "contact", "email"],
		category: "",
		description: "留言、邮件与交流主题。",
		kind: "page",
		lang: "zh",
	},
	{
		id: "page-library",
		title: "研究文献与开源资料",
		titleEn: "Research literature and open-source resources",
		href: "/library/",
		tags: ["文献", "论文", "资料", "library", "papers"],
		category: "",
		description: "论文、教材与开源项目导读。",
		kind: "page",
		lang: "zh",
	},
	{
		id: "page-rsi",
		title: "递归自我改进：机制与评估",
		titleEn: "Recursive self-improvement: mechanisms and evaluation",
		href: "/rsi/",
		tags: ["RSI", "研究", "导读", "self-improvement"],
		category: "",
		description: "RSI 的机制、评估与开放问题。",
		kind: "page",
		lang: "zh",
	},
	{
		id: "page-lab",
		title: "算法与模型交互实验",
		titleEn: "Interactive algorithm and model experiments",
		href: "/lab/",
		tags: ["实验", "互动", "MCTS", "因果推断", "lab", "experiments"],
		category: "",
		description: "MCTS、因果推断与模型全链路互动实验。",
		kind: "page",
		lang: "zh",
	},
	{
		id: "page-world",
		title: "人工智能研究工作台",
		titleEn: "AI research workspace",
		href: "/world/",
		tags: ["研究岛", "工作台", "3D", "world", "island"],
		category: "",
		description: "模型训练、搜索决策与因果评估的入口。",
		kind: "page",
		lang: "zh",
	},
	{
		id: "page-strandbeest",
		title: "Strandbeest 机构设计与运动仿真",
		titleEn: "Strandbeest mechanism design and simulation",
		href: "/essays/strandbeest/",
		tags: ["Strandbeest", "连杆", "仿真", "essay"],
		category: "",
		description: "连杆几何与运动仿真的视觉随笔。",
		kind: "page",
		lang: "zh",
	},
];

// Keep the payload small: titles and descriptions are what the palette ranks,
// article bodies stay out of this file.
const trim = (value: string, max = 160): string => {
	const text = value.replace(/\s+/g, " ").trim();
	return text.length > max ? `${text.slice(0, max - 1).trimEnd()}…` : text;
};

export const GET: APIRoute = async () => {
	const [posts, translations] = await Promise.all([
		getCollection("posts"),
		getCollection("enPosts"),
	]);

	const publishedPosts = posts.filter(
		(post) => !post.data.draft && !post.data.password,
	);
	const publishedIds = new Set(
		publishedPosts.map((post) => removeFileExtension(post.id)),
	);
	const translationBySource = new Map<string, (typeof translations)[number]>();
	for (const entry of translations) {
		translationBySource.set(removeFileExtension(entry.data.sourceId), entry);
	}

	const postItems: PaletteItem[] = publishedPosts.map((post) => {
		const id = removeFileExtension(post.id);
		const translation = translationBySource.get(id);
		return {
			id: `zh:${post.id}`,
			title: post.data.title,
			...(translation ? { titleEn: translation.data.title } : {}),
			href: `/posts/${id}/`,
			tags: post.data.tags.slice(0, 8),
			category: post.data.category ?? "",
			description: trim(post.data.description ?? ""),
			kind: "post",
			lang: "zh",
		};
	});

	// A translation is only public when its Chinese source is public, mirroring
	// the rule used by the /en/posts/[slug]/ routes.
	const enItems: PaletteItem[] = translations
		.filter((entry) =>
			publishedIds.has(removeFileExtension(entry.data.sourceId)),
		)
		.map((entry) => ({
			id: `en:${entry.id}`,
			title: entry.data.title,
			href: `/en/posts/${entry.data.slug}/`,
			tags: entry.data.tags.slice(0, 8),
			category: entry.data.category ?? "",
			description: trim(entry.data.description ?? ""),
			kind: "en-post",
			lang: "en",
		}));

	const items: PaletteItem[] = [...postItems, ...enItems, ...pages];

	return new Response(JSON.stringify(items), {
		headers: { "Content-Type": "application/json; charset=utf-8" },
	});
};
