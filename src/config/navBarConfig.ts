import { type NavBarConfig, type NavBarLink, type NavBarSearchConfig, NavBarSearchMethod } from "../types/navBarConfig";

/** The public navigation follows the new personal-site architecture. */
export const navBarSearchConfig: NavBarSearchConfig = {
	method: NavBarSearchMethod.PageFind,
};

export const LinkPresets: Record<string, NavBarLink> = {
	Home: { name: "主页", url: "/" },
	Work: { name: "研究档案", url: "/work/" },
	Notes: { name: "笔记", url: "/notes/" },
	About: { name: "关于", url: "/about/" },
	Contact: { name: "联系", url: "/contact/" },
	Archive: { name: "笔记", url: "/notes/" },
	Categories: { name: "分类", url: "/categories/" },
	Tags: { name: "标签", url: "/tags/" },
	Graph: { name: "知识图谱", url: "/graph/" },
	Sponsor: { name: "赞助", url: "/sponsor/" },
	Guestbook: { name: "留言", url: "/guestbook/" },
	Github: { name: "GitHub", url: "https://github.com/meiluosi", external: true },
};

export const navBarConfig: NavBarConfig = {
	links: [LinkPresets.Work, LinkPresets.Notes, LinkPresets.About, LinkPresets.Contact],
};
