import I18nKey from "@i18n/i18nKey";
import { i18n } from "@i18n/translation";

/**
 * 移除文件扩展名（.md, .mdx, .markdown）
 * 用于将 Astro v5 Content Layer API 的 id 转换为 URL 友好的 slug
 */
export function removeFileExtension(id: string): string {
	return id.replace(/\.(md|mdx|markdown)$/i, "");
}

export function pathsEqual(path1: string, path2: string): boolean {
	const normalizedPath1 = path1.replace(/^\/|\/$/g, "").toLowerCase();
	const normalizedPath2 = path2.replace(/^\/|\/$/g, "").toLowerCase();
	return normalizedPath1 === normalizedPath2;
}

/**
 * 智能拼接URL，正确处理网络URL和本地路径
 */
function joinUrl(...parts: string[]): string {
	// 如果第一个部分是网络URL，直接返回拼接后的结果（不处理协议头的//）
	if (
		parts[0]?.startsWith("http://") ||
		parts[0]?.startsWith("https://") ||
		parts[0]?.startsWith("//")
	) {
		return parts.join("").replace(/(?<!:)\/+/g, "/");
	}

	// 本地路径正常拼接
	const joined = parts.join("/");
	return joined.replace(/\/+/g, "/");
}

export function getPostUrlBySlug(slug: string): string {
	// 移除文件扩展名（如 .md, .mdx 等）
	const slugWithoutExt = removeFileExtension(slug);
	return url(`/posts/${slugWithoutExt}/`);
}

export function getTagUrl(tag: string): string {
	const normalized = tag.trim();
	if (!normalized) return url("/notes/");
	// 标签没有独立筛选维度，落到笔记页搜索框的 q 参数。
	return url(`/notes/?q=${encodeURIComponent(normalized)}`);
}

export function getCategoryUrl(category: string | null): string {
	const normalized = category?.trim() ?? "";
	// 笔记页 topic 的取值是文章 frontmatter 的原始分类，空分类统一为 Uncategorized。
	if (
		normalized === "" ||
		normalized.toLowerCase() === i18n(I18nKey.uncategorized).toLowerCase()
	)
		return url("/notes/?topic=Uncategorized");
	return url(`/notes/?topic=${encodeURIComponent(normalized)}`);
}

export function getDir(path: string): string {
	// 移除文件扩展名
	const pathWithoutExt = removeFileExtension(path);
	const lastSlashIndex = pathWithoutExt.lastIndexOf("/");
	if (lastSlashIndex < 0) {
		return "/";
	}
	return pathWithoutExt.substring(0, lastSlashIndex + 1);
}

export function getFileDirFromPath(filePath: string): string {
	return filePath.replace(/^src\//, "").replace(/\/[^/]+$/, "");
}

export function getSearchUrl(query: string): string {
	return url(`/search/?q=${encodeURIComponent(query.trim())}`);
}

export function url(path: string): string {
	// 关键修复：如果是网络URL，直接返回原地址
	if (
		path.startsWith("http://") ||
		path.startsWith("https://") ||
		path.startsWith("//")
	) {
		return path;
	}

	// 只有本地相对路径才添加BASE_URL
	return joinUrl("", import.meta.env.BASE_URL, path);
}
