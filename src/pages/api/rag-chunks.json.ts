import { getCollection } from "astro:content";
import type { APIRoute } from "astro";
import { removeFileExtension } from "@/utils/url-utils";

export interface RagChunk {
	id: string;
	postTitle: string;
	heading: string;
	url: string;
	tags: string[];
	category?: string | null;
	lang: "zh" | "en";
	text: string;
}

function cleanMarkdown(md: string): string {
	return md
		.replace(/```[\s\S]*?```/g, " [代码实现] ")
		.replace(/!\[[^\]]*\]\([^)]*\)/g, "")
		.replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
		.replace(/<[^>]*>/g, " ")
		.replace(/[#*`|>]/g, " ")
		.replace(/\s+/g, " ")
		.trim();
}

function slugifyHeading(text: string): string {
	return text
		.toLowerCase()
		.trim()
		.replace(
			/[`~!@#$%^&*()+=<>?:{}"|,./;'[\]·~！@#￥%……&*（）——+{}|《》？：“”【】、；‘'，。、]/g,
			"",
		)
		.replace(/\s+/g, "-");
}

export const GET: APIRoute = async () => {
	const [posts, translations] = await Promise.all([
		getCollection("posts"),
		getCollection("enPosts"),
	]);

	const chunks: RagChunk[] = [];

	// Process Chinese posts
	for (const post of posts) {
		if (post.data.draft || post.data.password) continue;

		const slug = removeFileExtension(post.id);
		const baseUrl = `/posts/${slug}/`;
		const title = post.data.title || slug;
		const tags = Array.isArray(post.data.tags) ? post.data.tags : [];
		const category = post.data.category;
		const body = post.body || "";

		const sections = body.split(/\n(?=#{2,3}\s+)/);
		let sectionIdx = 0;

		for (const sec of sections) {
			const trimmed = sec.trim();
			if (!trimmed) continue;

			let heading = "概述";
			let headingSlug = "";
			let text = trimmed;

			const headingMatch = trimmed.match(/^(#{2,3})\s+(.+)$/m);
			if (headingMatch) {
				heading = headingMatch[2].trim();
				headingSlug = slugifyHeading(heading);
				text = trimmed.replace(/^(#{2,3})\s+(.+)$/m, "").trim();
			}

			const cleaned = cleanMarkdown(text);
			if (cleaned.length < 30) continue;

			// Split long sections to preserve focus
			if (cleaned.length > 850) {
				const sentences = cleaned.split(/(?<=[。！？.!?])\s*/);
				let currentChunkText = "";
				let subIdx = 0;

				for (const s of sentences) {
					if (
						(currentChunkText + s).length > 650 &&
						currentChunkText.length > 200
					) {
						chunks.push({
							id: `${slug}-${sectionIdx}-${subIdx}`,
							postTitle: title,
							heading: subIdx === 0 ? heading : `${heading} (${subIdx + 1})`,
							url: headingSlug ? `${baseUrl}#${headingSlug}` : baseUrl,
							tags,
							category,
							lang: "zh",
							text: currentChunkText.trim(),
						});
						currentChunkText = s;
						subIdx++;
					} else {
						currentChunkText += (currentChunkText ? " " : "") + s;
					}
				}

				if (currentChunkText.trim().length > 30) {
					chunks.push({
						id: `${slug}-${sectionIdx}-${subIdx}`,
						postTitle: title,
						heading: subIdx === 0 ? heading : `${heading} (${subIdx + 1})`,
						url: headingSlug ? `${baseUrl}#${headingSlug}` : baseUrl,
						tags,
						category,
						lang: "zh",
						text: currentChunkText.trim(),
					});
				}
			} else {
				chunks.push({
					id: `${slug}-${sectionIdx}`,
					postTitle: title,
					heading,
					url: headingSlug ? `${baseUrl}#${headingSlug}` : baseUrl,
					tags,
					category,
					lang: "zh",
					text: cleaned,
				});
			}

			sectionIdx++;
		}
	}

	// Process English posts
	for (const post of translations) {
		const slug = post.data.slug || removeFileExtension(post.id);
		const baseUrl = `/en/posts/${slug}/`;
		const title = post.data.title || slug;
		const tags = Array.isArray(post.data.tags) ? post.data.tags : [];
		const category = post.data.category;
		const body = post.body || "";

		const sections = body.split(/\n(?=#{2,3}\s+)/);
		let sectionIdx = 0;

		for (const sec of sections) {
			const trimmed = sec.trim();
			if (!trimmed) continue;

			let heading = "Overview";
			let headingSlug = "";
			let text = trimmed;

			const headingMatch = trimmed.match(/^(#{2,3})\s+(.+)$/m);
			if (headingMatch) {
				heading = headingMatch[2].trim();
				headingSlug = slugifyHeading(heading);
				text = trimmed.replace(/^(#{2,3})\s+(.+)$/m, "").trim();
			}

			const cleaned = cleanMarkdown(text);
			if (cleaned.length < 30) continue;

			if (cleaned.length > 850) {
				const sentences = cleaned.split(/(?<=[.!?])\s*/);
				let currentChunkText = "";
				let subIdx = 0;

				for (const s of sentences) {
					if (
						(currentChunkText + s).length > 650 &&
						currentChunkText.length > 200
					) {
						chunks.push({
							id: `en-${slug}-${sectionIdx}-${subIdx}`,
							postTitle: title,
							heading: subIdx === 0 ? heading : `${heading} (${subIdx + 1})`,
							url: headingSlug ? `${baseUrl}#${headingSlug}` : baseUrl,
							tags,
							category,
							lang: "en",
							text: currentChunkText.trim(),
						});
						currentChunkText = s;
						subIdx++;
					} else {
						currentChunkText += (currentChunkText ? " " : "") + s;
					}
				}

				if (currentChunkText.trim().length > 30) {
					chunks.push({
						id: `en-${slug}-${sectionIdx}-${subIdx}`,
						postTitle: title,
						heading: subIdx === 0 ? heading : `${heading} (${subIdx + 1})`,
						url: headingSlug ? `${baseUrl}#${headingSlug}` : baseUrl,
						tags,
						category,
						lang: "en",
						text: currentChunkText.trim(),
					});
				}
			} else {
				chunks.push({
					id: `en-${slug}-${sectionIdx}`,
					postTitle: title,
					heading,
					url: headingSlug ? `${baseUrl}#${headingSlug}` : baseUrl,
					tags,
					category,
					lang: "en",
					text: cleaned,
				});
			}

			sectionIdx++;
		}
	}

	return new Response(JSON.stringify(chunks), {
		headers: {
			"Content-Type": "application/json; charset=utf-8",
			"Cache-Control": "public, max-age=3600, stale-while-revalidate=86400",
		},
	});
};
