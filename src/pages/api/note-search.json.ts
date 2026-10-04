import { getCollection } from "astro:content";
import type { APIRoute } from "astro";
import { removeFileExtension } from "@/utils/url-utils";

const plainText = (body: string): string =>
	body
		.replace(/```[^\n]*\n/g, " ")
		.replace(/!\[[^\]]*\]\([^)]*\)/g, " ")
		.replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
		.replace(/<[^>]*>/g, " ")
		.replace(/[#*`|>]/g, " ")
		.replace(/\s+/g, " ")
		.trim();

export const GET: APIRoute = async () => {
	const [posts, translations] = await Promise.all([
		getCollection("posts"),
		getCollection("enPosts"),
	]);
	const index = posts
		.filter((post) => !post.data.draft && !post.data.password)
		.map((post) => {
			const translation = translations.find(
				(entry) => entry.data.sourceId === removeFileExtension(post.id),
			);
			return {
				id: post.id,
				zh: plainText(
					`${post.data.title} ${post.data.description ?? ""} ${post.body ?? ""}`,
				),
				en: translation
					? plainText(
							`${translation.data.title} ${translation.data.description ?? ""} ${translation.body ?? ""}`,
						)
					: "",
			};
		});
	return new Response(JSON.stringify(index), {
		headers: { "Content-Type": "application/json; charset=utf-8" },
	});
};
