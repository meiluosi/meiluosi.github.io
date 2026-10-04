import { getCollection } from "astro:content";

export interface ResearchNote {
	id: string;
	title: string;
	titleEn: string;
	href: string;
	hrefEn: string;
	description: string;
	descriptionEn: string;
}

export async function getResearchNotes(): Promise<Map<string, ResearchNote>> {
	const [posts, english] = await Promise.all([
		getCollection("posts", ({ data }) => !data.draft && !data.password),
		getCollection("enPosts"),
	]);
	return new Map(
		posts.map((post) => {
			const id = post.id.replace(/\.mdx?$/, "");
			const translation = english.find((entry) => entry.data.sourceId === id);
			const href = `/posts/${id}/`;
			return [
				id,
				{
					id,
					title: post.data.title,
					titleEn: translation?.data.title ?? post.data.title,
					href,
					hrefEn: translation ? `/en/posts/${translation.data.slug}/` : href,
					description: post.data.description ?? "",
					descriptionEn:
						translation?.data.description ?? post.data.description ?? "",
				},
			];
		}),
	);
}

export function requireResearchNote(
	notes: Map<string, ResearchNote>,
	id: string,
): ResearchNote {
	const note = notes.get(id);
	if (!note)
		throw new Error(
			`Research entry references a missing or unpublished note: ${id}`,
		);
	return note;
}
