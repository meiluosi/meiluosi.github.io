import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { z } from "astro/zod";

type PostsData = {
	title: string;
	published: Date;
	updated?: Date;
	draft: boolean;
	description: string;
	image: string;
	tags: string[];
	category: string | null;
	/** 所属系列展示名（可选） */
	series?: string;
	/** 系列内序号，从 1 开始（可选） */
	seriesOrder?: number;
	format:
		| "plan"
		| "code-example"
		| "application"
		| "technical"
		| "experiment"
		| "reflection";
	/** 数字花园思想成熟度分类 */
	maturity?: "seedling" | "budding" | "evergreen" | "speculative";
	/** 关联的智能探索路线编号或 ID（可选） */
	route?: string | number;
	routes?: string[];
	lang: string;
	pinned: boolean;
	author: string;
	sourceLink: string;
	licenseName: string;
	licenseUrl: string;
	comment: boolean;
	password: string;
	passwordHint: string;
	aiSummary: string;
	prevTitle: string;
	prevSlug: string;
	nextTitle: string;
	nextSlug: string;
};

const postsSchema: z.ZodType<PostsData> = z.object({
	title: z.string(),
	published: z.date(),
	updated: z.date().optional(),
	draft: z.boolean().optional().default(false),
	description: z.string().optional().default(""),
	image: z.string().optional().default(""),
	tags: z.array(z.string()).optional().default([]),
	category: z.string().optional().nullable().default(""),
	series: z.string().optional(),
	seriesOrder: z.number().optional(),
	format: z
		.enum([
			"plan",
			"code-example",
			"application",
			"technical",
			"experiment",
			"reflection",
		])
		.optional()
		.default("technical"),
	maturity: z
		.enum(["seedling", "budding", "evergreen", "speculative"])
		.optional(),
	route: z.union([z.string(), z.number()]).optional(),
	routes: z.array(z.string()).optional().default([]),
	lang: z.string().optional().default(""),
	pinned: z.boolean().optional().default(false),
	author: z.string().optional().default(""),
	sourceLink: z.string().optional().default(""),
	licenseName: z.string().optional().default(""),
	licenseUrl: z.string().optional().default(""),
	comment: z.boolean().optional().default(true),
	password: z.string().optional().default(""),
	passwordHint: z.string().optional().default(""),
	aiSummary: z.string().optional().default(""),

	/* For internal use */
	prevTitle: z.string().default(""),
	prevSlug: z.string().default(""),
	nextTitle: z.string().default(""),
	nextSlug: z.string().default(""),
});

const postsCollection: ReturnType<typeof defineCollection<typeof postsSchema>> =
	defineCollection({
		loader: glob({ pattern: "**/*.{md,mdx}", base: "./src/content/posts" }),
		schema: postsSchema,
	});

const specSchema: z.ZodType<{}> = z.object({});
const specCollection: ReturnType<typeof defineCollection<typeof specSchema>> =
	defineCollection({
		loader: glob({ pattern: "**/*.{md,mdx}", base: "./src/content/spec" }),
		schema: specSchema,
	});

type EnglishPostData = {
	sourceId: string;
	slug: string;
	title: string;
	editionLabel?: string;
	published: Date;
	description: string;
	tags: string[];
	category: string | null;
	/** 所属系列展示名（可选） */
	series?: string;
	/** 系列内序号，从 1 开始（可选） */
	seriesOrder?: number;
	route?: string | number;
	routes?: string[];
	lang: "en";
};

const enPostsSchema: z.ZodType<EnglishPostData> = z.object({
	sourceId: z.string(),
	slug: z.string(),
	title: z.string(),
	editionLabel: z.string().optional(),
	published: z.date(),
	description: z.string().optional().default(""),
	tags: z.array(z.string()).optional().default([]),
	category: z.string().optional().nullable().default(""),
	series: z.string().optional(),
	seriesOrder: z.number().optional(),
	route: z.union([z.string(), z.number()]).optional(),
	routes: z.array(z.string()).optional().default([]),
	lang: z.literal("en").default("en"),
});

const enPostsCollection: ReturnType<
	typeof defineCollection<typeof enPostsSchema>
> = defineCollection({
	loader: glob({ pattern: "**/*.{md,mdx}", base: "./src/content/en-posts" }),
	schema: enPostsSchema,
});

export const collections: {
	posts: typeof postsCollection;
	enPosts: typeof enPostsCollection;
	spec: typeof specCollection;
} = {
	posts: postsCollection,
	enPosts: enPostsCollection,
	spec: specCollection,
};
