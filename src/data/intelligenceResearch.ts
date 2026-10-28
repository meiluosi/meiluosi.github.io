import evidence from "../../docs/intelligence-history/round-four/research.json";
import boundaries from "../../docs/intelligence-history/round-four/boundaries/research.json";
import meaning from "../../docs/intelligence-history/round-four/meaning/research.json";
import reviewData from "../../docs/intelligence-history/round-four/integration/research.json";

export interface ResearchClaim {
	id: string;
	text: string;
	locator: string;
	editorialCaution?: string;
}
export interface ResearchSource {
	id: string;
	title: string;
	authors: string[];
	year: number;
	url: string;
	version: string;
	readingStatus: string;
	readScope: string;
	unreadScope: string;
	problem: string;
	solution: string;
	remaining: string;
	sha256?: string | null;
	claims: ResearchClaim[];
}
export interface ResearchConclusion {
	id: string;
	title: string;
	statement: string;
	assumptions: string;
	objection: string;
	response: string;
	discriminator: string;
	scope: string;
	evidence: string[];
}
export interface ResearchArticleData {
	sources: ResearchSource[];
	conclusions: ResearchConclusion[];
	synthesis: { title: string; text: string }[];
}
export interface ResearchArticle {
	id: string;
	folder: string;
	title: string;
	titleEn: string;
	routes: string[];
	graphs: string[];
	conclusions: string[];
	data: ResearchArticleData;
}
export const researchReview: typeof reviewData = reviewData;
export const classificationLabels: Record<string, string> = {
	stable: "稳定的推理边界",
	contested: "依赖争议判据",
	bounded: "指定材料内的判断",
};
const originals: Record<string, ResearchArticleData> = {
	evidence: {
		...evidence,
		sources: evidence.sources.map((source) => {
			const guide = reviewData.sources.find(
				(item) => item.id === source.id,
			)!.readerGuide;
			return { ...source, ...guide };
		}),
	},
	boundaries,
	meaning,
};
export const researchArticles: ResearchArticle[] = reviewData.articles.map(
	(article) => ({ ...article, data: originals[article.id] }),
);
export const researchSources: (ResearchSource & { article: string })[] =
	researchArticles.flatMap((article) =>
		article.data.sources.map((source) => ({ ...source, article: article.id })),
	);
export function researchSourceForClaim(
	id: string,
): ResearchSource & { article: string } {
	const source = researchSources.find((source) =>
		source.claims.some((claim) => claim.id === id),
	);
	if (!source) throw new Error(`Missing research source: ${id}`);
	return source;
}
export function researchArticleForConclusion(id: string): ResearchArticle {
	const article = researchArticles.find((article) =>
		article.conclusions.includes(id),
	);
	if (!article) throw new Error(`Missing research conclusion: ${id}`);
	return article;
}
export const graphLabels: Record<string, string> = {
	"inference-map": "从表现到机制：推理边界",
	"collective-debate": "群体能力的证据与争论",
	"boundary-map": "系统边界的不同判据",
	timeline: "问题、方案与未决问题的演进",
	timescales: "发展与当下的时间尺度",
	"bridge-map": "意义与体验的解释桥梁",
	"prediction-map": "COGITATE逐项预测比较",
};
