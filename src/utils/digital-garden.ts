/**
 * Digital Garden & Knowledge Cosmos Utilities
 *
 * Implements Andy Matuschak's evergreen notes & digital garden taxonomy:
 * - 🌱 Seedling (萌芽): Nascent thoughts, raw reading logs, open questions.
 * - 🌿 Budding (演进): Structured formulations, ongoing experiments, active proofs.
 * - 🌳 Evergreen (常青): Rigorous synthesis, closed logic, durable foundational notes.
 * - 🔬 Speculative (待证): Hypothesis exploration tied to empirical lab workbenches.
 * - ⚙️ Lab (实验): Interactive computational apparatus & sandboxes.
 */

export type MaturityLevel =
	| "seedling"
	| "budding"
	| "evergreen"
	| "speculative"
	| "lab";

export interface MaturityMeta {
	level: MaturityLevel;
	icon: string;
	labelZh: string;
	labelEn: string;
	descriptionZh: string;
	descriptionEn: string;
	color: string;
	badgeBg: string;
}

export const MATURITY_DEFINITIONS: Record<MaturityLevel, MaturityMeta> = {
	seedling: {
		level: "seedling",
		icon: "🌱",
		labelZh: "萌芽想法",
		labelEn: "Seedling",
		descriptionZh: "初步构想、学习摘录或开放性猜想，结构精简，处于持续发酵中。",
		descriptionEn:
			"Nascent spark, reading notes, or open inquiry; brief and germinating.",
		color: "#2a9d8f",
		badgeBg: "rgba(42, 157, 143, 0.12)",
	},
	budding: {
		level: "budding",
		icon: "🌿",
		labelZh: "成长演进",
		labelEn: "Budding",
		descriptionZh: "具备清晰骨架、初步数学推导或工程验证，持续补充论据与实测。",
		descriptionEn:
			"Structured thesis with preliminary math or code; actively developing.",
		color: "#5c7a6b",
		badgeBg: "rgba(92, 122, 107, 0.14)",
	},
	evergreen: {
		level: "evergreen",
		icon: "🌳",
		labelZh: "常青基石",
		labelEn: "Evergreen",
		descriptionZh: "论证完备、逻辑闭环、代码与理论齐备，具备长期持久参考价值。",
		descriptionEn:
			"Rigorously argued, closed logic, mature code and theory; durable cornerstone.",
		color: "#35524a",
		badgeBg: "rgba(53, 82, 74, 0.18)",
	},
	speculative: {
		level: "speculative",
		icon: "🔬",
		labelZh: "实验待证",
		labelEn: "Speculative",
		descriptionZh: "配合仿真与实验探针的假设探索，等待更多证据或边界检验。",
		descriptionEn:
			"Empirical hypothesis tied to lab workbenches, pending boundary proofs.",
		color: "#e07a5f",
		badgeBg: "rgba(224, 122, 95, 0.14)",
	},
	lab: {
		level: "lab",
		icon: "⚙️",
		labelZh: "交互实验",
		labelEn: "Lab Sandbox",
		descriptionZh: "浏览器端可运行算法推演、参数拨盘与白盒化仿真仪器。",
		descriptionEn:
			"In-browser executable algorithms, live dials, and white-box instruments.",
		color: "#8367c7",
		badgeBg: "rgba(131, 103, 199, 0.14)",
	},
};

/**
 * Resolves post maturity level based on frontmatter or intelligent heuristics.
 */
export function getPostMaturity(post: {
	id: string;
	data: {
		maturity?: "seedling" | "budding" | "evergreen" | "speculative";
		format?: string;
		title?: string;
		category?: string | null;
	};
}): MaturityMeta {
	if (post.data.maturity && MATURITY_DEFINITIONS[post.data.maturity]) {
		return MATURITY_DEFINITIONS[post.data.maturity];
	}

	const title = post.data.title || "";
	const format = post.data.format || "";
	const category = post.data.category || "";

	// Specific topic classifications
	if (
		title.includes("思考") ||
		title.includes("调参") ||
		format === "reflection"
	) {
		return MATURITY_DEFINITIONS.seedling;
	}

	if (
		title.includes("因果图") ||
		title.includes("CFR") ||
		title.includes("Alpha Zero") ||
		title.includes("Transformer") ||
		title.includes("全链路") ||
		title.includes("Actor-Critic") ||
		category === "Deep Learning" ||
		category === "Reinforcement Learning"
	) {
		return MATURITY_DEFINITIONS.evergreen;
	}

	if (
		format === "experiment" ||
		title.includes("实战") ||
		title.includes("假设检验") ||
		title.includes("对齐")
	) {
		return MATURITY_DEFINITIONS.speculative;
	}

	return MATURITY_DEFINITIONS.budding;
}

export interface GraphNode {
	id: string;
	label: string;
	labelEn?: string;
	kind: "post" | "lab" | "hub";
	maturity: MaturityLevel;
	category: string;
	href: string;
	excerpt?: string;
	tags?: string[];
	val: number; // size / importance
	x?: number;
	y?: number;
	vx?: number;
	vy?: number;
}

export interface GraphEdge {
	source: string;
	target: string;
	type: "category" | "lab-link" | "tag-affinity" | "series";
	weight: number;
}

export interface KnowledgeGraphData {
	nodes: GraphNode[];
	edges: GraphEdge[];
}

export interface LabEntry {
	id: string;
	title: string;
	titleEn: string;
	href: string;
	category: string;
	maturity: MaturityLevel;
	tags: string[];
}

export interface CoreHub {
	id: string;
	label: string;
	labelEn: string;
	category: string;
}

export const LAB_ENTRIES: LabEntry[] = [
	{
		id: "lab-mcts",
		title: "MCTS 树搜索实验",
		titleEn: "Monte Carlo Tree Search",
		href: "/lab/mcts/",
		category: "Reinforcement Learning",
		maturity: "lab",
		tags: ["强化学习", "游戏AI与搜索", "MCTS"],
	},
	{
		id: "lab-causal",
		title: "因果效应与选择偏差",
		titleEn: "Causal Effects & Selection Bias",
		href: "/lab/causal-playground/",
		category: "Causal Inference",
		maturity: "lab",
		tags: ["因果推断", "统计学基础", "ATE"],
	},
	{
		id: "lab-cfr",
		title: "Kuhn Poker CFR 纳什均衡",
		titleEn: "Kuhn Poker CFR & Nash",
		href: "/lab/cfr-poker/",
		category: "Game Theory",
		maturity: "lab",
		tags: ["游戏AI与搜索", "博弈论", "CFR"],
	},
	{
		id: "lab-dpo",
		title: "DPO 偏好对齐与古德哈特",
		titleEn: "DPO Alignment & Goodhart Frontier",
		href: "/lab/preference-alignment/",
		category: "LLM Post-Training",
		maturity: "lab",
		tags: ["大模型训练", "后训练(RLHF/DPO)", "KL散度"],
	},
	{
		id: "lab-rope",
		title: "RoPE 旋转位置编码星盘",
		titleEn: "RoPE Rotary Position Astrolabe",
		href: "/lab/rope-attention/",
		category: "Deep Learning",
		maturity: "lab",
		tags: ["模型架构", "位置编码", "RoPE"],
	},
	{
		id: "lab-attention-lens",
		title: "Transformer 归纳回路探针",
		titleEn: "Attention Lens & Induction Circuits",
		href: "/lab/attention-lens/",
		category: "Mechanistic Interpretability",
		maturity: "lab",
		tags: ["模型架构", "注意力机制", "归纳头"],
	},
	{
		id: "lab-transformer-forward",
		title: "Transformer 前向推演与回路消融",
		titleEn: "Micro-Transformer Forward Pass",
		href: "/lab/transformer-forward/",
		category: "Deep Learning",
		maturity: "lab",
		tags: ["模型架构", "前向传播", "神经回路消融"],
	},
	{
		id: "lab-model-pipeline",
		title: "1B 模型训练旗舰工作台",
		titleEn: "1B Model Training Pipeline",
		href: "/lab/model-pipeline/",
		category: "LLM Training",
		maturity: "lab",
		tags: ["大模型训练", "LLM全链路", "从零预训练"],
	},
];

export const CORE_HUBS: CoreHub[] = [
	{
		id: "hub-rl",
		label: "强化学习与搜索",
		labelEn: "Reinforcement Learning",
		category: "Reinforcement Learning",
	},
	{
		id: "hub-llm",
		label: "大语言模型体系",
		labelEn: "Large Language Models",
		category: "Deep Learning",
	},
	{
		id: "hub-causal",
		label: "因果推断与统计评估",
		labelEn: "Causal Inference",
		category: "Causal Inference",
	},
	{
		id: "hub-interp",
		label: "机制可解释性与对齐",
		labelEn: "Mechanistic Interpretability & Alignment",
		category: "Mechanistic Interpretability",
	},
];

/**
 * Finds interactive computational labs connected to a specific article.
 */
export function findConnectedLabs(
	postTags: string[] = [],
	postCategory = "",
	postTitle = "",
): LabEntry[] {
	const titleLower = postTitle.toLowerCase();
	const catLower = postCategory.toLowerCase();
	const tagsSet = new Set(postTags.map((t) => t.toLowerCase()));

	const matches = LAB_ENTRIES.filter((lab) => {
		// Tag overlap
		const hasTagMatch = lab.tags.some(
			(t) =>
				tagsSet.has(t.toLowerCase()) ||
				postTags.some((pt) => pt.includes(t) || t.includes(pt)),
		);
		if (hasTagMatch) return true;

		// Category overlap
		if (postCategory && lab.category.toLowerCase() === catLower) return true;

		// Keyword heuristics
		if (
			titleLower.includes("transformer") &&
			(lab.id.includes("transformer") || lab.id.includes("attention"))
		) {
			return true;
		}
		if (titleLower.includes("cfr") && lab.id.includes("cfr")) return true;
		if (titleLower.includes("mcts") && lab.id.includes("mcts")) return true;
		if (titleLower.includes("rope") && lab.id.includes("rope")) return true;
		if (
			(titleLower.includes("dpo") || titleLower.includes("对齐")) &&
			lab.id.includes("dpo")
		) {
			return true;
		}
		if (
			(titleLower.includes("因果") || titleLower.includes("causal")) &&
			lab.id.includes("causal")
		) {
			return true;
		}
		if (
			(titleLower.includes("预训练") || titleLower.includes("1b")) &&
			lab.id.includes("model-pipeline")
		) {
			return true;
		}

		return false;
	});

	// If no match found, fallback to at least 1-2 representative labs
	if (matches.length === 0) {
		return LAB_ENTRIES.slice(0, 2);
	}
	return matches.slice(0, 3);
}
