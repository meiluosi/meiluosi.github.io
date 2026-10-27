import {
	getRouteById,
	getRouteByIndex,
	type RouteItem,
} from "@/data/intelligenceRoutes";

/**
 * 根据文章 frontmatter 显式 route 字段或 tags/category 智能推断归属的智能路线
 */
export function resolvePostRoute(postData: {
	route?: string | number;
	routes?: string[];
	tags?: string[];
	category?: string | null;
	title?: string;
}): RouteItem | undefined {
	// 1. 如果显式指定了 route
	if (postData.route !== undefined) {
		if (typeof postData.route === "number") {
			return getRouteByIndex(postData.route);
		}
		// 可能是数字字符串如 "07" 或 "7"
		const num = Number.parseInt(postData.route, 10);
		if (!Number.isNaN(num)) {
			const byNum = getRouteByIndex(num);
			if (byNum) return byNum;
		}
		// 或者是 route id 如 "causal-world-models"
		const byId = getRouteById(postData.route);
		if (byId) return byId;
	}

	if (postData.routes && postData.routes.length > 0) {
		const first = postData.routes[0];
		const byId = getRouteById(first);
		if (byId) return byId;
	}

	const tags = (postData.tags || []).map((t) => t.toLowerCase());
	const category = (postData.category || "").toLowerCase();
	const title = (postData.title || "").toLowerCase();

	// 2. 根据 tags / category / title 规则映射
	// 路线 12: 具身与动力系统 (Strandbeest / 连杆 / 步态)
	if (
		tags.some(
			(t) =>
				t.includes("strandbeest") || t.includes("仿生") || t.includes("连杆"),
		) ||
		title.includes("strandbeest") ||
		title.includes("步态")
	) {
		return getRouteById("embodied-dynamical");
	}

	// 路线 07: 概率因果与世界模型 (因果 / DUCG / YLearn / 反事实)
	if (
		category.includes("因果") ||
		tags.some(
			(t) => t.includes("因果") || t.includes("ducg") || t.includes("ylearn"),
		) ||
		title.includes("因果") ||
		title.includes("ducg")
	) {
		return getRouteById("causal-world-models");
	}

	// 路线 11: 开放式学习与 RSI (递归自我改进 / 自我进化)
	if (
		tags.some(
			(t) =>
				t.includes("rsi") || t.includes("自我改进") || t.includes("开放式"),
		) ||
		title.includes("rsi") ||
		title.includes("自我改进")
	) {
		return getRouteById("developmental-openended");
	}

	// 路线 08: 控制论与强化学习 (RL / 强化学习 / MCTS / CFR / 扑克 / 博弈)
	if (
		category.includes("强化学习") ||
		category.includes("博弈") ||
		tags.some(
			(t) =>
				t.includes("rl") ||
				t.includes("强化学习") ||
				t.includes("mcts") ||
				t.includes("cfr") ||
				t.includes("博弈"),
		) ||
		title.includes("强化学习") ||
		title.includes("cfr") ||
		title.includes("mcts")
	) {
		return getRouteById("cybernetics-rl");
	}

	// 路线 06: 连接主义、统计表征与大模型 (LLM / 训练 / Transformer / Attention / 微调)
	if (
		category.includes("llm") ||
		category.includes("大模型") ||
		tags.some(
			(t) =>
				t.includes("llm") ||
				t.includes("transformer") ||
				t.includes("预训练") ||
				t.includes("微调") ||
				t.includes("注意力") ||
				t.includes("attention") ||
				t.includes("vllm") ||
				t.includes("deepspeed"),
		) ||
		title.includes("llm") ||
		title.includes("transformer") ||
		title.includes("预训练")
	) {
		return getRouteById("connectionist");
	}

	// 路线 05: 符号与逻辑搜索
	if (
		tags.some(
			(t) => t.includes("逻辑") || t.includes("符号") || t.includes("知识图谱"),
		)
	) {
		return getRouteById("symbolic-logic");
	}

	// 路线 04: 理性与统计决策
	if (
		tags.some(
			(t) => t.includes("统计") || t.includes("决策") || t.includes("数据分析"),
		)
	) {
		return getRouteById("rational-resource");
	}

	// 默认返回连接主义表征路线或 undefined
	return undefined;
}
