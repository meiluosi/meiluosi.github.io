/**
 * 文章分类的双语展示名。
 *
 * 分类值本身是内容集合里的原始字符串，历史上以英文为主（"Deep Learning"），
 * 后期新增的系列改用中文（"大模型训练研究笔记"）。为了让两个界面都不泄漏另一种语言，
 * 这里同时维护两张表：
 *
 * - `categoryLabels`   中文界面显示名：原始值是英文时翻成中文
 * - `categoryLabelsEn` 英文界面显示名：原始值是中文时翻成英文
 *
 * 集中一份，避免 notes 目录、文章元信息、阅读链各写一套后逐渐不一致。
 */
export const categoryLabels: Record<string, string> = {
	"AI Applications": "AI 应用",
	"Data Science": "数据科学",
	"Deep Learning": "深度学习",
	"Fly Connectome": "果蝇连接组研究",
	"Game AI": "游戏 AI",
	"LLM Training Research": "大模型训练研究笔记",
	"Machine Learning": "机器学习",
	NLP: "自然语言处理",
	Probability: "概率论",
	"Reinforcement Learning": "强化学习",
	Research: "研究",
	Uncategorized: "未分类",
};

export const categoryLabelsEn: Record<string, string> = {
	"大模型训练研究笔记": "LLM Training Research",
	"果蝇连接组研究": "Fly Connectome",
};

/**
 * 取分类在指定语言下的展示名；没有对应译名时回落到原始值。
 */
export function getCategoryLabel(
	category: string,
	lang: "zh" | "en",
): string {
	if (!category) return category;
	if (lang === "en") return categoryLabelsEn[category] ?? category;
	return categoryLabels[category] ?? category;
}
