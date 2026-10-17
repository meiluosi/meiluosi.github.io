/**
 * 文章分类的展示名。
 *
 * 中文界面把英文分类值翻成中文，英文界面直接用原始值。
 * 这里集中一份，避免 notes 目录、阅读链等地方各写一套后逐渐不一致。
 */
export const categoryLabels: Record<string, string> = {
	"AI Applications": "AI 应用",
	"Data Science": "数据科学",
	"Deep Learning": "深度学习",
	"Game AI": "游戏 AI",
	"Machine Learning": "机器学习",
	NLP: "自然语言处理",
	Probability: "概率论",
	"Reinforcement Learning": "强化学习",
	Research: "研究",
	Uncategorized: "未分类",
};
