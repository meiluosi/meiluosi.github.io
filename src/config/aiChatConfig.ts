import type { AiChatConfig } from "@/types/aiChatConfig";

export const aiChatConfig: AiChatConfig = {
	// AI 对话总开关
	enable: false,

	// API 代理地址 — 部署 Cloudflare Worker 后填入
	// 例如: "https://ai-proxy.your-subdomain.workers.dev"
	apiEndpoint: "https://ai-proxy.your-subdomain.workers.dev",

	// 模型名称（OpenAI 兼容格式）
	// 推荐: deepseek-chat (便宜、中文好)、gpt-4o-mini、qwen-turbo
	model: "deepseek-chat",

	// 系统提示词 — 定义 AI 助手的角色
	systemPrompt: `你是「枫语」博客的 AI 助手，名叫小枫。你的特点：
- 你是一个友好、热情的技术伙伴，擅长 AI、强化学习、因果推断、大语言模型等领域
- 回答风格：简洁易懂，适当使用 emoji，像朋友聊天一样自然
- 当被问到博客相关内容时，可以引导用户浏览博客文章
- 如果被问到无关话题，礼貌地引导回技术讨论
- 始终使用中文回复（除非用户使用其他语言提问）`,

	// 最大输出 token 数
	maxTokens: 1024,

	// 温度参数 (0-2)，越高越有创意
	temperature: 0.7,

	// UI 文本
	title: "💬 与小枫对话",
	welcomeMessage: "你好呀！我是小枫，枫语博客的 AI 助手~ 有什么想聊的吗？😊",
	placeholder: "输入你的问题...",
	triggerHint: "点击和我聊天吧！",
};