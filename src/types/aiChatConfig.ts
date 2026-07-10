/** AI 对话看板娘配置类型 */
export interface AiChatConfig {
	/** 是否启用 AI 对话功能 */
	enable: boolean;
	/** API 代理地址（Cloudflare Worker 或其他 Serverless 代理） */
	apiEndpoint: string;
	/** 模型名称（OpenAI 兼容格式） */
	model: string;
	/** 系统提示词 */
	systemPrompt: string;
	/** 最大输出 token 数 */
	maxTokens: number;
	/** 温度参数 */
	temperature: number;
	/** 对话标题 */
	title: string;
	/** 欢迎消息 */
	welcomeMessage: string;
	/** 输入框占位符 */
	placeholder: string;
	/** 触发按钮提示文字 */
	triggerHint: string;
}