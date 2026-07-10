import type { AnnouncementConfig } from "../types/announcementConfig";

export const announcementConfig: AnnouncementConfig = {
	// 公告标题
	title: "公告",

	// 公告内容
	content: "LLM 训练/推理 · 分布式系统 · RLHF — 用 RL 的方法让大模型更聪明",

	// 是否允许用户关闭公告
	closable: true,

	link: {
		// 启用链接
		enable: true,
		// 链接文本
		text: "关于本站",
		// 链接 URL
		url: "/about/",
		// 内部链接
		external: false,
	},
};
