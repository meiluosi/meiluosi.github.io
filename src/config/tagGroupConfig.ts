/**
 * 个人品牌标签体系 — 按能力领域分组展示
 * 在标签页面 /tags/ 中会按此分组渲染
 */
export interface TagGroup {
	/** 分组名称 */
	name: string;
	/** 分组图标 (iconify) */
	icon: string;
	/** 该分组下的标签列表 */
	tags: string[];
}

export const tagGroups: TagGroup[] = [
	{
		name: "训练基础设施",
		icon: "material-symbols:hard-drive",
		tags: [
			"DeepSpeed",
			"FSDP",
			"Megatron-LM",
			"分布式训练",
			"混合精度",
			"ZeRO",
			"训练稳定性",
			"NCCL",
		],
	},
	{
		name: "推理优化",
		icon: "material-symbols:rocket-launch",
		tags: [
			"vLLM",
			"TensorRT-LLM",
			"量化",
			"Speculative Decoding",
			"MoE",
			"PagedAttention",
			"SGLang",
		],
	},
	{
		name: "后训练 · RLHF",
		icon: "material-symbols:psychology",
		tags: [
			"RLHF",
			"DPO",
			"GRPO",
			"Reward Modeling",
			"PPO",
			"强化学习",
			"RLVR",
			"ORPO",
		],
	},
	{
		name: "数据工程",
		icon: "material-symbols:database",
		tags: [
			"数据去重",
			"SFT数据构造",
			"数据配比",
			"预训练数据",
			"MinHash",
		],
	},
	{
		name: "模型架构",
		icon: "material-symbols:architecture",
		tags: [
			"Transformer",
			"Flash Attention",
			"RoPE",
			"大语言模型",
			"深度学习",
			"机器学习",
		],
	},
	{
		name: "论文解读",
		icon: "material-symbols:description",
		tags: [
			"ICLR",
			"NeurIPS",
			"ICML",
			"论文解读",
		],
	},
	{
		name: "因果推断",
		icon: "material-symbols:account-tree",
		tags: [
			"因果推断",
			"DUCG",
			"概率图模型",
			"DoWhy",
		],
	},
	{
		name: "其他",
		icon: "material-symbols:more-horiz",
		tags: [
			"数据科学",
			"人工智能",
			"AlphaZero",
			"技术博客",
		],
	},
];

/**
 * 根据标签名查找所属分组
 * 未匹配的标签归入"其他"
 */
export function getTagGroupName(tag: string): string {
	for (const group of tagGroups) {
		if (group.tags.includes(tag)) return group.name;
	}
	return "其他";
}

/**
 * 获取标签分组图标
 */
export function getTagGroupIcon(tag: string): string {
	for (const group of tagGroups) {
		if (group.tags.includes(tag)) return group.icon;
	}
	return "material-symbols:more-horiz";
}