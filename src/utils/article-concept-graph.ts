export interface ConceptNode {
	id: string;
	label: string;
	parent: string | null;
	slug: string;
	kind: "article" | "section" | "concept";
	evidence: string;
}
export interface ConceptGraph {
	nodes: ConceptNode[];
	links: { from: string; to: string; label: string }[];
}
interface Heading {
	depth: number;
	text: string;
	slug: string;
}
// Vocabulary identifies concepts only when they occur in the article; it creates no claims or edges by itself.
const vocabularyZh = [
	"预训练",
	"后训练",
	"微调",
	"强化学习",
	"监督学习",
	"无监督学习",
	"自监督学习",
	"因果推断",
	"因果效应",
	"混杂变量",
	"反事实",
	"随机变量",
	"条件概率",
	"贝叶斯",
	"假设检验",
	"置信区间",
	"梯度下降",
	"损失函数",
	"注意力",
	"多头注意力",
	"位置编码",
	"词嵌入",
	"分词",
	"采样",
	"去重",
	"污染",
	"验证集",
	"测试集",
	"训练集",
	"质量评分",
	"过滤阈值",
	"教育价值",
	"数据混合",
	"合成数据",
	"归一化",
	"规范化",
	"近重复",
	"精确去重",
	"语义",
	"奖励",
	"策略",
	"价值函数",
	"探索",
	"利用",
	"状态",
	"动作",
	"环境",
	"折扣因子",
	"经验回放",
	"目标网络",
	"策略梯度",
	"重要性采样",
	"离线强化学习",
	"多智能体",
	"纳什均衡",
	"遗憾",
	"博弈",
	"蒙特卡洛",
	"检索",
	"工具调用",
	"规划",
	"记忆",
	"推理",
	"自我改进",
	"反馈",
	"评估",
	"机械",
	"连杆",
	"步态",
	"优化",
	"种群",
	"变异",
	"选择",
	"实验",
	"因果",
	"审美",
	"技术",
	"人文",
	"MinHash",
	"Jaccard",
	"LSH",
	"SHA-256",
	"tokenizer",
	"token",
	"Transformer",
	"Attention",
	"LoRA",
	"QLoRA",
	"SFT",
	"DPO",
	"PPO",
	"GRPO",
	"RLHF",
	"RLAIF",
	"DQN",
	"MCTS",
	"CFR",
	"RAG",
	"LLM",
	"Agent",
	"RSI",
	"AGI",
	"HTML",
	"Unicode",
	"NFC",
	"NFKC",
	"BPE",
	"EOS",
	"FineWeb",
	"FineWeb-Edu",
	"DCLM",
	"DSIR",
	"DoReMi",
	"Common Crawl",
	"SkyPile",
	"Wanjuan",
	"ChineseWebText",
	"Dolma",
	"RefinedWeb",
	"C4",
	"Pile",
	"Nemotron",
	"DataTrove",
	"Parquet",
	"JSONL",
	"Git",
	"Python",
	"PyTorch",
	"CUDA",
	"KV Cache",
	"FlashAttention",
	"CLIP",
	"BLIP",
	"AlphaZero",
	"CartPole",
	"DUCG",
	"YLearn",
];

/**
 * 英文概念词表。
 *
 * 原词表只有中文概念，英文文章只能靠加粗词与拉丁缩写撑起概念层：
 * MCTS 概览能抓到的概念从中文原文的 114 个掉到 22 个，exploration / reward /
 * policy 这一层全部丢失。这里按同一套「出现在正文才算概念」的规则补上英文词，
 * 中英文章各用各的语言命中，不跨语言推断。
 */
const vocabularyEn = [
	// 学习与优化
	"pre-training",
	"pretraining",
	"fine-tuning",
	"reinforcement learning",
	"supervised learning",
	"unsupervised learning",
	"self-supervised learning",
	"policy gradient",
	"value function",
	"action value",
	"advantage function",
	"experience replay",
	"target network",
	"discount factor",
	"importance sampling",
	"online learning",
	"offline reinforcement learning",
	"multi-agent reinforcement learning",
	"Nash equilibrium",
	"exploration",
	"exploitation",
	"reward",
	"policy",
	"state",
	"action",
	"environment",
	"bootstrapping",
	"temporal difference",
	// 因果与统计
	"causal inference",
	"causal effect",
	"confounder",
	"counterfactual",
	"randomized experiment",
	"observational study",
	"instrumental variable",
	"difference-in-differences",
	"hypothesis testing",
	"confidence interval",
	"effect size",
	"statistical significance",
	// 数据
	"pre-training corpus",
	"data mixture",
	"synthetic data",
	"deduplication",
	"near-duplicate",
	"exact deduplication",
	"contamination",
	"validation set",
	"test set",
	"training set",
	"quality scoring",
	"filtering",
	"normalization",
	"tokenization",
	"web crawl",
	"data quality",
	// 网络结构
	"gradient descent",
	"loss function",
	"self-attention",
	"multi-head attention",
	"positional encoding",
	"word embedding",
	"residual connection",
	"layer normalization",
	"context window",
	"next-token prediction",
	// 系统与部署
	"retrieval-augmented generation",
	"tool use",
	"planning",
	"memory",
	"reasoning",
	"self-improvement",
	"feedback",
	"evaluation",
	"inference",
	"quantization",
	"KV cache",
	"paged attention",
	"distillation",
	"alignment",
	"benchmark",
	"ablation",
	"scaling law",
	// 人文与机械
	"mechanism",
	"linkage",
	"gait",
	"optimization",
	"population",
	"mutation",
	"selection",
	"experiment",
	"causality",
	"aesthetics",
	"technology",
	"humanities",
];

const vocabulary = [...vocabularyZh, ...vocabularyEn];
const clean = (text: string): string =>
	text
		.replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
		.replace(/[*_`]/g, "")
		.replace(/<[^>]*>/g, "")
		// 加粗片段常带尾随标点（"Selection:"、"Scope note."），去掉后再作为节点名。
		.replace(/[.,:;!?。，：；！？]+$/g, "")
		.trim();

/** Extract only explicit headings and marked terms; shared terms imply recurrence, not causality. */
export function buildConceptGraph(
	title: string,
	body: string,
	headings: Heading[],
): ConceptGraph {
	const nodes: ConceptNode[] = [
		{
			id: "root",
			label: title,
			parent: null,
			slug: "",
			kind: "article",
			evidence: title,
		},
	];
	const stack: { depth: number; id: string }[] = [{ depth: 0, id: "root" }];
	const terms = new Map<string, string[]>();
	let section = nodes[0];
	let headingIndex = 0;
	let fence = "";
	let count = 0;
	const prose = new Map<string, string[]>();
	const seen = new Set<string>();
	for (const line of body.split("\n")) {
		const marker = line.match(/^\s*(`{3,}|~{3,})/);
		if (marker) {
			if (!fence) fence = marker[1][0];
			else if (marker[1][0] === fence) fence = "";
			continue;
		}
		if (fence) continue;
		const heading = line.match(/^#{1,6}\s+(.+?)\s*#*$/);
		if (heading) {
			const text = clean(heading[1]);
			const index = headings.findIndex(
				(h, i) => i >= headingIndex && clean(h.text.replace(/#$/, "")) === text,
			);
			if (index < 0) continue;
			headingIndex = index + 1;
			const h = headings[index];
			while (stack.length > 1 && stack[stack.length - 1].depth >= h.depth)
				stack.pop();
			section = {
				id: `section-${index}`,
				label: clean(h.text.replace(/#$/, "")),
				parent: stack[stack.length - 1].id,
				slug: h.slug,
				kind: "section",
				evidence: clean(h.text.replace(/#$/, "")),
			};
			nodes.push(section);
			stack.push({ depth: h.depth, id: section.id });
			count = 0;
			continue;
		}
		prose.set(section.id, [...(prose.get(section.id) || []), line]);
		// Bold terms, inline terminology and named links are evidence-backed concept candidates.
		for (const match of line.matchAll(
			/\*\*([^*\n]+)\*\*|`([^`\n]+)`|\[([^\]\n]+)\]\([^)]*\)/g,
		)) {
			const label = clean(match[1] || match[2] || match[3]);
			if (/第.{1,3}篇|论文|报告|§|原文|下一篇|上一篇/.test(label)) continue;
			// 英文文章里的版块小标题（Scope note / Further reading）不是概念。
			if (
				/^(scope note|note|summary|overview|further reading|see also|references?|appendix)$/i.test(
					label,
				)
			)
				continue;
			if (
				label.length < 2 ||
				label.length > 36 ||
				/[。！？；{}=]|https?:|^\//.test(label)
			)
				continue;
			const key = label.toLocaleLowerCase();
			if (
				seen.has(`${section.id}:${key}`) ||
				count >= 8 ||
				key === section.label.toLocaleLowerCase()
			)
				continue;
			seen.add(`${section.id}:${key}`);
			count++;
			const id = `concept-${nodes.length}`;
			nodes.push({
				id,
				label,
				parent: section.id,
				slug: section.slug,
				kind: "concept",
				evidence: clean(line).slice(0, 280),
			});
			terms.set(key, [...(terms.get(key) || []), id]);
		}
	}
	// Add frequent named concepts from prose, retaining a source sentence for every node.
	for (const [sectionId, lines] of prose) {
		const owner = nodes.find((n) => n.id === sectionId)!;
		const text = clean(lines.join("\n"));
		const alternatives = [...vocabulary]
			.sort((a, b) => b.length - a.length)
			.map((label) => {
				const escaped = label.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
				return /[a-z]/i.test(label)
					? `(?<![a-zA-Z])${escaped}(?![a-zA-Z])`
					: escaped;
			});
		const counts = new Map<string, number>();
		for (const match of text.matchAll(
			new RegExp(alternatives.join("|"), "gi"),
		)) {
			const key = match[0].toLocaleLowerCase();
			counts.set(key, (counts.get(key) || 0) + 1);
		}
		const candidates = vocabulary
			.map((label) => ({
				label,
				count: counts.get(label.toLocaleLowerCase()) || 0,
			}))
			.filter((c) => c.count > 0)
			.sort((a, b) => b.count - a.count || b.label.length - a.label.length);

		let added = 0;
		for (const { label } of candidates) {
			const key = label.toLocaleLowerCase();
			if (seen.has(`${sectionId}:${key}`) || added >= 6) continue;
			seen.add(`${sectionId}:${key}`);
			added++;
			const id = `concept-${nodes.length}`;
			const evidence =
				lines.find((line) => line.toLocaleLowerCase().includes(key)) || text;
			nodes.push({
				id,
				label,
				parent: sectionId,
				slug: owner.slug,
				kind: "concept",
				evidence: clean(evidence).slice(0, 280),
			});
			terms.set(key, [...(terms.get(key) || []), id]);
		}
	}

	// Preserve headings even for HTML/setext headings that the line parser cannot associate with prose.
	for (const [index, h] of headings.entries()) {
		if (!nodes.some((n) => n.id === `section-${index}`))
			nodes.push({
				id: `section-${index}`,
				label: clean(h.text.replace(/#$/, "")),
				parent: "root",
				slug: h.slug,
				kind: "section",
				evidence: clean(h.text.replace(/#$/, "")),
			});
	}
	const links: ConceptGraph["links"] = [];
	for (const ids of terms.values())
		for (let i = 1; i < ids.length; i++)
			links.push({
				from: ids[0],
				to: ids[i],
				label: "同一概念 / Shared concept",
			});
	return { nodes, links };
}
