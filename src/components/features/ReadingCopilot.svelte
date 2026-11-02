<script lang="ts">
import type { MLCEngine } from "@mlc-ai/web-llm";
import { onDestroy, onMount } from "svelte";
import { getCopilotSelection } from "@/utils/copilot-selection";
import {
	formatRagContextForPrompt,
	type RagSearchResult,
	retrieveKnowledge,
} from "@/utils/rag-retriever";

interface Props {
	lang?: "zh" | "en";
	title?: string;
}

let { lang = "zh", title = "" }: Props = $props();

// Model configuration options
interface ModelOption {
	id: string;
	label: string;
	descZh: string;
	descEn: string;
	sizeMB: number;
	tagZh: string;
	tagEn: string;
}

const AVAILABLE_MODELS: ModelOption[] = [
	{
		id: "SmolLM2-360M-Instruct-q4f16_1-MLC",
		label: "SmolLM2 360M",
		descZh: "默认推荐 · 超轻量极速，启动只需数秒，显存占用极低",
		descEn: "Default · Ultra-compact, seconds to load, very low VRAM",
		sizeMB: 130,
		tagZh: "默认极速",
		tagEn: "Default",
	},
	{
		id: "Qwen2.5-0.5B-Instruct-q4f16_1-MLC",
		label: "Qwen2.5 0.5B",
		descZh: "进阶推荐 · 中英文双语逻辑好，分析更周全",
		descEn: "Advanced balanced, excellent bilingual technical chat",
		sizeMB: 350,
		tagZh: "全能",
		tagEn: "Balanced",
	},
	{
		id: "Qwen2.5-Coder-0.5B-Instruct-q4f16_1-MLC",
		label: "Qwen2.5-Coder 0.5B",
		descZh: "代码与算法强化，专为代码推导与实现微调",
		descEn: "Code & algorithmic reasoning specialist",
		sizeMB: 350,
		tagZh: "代码",
		tagEn: "Code",
	},
	{
		id: "Llama-3.2-1B-Instruct-q4f16_1-MLC",
		label: "Llama 3.2 1B",
		descZh: "Meta 开源模型，英文与国际化表述优异",
		descEn: "Meta open model, strong English fluency",
		sizeMB: 700,
		tagZh: "国际",
		tagEn: "English",
	},
	{
		id: "Qwen2.5-1.5B-Instruct-q4f16_1-MLC",
		label: "Qwen2.5 1.5B",
		descZh: "深度推理与复杂逻辑，适合高配设备",
		descEn: "Deeper reasoning for higher-spec hardware",
		sizeMB: 1100,
		tagZh: "高智",
		tagEn: "Deep",
	},
];

// State
let isOpen = $state(false);
let activeTab = $state<"explain" | "summary" | "chat">("explain");
let engineStatus = $state<
	"unloaded" | "loading" | "ready" | "generating" | "error" | "no-webgpu"
>("unloaded");
let loadProgress = $state(0);
let loadStatusText = $state("");
let errorMessage = $state("");

// Default model is SmolLM2-360M
let selectedModelId = $state<string>("SmolLM2-360M-Instruct-q4f16_1-MLC");
let cachedModelMap = $state<Record<string, boolean>>({});

// Selection Explainer
let selectedText = $state("");
let explainResult = $state("");
let showSelectionBubble = $state(false);
let bubblePos = $state({ top: 0, left: 0 });

// TL;DR / Summary
let summaryResult = $state("");

// Chat & RAG
let chatScope = $state<"site" | "post">("site");
let isRetrieving = $state(false);
let chatMessages = $state<
	Array<{
		role: "user" | "assistant";
		content: string;
		sources?: RagSearchResult[];
	}>
>([]);
let chatInput = $state("");

// Copy status tracking
let copiedTag = $state<string | null>(null);

// Engine reference
let engine: MLCEngine | null = null;

const isEn = $derived(lang === "en");
const currentModelMeta = $derived(
	AVAILABLE_MODELS.find((m) => m.id === selectedModelId) || AVAILABLE_MODELS[0],
);

// Suggestions for ChatGPT-style follow-ups
const explainSuggestions = $derived(
	isEn
		? [
				"💡 Give a concrete numerical calculation example",
				"⚠️ What are the common engineering pitfalls here?",
				"⚖️ How does it fundamentally differ from standard baselines?",
			]
		: [
				"💡 举一个具体的数值计算小例子",
				"⚠️ 在工程落地时通常有什么陷阱？",
				"⚖️ 它与传统基线方法的核心差异是什么？",
			],
);

const summarySuggestions = $derived(
	isEn
		? [
				"🔍 What is the most critical technical innovation in this post?",
				"🧭 For a beginner, which section is recommended to read first?",
				"📊 Are the experimental conclusions and empirical evidence solid?",
			]
		: [
				"🔍 本文最关键的技术突破或核心观点是什么？",
				"🧭 针对刚接触该领域的读者，推荐优先阅读哪一节？",
				"📊 文中的实验数据与论据是否足够可信与严密？",
			],
);

const siteRagSuggestions = $derived(
	isEn
		? [
				"⚖️ What are the key differences between GRPO and PPO?",
				"⚙️ How is the mobility of the Strandbeest linkage calculated?",
				"📐 What is the core divide between explicit physics models and latent world models?",
				"⚡ What is the vocabulary compatibility pitfall in BPE tokenization?",
			]
		: [
				"⚖️ GRPO 与 PPO 在组内优势计算与基准估计上有何本质差异？",
				"⚙️ Jansen 仿生步态连杆机构的自由度（Grübler 公式）如何推导？",
				"📐 显式物理世界模型与神经网络隐空间做梦的核心分野与收敛？",
				"⚡ 大模型预训练与微调中，BPE 分词器最容易忽视的编号兼容性陷阱是什么？",
			],
);

// Translations
const t = $derived({
	triggerTitle: isEn ? "AI Reading Copilot" : "AI 伴读",
	triggerHint: isEn
		? "Explain selection & article Q&A"
		: "划词解析 · 文章速读 · 对话问答",
	tabs: {
		explain: isEn ? "Selection" : "划词释义",
		summary: isEn ? "TL;DR" : "本文速读",
		chat: isEn ? "Ask Copilot" : "智能问答",
	},
	ragScopeSite: isEn ? "Site Knowledge (RAG)" : "全站知识库 (RAG)",
	ragScopePost: isEn ? "Current Post Only" : "仅限当前文章",
	retrievingSources: isEn
		? "Searching site archive for grounded chunks..."
		: "正在检索全站知识库切片...",
	sourcesFound: isEn ? "Referenced Sources" : "引用全站参考资料",
	chatScopeHint: isEn
		? "Search across 100+ articles & notes"
		: "跨全站 100+ 篇深度技术博文联合检索与问答",
	selectionPlaceholder: isEn
		? "Highlight text in the article or type any formula / concept here..."
		: "在文章正文中选中文字，或在此直接输入公式、概念...",
	btnExplain: isEn ? "Explain with AI" : "AI 深度解析",
	btnSummary: isEn ? "Generate Article TL;DR" : "生成核心导读",
	btnSend: isEn ? "Send" : "发送",
	chatPlaceholder: isEn
		? chatScope === "site"
			? "Ask across all articles (e.g. GRPO vs PPO, Jansen mobility)..."
			: "Ask anything about this post..."
		: chatScope === "site"
			? "提问全站知识库（例如：PPO 与 GRPO 的异同、Jansen 连杆自由度）..."
			: "针对本文内容提问...",
	loadingTitle: isEn
		? "Initializing On-Device Model..."
		: "正在初始化端侧大模型...",
	loadingCachedHint: isEn
		? "⚡ Loading model from local browser cache. 0 network data used."
		: "⚡ 正在从浏览器本地持久缓存直接挂载模型权重，无需重复下载。",
	loadingDownloadHint: isEn
		? `📥 Downloading model weights (~${currentModelMeta.sizeMB}MB). Weights are cached permanently in browser for future instant use.`
		: `📥 首次加载正在下载模型权重（约 ${currentModelMeta.sizeMB}MB），下载后永久缓存在本地磁盘，重启浏览器也无需重新下载。`,
	noWebGPU: isEn
		? "WebGPU is not enabled or supported in your current browser. You can still use the fallback heuristic extractor below."
		: "当前浏览器未开启或不支持 WebGPU 加速。推荐使用 Chrome 113+、Edge 113+ 或开启了 WebGPU 的 Safari。已为您切换至启发式结构提炼模式。",
	fallbackSummaryTitle: isEn
		? "Heuristic Structure & Key Terms"
		: "文章结构与核心知识点梳理",
	stopGen: isEn ? "Stop generating" : "停止生成",
	clearChat: isEn ? "Clear" : "清空",
	close: isEn ? "Close" : "关闭",
	clearCache: isEn ? "Purge model cache" : "清除当前模型本地缓存",
	cachedBadge: isEn ? "Local Cache Ready" : "本地已就绪",
	downloadBadge: isEn ? "Need Download" : "初次需下载",
	copyBtn: isEn ? "Copy" : "复制",
	copiedBtn: isEn ? "Copied!" : "已复制!",
	jumpSelection: isEn ? "Locate in Post" : "定位原文",
	followupTitle: isEn ? "Suggested follow-ups" : "猜你想问 / 进一步探索",
});

// Lazy loaded markdown & KaTeX libraries
type KatexModule = typeof import("katex");
type MarkedModule = typeof import("marked");

let katexLib: KatexModule | null = null;
let markedLib: MarkedModule["marked"] | null = null;
let markdownDepsPromise: Promise<void> | null = null;
let depsLoaded = $state(false);

function escapeHtml(str: string): string {
	return str
		.replace(/&/g, "&amp;")
		.replace(/</g, "&lt;")
		.replace(/>/g, "&gt;")
		.replace(/"/g, "&quot;")
		.replace(/'/g, "&#039;");
}

async function ensureMarkdownDeps(): Promise<void> {
	if (katexLib && markedLib) return;
	if (!markdownDepsPromise) {
		markdownDepsPromise = Promise.all([import("katex"), import("marked")])
			.then(([kMod, mMod]) => {
				katexLib = (kMod.default || kMod) as unknown as KatexModule;
				markedLib = mMod.marked;
				depsLoaded = true;
			})
			.catch((err) => {
				console.error("Failed to dynamically load katex/marked:", err);
				markdownDepsPromise = null;
			});
	}
	return markdownDepsPromise;
}

// Preload markdown/katex libraries on drawer open
$effect(() => {
	if (isOpen) {
		ensureMarkdownDeps();
		checkModelsCacheStatus();
	}
});

// Markdown & KaTeX rendering helper (gracefully falls back if libs still loading)
function renderMarkdown(raw: string): string {
	if (!raw) return "";
	if (!depsLoaded || !markedLib) {
		ensureMarkdownDeps();
		return escapeHtml(raw).replace(/\n/g, "<br/>");
	}
	try {
		let s = raw;
		// 1. Replace block math $$...$$
		if (katexLib) {
			const kl = katexLib;
			s = s.replace(/\$\$([\s\S]+?)\$\$/g, (_, tex) => {
				try {
					return kl.renderToString(tex.trim(), {
						displayMode: true,
						throwOnError: false,
					});
				} catch {
					return `$$${tex}$$`;
				}
			});
			// 2. Replace inline math $...$
			s = s.replace(/\$([^$\n]+?)\$/g, (_, tex) => {
				try {
					return kl.renderToString(tex.trim(), {
						displayMode: false,
						throwOnError: false,
					});
				} catch {
					return `$${tex}$`;
				}
			});
		}
		return markedLib.parse(s, { breaks: true, gfm: true }) as string;
	} catch {
		return raw;
	}
}

// Copy helper with feedback
async function copyToClipboard(text: string, tag: string) {
	try {
		await navigator.clipboard.writeText(text);
		copiedTag = tag;
		setTimeout(() => {
			if (copiedTag === tag) copiedTag = null;
		}, 1800);
	} catch (e) {
		console.warn("Clipboard copy failed:", e);
	}
}

// Jump to selection in article
function jumpToSelection() {
	if (!selectedText.trim()) return;
	const container = document.getElementById("post-container");
	if (!container) return;

	// Highlight effect: find text or scroll to post container
	const selection = window.getSelection();
	if (selection && selection.rangeCount > 0) {
		const range = selection.getRangeAt(0);
		const rect = range.getBoundingClientRect();
		window.scrollTo({
			top: window.scrollY + rect.top - 120,
			behavior: "smooth",
		});
	} else {
		container.scrollIntoView({ behavior: "smooth", block: "start" });
	}
}

// Check WebGPU availability
function checkWebGPUSupport(): boolean {
	return (
		typeof navigator !== "undefined" &&
		"gpu" in navigator &&
		Boolean((navigator as unknown as { gpu?: unknown }).gpu)
	);
}

// Refresh local cache status of models
async function checkModelsCacheStatus() {
	if (!checkWebGPUSupport()) return;
	try {
		const { hasModelInCache } = await import("@mlc-ai/web-llm");
		const map: Record<string, boolean> = {};
		for (const m of AVAILABLE_MODELS) {
			map[m.id] = await hasModelInCache(m.id);
		}
		cachedModelMap = map;
	} catch (e) {
		console.warn("Failed to check WebLLM cache status:", e);
	}
}

// Initialize WebLLM engine lazily
async function initEngine() {
	if (engineStatus === "ready" || engineStatus === "loading") return;
	if (!checkWebGPUSupport()) {
		engineStatus = "no-webgpu";
		return;
	}

	engineStatus = "loading";
	loadProgress = 5;
	const isCached = cachedModelMap[selectedModelId];
	loadStatusText = isCached
		? isEn
			? "Loading model from local storage..."
			: "正在从本地缓存快速挂载权重..."
		: isEn
			? "Connecting to model weights repo..."
			: "正在连接模型权重仓库...";

	try {
		const { CreateMLCEngine } = await import("@mlc-ai/web-llm");
		engine = await CreateMLCEngine(selectedModelId, {
			initProgressCallback: (report) => {
				loadProgress = Math.round((report.progress || 0) * 100);
				loadStatusText =
					report.text ||
					(isCached
						? isEn
							? "Reading cached weights..."
							: "正在读取本地缓存权重..."
						: isEn
							? "Downloading weights..."
							: "正在下载权重...");
			},
		});
		engineStatus = "ready";
		await checkModelsCacheStatus();
	} catch (err: unknown) {
		console.error("WebLLM initialization failed:", err);
		errorMessage = err instanceof Error ? err.message : String(err);
		engineStatus = "error";
	}
}

// Switch model
async function handleModelChange(newModelId: string) {
	if (newModelId === selectedModelId) return;
	if (engine) {
		try {
			await engine.unload();
		} catch {
			/* Ignore unload error */
		}
		engine = null;
	}
	selectedModelId = newModelId;
	try {
		localStorage.setItem("feng-yu-copilot-model", newModelId);
	} catch {
		/* Ignore storage error */
	}
	engineStatus = "unloaded";
	loadProgress = 0;
	loadStatusText = "";
	explainResult = "";
	summaryResult = "";
	await checkModelsCacheStatus();
}

// Purge current model cache
async function handleClearCache() {
	if (
		!confirm(
			isEn
				? `Clear local weights cache for ${currentModelMeta.label}? You can re-download it anytime.`
				: `确定要清除 ${currentModelMeta.label} 的本地缓存吗？清除后可随时重新下载。`,
		)
	) {
		return;
	}

	try {
		const { deleteModelAllInfoInCache } = await import("@mlc-ai/web-llm");
		if (engine) {
			await engine.unload();
			engine = null;
		}
		await deleteModelAllInfoInCache(selectedModelId);
		engineStatus = "unloaded";
		await checkModelsCacheStatus();
	} catch (e) {
		console.error("Failed to delete model cache:", e);
	}
}

// Stream text generation helper
async function streamCompletion(
	systemPrompt: string,
	userPrompt: string,
	onChunk: (chunk: string) => void,
) {
	if (!engine) {
		await initEngine();
	}
	if (engineStatus !== "ready" || !engine) return;

	engineStatus = "generating";
	try {
		const response = await engine.chat.completions.create({
			messages: [
				{ role: "system", content: systemPrompt },
				{ role: "user", content: userPrompt },
			],
			temperature: 0.5,
			max_tokens: 750,
			stream: true,
		});

		for await (const chunk of response) {
			const delta = chunk.choices[0]?.delta?.content || "";
			onChunk(delta);
		}
	} catch (err: unknown) {
		console.error("Inference error:", err);
		const errText =
			err instanceof Error ? err.message : "Inference interrupted";
		onChunk(`\n\n[Error: ${errText}]`);
	} finally {
		engineStatus = "ready";
	}
}

// Action: Explain selection
async function handleExplain() {
	ensureMarkdownDeps();
	const target = selectedText.trim();
	if (!target) return;

	explainResult = "";
	if (engineStatus === "no-webgpu") {
		generateFallbackExplain(target);
		return;
	}

	await initEngine();
	if (engineStatus === "error" || engineStatus === "no-webgpu") {
		generateFallbackExplain(target);
		return;
	}

	const systemPrompt = isEn
		? "You are an expert AI research tutor. Explain the user's selected mathematical formula, technical term, or sentence in 2-3 clear, insightful bullet points using Markdown with LaTeX math where appropriate ($...$ and $$...$$). Highlight intuition, function in context, and subtle nuances."
		: "你是一位兼具广度与深度的计算机与人工智能科研导师。请针对用户在博文中选中的公式、数学符号、专业术语或代码片段，用通俗、严谨且富有洞察力的语言进行解析，使用标准 Markdown 与 LaTeX 数学公式（行内 $...$ 或块级 $$...$$）：1. 核心直觉与物理含义；2. 在上下文/该算法中的关键作用；3. 容易混淆的要点。控制在 2~3 段以内。";

	await streamCompletion(systemPrompt, target, (delta) => {
		explainResult += delta;
	});
}

// Action: Generate TL;DR
async function handleSummary() {
	ensureMarkdownDeps();
	summaryResult = "";
	const articleTitle =
		title || document.querySelector("h1")?.textContent?.trim() || "";
	const headings = Array.from(
		document.querySelectorAll("#post-container h2, #post-container h3"),
	)
		.map((h) => h.textContent?.trim())
		.filter(Boolean)
		.slice(0, 12);

	if (engineStatus === "no-webgpu") {
		generateFallbackSummary(articleTitle, headings);
		return;
	}

	await initEngine();
	if (engineStatus === "error" || engineStatus === "no-webgpu") {
		generateFallbackSummary(articleTitle, headings);
		return;
	}

	const promptContent = `文章标题: ${articleTitle}\n章节目录:\n${headings.map((h, i) => `${i + 1}. ${h}`).join("\n")}`;
	const systemPrompt = isEn
		? "You are a research paper reading assistant. Based on the provided article title and section outline, generate formatted Markdown with: 1. Core motivation and target problem; 2. Three key technical takeaways; 3. Recommended reading sequence."
		: "你是一位资深研究员与博文导读助理。请根据提供的博文标题和章节骨架，提供一份结构优美的 Markdown 导读指引：1. 本文核心解决的问题或探讨的命题；2. 3个最具价值的技术要点/结论；3. 针对不同背景读者的建议阅读顺序。请使用规范的 Markdown 标题与无序列表。";

	await streamCompletion(systemPrompt, promptContent, (delta) => {
		summaryResult += delta;
	});
}

// Action: Chat Q&A
async function handleSendMessage(overrideQuery?: string) {
	ensureMarkdownDeps();
	const query = (overrideQuery || chatInput).trim();
	if (!query) return;

	chatMessages = [...chatMessages, { role: "user", content: query }];
	if (!overrideQuery) chatInput = "";

	let ragResults: RagSearchResult[] = [];
	if (chatScope === "site") {
		isRetrieving = true;
		try {
			ragResults = await retrieveKnowledge(query, {
				topK: 4,
				lang: isEn ? "en" : "all",
			});
		} catch (e) {
			console.warn("RAG retrieval error:", e);
		} finally {
			isRetrieving = false;
		}
	}

	const contentText =
		document.querySelector(".markdown-content")?.textContent?.slice(0, 2500) ||
		"";

	if (engineStatus === "no-webgpu") {
		let fallbackContent = "";
		if (chatScope === "site" && ragResults.length > 0) {
			fallbackContent = isEn
				? `⚡ *(WebGPU not active - Heuristic RAG mode)*\n\nFound **${ragResults.length} relevant knowledge chunks** in the site archive:\n\n` +
					ragResults
						.map(
							(r, i) =>
								`**[${i + 1}] [${r.chunk.postTitle} · ${r.chunk.heading}](${r.chunk.url})**\n> ${r.chunk.text.slice(0, 220)}...`,
						)
						.join("\n\n")
				: `⚡ *(当前未开启 WebGPU - 启发式切片检索模式)*\n\n已为您从全站检索到 **${ragResults.length} 个高相关度知识切片**：\n\n` +
					ragResults
						.map(
							(r, i) =>
								`**[${i + 1}] [${r.chunk.postTitle} · ${r.chunk.heading}](${r.chunk.url})**\n> ${r.chunk.text.slice(0, 220)}...`,
						)
						.join("\n\n");
		} else {
			fallbackContent = isEn
				? "Direct Q&A requires WebGPU acceleration. Please try Chrome 113+ or Edge to enable on-device real-time chat."
				: "自由对话问答需要 WebGPU 硬件加速支持。请在 Chrome 113+ 或 Edge 等支持 WebGPU 的浏览器中体验纯端侧实时交互。";
		}

		chatMessages = [
			...chatMessages,
			{
				role: "assistant",
				content: fallbackContent,
				sources: ragResults.length > 0 ? ragResults : undefined,
			},
		];
		return;
	}

	await initEngine();
	if (engineStatus !== "ready") return;

	let assistantReply = "";
	chatMessages = [
		...chatMessages,
		{
			role: "assistant",
			content: "",
			sources: ragResults.length > 0 ? ragResults : undefined,
		},
	];
	const assistantIndex = chatMessages.length - 1;

	let systemPrompt = "";
	let promptUserContent = query;

	if (chatScope === "site") {
		const ragContext = formatRagContextForPrompt(
			ragResults,
			isEn ? "en" : "zh",
		);
		systemPrompt = isEn
			? `You are an expert AI research tutor grounded in the author's technical blog knowledge base.
Answer the user's question with mathematical precision, physical intuition, and clarity using Markdown with LaTeX ($...$ and $$...$$).
Instructions:
1. Thoroughly leverage the provided [Relevant Knowledge Base Context].
2. Natural citing: Cite references as [1], [2] when stating key theorems, formulations, or empirical observations.
3. If the snippets don't completely cover all aspects of the query, augment using rigorous first-principles computer science and math, while noting what is grounded in the site archive.
4. Keep the explanation structured and insightful.`
			: `你是一位精通人工智能、强化学习、控制论与系统工程的博文学术导师。
请针对读者的问题，结合检索到的博文知识库切片给出严谨、通透且富有洞察力的回答。使用规范的 Markdown 与 LaTeX 数学公式（$...$ 与 $$...$$）。
回答规范：
1. 充分利用提供的【全站相关知识库参考资料】；
2. 严谨引用：在陈述关键定理、公式推导或工程结论时，自然标注引用序号（如 [1]、[2]）；
3. 若检索切片未完全覆盖问题，可基于第一性原理进行合理论证，并清晰区分“站内文章论证...”与“通用背景补充...”；
4. 语言精炼优雅，结构清晰，突出核心机制与物理直觉。`;

		promptUserContent = `${ragContext}\n\n读者问题: ${query}`;
	} else {
		systemPrompt = isEn
			? `You are an AI assistant grounded in this technical blog post. Answer the user's question concisely based on the context in Markdown:\n\n${contentText}`
			: `你是一个基于当前博文内容的智能助教。请根据以下提取的文章正文片段回答读者提问，保持严谨客观，使用清晰的 Markdown 排版：\n\n${contentText}`;
		promptUserContent = query;
	}

	await streamCompletion(systemPrompt, promptUserContent, (delta) => {
		assistantReply += delta;
		chatMessages[assistantIndex] = {
			role: "assistant",
			content: assistantReply,
			sources: ragResults.length > 0 ? ragResults : undefined,
		};
	});
}

// Quick trigger follow-up question
function triggerFollowup(prompt: string) {
	activeTab = "chat";
	handleSendMessage(prompt);
}

// Stop generation
function stopGeneration() {
	if (engine && typeof engine.interruptGenerate === "function") {
		engine.interruptGenerate();
	}
	engineStatus = "ready";
}

// Fallbacks for browsers without WebGPU
function generateFallbackExplain(text: string) {
	explainResult = isEn
		? `💡 **Term/Snippet Focus**: "${text}"\n\n*(WebGPU not active)* In this context, this expression specifies key constraints or updates within the mathematical formulation. Please verify relevant sections in the article for full derivational context.`
		: `💡 **划词焦点**：「${text}」\n\n*(当前处于轻量启发模式)* 该段文本/公式构成了本文论证或算法设计的核心步骤。若为数学符号或变量，请留意正文中前后的符号定义表与超参数约定。`;
}

function generateFallbackSummary(
	articleTitle: string,
	headings: (string | null | undefined)[],
) {
	summaryResult = isEn
		? `⚡ **Heuristic Overview for "${articleTitle}"**:\n\n` +
			"1. **Focus Area**: Addresses core mechanisms through structured theoretical derivation and experiments.\n" +
			`2. **Key Sections**:\n${headings.map((h) => `   - ${h}`).join("\n")}\n` +
			"3. **Reading Suggestion**: Start with the problem formulation before diving into experimental validations."
		: `⚡ **「${articleTitle}」核心结构速览**：\n\n` +
			"1. **核心议题**：围绕系统架构、算法推演与工程实证展开深入探讨。\n" +
			`2. **主要脉络**：\n${headings.map((h) => `   - ${h}`).join("\n")}\n` +
			"3. **阅读建议**：建议先通读背景与形式化定义，再对照实验结果验证设计结论。";
}

// Listen to mouseup selection inside article
function handleTextSelection() {
	const result = getCopilotSelection(
		window.getSelection(),
		document.getElementById("post-container"),
		window.scrollX,
		window.scrollY,
	);
	if (!result) {
		showSelectionBubble = false;
		return;
	}
	bubblePos = { top: result.top, left: result.left };
	selectedText = result.text;
	showSelectionBubble = true;
}

function triggerExplainFromBubble() {
	showSelectionBubble = false;
	isOpen = true;
	activeTab = "explain";
	handleExplain();
}

onMount(() => {
	// Restore saved model (default SmolLM2 if not set)
	try {
		const saved = localStorage.getItem("feng-yu-copilot-model");
		if (saved && AVAILABLE_MODELS.some((m) => m.id === saved)) {
			selectedModelId = saved;
		}
	} catch {
		/* Ignore storage error */
	}

	document.addEventListener("mouseup", handleTextSelection);
	const handleKeyDown = (e: KeyboardEvent) => {
		if (e.key === "Escape" && isOpen) {
			isOpen = false;
		}
	};
	window.addEventListener("keydown", handleKeyDown);

	return () => {
		document.removeEventListener("mouseup", handleTextSelection);
		window.removeEventListener("keydown", handleKeyDown);
		if (engine && typeof engine.unload === "function") {
			try {
				const eng = engine as unknown as { interruptGenerate?: () => void };
				if (typeof eng.interruptGenerate === "function") {
					eng.interruptGenerate();
				}
				engine.unload().catch(() => {});
			} catch {
				/* Ignore teardown errors */
			}
		}
	};
});
</script>

<!-- Selection Floating Quick Action Bubble -->
{#if showSelectionBubble}
	<div
		class="selection-copilot-bubble"
		style="top: {bubblePos.top}px; left: {bubblePos.left}px;"
		role="tooltip"
	>
		<button
			type="button"
			class="bubble-btn"
			onclick={triggerExplainFromBubble}
			title={t.tabs.explain}
		>
			<span class="bubble-spark">✨</span>
			<span class="bubble-text">{isEn ? "Explain" : "AI 解析"}</span>
		</button>
	</div>
{/if}

<!-- Floating Copilot Launch Pill -->
<div class="reading-copilot-launcher" class:is-hidden={isOpen}>
	<button
		type="button"
		class="launcher-pill"
		onmouseenter={() => ensureMarkdownDeps()}
		onclick={() => {
			isOpen = true;
			if (engineStatus === "unloaded") {
				initEngine();
			}
		}}
		title={t.triggerHint}
		aria-label={t.triggerTitle}
	>
		<span class="launcher-glow" aria-hidden="true"></span>
		<span class="launcher-icon">✨</span>
		<span class="launcher-label">{t.triggerTitle}</span>
		{#if cachedModelMap[selectedModelId]}
			<span class="launcher-cache-dot" title={t.cachedBadge}></span>
		{/if}
	</button>
</div>

<!-- Copilot Sliding Panel / Drawer -->
{#if isOpen}
	<div class="copilot-backdrop" onclick={() => (isOpen = false)} role="presentation"></div>
	<aside
		class="copilot-drawer"
		aria-label={t.triggerTitle}
		role="dialog"
		aria-modal="true"
	>
		<!-- Header -->
		<header class="copilot-header">
			<div class="copilot-title-group">
				<span class="copilot-brand-icon">✨</span>
				<div class="header-text-block">
					<div class="title-row">
						<h3 class="copilot-title">{t.triggerTitle}</h3>
						{#if cachedModelMap[selectedModelId]}
							<span class="header-cache-tag is-cached" title="模型权重已存储在浏览器本地 Cache Storage 中，关闭浏览器再开无需重复下载">
								⚡ {t.cachedBadge}
							</span>
						{:else}
							<span class="header-cache-tag is-uncached" title="初次使用需下载，随后自动永久缓存">
								📥 ~{currentModelMeta.sizeMB}MB
							</span>
						{/if}
					</div>
					<!-- Model Switcher Dropdown -->
					<div class="model-picker-row">
						<label for="copilot-model-select" class="sr-only">选择大模型</label>
						<select
							id="copilot-model-select"
							class="model-select"
							value={selectedModelId}
							onchange={(e) => handleModelChange(e.currentTarget.value)}
							disabled={engineStatus === "generating" || engineStatus === "loading"}
						>
							{#each AVAILABLE_MODELS as model}
								<option value={model.id}>
									{model.label} ({cachedModelMap[model.id] ? (isEn ? "⚡Cached" : "⚡已缓存") : `~${model.sizeMB}MB`}) · {isEn ? model.tagEn : model.tagZh}
								</option>
							{/each}
						</select>
						{#if cachedModelMap[selectedModelId]}
							<button
								type="button"
								class="cache-action-btn"
								onclick={handleClearCache}
								title={t.clearCache}
								aria-label={t.clearCache}
							>
								🗑️
							</button>
						{/if}
					</div>
				</div>
			</div>
			<button
				type="button"
				class="copilot-close-btn"
				onclick={() => (isOpen = false)}
				aria-label={t.close}
			>
				✕
			</button>
		</header>

		<!-- Model Loading Banner -->
		{#if engineStatus === "loading"}
			<div class="copilot-banner-loading">
				<div class="loading-header">
					<span class="spin-icon">⏳</span>
					<span class="loading-title">
						{cachedModelMap[selectedModelId] ? (isEn ? "Loading from Cache..." : "从本地持久缓存快速挂载...") : t.loadingTitle}
					</span>
					<span class="loading-percent">{loadProgress}%</span>
				</div>
				<div class="progress-bar-bg">
					<div class="progress-bar-fill" style="width: {loadProgress}%"></div>
				</div>
				<p class="loading-text">{loadStatusText}</p>
				<p class="loading-hint">
					{cachedModelMap[selectedModelId] ? t.loadingCachedHint : t.loadingDownloadHint}
				</p>
			</div>
		{:else if engineStatus === "no-webgpu"}
			<div class="copilot-banner-alert">
				<span class="alert-icon">⚡</span>
				<p class="alert-text">{t.noWebGPU}</p>
			</div>
		{/if}

		<!-- Tabs -->
		<nav class="copilot-tabs" role="tablist">
			<button
				type="button"
				role="tab"
				class="tab-btn"
				class:active={activeTab === "explain"}
				aria-selected={activeTab === "explain"}
				onclick={() => (activeTab = "explain")}
			>
				💡 {t.tabs.explain}
			</button>
			<button
				type="button"
				role="tab"
				class="tab-btn"
				class:active={activeTab === "summary"}
				aria-selected={activeTab === "summary"}
				onclick={() => {
					activeTab = "summary";
					if (!summaryResult) handleSummary();
				}}
			>
				⚡ {t.tabs.summary}
			</button>
			<button
				type="button"
				role="tab"
				class="tab-btn"
				class:active={activeTab === "chat"}
				aria-selected={activeTab === "chat"}
				onclick={() => (activeTab = "chat")}
			>
				💬 {t.tabs.chat}
			</button>
		</nav>

		<!-- Content Panels -->
		<div class="copilot-body">
			<!-- TAB 1: SELECTION EXPLAINER -->
			{#if activeTab === "explain"}
				<div class="tab-panel">
					<div class="input-card">
						<textarea
							class="selection-input"
							bind:value={selectedText}
							placeholder={t.selectionPlaceholder}
							rows="3"
						></textarea>
						<div class="action-row">
							<button
								type="button"
								class="primary-action-btn"
								disabled={!selectedText.trim() || engineStatus === "loading"}
								onclick={handleExplain}
							>
								{engineStatus === "generating" ? "..." : t.btnExplain}
							</button>
							{#if selectedText.trim()}
								<button
									type="button"
									class="secondary-action-btn"
									onclick={jumpToSelection}
									title="在文章正文中定位并查看选区"
								>
									🎯 {t.jumpSelection}
								</button>
							{/if}
							{#if engineStatus === "generating"}
								<button type="button" class="stop-btn" onclick={stopGeneration}>
									{t.stopGen}
								</button>
							{/if}
						</div>
					</div>

					{#if explainResult}
						<div class="result-card">
							<div class="result-header">
								<span class="result-tag">💡 {isEn ? "Explanation" : "解析阐释"}</span>
								<button
									type="button"
									class="copy-action-btn"
									onclick={() => copyToClipboard(explainResult, "explain")}
								>
									{copiedTag === "explain" ? `✓ ${t.copiedBtn}` : `📋 ${t.copyBtn}`}
								</button>
							</div>
							<div class="result-content markdown-rendered">
								{@html renderMarkdown(explainResult)}
								{#if engineStatus === "generating"}
									<span class="streaming-cursor" aria-hidden="true">▍</span>
								{/if}
							</div>

							<!-- Follow-up Suggestion Chips -->
							{#if engineStatus !== "generating"}
								<div class="followup-suggestions-block">
									<p class="suggestions-title">{t.followupTitle}</p>
									<div class="suggestions-chips">
										{#each explainSuggestions as prompt}
											<button
												type="button"
												class="suggestion-chip"
												onclick={() => triggerFollowup(prompt)}
											>
												{prompt} ↗
											</button>
										{/each}
									</div>
								</div>
							{/if}
						</div>
					{/if}
				</div>
			{/if}

			<!-- TAB 2: TL;DR SUMMARY -->
			{#if activeTab === "summary"}
				<div class="tab-panel">
					<div class="action-row mb-3">
						<button
							type="button"
							class="primary-action-btn"
							disabled={engineStatus === "loading"}
							onclick={handleSummary}
						>
							⚡ {t.btnSummary}
						</button>
						{#if summaryResult}
							<button
								type="button"
								class="secondary-action-btn"
								onclick={() => copyToClipboard(summaryResult, "summary")}
							>
								{copiedTag === "summary" ? `✓ ${t.copiedBtn}` : `📋 ${t.copyBtn}`}
							</button>
						{/if}
						{#if engineStatus === "generating"}
							<button type="button" class="stop-btn" onclick={stopGeneration}>
								{t.stopGen}
							</button>
						{/if}
					</div>

					{#if summaryResult}
						<div class="result-card">
							<div class="result-header">
								<span class="result-tag">⚡ {isEn ? "Key Takeaways" : "核心导读"}</span>
							</div>
							<div class="result-content markdown-rendered">
								{@html renderMarkdown(summaryResult)}
								{#if engineStatus === "generating"}
									<span class="streaming-cursor" aria-hidden="true">▍</span>
								{/if}
							</div>

							<!-- Follow-up Suggestion Chips -->
							{#if engineStatus !== "generating"}
								<div class="followup-suggestions-block">
									<p class="suggestions-title">{t.followupTitle}</p>
									<div class="suggestions-chips">
										{#each summarySuggestions as prompt}
											<button
												type="button"
												class="suggestion-chip"
												onclick={() => triggerFollowup(prompt)}
											>
												{prompt} ↗
											</button>
										{/each}
									</div>
								</div>
							{/if}
						</div>
					{/if}
				</div>
			{/if}

			<!-- TAB 3: CHAT Q&A & SITE RAG -->
			{#if activeTab === "chat"}
				<div class="tab-panel chat-panel">
					<!-- RAG Scope Toggle Bar -->
					<div class="chat-scope-bar" role="group" aria-label="RAG Scope Selector">
						<button
							type="button"
							class="scope-btn"
							class:is-active={chatScope === "site"}
							onclick={() => (chatScope = "site")}
							title={t.chatScopeHint}
						>
							<span class="scope-icon">🌐</span>
							<span class="scope-text">{t.ragScopeSite}</span>
						</button>
						<button
							type="button"
							class="scope-btn"
							class:is-active={chatScope === "post"}
							onclick={() => (chatScope = "post")}
						>
							<span class="scope-icon">📄</span>
							<span class="scope-text">{t.ragScopePost}</span>
						</button>
					</div>

					<div class="chat-history">
						{#if chatMessages.length === 0}
							<div class="chat-empty">
								<span class="empty-icon">{chatScope === "site" ? "🌐" : "💬"}</span>
								<p>
									{chatScope === "site"
										? isEn
											? "Ask any conceptual, mathematical, or empirical question across all 100+ articles in the library."
											: "提问全站 100+ 篇深度技术博文与智能演进路线，支持公式推导、理论辨析与跨篇考据。"
										: isEn
											? "Ask any question about this post's arguments, data, or math."
											: "针对本文的论证逻辑、实验数据或公式推导提出任何疑问。"}
								</p>
								<div class="chat-starters">
									{#each (chatScope === "site" ? siteRagSuggestions : summarySuggestions) as prompt}
										<button
											type="button"
											class="suggestion-chip chat-starter-chip"
											onclick={() => triggerFollowup(prompt)}
										>
											{prompt}
										</button>
									{/each}
								</div>
							</div>
						{:else}
							{#each chatMessages as msg, idx}
								<div class="chat-bubble-row" class:is-user={msg.role === "user"}>
									<div class="chat-bubble">
										{#if msg.role === "assistant"}
											<div class="bubble-toolbar">
												<span class="assistant-tag">🤖 {currentModelMeta.label}</span>
												<button
													type="button"
													class="copy-mini-btn"
													onclick={() => copyToClipboard(msg.content, `chat-${idx}`)}
													title="复制此条回复"
												>
													{copiedTag === `chat-${idx}` ? `✓` : `📋`}
												</button>
											</div>

											{#if msg.sources && msg.sources.length > 0}
												<div class="rag-sources-panel">
													<div class="rag-sources-header">
														<span class="rag-sources-icon">📚</span>
														<span class="rag-sources-title">{t.sourcesFound} ({msg.sources.length})</span>
													</div>
													<div class="rag-sources-chips">
														{#each msg.sources as res, sIdx}
															<a
																href={res.chunk.url}
																target="_blank"
																rel="noopener noreferrer"
																class="rag-source-chip"
																title="{res.chunk.text.slice(0, 140)}..."
															>
																<span class="rag-source-idx">[{sIdx + 1}]</span>
																<span class="rag-source-name">{res.chunk.postTitle} · {res.chunk.heading}</span>
																<span class="rag-source-arrow">↗</span>
															</a>
														{/each}
													</div>
												</div>
											{/if}

											<div class="markdown-rendered">
												{@html renderMarkdown(msg.content)}
												{#if engineStatus === "generating" && idx === chatMessages.length - 1}
													<span class="streaming-cursor" aria-hidden="true">▍</span>
												{/if}
											</div>
										{:else}
											<div class="user-text">{msg.content}</div>
										{/if}
									</div>
								</div>
							{/each}

							{#if isRetrieving}
								<div class="retrieval-indicator-row">
									<div class="retrieval-indicator">
										<span class="retrieval-spinner" aria-hidden="true"></span>
										<span>{t.retrievingSources}</span>
									</div>
								</div>
							{/if}
						{/if}
					</div>

					<form
						class="chat-input-bar"
						onsubmit={(e) => {
							e.preventDefault();
							handleSendMessage();
						}}
					>
						<input
							type="text"
							class="chat-input"
							bind:value={chatInput}
							placeholder={t.chatPlaceholder}
							disabled={engineStatus === "loading"}
						/>
						<button
							type="submit"
							class="primary-action-btn chat-send-btn"
							disabled={!chatInput.trim() || engineStatus === "generating"}
						>
							{t.btnSend}
						</button>
					</form>
				</div>
			{/if}
		</div>
	</aside>
{/if}

<style>
	/* Selection floating bubble */
	.selection-copilot-bubble {
		position: absolute;
		z-index: 100;
		transform: translateY(-100%);
		animation: popIn 0.18s cubic-bezier(0.16, 1, 0.3, 1);
	}
	@keyframes popIn {
		from { opacity: 0; transform: translateY(-90%) scale(0.92); }
		to { opacity: 1; transform: translateY(-100%) scale(1); }
	}
	.bubble-btn {
		display: inline-flex;
		align-items: center;
		gap: 5px;
		padding: 5px 11px;
		border-radius: 999px;
		border: 1px solid rgba(92, 122, 107, 0.4);
		background: var(--brand-paper, #fbfbf9);
		box-shadow: 0 4px 14px rgba(0, 0, 0, 0.16);
		color: var(--brand-accent, #5c7a6b);
		font-size: 12px;
		font-weight: 600;
		cursor: pointer;
		transition: all 0.15s ease;
		backdrop-filter: blur(8px);
	}
	.bubble-btn:hover {
		background: var(--brand-accent, #5c7a6b);
		color: #ffffff;
		transform: translateY(-1px);
	}

	/* Floating launcher pill */
	.reading-copilot-launcher {
		position: fixed;
		bottom: 26px;
		right: 26px;
		z-index: 45;
		transition: opacity 0.2s ease, transform 0.2s ease;
	}
	.reading-copilot-launcher.is-hidden {
		opacity: 0;
		pointer-events: none;
		transform: scale(0.9);
	}
	.launcher-pill {
		position: relative;
		display: inline-flex;
		align-items: center;
		gap: 8px;
		padding: 9px 16px;
		border-radius: 999px;
		border: 1px solid var(--brand-line, rgba(0, 0, 0, 0.12));
		background: var(--site-glass, rgba(255, 255, 255, 0.85));
		box-shadow: 0 6px 20px rgba(0, 0, 0, 0.12);
		backdrop-filter: blur(16px);
		color: var(--brand-ink, #2d3748);
		font-size: 13px;
		font-weight: 600;
		cursor: pointer;
		overflow: hidden;
		transition: all 0.2s ease;
	}
	.launcher-pill:hover {
		border-color: var(--brand-accent, #5c7a6b);
		color: var(--brand-accent, #5c7a6b);
		transform: translateY(-2px);
		box-shadow: 0 8px 24px rgba(92, 122, 107, 0.22);
	}
	.launcher-glow {
		position: absolute;
		inset: -1px;
		border-radius: inherit;
		background: radial-gradient(circle at 50% 0%, rgba(92, 122, 107, 0.2), transparent 70%);
		pointer-events: none;
	}
	.launcher-icon {
		font-size: 14px;
		animation: pulseSpark 2.4s ease-in-out infinite;
	}
	.launcher-cache-dot {
		width: 7px;
		height: 7px;
		border-radius: 50%;
		background: #10b981;
		box-shadow: 0 0 6px #10b981;
	}
	@keyframes pulseSpark {
		0%, 100% { transform: scale(1); opacity: 0.9; }
		50% { transform: scale(1.18); opacity: 1; }
	}

	/* Drawer Backdrop & Panel */
	.copilot-backdrop {
		position: fixed;
		inset: 0;
		z-index: 55;
		background: rgba(0, 0, 0, 0.35);
		backdrop-filter: blur(4px);
		animation: fadeIn 0.2s ease;
	}
	@keyframes fadeIn {
		from { opacity: 0; }
		to { opacity: 1; }
	}

	.copilot-drawer {
		position: fixed;
		top: 0;
		right: 0;
		bottom: 0;
		width: min(480px, 94vw);
		z-index: 60;
		background: var(--brand-paper, #fdfdfc);
		border-left: 1px solid var(--brand-line, rgba(0, 0, 0, 0.12));
		box-shadow: -8px 0 30px rgba(0, 0, 0, 0.18);
		display: flex;
		flex-direction: column;
		animation: slideIn 0.24s cubic-bezier(0.16, 1, 0.3, 1);
	}
	@keyframes slideIn {
		from { transform: translateX(100%); }
		to { transform: translateX(0); }
	}

	.copilot-header {
		display: flex;
		align-items: flex-start;
		justify-content: space-between;
		padding: 14px 18px;
		border-bottom: 1px solid var(--brand-line, rgba(0, 0, 0, 0.1));
		gap: 12px;
	}
	.copilot-title-group {
		display: flex;
		align-items: flex-start;
		gap: 10px;
		flex: 1;
		min-width: 0;
	}
	.copilot-brand-icon {
		font-size: 20px;
		padding-top: 2px;
	}
	.header-text-block {
		flex: 1;
		min-width: 0;
	}
	.title-row {
		display: flex;
		align-items: center;
		gap: 8px;
		flex-wrap: wrap;
	}
	.copilot-title {
		margin: 0;
		font-size: 15px;
		font-weight: 700;
		color: var(--brand-ink, #2d3748);
	}
	.header-cache-tag {
		font-size: 11px;
		font-weight: 600;
		padding: 1px 6px;
		border-radius: 4px;
	}
	.header-cache-tag.is-cached {
		background: rgba(16, 185, 129, 0.12);
		color: #059669;
		border: 1px solid rgba(16, 185, 129, 0.25);
	}
	.header-cache-tag.is-uncached {
		background: rgba(234, 179, 8, 0.12);
		color: #b45309;
		border: 1px solid rgba(234, 179, 8, 0.25);
	}

	.model-picker-row {
		display: flex;
		align-items: center;
		gap: 6px;
		margin-top: 6px;
	}
	.model-select {
		padding: 3px 8px;
		border-radius: 6px;
		border: 1px solid var(--brand-line, rgba(0, 0, 0, 0.15));
		background: var(--brand-surface, #ffffff);
		color: var(--brand-ink, #2d3748);
		font-size: 11.5px;
		font-family: inherit;
		outline: none;
		max-width: 250px;
		cursor: pointer;
	}
	.model-select:focus {
		border-color: var(--brand-accent, #5c7a6b);
	}
	.cache-action-btn {
		background: transparent;
		border: none;
		cursor: pointer;
		font-size: 12px;
		padding: 2px 4px;
		border-radius: 4px;
		opacity: 0.6;
		transition: opacity 0.15s ease;
	}
	.cache-action-btn:hover {
		opacity: 1;
		background: rgba(239, 68, 68, 0.1);
	}

	.copilot-close-btn {
		background: transparent;
		border: none;
		font-size: 16px;
		color: var(--brand-muted, #718096);
		cursor: pointer;
		padding: 6px;
		border-radius: 6px;
		transition: all 0.12s ease;
		flex-shrink: 0;
	}
	.copilot-close-btn:hover {
		background: var(--brand-surface, rgba(0, 0, 0, 0.05));
		color: var(--brand-ink, #2d3748);
	}

	/* Model load & WebGPU alert banners */
	.copilot-banner-loading {
		margin: 12px 16px;
		padding: 12px 14px;
		border-radius: 10px;
		background: rgba(92, 122, 107, 0.08);
		border: 1px solid rgba(92, 122, 107, 0.25);
	}
	.loading-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		font-size: 12px;
		font-weight: 600;
		color: var(--brand-accent, #5c7a6b);
		margin-bottom: 6px;
	}
	.progress-bar-bg {
		height: 6px;
		border-radius: 999px;
		background: rgba(92, 122, 107, 0.15);
		overflow: hidden;
		margin-bottom: 6px;
	}
	.progress-bar-fill {
		height: 100%;
		background: var(--brand-accent, #5c7a6b);
		transition: width 0.25s ease;
	}
	.loading-text {
		font-size: 11.5px;
		margin: 0 0 4px;
		color: var(--brand-body, #4a5568);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.loading-hint {
		font-size: 11px;
		margin: 0;
		color: var(--brand-muted, #718096);
		line-height: 1.45;
	}

	.copilot-banner-alert {
		margin: 10px 16px;
		padding: 10px 12px;
		border-radius: 8px;
		background: rgba(234, 179, 8, 0.1);
		border: 1px solid rgba(234, 179, 8, 0.3);
		display: flex;
		align-items: flex-start;
		gap: 8px;
	}
	.alert-icon { font-size: 14px; }
	.alert-text {
		margin: 0;
		font-size: 11.5px;
		line-height: 1.45;
		color: #92400e;
	}

	/* Navigation tabs */
	.copilot-tabs {
		display: flex;
		border-bottom: 1px solid var(--brand-line, rgba(0, 0, 0, 0.1));
		padding: 0 16px;
		gap: 6px;
	}
	.tab-btn {
		flex: 1;
		padding: 10px 4px;
		background: transparent;
		border: none;
		border-bottom: 2px solid transparent;
		color: var(--brand-muted, #718096);
		font-size: 12.5px;
		font-weight: 500;
		cursor: pointer;
		transition: all 0.15s ease;
		text-align: center;
	}
	.tab-btn:hover {
		color: var(--brand-ink, #2d3748);
	}
	.tab-btn.active {
		color: var(--brand-accent, #5c7a6b);
		border-bottom-color: var(--brand-accent, #5c7a6b);
		font-weight: 600;
	}

	/* Copilot Body */
	.copilot-body {
		flex: 1;
		overflow-y: auto;
		padding: 16px;
	}
	.tab-panel {
		display: flex;
		flex-direction: column;
		gap: 14px;
	}

	.input-card {
		display: flex;
		flex-direction: column;
		gap: 8px;
	}
	.selection-input {
		width: 100%;
		border-radius: 10px;
		border: 1px solid var(--brand-line, rgba(0, 0, 0, 0.12));
		background: var(--brand-surface, #ffffff);
		padding: 10px 12px;
		font-size: 13px;
		line-height: 1.5;
		color: var(--brand-ink, #2d3748);
		resize: vertical;
		outline: none;
		box-sizing: border-box;
		font-family: inherit;
	}
	.selection-input:focus {
		border-color: var(--brand-accent, #5c7a6b);
	}

	.action-row {
		display: flex;
		align-items: center;
		gap: 8px;
		flex-wrap: wrap;
	}
	.primary-action-btn {
		padding: 8px 16px;
		border-radius: 8px;
		border: none;
		background: var(--brand-accent, #5c7a6b);
		color: #ffffff;
		font-size: 13px;
		font-weight: 600;
		cursor: pointer;
		transition: all 0.15s ease;
	}
	.primary-action-btn:hover:not(:disabled) {
		opacity: 0.9;
		transform: translateY(-1px);
	}
	.primary-action-btn:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	.secondary-action-btn {
		padding: 7px 12px;
		border-radius: 8px;
		border: 1px solid var(--brand-line, rgba(0, 0, 0, 0.15));
		background: var(--brand-surface, #ffffff);
		color: var(--brand-body, #4a5568);
		font-size: 12.5px;
		font-weight: 500;
		cursor: pointer;
		transition: all 0.15s ease;
	}
	.secondary-action-btn:hover {
		border-color: var(--brand-accent, #5c7a6b);
		color: var(--brand-accent, #5c7a6b);
	}

	.stop-btn {
		padding: 7px 12px;
		border-radius: 8px;
		border: 1px solid var(--brand-line, rgba(0, 0, 0, 0.15));
		background: transparent;
		color: #ef4444;
		font-size: 12px;
		cursor: pointer;
	}

	.result-card {
		padding: 14px;
		border-radius: 12px;
		background: var(--brand-surface, rgba(0, 0, 0, 0.02));
		border: 1px solid var(--brand-line, rgba(0, 0, 0, 0.08));
	}
	.result-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		margin-bottom: 8px;
	}
	.result-tag {
		font-size: 11px;
		font-weight: 600;
		text-transform: uppercase;
		color: var(--brand-accent, #5c7a6b);
		letter-spacing: 0.04em;
	}
	.copy-action-btn {
		background: transparent;
		border: 1px solid var(--brand-line, rgba(0, 0, 0, 0.1));
		border-radius: 6px;
		padding: 3px 8px;
		font-size: 11.5px;
		color: var(--brand-muted, #718096);
		cursor: pointer;
		transition: all 0.12s ease;
	}
	.copy-action-btn:hover {
		border-color: var(--brand-accent, #5c7a6b);
		color: var(--brand-accent, #5c7a6b);
	}

	/* Streaming cursor */
	.streaming-cursor {
		display: inline-block;
		color: var(--brand-accent, #5c7a6b);
		font-weight: 700;
		animation: blink 0.9s infinite;
		margin-left: 2px;
		vertical-align: baseline;
	}
	@keyframes blink {
		0%, 100% { opacity: 1; }
		50% { opacity: 0; }
	}

	/* Follow-up suggestions block */
	.followup-suggestions-block {
		margin-top: 14px;
		padding-top: 12px;
		border-top: 1px dashed var(--brand-line, rgba(0, 0, 0, 0.1));
	}
	.suggestions-title {
		margin: 0 0 8px;
		font-size: 11.5px;
		font-weight: 600;
		color: var(--brand-muted, #718096);
	}
	.suggestions-chips {
		display: flex;
		flex-direction: column;
		gap: 6px;
	}
	.suggestion-chip {
		text-align: left;
		padding: 6px 10px;
		border-radius: 8px;
		border: 1px solid var(--brand-line, rgba(0, 0, 0, 0.12));
		background: var(--brand-surface, #ffffff);
		color: var(--brand-ink, #2d3748);
		font-size: 12px;
		cursor: pointer;
		transition: all 0.15s ease;
		line-height: 1.4;
	}
	.suggestion-chip:hover {
		border-color: var(--brand-accent, #5c7a6b);
		color: var(--brand-accent, #5c7a6b);
		background: rgba(92, 122, 107, 0.05);
		transform: translateX(2px);
	}

	/* Markdown Rendered Area */
	:global(.markdown-rendered) {
		font-size: 13.5px;
		line-height: 1.75;
		color: var(--brand-body, #4a5568);
	}
	:global(.markdown-rendered p) {
		margin: 0.55em 0;
	}
	:global(.markdown-rendered p:first-child) {
		margin-top: 0;
	}
	:global(.markdown-rendered p:last-child) {
		margin-bottom: 0;
	}
	:global(.markdown-rendered ul),
	:global(.markdown-rendered ol) {
		margin: 0.5em 0;
		padding-left: 1.3em;
	}
	:global(.markdown-rendered li) {
		margin: 0.25em 0;
	}
	:global(.markdown-rendered h1),
	:global(.markdown-rendered h2),
	:global(.markdown-rendered h3),
	:global(.markdown-rendered h4) {
		margin: 0.8em 0 0.4em;
		font-size: 14px;
		font-weight: 700;
		color: var(--brand-ink, #2d3748);
	}
	:global(.markdown-rendered blockquote) {
		margin: 0.6em 0;
		padding: 4px 12px;
		border-left: 3px solid var(--brand-accent, #5c7a6b);
		background: rgba(92, 122, 107, 0.06);
		color: var(--brand-muted, #718096);
		border-radius: 0 6px 6px 0;
	}
	:global(.markdown-rendered code) {
		background: rgba(92, 122, 107, 0.12);
		color: var(--brand-accent, #5c7a6b);
		padding: 2px 5px;
		border-radius: 4px;
		font-size: 12px;
		font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
	}
	:global(.markdown-rendered pre) {
		background: var(--brand-surface, #f4f4f2);
		border: 1px solid var(--brand-line, rgba(0, 0, 0, 0.1));
		padding: 10px 12px;
		border-radius: 8px;
		overflow-x: auto;
		margin: 0.6em 0;
	}
	:global(.markdown-rendered pre code) {
		background: transparent;
		padding: 0;
		color: var(--brand-ink, #2d3748);
	}
	:global(.markdown-rendered .katex) {
		font-size: 1.05em;
	}
	:global(.markdown-rendered .katex-display) {
		margin: 0.6em 0;
		overflow-x: auto;
		overflow-y: hidden;
		padding: 4px 0;
	}

	/* Chat panel */
	.chat-panel {
		height: 100%;
		min-height: 480px;
		display: flex;
		flex-direction: column;
	}
	/* Scope switch toolbar */
	.chat-scope-bar {
		display: flex;
		gap: 6px;
		margin-bottom: 10px;
		padding: 3px;
		background: var(--brand-surface, #f0f0ee);
		border-radius: 999px;
		border: 1px solid var(--brand-line, rgba(0, 0, 0, 0.08));
	}
	.scope-btn {
		flex: 1;
		display: inline-flex;
		align-items: center;
		justify-content: center;
		gap: 6px;
		padding: 5px 12px;
		border-radius: 999px;
		border: none;
		background: transparent;
		color: var(--brand-muted, #718096);
		font-size: 11.5px;
		font-weight: 600;
		cursor: pointer;
		transition: all 0.15s ease;
	}
	.scope-btn:hover {
		color: var(--brand-ink, #2d3748);
	}
	.scope-btn.is-active {
		background: var(--brand-paper, #ffffff);
		color: var(--brand-accent, #5c7a6b);
		box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08);
	}

	/* RAG Sources Panel inside assistant bubble */
	.rag-sources-panel {
		margin-bottom: 8px;
		padding-bottom: 8px;
		border-bottom: 1px dashed rgba(92, 122, 107, 0.25);
	}
	.rag-sources-header {
		display: flex;
		align-items: center;
		gap: 5px;
		font-size: 11px;
		font-weight: 700;
		color: var(--brand-accent, #5c7a6b);
		margin-bottom: 6px;
	}
	.rag-sources-chips {
		display: flex;
		flex-direction: column;
		gap: 4px;
	}
	.rag-source-chip {
		display: flex;
		align-items: center;
		gap: 6px;
		padding: 4px 8px;
		border-radius: 6px;
		background: var(--brand-paper, #ffffff);
		border: 1px solid var(--brand-line, rgba(0, 0, 0, 0.08));
		color: var(--brand-ink, #2d3748);
		font-size: 11.5px;
		text-decoration: none;
		transition: all 0.15s ease;
	}
	.rag-source-chip:hover {
		border-color: var(--brand-accent, #5c7a6b);
		background: rgba(92, 122, 107, 0.08);
		color: var(--brand-accent, #5c7a6b);
		transform: translateX(2px);
	}
	.rag-source-idx {
		font-weight: 700;
		color: var(--brand-accent, #5c7a6b);
		flex-shrink: 0;
	}
	.rag-source-name {
		flex: 1;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.rag-source-arrow {
		opacity: 0.6;
		font-size: 10px;
		flex-shrink: 0;
	}

	/* Retrieval in-flight indicator */
	.retrieval-indicator-row {
		display: flex;
		justify-content: flex-start;
	}
	.retrieval-indicator {
		display: inline-flex;
		align-items: center;
		gap: 8px;
		padding: 6px 12px;
		border-radius: 999px;
		background: var(--brand-surface, #f0f0ee);
		border: 1px solid var(--brand-line, rgba(0, 0, 0, 0.1));
		color: var(--brand-muted, #718096);
		font-size: 11.5px;
		animation: pulseSearch 1.5s ease-in-out infinite;
	}
	.retrieval-spinner {
		width: 10px;
		height: 10px;
		border: 2px solid var(--brand-accent, #5c7a6b);
		border-top-color: transparent;
		border-radius: 50%;
		animation: spin 0.8s linear infinite;
	}
	@keyframes spin {
		to { transform: rotate(360deg); }
	}
	@keyframes pulseSearch {
		0%, 100% { opacity: 0.8; }
		50% { opacity: 1; }
	}

	.chat-history {
		flex: 1;
		overflow-y: auto;
		display: flex;
		flex-direction: column;
		gap: 12px;
		padding-bottom: 12px;
	}
	.chat-empty {
		text-align: center;
		padding: 30px 16px;
		color: var(--brand-muted, #718096);
		font-size: 13px;
	}
	.empty-icon {
		font-size: 32px;
		display: block;
		margin-bottom: 8px;
		opacity: 0.6;
	}
	.chat-starters {
		display: flex;
		flex-direction: column;
		gap: 7px;
		margin-top: 14px;
		text-align: left;
	}
	.chat-starter-chip {
		background: var(--brand-surface, #ffffff);
	}
	.chat-bubble-row {
		display: flex;
		justify-content: flex-start;
	}
	.chat-bubble-row.is-user {
		justify-content: flex-end;
	}
	.chat-bubble {
		max-width: 88%;
		padding: 10px 14px;
		border-radius: 14px;
		font-size: 13px;
		line-height: 1.55;
		background: var(--brand-surface, #f0f0ee);
		color: var(--brand-ink, #2d3748);
	}
	.bubble-toolbar {
		display: flex;
		align-items: center;
		justify-content: space-between;
		margin-bottom: 6px;
		border-bottom: 1px solid rgba(0, 0, 0, 0.06);
		padding-bottom: 4px;
	}
	.assistant-tag {
		font-size: 11px;
		font-weight: 600;
		color: var(--brand-accent, #5c7a6b);
	}
	.copy-mini-btn {
		background: transparent;
		border: none;
		cursor: pointer;
		font-size: 11px;
		opacity: 0.6;
		padding: 2px 4px;
		border-radius: 4px;
		transition: opacity 0.15s ease;
	}
	.copy-mini-btn:hover {
		opacity: 1;
	}
	.user-text {
		white-space: pre-wrap;
	}
	.chat-bubble-row.is-user .chat-bubble {
		background: var(--brand-accent, #5c7a6b);
		color: #ffffff;
		border-bottom-right-radius: 4px;
	}
	.chat-bubble-row.is-user :global(.markdown-rendered) {
		color: #ffffff;
	}

	.chat-input-bar {
		display: flex;
		gap: 8px;
		padding-top: 10px;
		border-top: 1px solid var(--brand-line, rgba(0, 0, 0, 0.08));
	}
	.chat-input {
		flex: 1;
		border-radius: 8px;
		border: 1px solid var(--brand-line, rgba(0, 0, 0, 0.12));
		background: var(--brand-surface, #ffffff);
		padding: 8px 12px;
		font-size: 13px;
		color: var(--brand-ink, #2d3748);
		outline: none;
	}
	.chat-input:focus {
		border-color: var(--brand-accent, #5c7a6b);
	}
	.chat-send-btn {
		flex-shrink: 0;
	}
	.sr-only {
		position: absolute;
		width: 1px;
		height: 1px;
		padding: 0;
		margin: -1px;
		overflow: hidden;
		clip: rect(0, 0, 0, 0);
		white-space: nowrap;
		border-width: 0;
	}
</style>
