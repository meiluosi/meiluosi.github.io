<script lang="ts">
/**
 * Lens 17: 心灵哲学、功能主义与意向性路线 (Searle's Chinese Room & Symbol Grounding)
 * 揭示：塞尔‘中文房间’思想实验：纯粹按照形式语法规则搬运符号(Syntax)，无论对外多么对答如流，都不自动产生指向现实世界的真实语义理解(Semantics)与意向性(Intentionality)。
 */
let english = $state(false);

$effect(() => {
	const syncLang = () => {
		english = document.documentElement.lang.toLowerCase().startsWith("en");
	};
	syncLang();
	const obs = new MutationObserver(syncLang);
	obs.observe(document.documentElement, {
		attributes: true,
		attributeFilter: ["lang"],
	});
	return () => obs.disconnect();
});

const SYMBOLS = [
	{
		inputGlyph: "【火】",
		ruleLookup: "IF 见到笔画 [丶丿丿丶] → 输出符号串 [水、逃跑]",
		syntacticView: "纯几何折线编号 #4092 → 触发查表映射 #8810",
		semanticView: "灼热、发光氧化反应、生命危险感与滚烫触觉先验",
	},
	{
		inputGlyph: "【痛】",
		ruleLookup: "IF 见到字形 [疒+甬] → 输出符号串 [需要止痛药]",
		syntacticView: "离散 Token 向量内积最大化 → 采样下一个高概率 Token",
		semanticView: "C-纤维神经刺痛、具身逃避冲动与第一人称痛苦感受质",
	},
];

let selectedIdx = $state(0);
let inspectMode = $state<"inside" | "outside">("outside");

const activeItem = $derived(SYMBOLS[selectedIdx]);
</script>

<div class="lens-container">
	<div class="lens-header">
		<span class="lens-tag">COGNITIVE LENS 17</span>
		<h3 class="lens-title">{english ? "Searle's Chinese Room & Symbol Grounding" : "塞尔中文房间与符号接地问题沙盒"}</h3>
		<p class="lens-subtitle">
			{english
				? "John Searle, Franz Brentano & Stevan Harnad: Syntax is not semantics. A clerk inside a closed room can pass the Turing Test in Chinese using a giant rulebook without understanding a single word."
				: "约翰·塞尔 (John Searle)、布伦塔诺与哈纳德：语法不等于语义。房间里不懂中文的操作员凭一本详尽的‘字形变换手册’即可对答如流通过图灵测试，但他对符号所指代的真实世界毫无意向性理解。"}
		</p>
	</div>

	<div class="controls-bar">
		<div class="sym-picker">
			<span>传入纸条符号：</span>
			{#each SYMBOLS as s, idx}
				<button
					type="button"
					class="sym-btn"
					class:active={selectedIdx === idx}
					onclick={() => selectedIdx = idx}
				>
					{s.inputGlyph}
				</button>
			{/each}
		</div>

		<div class="view-picker">
			<button
				type="button"
				class="mode-btn"
				class:active={inspectMode === "outside"}
				onclick={() => inspectMode = "outside"}
			>
				👀 房间外部观察者 (功能主义视角)
			</button>
			<button
				type="button"
				class="mode-btn alert"
				class:active={inspectMode === "inside"}
				onclick={() => inspectMode = "inside"}
			>
				🚪 掀开房间屋顶 (意向性/接地视角)
			</button>
		</div>
	</div>

	<div class="room-stage">
		<div class="room-card">
			<span class="card-tag">1. 形式语法层 (Syntactic Manipulation)</span>
			<div class="code-line">{activeItem.ruleLookup}</div>
			<p class="desc">{activeItem.syntacticView}</p>
		</div>

		<div class="gap-divider">
			{#if inspectMode === "outside"}
				<span class="gap-pill ok">✓ 外部表现完美：被判定为“具备流利理解”</span>
			{:else}
				<span class="gap-pill broken">⚡ 符号接地鸿沟 (Symbol Grounding Gap)：字形与物理世界脱节</span>
			{/if}
		</div>

		<div class="room-card semantic" class:dimmed={inspectMode === "inside"}>
			<span class="card-tag">2. 真实语义与意向性接地层 (Semantic Grounding)</span>
			<p class="desc">{activeItem.semanticView}</p>
			<div class="status-note">
				{inspectMode === "inside"
					? "❌ 房间内部状态：仅存在无意义的符号循环查表（词典用词解释词），缺乏感知运动接地。"
					: "外部视角假定：既然回答正确，必然拥有内部语义。"}
			</div>
		</div>
	</div>

	<div class="insight-box">
		<div class="insight-icon">🏮</div>
		<div class="insight-text">
			<strong>{english ? "The Merry-Go-Round of Dictionary Definitions:" : "哈纳德的‘中文字典旋转木马’悖论："}</strong>
			{english
				? "If you try to learn Chinese using only a Chinese-to-Chinese dictionary, you loop endlessly from one ungrounded symbol to another. Meaning requires anchoring symbols into non-symbolic sensorimotor causal interactions with reality."
				: "如果你试图仅凭一本《新华字典》去学一门完全陌生的外语，你只会陷入‘用未知符号解释未知符号’的死循环。真正的‘理解’必须走出纯文本房间，让符号在物理世界的因果交互与具身感知中完成接地 (Grounding)。"}
		</div>
	</div>
</div>

<style>
.lens-container {
	border: 1px solid var(--border-color, rgba(120, 140, 130, 0.25));
	background: var(--brand-paper, #faf8f5);
	color: var(--brand-ink, #2d3748);
	border-radius: 12px;
	padding: 1.5rem;
	margin: 2rem 0;
	box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
}
:global(.dark) .lens-container { background: #141816; border-color: rgba(120, 150, 130, 0.2); color: #e2e8f0; }
.lens-tag { font-size: 0.75rem; font-weight: 700; letter-spacing: 0.1em; color: var(--brand-clay, #8a7356); text-transform: uppercase; }
.lens-title { font-size: 1.35rem; margin: 0.25rem 0 0.5rem; font-weight: 700; }
.lens-subtitle { font-size: 0.88rem; line-height: 1.5; color: var(--text-muted, #718096); margin-bottom: 1.25rem; }
.controls-bar {
	display: flex;
	justify-content: space-between;
	align-items: center;
	flex-wrap: wrap;
	gap: 0.75rem;
	padding: 0.85rem;
	background: rgba(0, 0, 0, 0.03);
	border-radius: 8px;
	margin-bottom: 1.25rem;
}
:global(.dark) .controls-bar { background: rgba(255, 255, 255, 0.03); }
.sym-picker, .view-picker { display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; font-size: 0.82rem; }
.sym-btn, .mode-btn {
	background: rgba(0, 0, 0, 0.05);
	border: 1px solid rgba(0, 0, 0, 0.1);
	color: inherit;
	padding: 0.35rem 0.75rem;
	border-radius: 6px;
	font-size: 0.8rem;
	cursor: pointer;
}
.sym-btn.active, .mode-btn.active { background: var(--brand-ink, #2d3748); color: #fff; }
:global(.dark) .sym-btn.active, :global(.dark) .mode-btn.active { background: #3b5043; }
.mode-btn.alert.active { background: #ea580c; border-color: #ea580c; }
.room-stage { display: flex; flex-direction: column; gap: 0.75rem; margin-bottom: 1.25rem; }
.room-card {
	padding: 1rem;
	border-radius: 8px;
	background: rgba(255, 255, 255, 0.65);
	border: 1px solid rgba(0, 0, 0, 0.08);
}
:global(.dark) .room-card { background: rgba(0, 0, 0, 0.25); border-color: rgba(255, 255, 255, 0.08); }
.room-card.dimmed { border-style: dashed; border-color: #ef4444; background: rgba(239, 68, 68, 0.05); }
.card-tag { font-size: 0.72rem; font-weight: 700; text-transform: uppercase; color: var(--brand-clay, #8a7356); display: block; margin-bottom: 0.35rem; }
.code-line { font-family: monospace; font-size: 0.82rem; padding: 0.35rem 0.6rem; background: rgba(0, 0, 0, 0.05); border-radius: 4px; margin-bottom: 0.35rem; }
:global(.dark) .code-line { background: rgba(255, 255, 255, 0.06); }
.desc { font-size: 0.84rem; margin: 0; }
.status-note { font-size: 0.76rem; color: #ef4444; margin-top: 0.4rem; font-weight: 600; }
.gap-divider { text-align: center; }
.gap-pill { display: inline-block; font-size: 0.75rem; font-weight: 700; padding: 0.25rem 0.75rem; border-radius: 99px; }
.gap-pill.ok { background: rgba(16, 185, 129, 0.15); color: #10b981; }
.gap-pill.broken { background: rgba(239, 68, 68, 0.15); color: #ef4444; }
.insight-box {
	display: flex;
	gap: 0.75rem;
	padding: 0.85rem 1rem;
	background: rgba(138, 115, 86, 0.08);
	border-left: 3px solid var(--brand-clay, #8a7356);
	border-radius: 4px;
	font-size: 0.84rem;
	line-height: 1.5;
}
.insight-icon { font-size: 1.1rem; }
</style>
