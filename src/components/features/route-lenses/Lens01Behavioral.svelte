<script lang="ts">
/**
 * Lens 01: 行为表现与操作性路线透镜 (Turing Operational Box)
 * 揭示：相同的外部可观察行为，内部可能源于完全不同的机制。
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

const prompts = [
	{
		qZh: "你如何看待未知的危险？",
		qEn: "How do you respond to unseen danger?",
		respA_Zh:
			"根据关键词‘危险’检索规则库[ID:904]：应保持静止，降低能耗并向控制中心发送预警信号。",
		respA_En:
			"Rule match [Key: danger, ID: 904]: Freeze movement, reduce power draw, ping base controller.",
		respB_Zh:
			"若无法直接观测，我会依赖过去的相似经验建立假设，并在保证退路的前提下主动试探观察。",
		respB_En:
			"If unobservable, I construct hypotheses based on past priors and probe cautiously with an exit route.",
		trueAgent: "B",
		typeA: "查表查找型 (Look-up Table)",
		typeB: "因果自适应型 (Causal Model)",
	},
	{
		qZh: "如果一个铁球从斜坡滚下，途中被一块石头挡住，会怎样？",
		qEn: "If an iron ball rolls down a slope and hits a stone, what happens?",
		respA_Zh:
			"重力与阻挡力学规则匹配：铁球动能转为冲击力，弹开或停止，符合牛顿运动描述范式。",
		respA_En:
			"Collision physics template matched: Kinetic energy converts to impact, deflects or halts.",
		respB_Zh:
			"石头会被撞偏或让铁球改变轨迹，这取决于石头的质量和坡度——如果石头够重，铁球就此停下。",
		respB_En:
			"Depends on mass: a heavy stone halts it; otherwise it deflects and continues along a altered path.",
		trueAgent: "B",
		typeA: "模板统计匹配 (Pattern Match)",
		typeB: "物理模拟推演 (Mental Simulation)",
	},
];

let selectedPromptIndex = $state(0);
let userGuess = $state<"A" | "B" | null>(null);
let revealed = $state(false);

function selectPrompt(idx: number) {
	selectedPromptIndex = idx;
	userGuess = null;
	revealed = false;
}

function makeGuess(guess: "A" | "B") {
	userGuess = guess;
	revealed = true;
}

function resetTest() {
	userGuess = null;
	revealed = false;
}
</script>

<div class="lens-container">
	<div class="lens-header">
		<span class="lens-tag">COGNITIVE LENS 01</span>
		<h3 class="lens-title">{english ? "The Operational Black Box" : "图灵操作性黑盒测试器"}</h3>
		<p class="lens-subtitle">
			{english
				? "Test whether behavioral indistinguishability guarantees equivalent internal understanding."
				: "测试外部交互的不可区分性，是否等价于系统内部拥有真正的认知理解。"}
		</p>
	</div>

	<!-- 题目选择 -->
	<div class="prompt-selector">
		{#each prompts as p, idx}
			<button
				type="button"
				class="prompt-btn"
				class:active={selectedPromptIndex === idx}
				onclick={() => selectPrompt(idx)}
			>
				{english ? `Test Probe ${idx + 1}` : `测试探针 0${idx + 1}`}
			</button>
		{/each}
	</div>

	<!-- 交互探问框 -->
	<div class="dialogue-box">
		<div class="probe-question">
			<span class="q-label">{english ? "Question / Stimulus:" : "探问刺激："}</span>
			<strong>{english ? prompts[selectedPromptIndex].qEn : prompts[selectedPromptIndex].qZh}</strong>
		</div>

		<div class="agents-grid">
			<!-- Agent A -->
			<div class="agent-card" class:revealed={revealed}>
				<div class="agent-label">
					<span class="agent-badge">Agent A</span>
					{#if revealed}
						<span class="agent-truth truth-a">{prompts[selectedPromptIndex].typeA}</span>
					{/if}
				</div>
				<p class="agent-output">
					{english ? prompts[selectedPromptIndex].respA_En : prompts[selectedPromptIndex].respA_Zh}
				</p>
				{#if !revealed}
					<button type="button" class="guess-btn" onclick={() => makeGuess("A")}>
						{english ? "This has True Understanding" : "判定此 Agent 具备内部理解"}
					</button>
				{/if}
			</div>

			<!-- Agent B -->
			<div class="agent-card" class:revealed={revealed}>
				<div class="agent-label">
					<span class="agent-badge">Agent B</span>
					{#if revealed}
						<span class="agent-truth truth-b">{prompts[selectedPromptIndex].typeB}</span>
					{/if}
				</div>
				<p class="agent-output">
					{english ? prompts[selectedPromptIndex].respB_En : prompts[selectedPromptIndex].respB_Zh}
				</p>
				{#if !revealed}
					<button type="button" class="guess-btn" onclick={() => makeGuess("B")}>
						{english ? "This has True Understanding" : "判定此 Agent 具备内部理解"}
					</button>
				{/if}
			</div>
		</div>

		{#if revealed}
			<div class="insight-reveal">
				<div class="insight-badge">
					{english ? "✦ BEHAVIORAL INSIGHT" : "✦ 行为主义路线的反思"}
				</div>
				<p>
					{english
						? "Both agents produced reasonable behavioral outputs. However, Agent A relied entirely on hardcoded lookup tables, whereas Agent B operated on causal counterfactual simulation. Operationalism tells us 'what it outputs', but remains blind to whether understanding actually happened inside."
						: "两个 Agent 在外部表现上都可以给出恰当的回答。然而 Agent A 只是机械地触发了死板的规则查找表，Agent B 则是基于因果反事实推断。这揭示了行为主义的核心局限：纯操作性表现无法分辨其底层机制究竟是机械拟态还是真实自适应。"}
				</p>
				<button type="button" class="reset-btn" onclick={resetTest}>
					{english ? "Try Another Probe ⟲" : "重置探针 ⟲"}
				</button>
			</div>
		{/if}
	</div>
</div>

<style>
	.lens-container {
		background: rgb(250 251 248 / 95%);
		border: 1px solid var(--brand-line);
		border-radius: 10px;
		padding: 24px 28px;
		margin-bottom: 32px;
	}

	.lens-tag {
		font-family: var(--font-mono, monospace);
		font-size: 11px;
		letter-spacing: 0.14em;
		color: var(--brand-clay);
		font-weight: 600;
	}

	.lens-title {
		margin: 4px 0 6px;
		font-family: var(--font-serif, "Songti SC", Georgia, serif);
		font-size: 22px;
		color: var(--brand-ink);
	}

	.lens-subtitle {
		margin: 0 0 18px;
		font-size: 13.5px;
		color: var(--brand-muted);
		line-height: 1.6;
	}

	.prompt-selector {
		display: flex;
		gap: 8px;
		margin-bottom: 16px;
	}

	.prompt-btn {
		padding: 4px 12px;
		border-radius: 6px;
		border: 1px solid var(--brand-line);
		background: var(--brand-surface);
		font-size: 12px;
		color: var(--brand-body);
		cursor: pointer;
		transition: all 0.18s ease;
	}

	.prompt-btn.active {
		background: var(--brand-clay);
		color: var(--brand-paper);
		border-color: var(--brand-clay);
	}

	.dialogue-box {
		display: flex;
		flex-direction: column;
		gap: 16px;
	}

	.probe-question {
		padding: 12px 16px;
		background: var(--brand-surface);
		border-left: 3px solid var(--brand-accent);
		border-radius: 0 6px 6px 0;
		font-size: 14px;
	}

	.q-label {
		display: block;
		font-size: 11px;
		color: var(--brand-muted);
		margin-bottom: 2px;
	}

	.agents-grid {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 16px;
	}

	.agent-card {
		padding: 16px 18px;
		border: 1px solid var(--brand-line);
		border-radius: 8px;
		background: var(--brand-paper);
		display: flex;
		flex-direction: column;
		gap: 12px;
		transition: all 0.2s ease;
	}

	.agent-card.revealed {
		border-color: var(--brand-clay);
	}

	.agent-label {
		display: flex;
		justify-content: space-between;
		align-items: center;
	}

	.agent-badge {
		font-family: var(--font-mono, monospace);
		font-weight: 700;
		font-size: 12px;
		color: var(--brand-ink);
	}

	.agent-truth {
		font-size: 11px;
		padding: 1px 6px;
		border-radius: 4px;
		background: rgb(145 86 56 / 15%);
		color: var(--brand-clay);
		font-weight: 500;
	}

	.agent-output {
		margin: 0;
		font-size: 13.5px;
		line-height: 1.7;
		color: var(--brand-body);
		font-family: var(--font-serif, serif);
		flex: 1;
	}

	.guess-btn {
		padding: 7px 12px;
		border-radius: 6px;
		border: 1px dashed var(--brand-line);
		background: var(--brand-surface);
		color: var(--brand-accent);
		font-size: 12px;
		cursor: pointer;
		transition: all 0.2s ease;
	}

	.guess-btn:hover {
		background: var(--brand-accent);
		color: #fff;
		border-style: solid;
	}

	.insight-reveal {
		padding: 14px 18px;
		border-radius: 8px;
		background: rgb(243 199 172 / 16%);
		border: 1px solid rgb(145 86 56 / 25%);
	}

	.insight-badge {
		font-size: 11px;
		font-weight: 700;
		letter-spacing: 0.08em;
		color: var(--brand-clay);
		margin-bottom: 6px;
	}

	.insight-reveal p {
		margin: 0 0 10px;
		font-size: 13px;
		line-height: 1.75;
		color: var(--brand-body);
	}

	.reset-btn {
		padding: 4px 10px;
		border-radius: 4px;
		border: 1px solid var(--brand-clay);
		background: transparent;
		color: var(--brand-clay);
		font-size: 11.5px;
		cursor: pointer;
	}

	.reset-btn:hover {
		background: var(--brand-clay);
		color: #fff;
	}

	@media (max-width: 640px) {
		.agents-grid {
			grid-template-columns: 1fr;
		}
	}
</style>
