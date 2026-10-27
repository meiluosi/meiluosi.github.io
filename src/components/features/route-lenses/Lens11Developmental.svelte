<script lang="ts">
/**
 * Lens 11: 发生认识论与发展智能路线 (Piagetian Equilibration & Curiosity Progress)
 * 揭示：智能并非预装成品，而是主客体在自组织同化(Assimilation)与顺应(Accommodation)中动态涌现；内在好奇心驱动智能体寻找‘学习进展最大化’的适度挑战区。
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

// Task difficulty: 1 (Too Simple / White Noise), 2 (Goldilocks Zone / Maximum Progress), 3 (Pure Chaos / Incomprehensible)
let taskComplexity = $state(2);
let currentSchema = $state<"four-legged" | "bird" | "refined">("four-legged");
let eventLog = $state<string[]>(["初始认知基式：四条腿、有毛皮的都是‘小狗’。"]);

// Intrinsic Motivation derivative curve (Oudeyer's Learning Progress)
const stats = $derived.by(() => {
	if (taskComplexity === 1) {
		return {
			labelZh: "极度简易 (可完全预测)",
			labelEn: "Boring (Already Mastered)",
			learningRate: "0.02 (停滞)",
			curiosityDrive: "极低 (无新信息)",
			stateType: "boring",
		};
	}
	if (taskComplexity === 2) {
		return {
			labelZh: "适度新奇 (最近发展区 / Goldilocks Zone)",
			labelEn: "Optimal Challenge (Goldilocks Zone)",
			learningRate: "+0.85 (进展最大化)",
			curiosityDrive: "极高 (好奇心爆发)",
			stateType: "optimal",
		};
	}
	return {
		labelZh: "不可预测混沌 (随机白噪声)",
		labelEn: "Unlearnable Chaos",
		learningRate: "0.00 (无法拟合)",
		curiosityDrive: "受挫放弃",
		stateType: "chaos",
	};
});

function triggerPiagetianEncounter(animal: "cat" | "flying-squirrel") {
	if (animal === "cat") {
		// Assimilation: fits schema (four legs, fur)
		eventLog = [
			"[同化 Assimilation] 遭遇猫咪：四腿毛茸茸 -> 自动纳入现有基式‘小狗’。认知平衡维持。",
			...eventLog.slice(0, 4),
		];
	} else {
		// Accommodation: conflict! A flying creature with fur
		currentSchema = "refined";
		eventLog = [
			"[顺应 Accommodation 突变] 遭遇飞鼠：会飞但非鸟类！旧基式破裂 -> 认知失衡 (Disequilibrium) -> 触发基式重构，分化出新概念范畴！",
			...eventLog.slice(0, 4),
		];
	}
}

function resetSchema() {
	currentSchema = "four-legged";
	eventLog = ["重置认知基式为初始状态。"];
}
</script>

<div class="lens-container">
	<div class="lens-header">
		<span class="lens-tag">COGNITIVE LENS 11</span>
		<h3 class="lens-title">{english ? "Piagetian Equilibration & Curiosity Drive" : "皮亚杰平衡化与内在好奇心沙盒"}</h3>
		<p class="lens-subtitle">
			{english
				? "Jean Piaget & Pierre-Yves Oudeyer: Intelligence develops through self-organizing assimilation, disequilibrium, and accommodation. Intrinsic curiosity gravitates neither to the trivial nor the chaotic, but to where learning progress is maximized."
				: "皮亚杰发生认识论与欧迪耶 (Oudeyer) 内在动机假说：智能不是静态数据拟合，而是通过同化与顺应的动态平衡化推进；好奇心驱使系统自主绕开已知的平庸与不可解的混沌，投向学习进展增量最高的适度挑战区。"}
		</p>
	</div>

	<!-- Complexity & Zone Selector -->
	<div class="controls-bar">
		<div class="zone-picker">
			<span class="ctrl-label">{english ? "Environment Predictability:" : "所处探索任务的可预测性与复杂度："}</span>
			<div class="btn-group">
				<button
					type="button"
					class="zone-btn"
					class:active={taskComplexity === 1}
					onclick={() => taskComplexity = 1}
				>
					1. {english ? "Trivial Routine (Predictable)" : "平庸例行 (已掌握/无进展)"}
				</button>
				<button
					type="button"
					class="zone-btn goldilocks"
					class:active={taskComplexity === 2}
					onclick={() => taskComplexity = 2}
				>
					★ 2. {english ? "Goldilocks Zone (Max Progress)" : "黄金探索带 (学习进展最大化)"}
				</button>
				<button
					type="button"
					class="zone-btn"
					class:active={taskComplexity === 3}
					onclick={() => taskComplexity = 3}
				>
					3. {english ? "Unstructured Noise (Incomprehensible)" : "不可解纯噪声 (认知超载)"}
				</button>
			</div>
		</div>
	</div>

	<!-- Interactive Equilibration Sandbox -->
	<div class="sandbox-stage">
		<div class="schema-box">
			<div class="schema-header">
				<span class="box-tag">{english ? "Current Mental Schema" : "主体当前内部心智基式 (Schema)"}</span>
				<button type="button" class="mini-reset" onclick={resetSchema}>重置心智</button>
			</div>
			<div class="schema-body">
				{#if currentSchema === "four-legged"}
					<div class="schema-node primitive">
						<strong>低阶基式：</strong>四足毛皮 = 小狗
					</div>
				{:else}
					<div class="schema-node advanced">
						<strong>解耦重构后基式：</strong>[哺乳类: 陆生/滑翔] + [形态分化特征] (顺应完成)
					</div>
				{/if}
			</div>

			<div class="encounter-actions">
				<span class="action-label">{english ? "Introduce Novel Stimulus:" : "注入外部反例刺激："}</span>
				<div class="stimuli-btns">
					<button type="button" class="action-pill" onclick={() => triggerPiagetianEncounter("cat")}>
						🐱 猫咪 (可同化)
					</button>
					<button type="button" class="action-pill alert" onclick={() => triggerPiagetianEncounter("flying-squirrel")}>
						🐿️ 飞鼠 (诱发失衡顺应)
					</button>
				</div>
			</div>
		</div>

		<!-- Event Stream -->
		<div class="log-stream">
			<span class="log-title">{english ? "Developmental Epistemology Log:" : "认识论演化事件流："}</span>
			<div class="log-entries">
				{#each eventLog as entry}
					<div class="log-item">{entry}</div>
				{/each}
			</div>
		</div>
	</div>

	<!-- Learning Progress Gauge -->
	<div class="metrics-row">
		<div class="metric-card">
			<span class="m-label">{english ? "Learning Progress Derivative (dE/dt)" : "学习误差改善率 (dE/dt)"}</span>
			<span class="m-val" class:high={taskComplexity === 2}>{stats.learningRate}</span>
		</div>
		<div class="metric-card">
			<span class="m-label">{english ? "Intrinsic Curiosity Drive" : "内在好奇心激活强度"}</span>
			<span class="m-val" class:high={taskComplexity === 2}>{stats.curiosityDrive}</span>
		</div>
	</div>

	<!-- Insight Box -->
	<div class="insight-box">
		<div class="insight-icon">🌱</div>
		<div class="insight-text">
			<strong>{english ? "Autonomous Self-Development:" : "发展智能的根本真谛："}</strong>
			{english
				? "Real intelligence cannot be dumped into a model at timestamp zero. It requires ontogenetic development—an autonomous exploratory trajectory driven by intrinsic rewards that systematically expands its own cognitive frontier."
				: "真正的通用智能无法像静态数据库一样在第 0 秒整体灌注。它必须经历发生学的自主发展轨迹：在内在探索动机的驱动下，从感觉运动阶段向抽象形式运思自主跃迁。"}
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
:global(.dark) .lens-container {
	background: #141816;
	border-color: rgba(120, 150, 130, 0.2);
	color: #e2e8f0;
}
.lens-tag {
	font-size: 0.75rem;
	font-weight: 700;
	letter-spacing: 0.1em;
	color: var(--brand-clay, #8a7356);
	text-transform: uppercase;
}
.lens-title {
	font-size: 1.35rem;
	margin: 0.25rem 0 0.5rem;
	font-weight: 700;
}
.lens-subtitle {
	font-size: 0.88rem;
	line-height: 1.5;
	color: var(--text-muted, #718096);
	margin-bottom: 1.25rem;
}
.controls-bar {
	margin-bottom: 1.25rem;
	padding: 0.85rem;
	background: rgba(0, 0, 0, 0.03);
	border-radius: 8px;
}
:global(.dark) .controls-bar {
	background: rgba(255, 255, 255, 0.03);
}
.ctrl-label {
	font-size: 0.8rem;
	font-weight: 700;
	display: block;
	margin-bottom: 0.5rem;
}
.btn-group {
	display: flex;
	gap: 0.5rem;
	flex-wrap: wrap;
}
.zone-btn {
	background: rgba(0, 0, 0, 0.04);
	border: 1px solid rgba(0, 0, 0, 0.08);
	color: inherit;
	padding: 0.4rem 0.8rem;
	border-radius: 6px;
	font-size: 0.8rem;
	cursor: pointer;
	transition: all 0.2s;
}
:global(.dark) .zone-btn {
	background: rgba(255, 255, 255, 0.04);
	border-color: rgba(255, 255, 255, 0.08);
}
.zone-btn.active {
	background: var(--brand-ink, #2d3748);
	color: #fff;
}
:global(.dark) .zone-btn.active {
	background: #3b5043;
}
.zone-btn.goldilocks.active {
	background: #0284c7;
	border-color: #0284c7;
}
.sandbox-stage {
	display: grid;
	grid-template-columns: 1fr 1.2fr;
	gap: 1.25rem;
	margin-bottom: 1.25rem;
}
@media (max-width: 640px) {
	.sandbox-stage {
		grid-template-columns: 1fr;
	}
}
.schema-box, .log-stream {
	background: rgba(255, 255, 255, 0.6);
	border: 1px solid rgba(0, 0, 0, 0.06);
	border-radius: 8px;
	padding: 1rem;
	display: flex;
	flex-direction: column;
}
:global(.dark) .schema-box, :global(.dark) .log-stream {
	background: rgba(0, 0, 0, 0.25);
	border-color: rgba(255, 255, 255, 0.06);
}
.schema-header {
	display: flex;
	justify-content: space-between;
	align-items: center;
	margin-bottom: 0.75rem;
}
.box-tag {
	font-size: 0.72rem;
	font-weight: 700;
	color: var(--text-muted, #718096);
	text-transform: uppercase;
}
.mini-reset {
	background: transparent;
	border: none;
	color: var(--brand-clay, #8a7356);
	font-size: 0.74rem;
	cursor: pointer;
	text-decoration: underline;
}
.schema-body {
	min-height: 60px;
	display: flex;
	align-items: center;
	margin-bottom: 1rem;
}
.schema-node {
	padding: 0.6rem 0.85rem;
	border-radius: 6px;
	font-size: 0.82rem;
	width: 100%;
}
.schema-node.primitive {
	background: rgba(0, 0, 0, 0.05);
	border: 1px solid rgba(0, 0, 0, 0.1);
}
:global(.dark) .schema-node.primitive {
	background: rgba(255, 255, 255, 0.05);
	border-color: rgba(255, 255, 255, 0.1);
}
.schema-node.advanced {
	background: rgba(16, 185, 129, 0.1);
	border: 1px solid rgba(16, 185, 129, 0.3);
	color: #065f46;
}
:global(.dark) .schema-node.advanced {
	color: #a7f3d0;
}
.action-label {
	font-size: 0.74rem;
	color: var(--text-muted, #718096);
	display: block;
	margin-bottom: 0.35rem;
}
.stimuli-btns {
	display: flex;
	gap: 0.5rem;
}
.action-pill {
	flex: 1;
	background: rgba(0, 0, 0, 0.04);
	border: 1px solid rgba(0, 0, 0, 0.1);
	color: inherit;
	padding: 0.4rem;
	border-radius: 6px;
	font-size: 0.76rem;
	cursor: pointer;
}
:global(.dark) .action-pill {
	background: rgba(255, 255, 255, 0.04);
	border-color: rgba(255, 255, 255, 0.1);
}
.action-pill.alert {
	border-color: #f59e0b;
	color: #d97706;
}
.log-title {
	font-size: 0.72rem;
	font-weight: 700;
	color: var(--text-muted, #718096);
	text-transform: uppercase;
	margin-bottom: 0.5rem;
}
.log-entries {
	display: flex;
	flex-direction: column;
	gap: 0.4rem;
	max-height: 140px;
	overflow-y: auto;
}
.log-item {
	font-size: 0.74rem;
	line-height: 1.4;
	padding: 0.35rem 0.5rem;
	background: rgba(0, 0, 0, 0.03);
	border-radius: 4px;
}
:global(.dark) .log-item {
	background: rgba(255, 255, 255, 0.04);
}
.metrics-row {
	display: grid;
	grid-template-columns: 1fr 1fr;
	gap: 1rem;
	margin-bottom: 1.25rem;
}
.metric-card {
	padding: 0.85rem;
	background: rgba(0, 0, 0, 0.02);
	border-radius: 6px;
	border: 1px solid rgba(0, 0, 0, 0.05);
	text-align: center;
}
:global(.dark) .metric-card {
	background: rgba(255, 255, 255, 0.02);
	border-color: rgba(255, 255, 255, 0.05);
}
.m-label {
	font-size: 0.72rem;
	color: var(--text-muted, #718096);
	display: block;
	margin-bottom: 0.25rem;
}
.m-val {
	font-size: 1.2rem;
	font-weight: 700;
	color: var(--text-muted, #718096);
}
.m-val.high {
	color: #0284c7;
}
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
.insight-icon {
	font-size: 1.1rem;
}
</style>
