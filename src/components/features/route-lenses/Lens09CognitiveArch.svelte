<script lang="ts">
/**
 * Lens 09: 认知架构与功能模块整合路线 (ACT-R / SOAR Working Memory Bus)
 * 揭示：智能是多模块协同的整体工程系统。感知缓冲区、工作记忆总线、陈述性记忆检索与程序性规则匹配构成核心认知循环。
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

type CognitivePhase = "perceive" | "match" | "retrieve" | "execute";

let currentPhase = $state<CognitivePhase>("perceive");
let cycleStep = $state(0);
let workingMemoryChunk = $state("Goal: Solve(Equation)");
let firedRule = $state("None");
let isAutoRunning = $state(false);

const PHASES: Record<
	CognitivePhase,
	{ nameZh: string; nameEn: string; descZh: string; descEn: string }
> = {
	perceive: {
		nameZh: "1. 感觉缓冲装载 (Perception)",
		nameEn: "1. Sensory Buffer Encoding",
		descZh: "视觉模块捕捉外部符号流，转化为感知块注入工作记忆区。",
		descEn: "Visual buffer decodes visual stream into symbolic chunk in WM.",
	},
	match: {
		nameZh: "2. 产生式规则模式匹配 (Conflict Resolution)",
		nameEn: "2. Production Rule Pattern Match",
		descZh: "程序性记忆匹配‘IF 工作记忆存在目标 THEN 触发操作’的规则集合。",
		descEn:
			"Procedural system checks IF/THEN condition against current WM chunk.",
	},
	retrieve: {
		nameZh: "3. 陈述性事实长时提取 (Declarative Retrieval)",
		nameEn: "3. Declarative Fact Retrieval",
		descZh: "激活长时记忆库，按基线激活度检索关联知识块（如代数运算法则）。",
		descEn:
			"Spreading activation pulls long-term declarative fact based on association.",
	},
	execute: {
		nameZh: "4. 动作执行与工作记忆更新 (Motor Action & WM Write)",
		nameEn: "4. Action & Working Memory Update",
		descZh: "执行运动输出或重写目标缓冲栈，完成一个 50ms 认知节拍循环。",
		descEn:
			"Motor command is dispatched and WM buffer is rewritten for next 50ms cycle.",
	},
};

function advanceCycle() {
	if (currentPhase === "perceive") {
		currentPhase = "match";
		workingMemoryChunk = "Goal: Isolate(x) & Exp: 2x+4=10";
		firedRule = "Rule#104: IF 2x+c=d THEN move c to RHS";
	} else if (currentPhase === "match") {
		currentPhase = "retrieve";
		firedRule = "Retrieving Fact: [Arithmetic: 10 - 4 = 6]";
	} else if (currentPhase === "retrieve") {
		currentPhase = "execute";
		workingMemoryChunk = "Goal: Solve 2x = 6";
	} else {
		currentPhase = "perceive";
		cycleStep++;
		workingMemoryChunk = "Goal: Division (x = 3) -> Finished";
		firedRule = "Rule#200: Done";
	}
}

function resetCycle() {
	currentPhase = "perceive";
	cycleStep = 0;
	workingMemoryChunk = "Goal: Solve(Equation)";
	firedRule = "None";
}
</script>

<div class="lens-container">
	<div class="lens-header">
		<span class="lens-tag">COGNITIVE LENS 09</span>
		<h3 class="lens-title">{english ? "ACT-R Cognitive Bus & Production Cycle" : "ACT-R 认知总线与产生式循环沙盒"}</h3>
		<p class="lens-subtitle">
			{english
				? "John R. Anderson & Allen Newell: Intelligence is not an isolated black-box algorithm, but an integrated architecture coordinating perception, declarative memory, and procedural production rules."
				: "约翰·安德森 (ACT-R) 与纽厄尔 (Soar)：智能不是孤立的单一算法，而是感知、工作记忆、长时陈述记忆与程序性产生式规则协同运行的系统级集成架构。"}
		</p>
	</div>

	<!-- Architecture Bus Diagram -->
	<div class="arch-stage">
		<div class="modules-grid">
			<!-- Visual / Sensory Buffer -->
			<div class="module-card sensory" class:active={currentPhase === "perceive"}>
				<div class="mod-icon">👁️</div>
				<div class="mod-name">{english ? "Sensory Buffer" : "感觉感知缓冲器"}</div>
				<div class="mod-val">Input: "2x + 4 = 10"</div>
			</div>

			<!-- Working Memory / Central Bottleneck -->
			<div class="module-card wm" class:active={currentPhase === "perceive" || currentPhase === "execute"}>
				<div class="mod-icon">⚡</div>
				<div class="mod-name">{english ? "Working Memory Bus (Central Bottleneck)" : "工作记忆中央总线 (瓶颈容量)"}</div>
				<div class="mod-chunk">{workingMemoryChunk}</div>
			</div>

			<!-- Motor / Output -->
			<div class="module-card motor" class:active={currentPhase === "execute"}>
				<div class="mod-icon">🦾</div>
				<div class="mod-name">{english ? "Motor Execution" : "运动/输出执行器"}</div>
				<div class="mod-val">Output: x = 3</div>
			</div>
		</div>

		<div class="bus-divider">
			<span class="bus-line"></span>
			<span class="bus-tag">{english ? "50ms Production Matching Cycle" : "50ms 认知产生式协调循环"}</span>
			<span class="bus-line"></span>
		</div>

		<div class="modules-grid lower">
			<!-- Declarative Memory -->
			<div class="module-card declarative" class:active={currentPhase === "retrieve"}>
				<div class="mod-icon">📚</div>
				<div class="mod-name">{english ? "Declarative Memory (Chunks)" : "陈述性记忆库 (Chunks)"}</div>
				<div class="mod-desc">长时事实：加减乘除先验知识</div>
			</div>

			<!-- Procedural Production Rules -->
			<div class="module-card procedural" class:active={currentPhase === "match"}>
				<div class="mod-icon">⚙️</div>
				<div class="mod-name">{english ? "Procedural Memory (IF-THEN)" : "程序性规则系统 (IF-THEN)"}</div>
				<div class="mod-rule">{firedRule}</div>
			</div>
		</div>
	</div>

	<!-- Step Controls & Explanation -->
	<div class="action-footer">
		<div class="phase-info">
			<span class="phase-badge">{english ? PHASES[currentPhase].nameEn : PHASES[currentPhase].nameZh}</span>
			<p class="phase-desc">{english ? PHASES[currentPhase].descEn : PHASES[currentPhase].descZh}</p>
		</div>

		<div class="btn-cluster">
			<button type="button" class="cycle-btn" onclick={advanceCycle}>
				⏭️ {english ? "Next Cognitive Step (50ms)" : "推进下一步节拍 (50ms)"}
			</button>
			<button type="button" class="reset-btn" onclick={resetCycle}>
				{english ? "Reset" : "重置"}
			</button>
		</div>
	</div>

	<!-- Insight Box -->
	<div class="insight-box">
		<div class="insight-icon">🧩</div>
		<div class="insight-text">
			<strong>{english ? "Architectural View of Intelligence:" : "认知架构视角的启发："}</strong>
			{english
				? "Human cognitive limits (7±2 working memory slots, 50ms decision cycles) are not biological bugs—they are critical architectural choices that prevent infinite internal recursion and enable unified coordination."
				: "人类的认知局限（如米勒 7±2 工作记忆容量限制、50ms 产生式匹配时钟）绝非生物缺陷，而是工程架构在并行感知与串行动作之间防止内部死锁的关键解耦设计。"}
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
.arch-stage {
	background: rgba(255, 255, 255, 0.6);
	border: 1px solid rgba(0, 0, 0, 0.06);
	border-radius: 8px;
	padding: 1.25rem;
	margin-bottom: 1.25rem;
}
:global(.dark) .arch-stage {
	background: rgba(0, 0, 0, 0.25);
	border-color: rgba(255, 255, 255, 0.06);
}
.modules-grid {
	display: grid;
	grid-template-columns: 1fr 1.5fr 1fr;
	gap: 1rem;
}
.modules-grid.lower {
	grid-template-columns: 1fr 1fr;
}
@media (max-width: 640px) {
	.modules-grid, .modules-grid.lower {
		grid-template-columns: 1fr;
	}
}
.module-card {
	background: rgba(0, 0, 0, 0.03);
	border: 1px solid rgba(0, 0, 0, 0.08);
	border-radius: 8px;
	padding: 0.85rem;
	display: flex;
	flex-direction: column;
	align-items: center;
	text-align: center;
	transition: all 0.3s;
}
:global(.dark) .module-card {
	background: rgba(255, 255, 255, 0.03);
	border-color: rgba(255, 255, 255, 0.08);
}
.module-card.active {
	border-color: #0284c7;
	background: rgba(2, 132, 199, 0.08);
	box-shadow: 0 0 14px rgba(2, 132, 199, 0.2);
	transform: translateY(-2px);
}
.mod-icon {
	font-size: 1.4rem;
	margin-bottom: 0.25rem;
}
.mod-name {
	font-size: 0.78rem;
	font-weight: 700;
	margin-bottom: 0.35rem;
}
.mod-val, .mod-desc {
	font-size: 0.72rem;
	color: var(--text-muted, #718096);
}
.mod-chunk {
	font-family: monospace;
	font-size: 0.76rem;
	background: rgba(0, 0, 0, 0.06);
	padding: 0.25rem 0.5rem;
	border-radius: 4px;
	color: #0284c7;
	font-weight: 600;
}
:global(.dark) .mod-chunk {
	background: rgba(255, 255, 255, 0.08);
	color: #38bdf8;
}
.mod-rule {
	font-family: monospace;
	font-size: 0.74rem;
	color: #ea580c;
}
.bus-divider {
	display: flex;
	align-items: center;
	gap: 1rem;
	margin: 1rem 0;
}
.bus-line {
	flex: 1;
	height: 1px;
	background: var(--border-color, rgba(120, 140, 130, 0.3));
}
.bus-tag {
	font-size: 0.7rem;
	font-weight: 700;
	color: var(--text-muted, #718096);
	letter-spacing: 0.05em;
	text-transform: uppercase;
}
.action-footer {
	display: flex;
	justify-content: space-between;
	align-items: center;
	gap: 1rem;
	margin-bottom: 1.25rem;
	padding: 0.85rem;
	background: rgba(0, 0, 0, 0.02);
	border-radius: 8px;
}
@media (max-width: 600px) {
	.action-footer {
		flex-direction: column;
		align-items: stretch;
	}
}
.phase-badge {
	font-size: 0.8rem;
	font-weight: 700;
	color: var(--brand-clay, #8a7356);
}
.phase-desc {
	font-size: 0.78rem;
	margin-top: 0.2rem;
	color: var(--text-muted, #718096);
}
.btn-cluster {
	display: flex;
	gap: 0.5rem;
}
.cycle-btn {
	background: var(--brand-ink, #2d3748);
	color: #fff;
	border: none;
	padding: 0.45rem 0.85rem;
	border-radius: 6px;
	font-size: 0.8rem;
	cursor: pointer;
	white-space: nowrap;
}
:global(.dark) .cycle-btn {
	background: #3b5043;
}
.reset-btn {
	background: transparent;
	border: 1px solid var(--border-color, rgba(120, 140, 130, 0.3));
	color: inherit;
	padding: 0.45rem 0.7rem;
	border-radius: 6px;
	font-size: 0.8rem;
	cursor: pointer;
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
