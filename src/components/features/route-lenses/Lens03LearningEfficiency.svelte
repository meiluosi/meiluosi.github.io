<script lang="ts">
/**
 * Lens 03: 学习效率与泛化路线透镜 (Chollet ARC / Few-Shot Acquisition)
 * 揭示：智能并非静态做题能力，而是面对全新环境时，将先验知识快速转化为解决未知问题技能的效率。
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

// 3x3 Grid pattern transformation
type Grid = number[][];

const PATTERNS = [
	{
		nameZh: "对称补全 (Symmetry Completion)",
		nameEn: "Symmetry Completion",
		descZh:
			"根据核心先验[对称性与几何反射]，将左半图案沿中轴镜像填充右半部分。",
		descEn:
			"Using core prior [reflectional symmetry], mirror the left axis onto the right.",
		input: [
			[1, 0, 0],
			[1, 1, 0],
			[0, 1, 0],
		],
		target: [
			[1, 0, 1],
			[1, 1, 1],
			[0, 1, 0],
		],
		ruleHintZh: "先验规则：列 2 = 列 0",
		ruleHintEn: "Prior rule: Col 2 = Col 0",
	},
	{
		nameZh: "重力坠落 (Gravity Fall)",
		nameEn: "Gravity Fall",
		descZh: "根据核心先验[物体性与重力连续性]，所有悬浮像素受重力向底部堆叠。",
		descEn:
			"Using core prior [objectness & gravity], floating pixels settle to the floor.",
		input: [
			[2, 0, 2],
			[0, 0, 0],
			[0, 2, 0],
		],
		target: [
			[0, 0, 0],
			[0, 0, 0],
			[2, 2, 2],
		],
		ruleHintZh: "先验规则：非零元素落入最底层空位",
		ruleHintEn: "Prior rule: Active elements drop to bottom rows",
	},
	{
		nameZh: "拓扑填充 (Enclosure Filling)",
		nameEn: "Enclosure Filling",
		descZh: "根据核心先验[内外拓扑连通]，被边界包围的空位被填色染色。",
		descEn:
			"Using core prior [topological enclosure], bounded inner space is colored.",
		input: [
			[3, 3, 3],
			[3, 0, 3],
			[3, 3, 3],
		],
		target: [
			[3, 3, 3],
			[3, 4, 3],
			[3, 3, 3],
		],
		ruleHintZh: "先验规则：内部中点 (1,1) 从 0 变为 4",
		ruleHintEn: "Prior rule: Center cavity (1,1) fills with color 4",
	},
];

let selectedPatternIdx = $state(0);
let mode = $state<"fewshot" | "bruteforce">("fewshot");
let samplesObserved = $state(1);
let currentGrid = $state<Grid>([
	[0, 0, 0],
	[0, 0, 0],
	[0, 0, 0],
]);
let solved = $state(false);

const activePattern = $derived(PATTERNS[selectedPatternIdx]);

// Color map for grid values
const COLOR_CLASSES: Record<number, string> = {
	0: "bg-surface",
	1: "bg-cyan",
	2: "bg-amber",
	3: "bg-rose",
	4: "bg-emerald",
};

function loadPattern(idx: number) {
	selectedPatternIdx = idx;
	currentGrid = JSON.parse(JSON.stringify(PATTERNS[idx].input));
	solved = false;
	samplesObserved = mode === "fewshot" ? 1 : 10000;
}

// Reset whenever mode or pattern changes
$effect(() => {
	loadPattern(selectedPatternIdx);
});

function toggleCell(r: number, c: number) {
	if (solved) return;
	// Cycle through color values 0-4
	currentGrid[r][c] = (currentGrid[r][c] + 1) % 5;
	checkSolution();
}

function checkSolution() {
	const target = activePattern.target;
	let match = true;
	for (let r = 0; r < 3; r++) {
		for (let c = 0; c < 3; c++) {
			if (currentGrid[r][c] !== target[r][c]) {
				match = false;
				break;
			}
		}
	}
	solved = match;
}

function autoSolveWithPrior() {
	currentGrid = JSON.parse(JSON.stringify(activePattern.target));
	solved = true;
}
</script>

<div class="lens-container">
	<div class="lens-header">
		<span class="lens-tag">COGNITIVE LENS 03</span>
		<h3 class="lens-title">{english ? "ARC Few-Shot Induction Puzzle" : "ARC 技能习得与先验泛化沙盒"}</h3>
		<p class="lens-subtitle">
			{english
				? "François Chollet's ARC proposition: Intelligence is not static memory, but the efficiency of acquiring novel skills using core priors."
				: "弗朗索瓦·肖莱 (François Chollet) 命题：智能不是静态知识库的大小，而是利用核心先验在极少量样本下习得新技能的效率。"}
		</p>
	</div>

	<!-- Mode Switcher & Pattern Pills -->
	<div class="controls-bar">
		<div class="pattern-selector">
			{#each PATTERNS as p, idx}
				<button
					type="button"
					class="pattern-tab"
					class:active={selectedPatternIdx === idx}
					onclick={() => loadPattern(idx)}
				>
					{english ? p.nameEn : p.nameZh}
				</button>
			{/each}
		</div>

		<div class="mode-toggles">
			<button
				type="button"
				class="mode-btn"
				class:active={mode === "fewshot"}
				onclick={() => { mode = "fewshot"; samplesObserved = 1; }}
			>
				{english ? "Few-Shot Prior (Humans/ARC)" : "核心先验少样本 (人类/ARC)"}
			</button>
			<button
				type="button"
				class="mode-btn"
				class:active={mode === "bruteforce"}
				onclick={() => { mode = "bruteforce"; samplesObserved = 50000; }}
			>
				{english ? "Massive Pretraining (10k+ epochs)" : "蛮力过拟合 (海量经验驱动)"}
			</button>
		</div>
	</div>

	<!-- Interactive Workspace -->
	<div class="workspace-grid">
		<!-- Left: Task Description & Input -->
		<div class="card panel">
			<div class="panel-header">
				<span class="step-badge">INPUT</span>
				<h4>{english ? "Given Problem Instance" : "输入问题示例"}</h4>
			</div>
			<div class="matrix-display">
				{#each activePattern.input as row, r}
					<div class="matrix-row">
						{#each row as val, c}
							<div class="cell {COLOR_CLASSES[val]}"></div>
						{/each}
					</div>
				{/each}
			</div>
			<p class="matrix-caption">
				{english ? activePattern.descEn : activePattern.descZh}
			</p>
		</div>

		<!-- Middle: Transformation Arrow & Metric -->
		<div class="card metrics-panel">
			<div class="efficiency-gauge">
				<span class="gauge-label">{english ? "Conversion Efficiency" : "转化效率"}</span>
				<div class="gauge-value" class:high={mode === "fewshot"}>
					{mode === "fewshot" ? "O(1) Sample" : "O(N) Samples"}
				</div>
				<div class="sample-count">
					{english ? "Observed Examples: " : "已观察样本量: "}
					<strong>{samplesObserved.toLocaleString()}</strong>
				</div>
			</div>
			<div class="arrow-indicator">➔</div>
			<button
				type="button"
				class="solve-action-btn"
				onclick={autoSolveWithPrior}
			>
				{english ? "Apply Prior Hypothesis" : "执行核心先验假设"}
			</button>
		</div>

		<!-- Right: Output Grid (Editable) -->
		<div class="card panel">
			<div class="panel-header">
				<span class="step-badge output">TARGET</span>
				<h4>{english ? "Your Deduction (Click cells)" : "目标输出 (点击单元格调试)"}</h4>
			</div>
			<div class="matrix-display editable">
				{#each currentGrid as row, r}
					<div class="matrix-row">
						{#each row as val, c}
							<button
								type="button"
								class="cell clickable {COLOR_CLASSES[val]}"
								onclick={() => toggleCell(r, c)}
								aria-label={`Row ${r} Col ${c} value ${val}`}
							></button>
						{/each}
					</div>
				{/each}
			</div>
			<div class="status-indicator" class:success={solved}>
				{#if solved}
					✓ {english ? "Generalization Verified! Rule Inducted." : "泛化成功！已通过核心先验完成规律归纳。"}
				{:else}
					{english ? "Transform incomplete. Match the target." : "尚未完成推演。点击色块或应用先验。"}
				{/if}
			</div>
		</div>
	</div>

	<!-- Theoretical Takeaway -->
	<div class="insight-box">
		<div class="insight-icon">💡</div>
		<div class="insight-text">
			<strong>{english ? "Theoretical Takeaway:" : "理论核心洞察："}</strong>
			{english
				? "Memorizing all 3^9 = 19,683 states is not intelligence. Generalization occurs when an agent synthesizes an abstract program from 1 example by leveraging innate geometric, topological, or physical priors."
				: "死记硬背所有 3^9 = 19,683 种网格排列不是智能。真正的智能在于：依托几何对称、拓扑连通、重力等人类固有的低维核心先验，仅凭 1 个样例即可瞬间合成出抽象变换程序。"}
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
	display: flex;
	flex-wrap: wrap;
	gap: 0.75rem;
	justify-content: space-between;
	align-items: center;
	margin-bottom: 1.25rem;
	padding-bottom: 1rem;
	border-bottom: 1px dashed var(--border-color, rgba(120, 140, 130, 0.2));
}
.pattern-selector, .mode-toggles {
	display: flex;
	gap: 0.5rem;
	flex-wrap: wrap;
}
.pattern-tab, .mode-btn {
	background: rgba(0, 0, 0, 0.04);
	border: 1px solid rgba(0, 0, 0, 0.08);
	color: inherit;
	padding: 0.35rem 0.75rem;
	border-radius: 6px;
	font-size: 0.82rem;
	cursor: pointer;
	transition: all 0.2s ease;
}
:global(.dark) .pattern-tab, :global(.dark) .mode-btn {
	background: rgba(255, 255, 255, 0.05);
	border-color: rgba(255, 255, 255, 0.1);
}
.pattern-tab.active, .mode-btn.active {
	background: var(--brand-ink, #2d3748);
	color: #fff;
	border-color: var(--brand-ink, #2d3748);
}
:global(.dark) .pattern-tab.active, :global(.dark) .mode-btn.active {
	background: #3b5043;
	border-color: #526f5d;
	color: #fff;
}
.workspace-grid {
	display: grid;
	grid-template-columns: 1fr 140px 1fr;
	gap: 1.25rem;
	align-items: stretch;
}
@media (max-width: 768px) {
	.workspace-grid {
		grid-template-columns: 1fr;
	}
}
.card {
	background: rgba(255, 255, 255, 0.7);
	border: 1px solid rgba(0, 0, 0, 0.06);
	border-radius: 8px;
	padding: 1rem;
	display: flex;
	flex-direction: column;
	align-items: center;
}
:global(.dark) .card {
	background: rgba(255, 255, 255, 0.02);
	border-color: rgba(255, 255, 255, 0.06);
}
.panel-header {
	display: flex;
	align-items: center;
	gap: 0.5rem;
	margin-bottom: 0.75rem;
	width: 100%;
}
.step-badge {
	font-size: 0.65rem;
	font-weight: 700;
	padding: 0.15rem 0.4rem;
	border-radius: 4px;
	background: #e2e8f0;
	color: #475569;
}
:global(.dark) .step-badge {
	background: #334155;
	color: #cbd5e1;
}
.step-badge.output {
	background: #ccfbf1;
	color: #0f766e;
}
:global(.dark) .step-badge.output {
	background: #115e59;
	color: #ccfbf1;
}
.panel-header h4 {
	font-size: 0.9rem;
	margin: 0;
}
.matrix-display {
	display: flex;
	flex-direction: column;
	gap: 4px;
	margin: 0.5rem 0;
	padding: 6px;
	background: rgba(0, 0, 0, 0.08);
	border-radius: 6px;
}
:global(.dark) .matrix-display {
	background: rgba(0, 0, 0, 0.4);
}
.matrix-row {
	display: flex;
	gap: 4px;
}
.cell {
	width: 40px;
	height: 40px;
	border-radius: 4px;
	transition: background 0.15s ease;
}
.cell.clickable {
	cursor: pointer;
	border: 1px solid rgba(0, 0, 0, 0.15);
}
.cell.clickable:hover {
	transform: scale(0.96);
}
.bg-surface { background: #f1f5f9; }
:global(.dark) .bg-surface { background: #1e293b; }
.bg-cyan { background: #06b6d4; }
.bg-amber { background: #f59e0b; }
.bg-rose { background: #f43f5e; }
.bg-emerald { background: #10b981; }

.matrix-caption {
	font-size: 0.78rem;
	color: var(--text-muted, #64748b);
	text-align: center;
	margin-top: 0.5rem;
	line-height: 1.4;
}
.metrics-panel {
	justify-content: center;
	text-align: center;
	gap: 0.75rem;
}
.gauge-label {
	font-size: 0.72rem;
	text-transform: uppercase;
	color: var(--text-muted, #64748b);
}
.gauge-value {
	font-size: 1.1rem;
	font-weight: 700;
	margin: 0.2rem 0;
	color: #ef4444;
}
.gauge-value.high {
	color: #10b981;
}
.sample-count {
	font-size: 0.75rem;
}
.arrow-indicator {
	font-size: 1.5rem;
	color: var(--brand-clay, #8a7356);
}
.solve-action-btn {
	background: var(--brand-clay, #8a7356);
	color: white;
	border: none;
	padding: 0.4rem 0.6rem;
	border-radius: 6px;
	font-size: 0.78rem;
	cursor: pointer;
	transition: opacity 0.2s;
}
.solve-action-btn:hover {
	opacity: 0.9;
}
.status-indicator {
	font-size: 0.8rem;
	margin-top: 0.5rem;
	color: #ef4444;
	font-weight: 600;
	text-align: center;
}
.status-indicator.success {
	color: #10b981;
}
.insight-box {
	display: flex;
	gap: 0.75rem;
	margin-top: 1.25rem;
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
