<script lang="ts">
/**
 * Lens 04: 理性、适应与资源约束路线 (Bounded & Resource Rationality)
 * 揭示：计算资源与时间是有限的。最优决策不是无限深度的全知计算，而是计算成本与期望效用之间的帕累托最优平衡。
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

// Secretary / Optimal Stopping Simulation
// N candidates arrive sequentially. The goal is to find the best candidate.
// Deliberation cost per candidate inspected = c
let candidateCount = $state(20);
let stoppingThreshold = $state(7); // default ~ 37% of 20 = 7.4
let deliberationCost = $state(2.5); // cost per search unit
let simulationSeed = $state(1);

interface Candidate {
	id: number;
	score: number; // 0 - 100
	inspected: boolean;
	chosen: boolean;
}

// Generate deterministic random candidates from seed
function generateCandidates(seed: number, count: number): Candidate[] {
	const list: Candidate[] = [];
	let s = seed * 9301 + 49297;
	for (let i = 1; i <= count; i++) {
		s = (s * 9301 + 49297) % 233280;
		const score = Math.floor((s / 233280) * 85) + 15;
		list.push({ id: i, score, inspected: false, chosen: false });
	}
	return list;
}

const candidates = $derived(generateCandidates(simulationSeed, candidateCount));

// Run the 37% / threshold strategy
const runResult = $derived.by(() => {
	const list = candidates.map((c) => ({ ...c }));
	const r = Math.max(1, Math.min(stoppingThreshold, list.length - 1));

	// Phase 1: Observation phase (1 to r), find best score seen so far
	let maxSeen = 0;
	for (let i = 0; i < r; i++) {
		list[i].inspected = true;
		if (list[i].score > maxSeen) {
			maxSeen = list[i].score;
		}
	}

	// Phase 2: Selection phase (r to N), pick first candidate strictly > maxSeen
	let pickedIndex = -1;
	for (let i = r; i < list.length; i++) {
		list[i].inspected = true;
		if (list[i].score > maxSeen) {
			pickedIndex = i;
			list[i].chosen = true;
			break;
		}
	}

	// If none exceeded maxSeen, forced to take the last candidate
	if (pickedIndex === -1) {
		pickedIndex = list.length - 1;
		list[pickedIndex].inspected = true;
		list[pickedIndex].chosen = true;
	}

	const inspectedCount = list.filter((c) => c.inspected).length;
	const rawScore = list[pickedIndex].score;
	const totalComputeCost = inspectedCount * deliberationCost;
	const netUtility = Math.max(0, rawScore - totalComputeCost);
	const bestPossible = Math.max(...list.map((c) => c.score));

	return {
		candidates: list,
		pickedIndex,
		inspectedCount,
		rawScore,
		computeCost: totalComputeCost,
		netUtility: Number(netUtility.toFixed(1)),
		bestPossible,
		efficiency: Number(((rawScore / bestPossible) * 100).toFixed(1)),
	};
});

function reseed() {
	simulationSeed += 1;
}
</script>

<div class="lens-container">
	<div class="lens-header">
		<span class="lens-tag">COGNITIVE LENS 04</span>
		<h3 class="lens-title">{english ? "Resource-Rational Stopping Frontier" : "有限理性与最优停止期界沙盒"}</h3>
		<p class="lens-subtitle">
			{english
				? "Herbert Simon & Tom Griffiths: True intelligence balances search depth with deliberation costs. Infinite computation is neither rational nor viable."
				: "西蒙 (Herbert Simon) 与资源理性假说：智能不是无节制地穷举搜索，而是在时间、能耗与认知负荷约束下，寻求满意解与计算成本的最优折衷。"}
		</p>
	</div>

	<!-- Interactive Sliders -->
	<div class="controls-grid">
		<div class="control-item">
			<div class="label-row">
				<span>{english ? "Sample Observation Phase (r)" : "先验观察截断阈值 (r)"}</span>
				<strong class="val-pill">{stoppingThreshold} / {candidateCount} ({Math.round((stoppingThreshold / candidateCount) * 100)}%)</strong>
			</div>
			<input
				type="range"
				min="1"
				max={candidateCount - 1}
				bind:value={stoppingThreshold}
				class="slider"
			/>
			<div class="sub-hint">
				{english ? "Classic optimal without search cost = 1/e ≈ 37%" : "经典无计算成本最优解：1/e ≈ 37% (即第 7 位)"}
			</div>
		</div>

		<div class="control-item">
			<div class="label-row">
				<span>{english ? "Deliberation Cost per Step" : "每步思考 / 搜索能耗成本"}</span>
				<strong class="val-pill">-{deliberationCost} pts/step</strong>
			</div>
			<input
				type="range"
				min="0"
				max="6"
				step="0.5"
				bind:value={deliberationCost}
				class="slider"
			/>
			<div class="sub-hint">
				{english ? "Higher cost shifts optimal stopping leftward (earlier satisficing)." : "计算成本越高，理性主体越应提前满足并终止探索。"}
			</div>
		</div>

		<div class="button-box">
			<button type="button" class="action-btn" onclick={reseed}>
				🎲 {english ? "Reseed Environment" : "生成新序列环境"}
			</button>
		</div>
	</div>

	<!-- Candidates Stream Visualization -->
	<div class="stream-container">
		<div class="stream-title">
			<span>{english ? "Sequential Stream (Left to Right)" : "按时序展开的候选环境流 (从左到右)"}</span>
			<span class="legend">
				<span class="dot obs"></span> {english ? "Observation Window" : "基准观察窗"}
				<span class="dot pick"></span> {english ? "Chosen Action" : "最终选择"}
			</span>
		</div>

		<div class="bars-wrapper">
			{#each runResult.candidates as c, idx}
				<div
					class="bar-col"
					class:is-observation={idx < stoppingThreshold}
					class:is-chosen={c.chosen}
					class:is-inspected={c.inspected}
				>
					<div class="bar-val">{c.score}</div>
					<div class="bar-track">
						<div
							class="bar-fill"
							style="height: {c.score}%;"
						></div>
					</div>
					<div class="bar-idx">#{c.id}</div>
				</div>
			{/each}
		</div>
	</div>

	<!-- Outcome Metrics Cards -->
	<div class="stats-row">
		<div class="stat-card">
			<span class="stat-label">{english ? "Raw Outcome Score" : "客观得分 (解决品质)"}</span>
			<span class="stat-num">{runResult.rawScore} <small>/ {runResult.bestPossible}</small></span>
			<span class="stat-sub">{runResult.efficiency}% {english ? "of global maximum" : "理论全局最优"}</span>
		</div>

		<div class="stat-card">
			<span class="stat-label">{english ? "Cognitive Computation Cost" : "认知搜索损耗"}</span>
			<span class="stat-num cost">-{runResult.computeCost.toFixed(1)}</span>
			<span class="stat-sub">{runResult.inspectedCount} {english ? "steps deliberated" : "步探索消耗"}</span>
		</div>

		<div class="stat-card highlight">
			<span class="stat-label">{english ? "Net Ecological Utility" : "净生态效用 (得分 - 损耗)"}</span>
			<span class="stat-num net">{runResult.netUtility}</span>
			<span class="stat-sub">{english ? "Realistic survival metric" : "真实环境适应度度量"}</span>
		</div>
	</div>

	<!-- Theoretical Insight -->
	<div class="insight-box">
		<div class="insight-icon">⚖️</div>
		<div class="insight-text">
			<strong>{english ? "Bounded Rationality Principle:" : "有限理性核心法则："}</strong>
			{english
				? "If thinking itself burns energy and time, an agent that strives for perfect calculation will perish while calculating. Rationality is not maximization; it is 'satisficing' under resource scarcity."
				: "当思考本身消耗能量与时间时，追求‘绝对全知最优’的智能体往往在计算完成前就已被环境淘汰。智能的本质是王培教授所称的‘知识与资源不足下的适应’，以及西蒙提出的满意化 (Satisficing)。"}
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
.controls-grid {
	display: grid;
	grid-template-columns: 1fr 1fr auto;
	gap: 1rem;
	align-items: center;
	background: rgba(0, 0, 0, 0.03);
	padding: 1rem;
	border-radius: 8px;
	margin-bottom: 1.25rem;
}
:global(.dark) .controls-grid {
	background: rgba(255, 255, 255, 0.03);
}
@media (max-width: 768px) {
	.controls-grid {
		grid-template-columns: 1fr;
	}
}
.control-item {
	display: flex;
	flex-direction: column;
	gap: 0.35rem;
}
.label-row {
	display: flex;
	justify-content: space-between;
	font-size: 0.82rem;
	font-weight: 600;
}
.val-pill {
	color: var(--brand-clay, #8a7356);
	font-family: monospace;
}
.slider {
	width: 100%;
	accent-color: var(--brand-clay, #8a7356);
}
.sub-hint {
	font-size: 0.74rem;
	color: var(--text-muted, #718096);
}
.action-btn {
	background: var(--brand-ink, #2d3748);
	color: #fff;
	border: none;
	padding: 0.5rem 0.85rem;
	border-radius: 6px;
	font-size: 0.82rem;
	cursor: pointer;
	white-space: nowrap;
	height: fit-content;
}
:global(.dark) .action-btn {
	background: #3b5043;
}
.stream-container {
	margin-bottom: 1.25rem;
	padding: 1rem;
	background: rgba(255, 255, 255, 0.6);
	border: 1px solid rgba(0, 0, 0, 0.06);
	border-radius: 8px;
}
:global(.dark) .stream-container {
	background: rgba(0, 0, 0, 0.2);
	border-color: rgba(255, 255, 255, 0.06);
}
.stream-title {
	display: flex;
	justify-content: space-between;
	font-size: 0.8rem;
	margin-bottom: 0.75rem;
	color: var(--text-muted, #718096);
}
.legend {
	display: flex;
	gap: 0.75rem;
	align-items: center;
}
.dot {
	display: inline-block;
	width: 8px;
	height: 8px;
	border-radius: 50%;
}
.dot.obs { background: #94a3b8; }
.dot.pick { background: #10b981; }

.bars-wrapper {
	display: flex;
	gap: 6px;
	align-items: flex-end;
	height: 120px;
	padding-top: 15px;
}
.bar-col {
	flex: 1;
	display: flex;
	flex-direction: column;
	align-items: center;
	height: 100%;
	justify-content: flex-end;
	opacity: 0.35;
	transition: opacity 0.2s;
}
.bar-col.is-inspected {
	opacity: 1;
}
.bar-col.is-observation .bar-fill {
	background: #94a3b8;
}
.bar-col.is-chosen .bar-fill {
	background: #10b981;
	box-shadow: 0 0 10px rgba(16, 185, 129, 0.6);
}
.bar-val {
	font-size: 0.65rem;
	margin-bottom: 2px;
	font-family: monospace;
}
.bar-track {
	width: 100%;
	background: rgba(0, 0, 0, 0.06);
	height: 80px;
	display: flex;
	align-items: flex-end;
	border-radius: 3px 3px 0 0;
	overflow: hidden;
}
:global(.dark) .bar-track {
	background: rgba(255, 255, 255, 0.06);
}
.bar-fill {
	width: 100%;
	background: var(--brand-clay, #8a7356);
	transition: height 0.3s;
}
.bar-idx {
	font-size: 0.65rem;
	color: var(--text-muted, #718096);
	margin-top: 4px;
}
.stats-row {
	display: grid;
	grid-template-columns: repeat(3, 1fr);
	gap: 1rem;
	margin-bottom: 1.25rem;
}
@media (max-width: 600px) {
	.stats-row {
		grid-template-columns: 1fr;
	}
}
.stat-card {
	padding: 0.85rem;
	background: rgba(0, 0, 0, 0.02);
	border: 1px solid rgba(0, 0, 0, 0.06);
	border-radius: 8px;
	display: flex;
	flex-direction: column;
}
:global(.dark) .stat-card {
	background: rgba(255, 255, 255, 0.02);
	border-color: rgba(255, 255, 255, 0.06);
}
.stat-card.highlight {
	background: rgba(16, 185, 129, 0.08);
	border-color: rgba(16, 185, 129, 0.3);
}
.stat-label {
	font-size: 0.72rem;
	text-transform: uppercase;
	color: var(--text-muted, #718096);
	margin-bottom: 0.25rem;
}
.stat-num {
	font-size: 1.35rem;
	font-weight: 700;
	font-family: monospace;
}
.stat-num.cost { color: #ef4444; }
.stat-num.net { color: #10b981; }
.stat-sub {
	font-size: 0.74rem;
	color: var(--text-muted, #718096);
	margin-top: 0.25rem;
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
