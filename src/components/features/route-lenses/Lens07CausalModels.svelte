<script lang="ts">
/**
 * Lens 07: 因果推断、世界模型与反事实路线 (Pearl's Ladder of Causation)
 * 揭示：统计关联 (Seeing) 无法推导干预效应 (Doing)，更无法回答反事实问题 (Imagining)。
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

type CausalRung = 1 | 2 | 3;
let selectedRung = $state<CausalRung>(1);

// Confounder setting: Severe Weather (Z) influences both Umbrella Use (X) and Car Accidents (Y)
let doIntervention = $state(false); // If true, cuts Z -> X (Pearl's do-operator)
let umbrellaGivenByPolicy = $state(true); // Forcing X = 1

// Simulation calculation:
// Without intervention: Severe Weather causes Umbrellas (correlation is positive, accidents are high)
// With do(X=1): Handing everyone umbrellas on a sunny day DOES NOT cause accidents!
const stats = $derived.by(() => {
	if (selectedRung === 1) {
		// Observational Association: P(Accident | Umbrella)
		// Because rainy weather makes people hold umbrellas AND makes roads slippery, umbrellas correlate with accidents!
		return {
			pObs: "82%",
			interpretationZh: "统计关联 P(事故 | 雨伞) = 82% (呈现虚假强相关)",
			interpretationEn:
				"Statistical Association P(Accident | Umbrella) = 82% (Spurious correlation due to weather confounder)",
			formula: "P(Y | X) = \\sum_z P(Y | X, z) P(z | X)",
			insightZh:
				"第 1 层 [观察]：纯统计相关，分不清是雨伞导致事故，还是恶劣天气同时导致了两者。",
			insightEn:
				"Rung 1 [Association]: Pure observational data cannot discern whether umbrellas cause crashes or bad weather causes both.",
		};
	}
	if (selectedRung === 2) {
		// Intervention: P(Accident | do(Umbrella))
		// Cutting the incoming arrow from Weather to Umbrella
		return {
			pObs: doIntervention ? (umbrellaGivenByPolicy ? "12%" : "10%") : "82%",
			interpretationZh: doIntervention
				? "行动干预 P(事故 | do(雨伞=1)) = 12% (晴天发伞不增事故，因果箭头斩断)"
				: "尚未实施干预操作 do(X)，仍处于混杂污染状态",
			interpretationEn: doIntervention
				? "Causal Intervention P(Accident | do(Umbrella=1)) = 12% (Distributing umbrellas in sunlight doesn't cause accidents)"
				: "do(X) not activated; graph remains confounded",
			formula: "P(Y | do(X=x)) = \\sum_z P(Y | X=x, z) P(z)",
			insightZh:
				"第 2 层 [干预]：通过强制设定 X 并切断所有指向 X 的父节点箭头，消除混杂因子 Z 的后门路径。",
			insightEn:
				"Rung 2 [Intervention]: By clamping X and severing incoming parents, the backdoor confounding path is eliminated.",
		};
	}
	// Counterfactual: P(Y_{X=0} | X=1, Y=1)
	return {
		pObs: "94% (归因概率)",
		interpretationZh:
			"反事实回顾：‘这起事故如果当时没有暴风雨，即使路人撑伞是否依然会发生？’",
		interpretationEn:
			"Counterfactual Retrospective: 'Would this crash have occurred had there been no storm, even if umbrellas were open?'",
		formula: "P(Y_{X=x'} | X=x, Y=y)",
		insightZh:
			"第 3 层 [反事实]：想象未曾发生的世界（平行历史），追问必要责任、归因与回溯性推演。",
		insightEn:
			"Rung 3 [Counterfactuals]: Reasoning across alternative counter-histories to establish structural credit assignment and necessity.",
	};
});
</script>

<div class="lens-container">
	<div class="lens-header">
		<span class="lens-tag">COGNITIVE LENS 07</span>
		<h3 class="lens-title">{english ? "Pearl's Causal Ladder & Graph Surgery" : "珀尔因果三层阶梯与图切断沙盒"}</h3>
		<p class="lens-subtitle">
			{english
				? "Judea Pearl: Deep neural networks remain stuck on Rung 1 (curve fitting / association). Genuine intelligence requires Rung 2 (intervention) and Rung 3 (counterfactual imagination)."
				: "朱迪亚·珀尔 (Judea Pearl) 命题：当今大模型与统计学习仍停留在因果之梯的第 1 层 (关联拟合)。真正的世界模型必须能够跨越至第 2 层 (干预 do-算子) 与第 3 层 (反事实推演)。"}
		</p>
	</div>

	<!-- Rung Selector -->
	<div class="rungs-nav">
		<button
			type="button"
			class="rung-tab"
			class:active={selectedRung === 1}
			onclick={() => selectedRung = 1}
		>
			<span class="rung-num">RUNG 1</span>
			<span class="rung-name">{english ? "Association (Seeing)" : "第一层：关联 (观察)"}</span>
		</button>
		<button
			type="button"
			class="rung-tab"
			class:active={selectedRung === 2}
			onclick={() => selectedRung = 2}
		>
			<span class="rung-num">RUNG 2</span>
			<span class="rung-name">{english ? "Intervention (Doing)" : "第二层：干预 (行动)"}</span>
		</button>
		<button
			type="button"
			class="rung-tab"
			class:active={selectedRung === 3}
			onclick={() => selectedRung = 3}
		>
			<span class="rung-num">RUNG 3</span>
			<span class="rung-name">{english ? "Counterfactuals (Imagining)" : "第三层：反事实 (想象)"}</span>
		</button>
	</div>

	<!-- Interactive Causal DAG Visualizer -->
	<div class="dag-stage">
		<div class="dag-header">
			<span class="stage-badge">{english ? "Causal DAG Representation" : "因果有向无环图 (DAG)"}</span>
			{#if selectedRung === 2}
				<label class="toggle-label">
					<input
						type="checkbox"
						bind:checked={doIntervention}
					/>
					<span>{english ? "Execute Graph Surgery: do(X)" : "执行图结构手术：do(X)"}</span>
				</label>
			{/if}
		</div>

		<div class="graph-svg-container">
			<svg viewBox="0 0 500 240" class="dag-svg">
				<defs>
					<marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
						<path d="M 0 1 L 10 5 L 0 9 z" fill="var(--brand-ink, #475569)" />
					</marker>
					<marker id="arrow-cut" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
						<path d="M 0 1 L 10 5 L 0 9 z" fill="#ef4444" />
					</marker>
				</defs>

				<!-- Confounder Node Z: Weather -->
				<g class="node-group" transform="translate(250, 45)">
					<circle r="32" class="node-circle z-node" />
					<text y="-4" text-anchor="middle" class="node-title">混杂因子 Z</text>
					<text y="14" text-anchor="middle" class="node-sub">暴风雨天气</text>
				</g>

				<!-- Treatment Node X: Umbrella -->
				<g class="node-group" transform="translate(110, 185)">
					<circle r="32" class="node-circle x-node" class:clamped={selectedRung === 2 && doIntervention} />
					<text y="-4" text-anchor="middle" class="node-title">原因变量 X</text>
					<text y="14" text-anchor="middle" class="node-sub">撑伞人群</text>
				</g>

				<!-- Outcome Node Y: Accident -->
				<g class="node-group" transform="translate(390, 185)">
					<circle r="32" class="node-circle y-node" />
					<text y="-4" text-anchor="middle" class="node-title">结果变量 Y</text>
					<text y="14" text-anchor="middle" class="node-sub">交通事故概率</text>
				</g>

				<!-- Edge Z -> Y -->
				<line x1="265" y1="72" x2="370" y2="158" class="causal-edge" marker-end="url(#arrow)" />

				<!-- Edge Z -> X (Cut if doIntervention in Rung 2) -->
				{#if selectedRung === 2 && doIntervention}
					<line x1="235" y1="72" x2="130" y2="158" class="causal-edge cut" stroke-dasharray="4" marker-end="url(#arrow-cut)" />
					<text x="175" y="105" fill="#ef4444" font-size="16" font-weight="bold">✂ CUT</text>
				{:else}
					<line x1="235" y1="72" x2="130" y2="158" class="causal-edge" marker-end="url(#arrow)" />
				{/if}

				<!-- Direct Edge X -> Y (Hypothesized causal link) -->
				<line x1="145" y1="185" x2="355" y2="185" class="causal-edge direct" stroke-dasharray="2" marker-end="url(#arrow)" />
				<text x="250" y="175" text-anchor="middle" class="edge-label">因果效应 = 0</text>
			</svg>
		</div>

		<!-- Calculation & Deduction Display -->
		<div class="result-banner">
			<div class="res-stat">
				<span class="label">{english ? "Computed Probability:" : "算子推演概率值："}</span>
				<span class="val">{stats.pObs}</span>
			</div>
			<div class="res-desc">
				<div class="formula"><code>{stats.formula}</code></div>
				<p class="summary">{english ? stats.interpretationEn : stats.interpretationZh}</p>
			</div>
		</div>
	</div>

	<!-- Insight Box -->
	<div class="insight-box">
		<div class="insight-icon">🪜</div>
		<div class="insight-text">
			<strong>{english ? "Why Associative LLMs Aren't Enough:" : "为什么仅靠大模型关联永远不够："}</strong>
			{english
				? stats.insightEn
				: stats.insightZh}
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
.rungs-nav {
	display: grid;
	grid-template-columns: repeat(3, 1fr);
	gap: 0.75rem;
	margin-bottom: 1.25rem;
}
@media (max-width: 640px) {
	.rungs-nav {
		grid-template-columns: 1fr;
	}
}
.rung-tab {
	background: rgba(0, 0, 0, 0.03);
	border: 1px solid rgba(0, 0, 0, 0.08);
	border-radius: 8px;
	padding: 0.75rem;
	display: flex;
	flex-direction: column;
	align-items: center;
	gap: 0.25rem;
	cursor: pointer;
	transition: all 0.2s;
	color: inherit;
}
:global(.dark) .rung-tab {
	background: rgba(255, 255, 255, 0.03);
	border-color: rgba(255, 255, 255, 0.08);
}
.rung-tab.active {
	background: var(--brand-ink, #2d3748);
	border-color: var(--brand-ink, #2d3748);
	color: #fff;
}
:global(.dark) .rung-tab.active {
	background: #3b5043;
}
.rung-num {
	font-size: 0.65rem;
	font-weight: 700;
	letter-spacing: 0.08em;
	opacity: 0.8;
}
.rung-name {
	font-size: 0.84rem;
	font-weight: 600;
}
.dag-stage {
	background: rgba(255, 255, 255, 0.6);
	border: 1px solid rgba(0, 0, 0, 0.06);
	border-radius: 8px;
	padding: 1.25rem;
	margin-bottom: 1.25rem;
}
:global(.dark) .dag-stage {
	background: rgba(0, 0, 0, 0.25);
	border-color: rgba(255, 255, 255, 0.06);
}
.dag-header {
	display: flex;
	justify-content: space-between;
	align-items: center;
	margin-bottom: 0.75rem;
}
.stage-badge {
	font-size: 0.75rem;
	font-weight: 700;
	color: var(--text-muted, #718096);
	text-transform: uppercase;
}
.toggle-label {
	display: flex;
	align-items: center;
	gap: 0.5rem;
	font-size: 0.82rem;
	font-weight: 600;
	cursor: pointer;
	color: #ef4444;
}
.graph-svg-container {
	width: 100%;
	max-width: 500px;
	margin: 0 auto;
}
.dag-svg {
	width: 100%;
	height: auto;
}
.node-circle {
	stroke-width: 2;
	fill: #f8fafc;
	stroke: #64748b;
}
:global(.dark) .node-circle {
	fill: #1e293b;
	stroke: #94a3b8;
}
.node-circle.z-node {
	stroke: #8b5cf6;
}
.node-circle.x-node {
	stroke: #0284c7;
}
.node-circle.x-node.clamped {
	stroke: #ef4444;
	stroke-dasharray: 4;
}
.node-circle.y-node {
	stroke: #ea580c;
}
.node-title {
	font-size: 11px;
	font-weight: 700;
	fill: var(--brand-ink, #334155);
}
:global(.dark) .node-title { fill: #f1f5f9; }
.node-sub {
	font-size: 10px;
	fill: var(--text-muted, #64748b);
}
.causal-edge {
	stroke: var(--brand-ink, #475569);
	stroke-width: 2;
}
.causal-edge.cut {
	stroke: #ef4444;
	stroke-width: 2;
}
.causal-edge.direct {
	stroke: #94a3b8;
}
.edge-label {
	font-size: 10px;
	fill: #94a3b8;
}
.result-banner {
	display: flex;
	gap: 1.5rem;
	align-items: center;
	margin-top: 1rem;
	padding-top: 1rem;
	border-top: 1px dashed var(--border-color, rgba(120, 140, 130, 0.2));
}
@media (max-width: 600px) {
	.result-banner {
		flex-direction: column;
		align-items: flex-start;
		gap: 0.5rem;
	}
}
.res-stat {
	display: flex;
	flex-direction: column;
}
.res-stat .label {
	font-size: 0.72rem;
	color: var(--text-muted, #718096);
}
.res-stat .val {
	font-size: 1.5rem;
	font-weight: 700;
	color: var(--brand-clay, #8a7356);
}
.formula code {
	font-family: monospace;
	font-size: 0.8rem;
	background: rgba(0, 0, 0, 0.05);
	padding: 0.2rem 0.4rem;
	border-radius: 4px;
}
:global(.dark) .formula code {
	background: rgba(255, 255, 255, 0.06);
}
.summary {
	font-size: 0.82rem;
	margin-top: 0.35rem;
	line-height: 1.4;
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
