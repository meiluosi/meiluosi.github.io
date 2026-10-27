<script lang="ts">
/**
 * Lens 13: 演化、自组织与生命起源路线 (Autopoiesis & Dissipative Boundary)
 * 揭示：马图拉纳与瓦雷拉的‘自创生’(Autopoiesis)假说：原初智能即生命体吸收外部自由能、主动修复并维持自身与环境物理边界的自组织过程。
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

// Nutrient influx rate from environment
let nutrientFlux = $state(6); // 0 to 10
// Environmental entropy / membrane decay rate
let entropyDecay = $state(4); // 1 to 9

// Membrane integrity calculation
const vesicleState = $derived.by(() => {
	const netRepair = nutrientFlux * 1.1 - entropyDecay;
	const integrity = Math.max(0, Math.min(100, Math.round(50 + netRepair * 10)));
	const intactSegments = Math.round((integrity / 100) * 12);

	return {
		integrity,
		intactSegments,
		alive: integrity >= 45,
		stateLabelZh:
			integrity >= 75
				? "自创生稳态旺盛 (Autopoietic Flourishing)"
				: integrity >= 45
					? "临界自维持边界 (Critical Self-Maintenance)"
					: "边界解体 / 热力学平衡死态 (Membrane Dissolution)",
	};
});
</script>

<div class="lens-container">
	<div class="lens-header">
		<span class="lens-tag">COGNITIVE LENS 13</span>
		<h3 class="lens-title">{english ? "Autopoietic Membrane & Proto-Intelligence" : "自创生胞膜与耗散自组织沙盒"}</h3>
		<p class="lens-subtitle">
			{english
				? "Maturana, Varela & Michael Levin: Life and cognition are co-extensive. Before neurons existed, proto-intelligence emerged when a chemical network enclosed itself in a membrane and actively maintained that boundary against entropy."
				: "马图拉纳与瓦雷拉 (Autopoiesis) 及迈克尔·莱文：生命与认知同源。在神经元诞生前，原初智能就已涌现——系统通过代谢循环生成自身边界（细胞膜），而边界又反过来维系内部催化网络免于热力学消散。"}
		</p>
	</div>

	<div class="controls-grid">
		<div class="ctrl-col">
			<div class="label-row">
				<span>{english ? "Metabolic Free-Energy Intake" : "跨膜自由能/营养摄取率 (催化修复)"}</span>
				<strong class="val-pill green">+{nutrientFlux}</strong>
			</div>
			<input type="range" min="0" max="10" step="1" bind:value={nutrientFlux} class="slider" />
		</div>

		<div class="ctrl-col">
			<div class="label-row">
				<span>{english ? "Environmental Thermal Entropy" : "外部环境热力学冲刷/熵增率"}</span>
				<strong class="val-pill red">-{entropyDecay}</strong>
			</div>
			<input type="range" min="1" max="9" step="1" bind:value={entropyDecay} class="slider" />
		</div>
	</div>

	<div class="vesicle-stage">
		<div class="svg-box">
			<svg viewBox="0 0 240 200" class="vesicle-svg">
				<!-- 12 Membrane Lipid Segments around center (120, 100) -->
				{#each Array.from({ length: 12 }) as _, i}
					{@const angle = (i * 30 * Math.PI) / 180}
					{@const cx = 120 + Math.cos(angle) * 62}
					{@const cy = 100 + Math.sin(angle) * 62}
					{@const isIntact = i < vesicleState.intactSegments}
					<circle
						cx={cx}
						cy={cy}
						r="10"
						fill={isIntact ? "#10b981" : "rgba(239, 68, 68, 0.25)"}
						stroke={isIntact ? "#059669" : "#ef4444"}
						stroke-width="2"
						stroke-dasharray={isIntact ? "none" : "3 2"}
					/>
				{/each}

				<!-- Internal Autocatalytic Core -->
				<circle
					cx="120"
					cy="100"
					r="32"
					fill={vesicleState.alive ? "rgba(2, 132, 199, 0.18)" : "rgba(148, 163, 184, 0.1)"}
					stroke={vesicleState.alive ? "#0284c7" : "#94a3b8"}
					stroke-width="2"
				/>
				<text x="120" y="96" text-anchor="middle" font-size="10" font-weight="700" fill="currentColor">
					{vesicleState.alive ? "自催化核心" : "反应耗散"}
				</text>
				<text x="120" y="110" text-anchor="middle" font-size="9" fill="var(--text-muted, #64748b)">
					{vesicleState.integrity}% 完整度
				</text>
			</svg>
		</div>

		<div class="status-panel">
			<div class="stat-badge" class:alive={vesicleState.alive}>
				{vesicleState.stateLabelZh}
			</div>
			<p class="loop-explanation">
				<strong>环形因果闭环 (Circular Causality)：</strong><br />
				1. 内部代谢网络合成脂质分子 → 修补外围膜边界；<br />
				2. 完整膜边界阻挡核心催化剂流失 → 维持内部代谢持续进行。
			</p>
			<div class="integrity-meter">
				<span>马尔可夫毯 / 边界完整度：{vesicleState.integrity}% ({vesicleState.intactSegments}/12 脂质节段)</span>
			</div>
		</div>
	</div>

	<div class="insight-box">
		<div class="insight-icon">🦠</div>
		<div class="insight-text">
			<strong>{english ? "Why AI Lacks Intrinsic Purpose:" : "自组织路线对人工系统的拷问："}</strong>
			{english
				? "A computer program doesn't care if its silicon chip melts; its goals are externally imposed. An autopoietic organism's very existence depends on its own ongoing activity—giving rise to genuine intrinsic teleology (purpose) and care."
				: "传统 AI 程序并不在乎承载它的硅芯片是否断电融化，它的目标函数由人类强加；而自创生生命体的‘存在本身’就是它自身行动的产物，由此涌现出真正的内生目的性 (Intrinsic Teleology) 与生存关切。"}
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
	grid-template-columns: 1fr 1fr;
	gap: 1.25rem;
	padding: 1rem;
	background: rgba(0, 0, 0, 0.03);
	border-radius: 8px;
	margin-bottom: 1.25rem;
}
:global(.dark) .controls-grid { background: rgba(255, 255, 255, 0.03); }
@media (max-width: 640px) { .controls-grid { grid-template-columns: 1fr; } }
.label-row {
	display: flex;
	justify-content: space-between;
	font-size: 0.82rem;
	font-weight: 600;
	margin-bottom: 0.35rem;
}
.val-pill.green { color: #10b981; font-family: monospace; }
.val-pill.red { color: #ef4444; font-family: monospace; }
.slider { width: 100%; accent-color: var(--brand-clay, #8a7356); }
.vesicle-stage {
	display: grid;
	grid-template-columns: 240px 1fr;
	gap: 1.25rem;
	align-items: center;
	background: rgba(255, 255, 255, 0.6);
	border: 1px solid rgba(0, 0, 0, 0.06);
	border-radius: 8px;
	padding: 1rem;
	margin-bottom: 1.25rem;
}
:global(.dark) .vesicle-stage {
	background: rgba(0, 0, 0, 0.25);
	border-color: rgba(255, 255, 255, 0.06);
}
@media (max-width: 640px) { .vesicle-stage { grid-template-columns: 1fr; } }
.vesicle-svg { width: 100%; max-width: 240px; margin: 0 auto; display: block; }
.stat-badge {
	display: inline-block;
	padding: 0.3rem 0.65rem;
	border-radius: 6px;
	font-size: 0.8rem;
	font-weight: 700;
	background: rgba(239, 68, 68, 0.15);
	color: #ef4444;
	margin-bottom: 0.65rem;
}
.stat-badge.alive {
	background: rgba(16, 185, 129, 0.15);
	color: #10b981;
}
.loop-explanation {
	font-size: 0.82rem;
	line-height: 1.55;
	margin-bottom: 0.65rem;
}
.integrity-meter {
	font-size: 0.76rem;
	font-family: monospace;
	color: var(--text-muted, #718096);
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
.insight-icon { font-size: 1.1rem; }
</style>
