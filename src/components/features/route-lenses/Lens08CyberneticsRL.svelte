<script lang="ts">
/**
 * Lens 08: 控制论、强化学习与自主行动路线 (Ashby's Law of Requisite Variety)
 * 揭示：‘只有多样性才能消解多样性’(Only variety can absorb variety)。智能系统的核心在于通过负反馈回路对抗外界熵增干扰。
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

// Disturbance intensity from environment (D)
let disturbance = $state(5); // 0 to 10
// Controller variety / repertoire capacity (R)
let regulatorVariety = $state(4); // 1 to 10
let isOscillating = $state(true);

// State deviation (E = D - R)
const systemError = $derived(Math.max(0, disturbance - regulatorVariety));
const stability = $derived(Math.max(0, 100 - systemError * 16));
const isHomeostatic = $derived(regulatorVariety >= disturbance);

function injectDisturbance(level: number) {
	disturbance = level;
}
</script>

<div class="lens-container">
	<div class="lens-header">
		<span class="lens-tag">COGNITIVE LENS 08</span>
		<h3 class="lens-title">{english ? "Ashby's Law & Homeostatic Regulator" : "阿什比必要多样性定律与稳态回路沙盒"}</h3>
		<p class="lens-subtitle">
			{english
				? "W. Ross Ashby & Norbert Wiener: Intelligence is closed-loop negative feedback control. Only variety in the regulator can absorb variety from environmental perturbations."
				: "诺伯特·维纳与阿什比 (W. Ross Ashby)：智能起源于闭环负反馈调节。只有当控制系统内部的可调控多样性 (Variety) 大于或等于外界环境的扰动多样性时，系统才能维持生存稳态 (Homeostasis)。"}
		</p>
	</div>

	<!-- Interactive Controls -->
	<div class="controls-grid">
		<div class="control-col">
			<div class="label-row">
				<span>{english ? "Environmental Perturbation (Entropy/Disturbance D)" : "外部环境扰动多样性 (熵增/冲击 D)"}</span>
				<strong class="val-pill red">D = {disturbance}</strong>
			</div>
			<input
				type="range"
				min="1"
				max="10"
				step="1"
				bind:value={disturbance}
				class="slider"
			/>
			<div class="preset-row">
				<button type="button" class="mini-btn" onclick={() => injectDisturbance(2)}>轻微微风</button>
				<button type="button" class="mini-btn" onclick={() => injectDisturbance(6)}>剧烈乱流</button>
				<button type="button" class="mini-btn alert" onclick={() => injectDisturbance(10)}>极端风暴</button>
			</div>
		</div>

		<div class="control-col">
			<div class="label-row">
				<span>{english ? "Regulator Internal Variety (Capacity R)" : "智能主体调节器内部多样性 (策略库容量 R)"}</span>
				<strong class="val-pill green">R = {regulatorVariety}</strong>
			</div>
			<input
				type="range"
				min="1"
				max="10"
				step="1"
				bind:value={regulatorVariety}
				class="slider"
			/>
			<div class="sub-hint">
				{english
					? "Ashby Theorem: V(E) >= V(D) - V(R). If R < D, error is inevitable."
					: "阿什比定理：残余误差 V(E) ≥ V(D) - V(R)。当 R < D 时，稳态必然崩溃。"}
			</div>
		</div>
	</div>

	<!-- Visual Homeostasis Chamber -->
	<div class="chamber-box" class:stable={isHomeostatic} class:critical={!isHomeostatic && systemError >= 4}>
		<div class="chamber-status">
			<span class="status-badge" class:ok={isHomeostatic}>
				{isHomeostatic ? "● 稳态维持 (HOMEOSTASIS SECURED)" : "▲ 稳态失控 (ENTROPIC DRIFT)"}
			</span>
			<span class="stability-rate">{english ? "Stability Index: " : "系统稳定性指数："}<strong>{stability}%</strong></span>
		</div>

		<div class="feedback-circuit">
			<div class="circuit-node env">
				<div class="node-icon">🌪️</div>
				<div class="node-meta">
					<span class="t">环境冲击 D</span>
					<span class="v">多样性 {disturbance}</span>
				</div>
			</div>

			<div class="circuit-arrow env-to-sys">➔</div>

			<div class="circuit-node core" style="transform: rotate({systemError * 3}deg);">
				<div class="pendulum-graphic">
					<div class="pivot"></div>
					<div class="arm" style="transform: rotate({(disturbance - regulatorVariety) * 6}deg);">
						<div class="bob"></div>
					</div>
				</div>
				<span class="core-label">{english ? "Organism / Agent State" : "主体核心受力态"}</span>
			</div>

			<div class="circuit-arrow sys-to-reg">➔</div>

			<div class="circuit-node reg">
				<div class="node-icon">🛡️</div>
				<div class="node-meta">
					<span class="t">负反馈调节器 R</span>
					<span class="v">反制容量 {regulatorVariety}</span>
				</div>
			</div>
		</div>
	</div>

	<!-- Insight Box -->
	<div class="insight-box">
		<div class="insight-icon">⚖️</div>
		<div class="insight-text">
			<strong>{english ? "Cybernetics Perspective on AGI:" : "控制论视角的智能本质："}</strong>
			{english
				? "Intelligence is not passive knowledge retention; it is active environmental survival through circular causality. If a system lacks the internal variety to counter world volatility, no amount of passive memorization will prevent entropy collapse."
				: "控制论将智能视为闭环因果中的能动生存机制。一个仅有海量静态文本记忆、却缺乏足够应对行动多态性 (Action Variety) 的系统，在动态开放环境中注定无法维持自洽稳态。"}
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
:global(.dark) .controls-grid {
	background: rgba(255, 255, 255, 0.03);
}
@media (max-width: 640px) {
	.controls-grid {
		grid-template-columns: 1fr;
	}
}
.label-row {
	display: flex;
	justify-content: space-between;
	font-size: 0.82rem;
	font-weight: 600;
	margin-bottom: 0.35rem;
}
.val-pill {
	font-family: monospace;
	font-weight: 700;
}
.val-pill.red { color: #ef4444; }
.val-pill.green { color: #10b981; }
.slider {
	width: 100%;
	accent-color: var(--brand-clay, #8a7356);
}
.preset-row {
	display: flex;
	gap: 0.5rem;
	margin-top: 0.5rem;
}
.mini-btn {
	background: rgba(0, 0, 0, 0.05);
	border: 1px solid rgba(0, 0, 0, 0.08);
	border-radius: 4px;
	padding: 0.2rem 0.5rem;
	font-size: 0.72rem;
	cursor: pointer;
	color: inherit;
}
:global(.dark) .mini-btn {
	background: rgba(255, 255, 255, 0.05);
	border-color: rgba(255, 255, 255, 0.1);
}
.mini-btn.alert { color: #ef4444; }
.sub-hint {
	font-size: 0.74rem;
	color: var(--text-muted, #718096);
	margin-top: 0.5rem;
}
.chamber-box {
	background: rgba(255, 255, 255, 0.6);
	border: 2px solid #10b981;
	border-radius: 8px;
	padding: 1.25rem;
	margin-bottom: 1.25rem;
	transition: all 0.3s;
}
:global(.dark) .chamber-box {
	background: rgba(0, 0, 0, 0.25);
}
.chamber-box.critical {
	border-color: #ef4444;
	box-shadow: 0 0 16px rgba(239, 68, 68, 0.15);
}
.chamber-status {
	display: flex;
	justify-content: space-between;
	align-items: center;
	margin-bottom: 1rem;
}
.status-badge {
	font-size: 0.76rem;
	font-weight: 700;
	color: #ef4444;
}
.status-badge.ok { color: #10b981; }
.stability-rate {
	font-size: 0.85rem;
}
.feedback-circuit {
	display: flex;
	align-items: center;
	justify-content: space-around;
	gap: 1rem;
	padding: 1rem 0;
}
@media (max-width: 600px) {
	.feedback-circuit {
		flex-direction: column;
	}
}
.circuit-node {
	display: flex;
	align-items: center;
	gap: 0.75rem;
	padding: 0.65rem 1rem;
	background: rgba(0, 0, 0, 0.04);
	border-radius: 8px;
}
:global(.dark) .circuit-node {
	background: rgba(255, 255, 255, 0.05);
}
.circuit-node.core {
	flex-direction: column;
	background: transparent;
	padding: 0;
	transition: transform 0.3s;
}
.pendulum-graphic {
	width: 60px;
	height: 70px;
	position: relative;
	display: flex;
	justify-content: center;
}
.pivot {
	width: 8px;
	height: 8px;
	background: #64748b;
	border-radius: 50%;
}
.arm {
	position: absolute;
	top: 4px;
	width: 3px;
	height: 50px;
	background: #64748b;
	transform-origin: top center;
	transition: transform 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
	display: flex;
	align-items: flex-end;
	justify-content: center;
}
.bob {
	width: 14px;
	height: 14px;
	background: var(--brand-clay, #8a7356);
	border-radius: 50%;
}
.core-label {
	font-size: 0.74rem;
	margin-top: 0.35rem;
	color: var(--text-muted, #718096);
}
.circuit-arrow {
	font-size: 1.2rem;
	color: var(--brand-clay, #8a7356);
}
.node-meta {
	display: flex;
	flex-direction: column;
}
.node-meta .t {
	font-size: 0.8rem;
	font-weight: 600;
}
.node-meta .v {
	font-size: 0.7rem;
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
.insight-icon {
	font-size: 1.1rem;
}
</style>
