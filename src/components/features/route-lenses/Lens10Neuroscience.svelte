<script lang="ts">
/**
 * Lens 10: 神经科学、脑网络与计算神经科学路线 (P-FIT Frontoparietal Synchrony)
 * 揭示：生物智能依赖全脑大尺度功能连接网络。顶额叶整合理论 (P-FIT) 揭示智力与神经元振荡同调性 (Phase Synchrony) 的紧密联系。
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

// Cognitive Load Level: 1 (Resting) to 3 (Intense Abstract Reasoning)
let taskLoad = $state(1);
let inhibitoryBalance = $state(0.7); // E/I (Excitatory/Inhibitory) balance

// Simulated brain network hubs
interface BrainHub {
	id: string;
	nameZh: string;
	nameEn: string;
	roleZh: string;
	x: number;
	y: number;
}

const HUBS: BrainHub[] = [
	{
		id: "DLPFC",
		nameZh: "背外侧前额叶",
		nameEn: "dlPFC",
		roleZh: "工作记忆与目标控制",
		x: 90,
		y: 70,
	},
	{
		id: "ACC",
		nameZh: "前扣带回",
		nameEn: "ACC",
		roleZh: "冲突监测与注意分配",
		x: 170,
		y: 60,
	},
	{
		id: "IPL",
		nameZh: "下顶叶小叶",
		nameEn: "IPL",
		roleZh: "感觉抽象整合与符号处理",
		x: 260,
		y: 110,
	},
	{
		id: "Temp",
		nameZh: "颞叶感觉联合区",
		nameEn: "Temporal",
		roleZh: "语义联想匹配",
		x: 150,
		y: 150,
	},
	{
		id: "Visual",
		nameZh: "初级视皮层",
		nameEn: "Occipital",
		roleZh: "特征解构输入",
		x: 310,
		y: 160,
	},
];

// Connectivity & Coherence
const networkState = $derived.by(() => {
	const synchronyGamma = taskLoad === 1 ? 0.28 : taskLoad === 2 ? 0.65 : 0.94;
	const neuralEfficiency = (
		synchronyGamma *
		(1.1 - Math.abs(inhibitoryBalance - 0.7))
	).toFixed(2);
	const activeParietoFrontal = taskLoad >= 2;

	return {
		synchrony: Math.round(synchronyGamma * 100),
		efficiency: neuralEfficiency,
		activeParietoFrontal,
	};
});
</script>

<div class="lens-container">
	<div class="lens-header">
		<span class="lens-tag">COGNITIVE LENS 10</span>
		<h3 class="lens-title">{english ? "P-FIT Parieto-Frontal Synchrony Simulator" : "顶额整合理论 (P-FIT) 与神经同调沙盒"}</h3>
		<p class="lens-subtitle">
			{english
				? "Jung & Haier: General intelligence emerges not from a single 'smart' brain area, but from fast, synchronized parieto-frontal functional connectivity and neural efficiency."
				: "容格与海尔 (P-FIT 假说)：人类智力并不源于单一脑区，而是取决于感觉联合区（顶叶）与控制中枢（额叶）之间高频振荡的大尺度功能同调连接与神经效率假说。"}
		</p>
	</div>

	<!-- Interactive Controls -->
	<div class="controls-bar">
		<div class="task-load-selector">
			<span class="ctrl-title">{english ? "Cognitive Task Demand:" : "认知任务负荷水平："}</span>
			<div class="btn-group">
				<button
					type="button"
					class="load-btn"
					class:active={taskLoad === 1}
					onclick={() => taskLoad = 1}
				>
					{english ? "Resting State (Default Mode)" : "静息态 (默认网络活跃)"}
				</button>
				<button
					type="button"
					class="load-btn"
					class:active={taskLoad === 2}
					onclick={() => taskLoad = 2}
				>
					{english ? "Fluid Reasoning (Working Memory)" : "流体推理 (双任务负荷)"}
				</button>
				<button
					type="button"
					class="load-btn intense"
					class:active={taskLoad === 3}
					onclick={() => taskLoad = 3}
				>
					{english ? "High-Complexity Insight" : "高复杂度顿悟推演 (P-FIT 强同调)"}
				</button>
			</div>
		</div>
	</div>

	<!-- Brain Connectome Visualizer -->
	<div class="connectome-stage">
		<div class="svg-container">
			<svg viewBox="0 0 380 220" class="brain-svg">
				<!-- Brain outline contour -->
				<path
					d="M 60 120 C 50 60, 120 20, 240 25 C 330 30, 360 100, 350 160 C 340 200, 290 205, 230 190 C 200 185, 140 195, 100 185 C 60 175, 50 150, 60 120 Z"
					fill="rgba(120, 140, 130, 0.08)"
					stroke="rgba(120, 140, 130, 0.3)"
					stroke-width="1.5"
					stroke-dasharray="4 2"
				/>

				<!-- Functional Sync Edges -->
				<!-- Visual -> Temp -->
				<line x1="310" y1="160" x2="150" y2="150" stroke="rgba(120, 140, 130, 0.4)" stroke-width="2" />
				<!-- Temp -> IPL -->
				<line x1="150" y1="150" x2="260" y2="110" stroke="rgba(120, 140, 130, 0.4)" stroke-width="2" />

				<!-- Parieto-Frontal Hub (IPL <-> dlPFC / ACC) -->
				<line
					x1="260"
					y1="110"
					x2="90"
					y2="70"
					class="sync-edge"
					class:active={networkState.activeParietoFrontal}
					stroke-width={taskLoad === 3 ? "4" : "2"}
				/>
				<line
					x1="260"
					y1="110"
					x2="170"
					y2="60"
					class="sync-edge"
					class:active={networkState.activeParietoFrontal}
					stroke-width={taskLoad === 3 ? "3" : "1.5"}
				/>
				<line
					x1="170"
					y1="60"
					x2="90"
					y2="70"
					stroke="rgba(120, 140, 130, 0.5)"
					stroke-width="2"
				/>

				<!-- Hub Nodes -->
				{#each HUBS as hub}
					<g class="hub-group" transform="translate({hub.x}, {hub.y})">
						<circle
							r={taskLoad === 3 ? "12" : "9"}
							class="hub-circle"
							class:pulsing={taskLoad >= 2 && (hub.id === "DLPFC" || hub.id === "IPL")}
						/>
						<text y="-14" text-anchor="middle" class="hub-text">{hub.nameEn}</text>
						<text y="22" text-anchor="middle" class="hub-role">{hub.roleZh}</text>
					</g>
				{/each}
			</svg>
		</div>

		<!-- Coherence Readouts -->
		<div class="readouts-panel">
			<div class="readout-card">
				<span class="readout-label">{english ? "Phase Synchrony (Gamma Band 40Hz)" : "伽马波段相位同调率 (40Hz)"}</span>
				<div class="readout-bar-bg">
					<div class="readout-bar-fill" style="width: {networkState.synchrony}%;"></div>
				</div>
				<span class="readout-num">{networkState.synchrony}% 同调性</span>
			</div>

			<div class="readout-card">
				<span class="readout-label">{english ? "Neural Efficiency (Energy per Solution)" : "神经效率比 (解题能耗产出比)"}</span>
				<span class="readout-efficiency">{networkState.efficiency}x</span>
				<span class="readout-sub">高智力系统消耗更少代谢葡萄糖</span>
			</div>
		</div>
	</div>

	<!-- Insight Box -->
	<div class="insight-box">
		<div class="insight-icon">🧠</div>
		<div class="insight-text">
			<strong>{english ? "Neural Efficiency Hypothesis:" : "神经科学核心启示："}</strong>
			{english
				? "Brains are not uniform tensor arrays. High intelligence is characterized by higher metabolic efficiency—pruning noisy pathways, maintaining crisp E/I balance, and selectively synchronizing parieto-frontal circuits only when novel reasoning is required."
				: "人脑不是无差别的张量矩阵堆叠。高智力表现的核心特征恰恰是‘神经效率’：在低负荷时保持极低背景能耗，仅在遭遇高阶抽象问题时，精准诱发顶叶感知与前额叶执行回路的瞬态强相位锁定。"}
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
.ctrl-title {
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
.load-btn {
	background: rgba(0, 0, 0, 0.04);
	border: 1px solid rgba(0, 0, 0, 0.08);
	color: inherit;
	padding: 0.4rem 0.8rem;
	border-radius: 6px;
	font-size: 0.8rem;
	cursor: pointer;
	transition: all 0.2s;
}
:global(.dark) .load-btn {
	background: rgba(255, 255, 255, 0.04);
	border-color: rgba(255, 255, 255, 0.08);
}
.load-btn.active {
	background: var(--brand-ink, #2d3748);
	color: #fff;
}
:global(.dark) .load-btn.active {
	background: #3b5043;
}
.connectome-stage {
	display: grid;
	grid-template-columns: 1.5fr 1fr;
	gap: 1.25rem;
	background: rgba(255, 255, 255, 0.6);
	border: 1px solid rgba(0, 0, 0, 0.06);
	border-radius: 8px;
	padding: 1.25rem;
	align-items: center;
	margin-bottom: 1.25rem;
}
:global(.dark) .connectome-stage {
	background: rgba(0, 0, 0, 0.25);
	border-color: rgba(255, 255, 255, 0.06);
}
@media (max-width: 700px) {
	.connectome-stage {
		grid-template-columns: 1fr;
	}
}
.brain-svg {
	width: 100%;
	height: auto;
}
.hub-circle {
	fill: var(--brand-clay, #8a7356);
	stroke: #fff;
	stroke-width: 2;
	transition: all 0.3s ease;
}
.hub-circle.pulsing {
	fill: #0284c7;
	filter: drop-shadow(0 0 6px rgba(2, 132, 199, 0.6));
}
.hub-text {
	font-size: 10px;
	font-weight: 700;
	fill: var(--brand-ink, #2d3748);
}
:global(.dark) .hub-text { fill: #f1f5f9; }
.hub-role {
	font-size: 8px;
	fill: var(--text-muted, #718096);
}
.sync-edge {
	stroke: rgba(120, 140, 130, 0.4);
	transition: all 0.3s;
}
.sync-edge.active {
	stroke: #0284c7;
	stroke-dasharray: 4 2;
	animation: pulseLine 1.5s infinite linear;
}
@keyframes pulseLine {
	to { stroke-dashoffset: -12; }
}
.readouts-panel {
	display: flex;
	flex-direction: column;
	gap: 1rem;
}
.readout-card {
	padding: 0.85rem;
	background: rgba(0, 0, 0, 0.02);
	border-radius: 6px;
	border: 1px solid rgba(0, 0, 0, 0.05);
}
:global(.dark) .readout-card {
	background: rgba(255, 255, 255, 0.02);
	border-color: rgba(255, 255, 255, 0.05);
}
.readout-label {
	font-size: 0.72rem;
	color: var(--text-muted, #718096);
	display: block;
	margin-bottom: 0.35rem;
}
.readout-bar-bg {
	width: 100%;
	height: 8px;
	background: rgba(0, 0, 0, 0.08);
	border-radius: 4px;
	overflow: hidden;
	margin-bottom: 0.25rem;
}
:global(.dark) .readout-bar-bg {
	background: rgba(255, 255, 255, 0.1);
}
.readout-bar-fill {
	height: 100%;
	background: #0284c7;
	transition: width 0.4s ease;
}
.readout-num {
	font-size: 0.78rem;
	font-weight: 700;
	color: #0284c7;
}
.readout-efficiency {
	font-size: 1.4rem;
	font-weight: 700;
	font-family: monospace;
	color: var(--brand-clay, #8a7356);
	display: block;
}
.readout-sub {
	font-size: 0.72rem;
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
