<script lang="ts">
import { onMount } from "svelte";
import {
	DIVERGENCE_AXES,
	INTELLIGENCE_ROUTES,
	ROUTE_AXIS_MATRIX,
	type RouteItem,
} from "@/data/intelligenceRoutes";

let english = $state(false);
let axisX = $state(1); // Default X: 怎样解释行为？ (内部表征 vs 实时环境耦合)
let axisY = $state(3); // Default Y: 智能的单位是什么？ (孤立大脑 vs 具身-工具-社会系统)
let selectedRouteId = $state("embodied-dynamical");
let activeClusterFilter = $state<string>("all");

const PRESETS = [
	{
		x: 1,
		y: 3,
		labelZh: "🧭 具身与延展象限 (表征计算 × 系统边界)",
		labelEn: "🧭 Embodied & Extended Quadrant",
	},
	{
		x: 4,
		y: 0,
		labelZh: "⚡ 泛化与先验象限 (统计缩放/先验 × 静态/泛化效率)",
		labelEn: "⚡ Generalization & Priors Quadrant",
	},
	{
		x: 2,
		y: 7,
		labelZh: "⚖️ 目的与智慧象限 (外生奖励/自维持 × 工具/价值智慧)",
		labelEn: "⚖️ Teleology & Wisdom Quadrant",
	},
	{
		x: 5,
		y: 6,
		labelZh: "🕯️ 架构与意识象限 (单一主算法/多模块 × 现象意识)",
		labelEn: "🕯️ Architecture & Consciousness Quadrant",
	},
];

const CLUSTER_COLORS: Record<RouteItem["cluster"], string> = {
	measurement: "#0d9488",
	representation: "#0284c7",
	embodiment: "#16a34a",
	collective: "#d97706",
	mind: "#7c3aed",
};

const currentXAxis = $derived(DIVERGENCE_AXES[axisX]);
const currentYAxis = $derived(DIVERGENCE_AXES[axisY]);

const plottedRoutes = $derived.by(() => {
	return INTELLIGENCE_ROUTES.map((route) => {
		const coords = ROUTE_AXIS_MATRIX[route.id] ?? [0, 0, 0, 0, 0, 0, 0, 0];
		const vx = coords[axisX] ?? 0;
		const vy = coords[axisY] ?? 0;
		// Convert [-1, 1] to percentage [8%, 92%]
		const leftPct = Number((50 + vx * 40).toFixed(1));
		const topPct = Number((50 - vy * 40).toFixed(1));
		return {
			route,
			vx,
			vy,
			leftPct,
			topPct,
			color: CLUSTER_COLORS[route.cluster],
			dimmed:
				activeClusterFilter !== "all" && route.cluster !== activeClusterFilter,
		};
	});
});

const selectedPoint = $derived(
	plottedRoutes.find((p) => p.route.id === selectedRouteId) ?? plottedRoutes[0],
);

function selectPreset(x: number, y: number) {
	axisX = x;
	axisY = y;
}

onMount(() => {
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
</script>

<div class="compass-shell">
	<div class="compass-header">
		<div class="compass-title-block">
			<span class="compass-kicker">
				{english ? "2D PARADIGM GRAVITY COMPASS" : "二维思想引力象限仪 · 动态分歧投影"}
			</span>
			<h3>
				{english
					? "Project All 20 Routes Across Any Two Philosophical Axes"
					: "任选两条根本分歧轴，观察 20 大智能路线如何在思想坐标系中聚类与对峙"}
			</h3>
		</div>

		<div class="preset-pills">
			{#each PRESETS as preset}
				<button
					type="button"
					class="preset-btn"
					class:active={axisX === preset.x && axisY === preset.y}
					onclick={() => selectPreset(preset.x, preset.y)}
				>
					{english ? preset.labelEn : preset.labelZh}
				</button>
			{/each}
		</div>
	</div>

	<!-- Axis Selectors & Cluster Filter -->
	<div class="axis-controls-bar">
		<label class="axis-select-group">
			<span class="axis-tag">X 轴 (水平分歧)</span>
			<select bind:value={axisX} class="axis-select">
				{#each DIVERGENCE_AXES as ax, idx}
					<option value={idx}>
						{idx + 1}. {english ? ax.nameEn : ax.nameZh} ({english ? ax.poleAEn : ax.poleAZh} ↔ {english ? ax.poleBEn : ax.poleBZh})
					</option>
				{/each}
			</select>
		</label>

		<label class="axis-select-group">
			<span class="axis-tag">Y 轴 (垂直分歧)</span>
			<select bind:value={axisY} class="axis-select">
				{#each DIVERGENCE_AXES as ax, idx}
					<option value={idx}>
						{idx + 1}. {english ? ax.nameEn : ax.nameZh} ({english ? ax.poleAEn : ax.poleAZh} ↔ {english ? ax.poleBEn : ax.poleBZh})
					</option>
				{/each}
			</select>
		</label>
	</div>

	<div class="compass-main-grid">
		<!-- 2D Coordinate Plane -->
		<div class="plane-stage">
			<!-- Quadrant background grid & crosshair -->
			<div class="plane-grid-lines" aria-hidden="true">
				<div class="crosshair-h"></div>
				<div class="crosshair-v"></div>
				<div class="orbit-ring r1"></div>
				<div class="orbit-ring r2"></div>
			</div>

			<!-- Axis Pole Labels -->
			<div class="pole-badge top">
				↑ +Y: {english ? currentYAxis.poleBEn : currentYAxis.poleBZh}
			</div>
			<div class="pole-badge bottom">
				↓ -Y: {english ? currentYAxis.poleAEn : currentYAxis.poleAZh}
			</div>
			<div class="pole-badge left">
				← -X: {english ? currentXAxis.poleAEn : currentXAxis.poleAZh}
			</div>
			<div class="pole-badge right">
				+X: {english ? currentXAxis.poleBEn : currentXAxis.poleBZh} →
			</div>

			<!-- 20 Orbiting Route Nodes -->
			{#each plottedRoutes as item (item.route.id)}
				<button
					type="button"
					class="route-star-node"
					class:selected={selectedRouteId === item.route.id}
					class:dimmed={item.dimmed}
					style="left: {item.leftPct}%; top: {item.topPct}%; --node-color: {item.color};"
					onclick={() => (selectedRouteId = item.route.id)}
					onmouseenter={() => (selectedRouteId = item.route.id)}
					aria-label={`${item.route.roman} ${item.route.nameZh}`}
				>
					<span class="star-dot">{item.route.roman}</span>
					<span class="star-label">{english ? item.route.nameEn.split(",")[0] : item.route.nameZh.replace("路线", "")}</span>
				</button>
			{/each}
		</div>

		<!-- Right Inspector Panel -->
		<aside class="inspector-card" style="--active-color: {selectedPoint.color};">
			<div class="insp-top">
				<span class="insp-roman">ROUTE {selectedPoint.route.roman}</span>
				<span class="insp-cluster">{english ? selectedPoint.route.clusterEn : selectedPoint.route.clusterZh}</span>
			</div>

			<h4 class="insp-title">
				{english ? selectedPoint.route.nameEn : selectedPoint.route.nameZh}
			</h4>

			<p class="insp-thesis">
				{english ? selectedPoint.route.coreThesisEn : selectedPoint.route.coreThesisZh}
			</p>

			<div class="coord-readout-list">
				<div class="coord-item">
					<span class="coord-axis">X · {english ? currentXAxis.nameEn : currentXAxis.nameZh}</span>
					<strong class="coord-val">
						{selectedPoint.vx > 0 ? `+${selectedPoint.vx}` : selectedPoint.vx}
						({selectedPoint.vx >= 0
							? english ? currentXAxis.poleBEn : currentXAxis.poleBZh
							: english ? currentXAxis.poleAEn : currentXAxis.poleAZh})
					</strong>
				</div>
				<div class="coord-item">
					<span class="coord-axis">Y · {english ? currentYAxis.nameEn : currentYAxis.nameZh}</span>
					<strong class="coord-val">
						{selectedPoint.vy > 0 ? `+${selectedPoint.vy}` : selectedPoint.vy}
						({selectedPoint.vy >= 0
							? english ? currentYAxis.poleBEn : currentYAxis.poleBZh
							: english ? currentYAxis.poleAEn : currentYAxis.poleAZh})
					</strong>
				</div>
			</div>

			<a href={`/routes/${selectedPoint.route.id}/`} class="insp-cta">
				{english ? "Enter Route Monograph & Interactive Lens →" : "进入该路线专题页与交互透镜 →"}
			</a>
		</aside>
	</div>
</div>

<style>
.compass-shell {
	margin: 2rem 0 2.5rem;
	padding: 1.5rem;
	border-radius: 16px;
	border: 1px solid rgba(47, 107, 86, 0.22);
	background: linear-gradient(145deg, rgba(255, 255, 255, 0.9), rgba(246, 244, 238, 0.92));
	box-shadow: 0 12px 32px rgba(20, 30, 24, 0.05);
}
:global(.dark) .compass-shell {
	background: linear-gradient(145deg, rgba(22, 28, 25, 0.95), rgba(16, 20, 18, 0.95));
	border-color: rgba(120, 160, 140, 0.2);
}
.compass-header {
	display: flex;
	flex-direction: column;
	gap: 0.85rem;
	margin-bottom: 1.1rem;
}
.compass-kicker {
	font-size: 0.72rem;
	font-weight: 700;
	letter-spacing: 0.14em;
	color: var(--brand-accent, #2f6b56);
	text-transform: uppercase;
}
.compass-title-block h3 {
	font-size: 1.2rem;
	margin: 0.2rem 0 0;
	font-weight: 700;
}
.preset-pills {
	display: flex;
	flex-wrap: wrap;
	gap: 0.5rem;
}
.preset-btn {
	padding: 0.38rem 0.75rem;
	border-radius: 999px;
	border: 1px solid rgba(47, 107, 86, 0.22);
	background: rgba(255, 255, 255, 0.7);
	color: inherit;
	font-size: 0.78rem;
	font-weight: 600;
	cursor: pointer;
	transition: all 0.2s ease;
}
:global(.dark) .preset-btn {
	background: rgba(255, 255, 255, 0.04);
	border-color: rgba(255, 255, 255, 0.12);
}
.preset-btn.active {
	background: var(--brand-accent, #2f6b56);
	border-color: var(--brand-accent, #2f6b56);
	color: #fff;
}
.axis-controls-bar {
	display: grid;
	grid-template-columns: 1fr 1fr;
	gap: 1rem;
	margin-bottom: 1.25rem;
	padding: 0.85rem 1rem;
	border-radius: 10px;
	background: rgba(0, 0, 0, 0.03);
}
:global(.dark) .axis-controls-bar {
	background: rgba(255, 255, 255, 0.03);
}
@media (max-width: 720px) {
	.axis-controls-bar { grid-template-columns: 1fr; }
}
.axis-select-group {
	display: flex;
	flex-direction: column;
	gap: 0.3rem;
}
.axis-tag {
	font-size: 0.72rem;
	font-weight: 700;
	color: var(--brand-clay, #8a7356);
	text-transform: uppercase;
}
.axis-select {
	padding: 0.45rem 0.65rem;
	border-radius: 6px;
	border: 1px solid rgba(0, 0, 0, 0.14);
	background: rgba(255, 255, 255, 0.9);
	color: inherit;
	font-size: 0.82rem;
}
:global(.dark) .axis-select {
	background: #1c2220;
	border-color: rgba(255, 255, 255, 0.15);
}
.compass-main-grid {
	display: grid;
	grid-template-columns: 1.45fr 1fr;
	gap: 1.25rem;
	align-items: stretch;
}
@media (max-width: 900px) {
	.compass-main-grid { grid-template-columns: 1fr; }
}
.plane-stage {
	position: relative;
	height: 410px;
	border-radius: 12px;
	background: radial-gradient(circle at center, rgba(47, 107, 86, 0.06) 0%, rgba(0, 0, 0, 0.02) 100%);
	border: 1px solid rgba(47, 107, 86, 0.18);
	overflow: hidden;
}
:global(.dark) .plane-stage {
	background: radial-gradient(circle at center, rgba(47, 107, 86, 0.14) 0%, rgba(0, 0, 0, 0.4) 100%);
}
.plane-grid-lines {
	position: absolute;
	inset: 0;
	pointer-events: none;
}
.crosshair-h {
	position: absolute;
	top: 50%;
	left: 0;
	right: 0;
	height: 1px;
	background: rgba(120, 140, 130, 0.3);
}
.crosshair-v {
	position: absolute;
	left: 50%;
	top: 0;
	bottom: 0;
	width: 1px;
	background: rgba(120, 140, 130, 0.3);
}
.orbit-ring {
	position: absolute;
	top: 50%;
	left: 50%;
	transform: translate(-50%, -50%);
	border-radius: 50%;
	border: 1px dashed rgba(120, 140, 130, 0.2);
}
.orbit-ring.r1 { width: 42%; height: 42%; }
.orbit-ring.r2 { width: 78%; height: 78%; }

.pole-badge {
	position: absolute;
	font-size: 0.7rem;
	font-weight: 700;
	padding: 0.2rem 0.55rem;
	border-radius: 4px;
	background: rgba(255, 255, 255, 0.82);
	color: var(--brand-ink, #334155);
	pointer-events: none;
	z-index: 2;
	max-width: 44%;
	white-space: nowrap;
	overflow: hidden;
	text-overflow: ellipsis;
}
:global(.dark) .pole-badge {
	background: rgba(20, 24, 22, 0.85);
	color: #cbd5e1;
}
.pole-badge.top { top: 8px; left: 50%; transform: translateX(-50%); }
.pole-badge.bottom { bottom: 8px; left: 50%; transform: translateX(-50%); }
.pole-badge.left { left: 8px; top: 50%; transform: translateY(-50%); }
.pole-badge.right { right: 8px; top: 50%; transform: translateY(-50%); }

.route-star-node {
	position: absolute;
	transform: translate(-50%, -50%);
	display: inline-flex;
	align-items: center;
	gap: 0.3rem;
	padding: 0.2rem 0.48rem 0.2rem 0.25rem;
	border-radius: 999px;
	background: rgba(255, 255, 255, 0.92);
	border: 1.5px solid var(--node-color, #0284c7);
	color: var(--brand-ink, #1e293b);
	cursor: pointer;
	z-index: 3;
	transition:
		left 0.55s cubic-bezier(0.22, 1, 0.36, 1),
		top 0.55s cubic-bezier(0.22, 1, 0.36, 1),
		transform 0.2s ease,
		box-shadow 0.2s ease;
	box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
}
:global(.dark) .route-star-node {
	background: rgba(24, 30, 27, 0.92);
	color: #f1f5f9;
}
.route-star-node:hover,
.route-star-node.selected {
	z-index: 10;
	transform: translate(-50%, -50%) scale(1.14);
	background: var(--node-color, #0284c7);
	color: #fff;
	box-shadow: 0 0 18px var(--node-color, rgba(2, 132, 199, 0.45));
}
.star-dot {
	width: 18px;
	height: 18px;
	border-radius: 50%;
	background: var(--node-color, #0284c7);
	color: #fff;
	font-size: 0.65rem;
	font-weight: 700;
	display: inline-flex;
	align-items: center;
	justify-content: center;
	font-family: monospace;
}
.route-star-node.selected .star-dot,
.route-star-node:hover .star-dot {
	background: rgba(255, 255, 255, 0.25);
}
.star-label {
	font-size: 0.7rem;
	font-weight: 600;
	white-space: nowrap;
}
@media (max-width: 600px) {
	.star-label { display: none; }
	.route-star-node { padding: 0.2rem; }
}

.inspector-card {
	display: flex;
	flex-direction: column;
	justify-content: space-between;
	padding: 1.25rem;
	border-radius: 12px;
	background: rgba(255, 255, 255, 0.82);
	border: 1px solid rgba(0, 0, 0, 0.08);
	border-top: 4px solid var(--active-color, #2f6b56);
}
:global(.dark) .inspector-card {
	background: rgba(255, 255, 255, 0.03);
	border-color: rgba(255, 255, 255, 0.08);
	border-top-color: var(--active-color, #2f6b56);
}
.insp-top {
	display: flex;
	justify-content: space-between;
	align-items: center;
	font-size: 0.74rem;
	font-weight: 700;
}
.insp-roman {
	font-family: monospace;
	color: var(--active-color, #2f6b56);
}
.insp-cluster {
	color: var(--text-muted, #64748b);
}
.insp-title {
	font-size: 1.15rem;
	margin: 0.5rem 0;
	font-weight: 700;
}
.insp-thesis {
	font-size: 0.84rem;
	line-height: 1.55;
	color: var(--text-muted, #475569);
	margin-bottom: 1rem;
}
:global(.dark) .insp-thesis { color: #cbd5e1; }
.coord-readout-list {
	display: flex;
	flex-direction: column;
	gap: 0.6rem;
	margin-bottom: 1.1rem;
	padding: 0.75rem;
	border-radius: 8px;
	background: rgba(0, 0, 0, 0.03);
}
:global(.dark) .coord-readout-list { background: rgba(255, 255, 255, 0.04); }
.coord-item {
	display: flex;
	flex-direction: column;
	gap: 0.15rem;
}
.coord-axis {
	font-size: 0.7rem;
	color: var(--text-muted, #64748b);
}
.coord-val {
	font-size: 0.8rem;
	font-family: monospace;
}
.insp-cta {
	display: inline-flex;
	align-items: center;
	justify-content: center;
	padding: 0.65rem 1rem;
	border-radius: 8px;
	background: var(--active-color, #2f6b56);
	color: #fff;
	font-size: 0.84rem;
	font-weight: 700;
	text-decoration: none;
	transition: opacity 0.2s;
}
.insp-cta:hover {
	opacity: 0.92;
}
</style>
