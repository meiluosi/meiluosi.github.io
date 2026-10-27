<script lang="ts">
import { onMount } from "svelte";

/**
 * Lens 12: 具身认知、生成认知与动态系统路线 (Passive Dynamic Walking & 4E Cognition)
 * 揭示：智能并不完全居于大脑中央处理器中。身体形态学结构(Morphology)、物理重力与环境的动态耦合本身即可卸载大量计算负荷。
 */
let english = $state(false);

// Slope angle in degrees (optimal around 3.5 - 4.5 deg for McGeer's passive walker)
let slopeAngle = $state(4.0);
// Joint stiffness / damping ratio
let legStiffness = $state(0.5);
// Continuous phase animation
let phase = $state(0);
let isPlaying = $state(true);

const gaitMetrics = $derived.by(() => {
	const angleDiff = Math.abs(slopeAngle - 4.0);
	const stiffDiff = Math.abs(legStiffness - 0.5);
	const couplingError = angleDiff * 0.25 + stiffDiff * 1.2;

	if (couplingError < 0.35) {
		return {
			statusZh: "稳态极限环步态 (Limit-Cycle Attractor)",
			statusEn: "Stable Limit-Cycle Gait",
			cpuLoad: "0 FLOPs (零大脑控制)",
			cpuLoadEn: "0 FLOPs (Zero Brain Control)",
			isStable: true,
			cadence: Number((1.2 + slopeAngle * 0.1).toFixed(2)),
		};
	}
	if (slopeAngle < 2.0) {
		return {
			statusZh: "动能耗散停摆 (Damped Halt)",
			statusEn: "Kinetic Energy Depleted",
			cpuLoad: "0 FLOPs",
			cpuLoadEn: "0 FLOPs",
			isStable: false,
			cadence: 0,
		};
	}
	return {
		statusZh: "混沌失稳倾覆 (Chaotic Bifurcation)",
		statusEn: "Chaotic Fall / Bifurcation",
		cpuLoad: "0 FLOPs",
		cpuLoadEn: "0 FLOPs",
		isStable: false,
		cadence: 0,
	};
});

// Leg kinematics derived from phase
const kinematics = $derived.by(() => {
	if (!gaitMetrics.isStable) {
		return {
			hipX: slopeAngle < 2.0 ? -5 : 28,
			hipY: slopeAngle < 2.0 ? -66 : -42,
			swingFootX: slopeAngle < 2.0 ? 8 : -38,
			swingFootY: slopeAngle < 2.0 ? -2 : -22,
		};
	}
	const swing = Math.sin(phase);
	const hipX = swing * 12;
	const hipY = -65 + Math.abs(swing) * 4;
	const swingFootX = -swing * 26;
	const swingFootY = -4 - Math.cos(phase) * 8;
	return { hipX, hipY, swingFootX, swingFootY };
});

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

	let rafId = 0;
	const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
	if (reduced) isPlaying = false;

	const tick = () => {
		if (isPlaying && gaitMetrics.isStable) {
			phase = (phase + 0.055 * gaitMetrics.cadence) % (Math.PI * 2);
		}
		rafId = requestAnimationFrame(tick);
	};
	rafId = requestAnimationFrame(tick);

	return () => {
		obs.disconnect();
		cancelAnimationFrame(rafId);
	};
});
</script>

<div class="lens-container">
	<div class="lens-header">
		<div class="header-top-row">
			<span class="lens-tag">COGNITIVE LENS 12</span>
			<button type="button" class="anim-toggle" onclick={() => (isPlaying = !isPlaying)}>
				{isPlaying
					? english ? "⏸ Pause Dynamics" : "⏸ 暂停步态律动"
					: english ? "▶ Resume Dynamics" : "▶ 开启 60fps 物理律动"}
			</button>
		</div>
		<h3 class="lens-title">{english ? "Passive Dynamic Walker & Morphological Computation" : "被动动态行走与形态学计算沙盒"}</h3>
		<p class="lens-subtitle">
			{english
				? "Tad McGeer, Rodney Brooks & Varela: A biped mechanism without motors, sensors, or microprocessors can walk smoothly down a gentle slope purely through gravity-body coupling."
				: "麦吉尔 (Tad McGeer)、布鲁克斯与瓦雷拉 (4E 认知)：没有任何电机、传感器或芯片的双足机械，仅凭身体摆锤结构与斜坡重力的物理耦合，即可涌现出自然的稳态行走步态。"}
		</p>
	</div>

	<!-- Interactive Controls -->
	<div class="controls-grid">
		<div class="ctrl-box">
			<div class="label-row">
				<span>{english ? "Incline Slope Angle (γ)" : "环境斜坡倾角 (γ)"}</span>
				<strong class="val-pill">{slopeAngle.toFixed(1)}°</strong>
			</div>
			<input type="range" min="0.5" max="8.0" step="0.5" bind:value={slopeAngle} class="slider" />
			<span class="sub-hint">{english ? "Golden resonance slope: 3.5° ~ 4.5°" : "黄金共振倾角：3.5° ~ 4.5°"}</span>
		</div>

		<div class="ctrl-box">
			<div class="label-row">
				<span>{english ? "Hip Passive Spring Stiffness (k)" : "髋关节被动弹性耦合度 (k)"}</span>
				<strong class="val-pill">{legStiffness.toFixed(2)}</strong>
			</div>
			<input type="range" min="0.1" max="1.0" step="0.05" bind:value={legStiffness} class="slider" />
			<span class="sub-hint">{english ? "Natural pendulum frequency: 0.45 ~ 0.55" : "匹配摆腿固有频率：0.45 ~ 0.55"}</span>
		</div>
	</div>

	<!-- Visual Slope & Walker Simulation -->
	<div class="stage-box" class:stable={gaitMetrics.isStable}>
		<svg viewBox="0 0 420 180" class="walker-svg">
			<!-- Inclined Ground Plane -->
			<g transform="rotate({slopeAngle}, 60, 130)">
				<line x1="20" y1="130" x2="390" y2="130" stroke="var(--brand-clay, #8a7356)" stroke-width="3" />
				<line x1="20" y1="134" x2="390" y2="134" stroke="rgba(138, 115, 86, 0.3)" stroke-width="6" stroke-dasharray="8 4" />

				<!-- Biped Linkage on Slope -->
				<g transform="translate(195, 130)">
					<!-- Stance Leg -->
					<line
						x1="0"
						y1="0"
						x2={kinematics.hipX}
						y2={kinematics.hipY}
						stroke="#0284c7"
						stroke-width="4.5"
						stroke-linecap="round"
					/>
					<!-- Swing Leg -->
					<line
						x1={kinematics.hipX}
						y1={kinematics.hipY}
						x2={kinematics.swingFootX}
						y2={kinematics.swingFootY}
						stroke="#ea580c"
						stroke-width="4.5"
						stroke-linecap="round"
					/>
					<!-- Hip Mass -->
					<circle cx={kinematics.hipX} cy={kinematics.hipY} r="9" fill="var(--brand-ink, #2d3748)" />
					<!-- Curved Foot Arcs -->
					<path d="M -9 0 Q 0 4 9 0" fill="none" stroke="#0284c7" stroke-width="2.5" />
				</g>
			</g>
			<text x="20" y="28" font-size="11" fill="var(--text-muted, #64748b)">
				{english ? `Phase Portrait: ${gaitMetrics.statusEn}` : `相空间轨迹：${gaitMetrics.statusZh}`}
			</text>
		</svg>

		<div class="metrics-strip">
			<div class="m-item">
				<span class="k">{english ? "Central Brain Compute" : "中央大脑算力消耗"}</span>
				<strong class="v zero">{english ? gaitMetrics.cpuLoadEn : gaitMetrics.cpuLoad}</strong>
			</div>
			<div class="m-item">
				<span class="k">{english ? "Attractor Regime" : "步态吸引子状态"}</span>
				<strong class="v" class:ok={gaitMetrics.isStable}>{english ? gaitMetrics.statusEn : gaitMetrics.statusZh}</strong>
			</div>
			<div class="m-item">
				<span class="k">{english ? "Resonant Cadence" : "步频共振节拍"}</span>
				<strong class="v">{gaitMetrics.cadence.toFixed(2)} Hz</strong>
			</div>
		</div>
	</div>

	<div class="insight-box">
		<div class="insight-icon">🦿</div>
		<div class="insight-text">
			<strong>{english ? "Morphological Computation:" : "形态学计算与具身启示："}</strong>
			{english
				? "Before evolving a gigantic cortex to compute joint trajectories via inverse kinematics, nature offloads control into compliant tendons, pendular dynamics, and environmental affordances."
				: "传统符号主义试图用中央处理器精确解算每个关节的逆运动学微分方程；而具身认知指出：顺应性的肌腱、摆锤形态与地面反作用力的物理闭环，本身就在以零算力完成精妙的‘形态学计算’。"}
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
.header-top-row {
	display: flex;
	justify-content: space-between;
	align-items: center;
}
.anim-toggle {
	font-size: 0.74rem;
	padding: 0.22rem 0.6rem;
	border-radius: 99px;
	border: 1px solid rgba(0, 0, 0, 0.14);
	background: rgba(0, 0, 0, 0.03);
	color: inherit;
	cursor: pointer;
}
:global(.dark) .anim-toggle {
	border-color: rgba(255, 255, 255, 0.14);
	background: rgba(255, 255, 255, 0.05);
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
	.controls-grid { grid-template-columns: 1fr; }
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
	color: var(--brand-clay, #8a7356);
}
.slider {
	width: 100%;
	accent-color: var(--brand-clay, #8a7356);
}
.sub-hint {
	font-size: 0.72rem;
	color: var(--text-muted, #718096);
}
.stage-box {
	background: rgba(255, 255, 255, 0.6);
	border: 1px solid rgba(0, 0, 0, 0.08);
	border-radius: 8px;
	padding: 1rem;
	margin-bottom: 1.25rem;
}
:global(.dark) .stage-box {
	background: rgba(0, 0, 0, 0.25);
	border-color: rgba(255, 255, 255, 0.08);
}
.stage-box.stable {
	border-color: #10b981;
}
.walker-svg {
	width: 100%;
	max-height: 180px;
}
.metrics-strip {
	display: grid;
	grid-template-columns: repeat(3, 1fr);
	gap: 0.75rem;
	padding-top: 0.75rem;
	border-top: 1px dashed rgba(120, 140, 130, 0.25);
}
@media (max-width: 600px) {
	.metrics-strip { grid-template-columns: 1fr; }
}
.m-item {
	display: flex;
	flex-direction: column;
}
.m-item .k {
	font-size: 0.7rem;
	color: var(--text-muted, #718096);
}
.m-item .v {
	font-size: 0.88rem;
	font-family: monospace;
}
.m-item .v.zero { color: #0284c7; }
.m-item .v.ok { color: #10b981; }
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
