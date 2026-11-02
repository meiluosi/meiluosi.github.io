<script lang="ts">
import { onMount } from "svelte";

/**
 * LensWorldModelIntegrator: 显式代码世界模型解剖台 (Explicit Code World Model & Symplectic Phase Space)
 * 揭示：一段 60fps 的程序化动画本质上就是状态转移定律 s_{t+1} = f(s_t, a_t, Δt) 与观测投影 o_t = g(s_t)。
 * 对比辛几何积分器 (Symplectic Euler)、显式欧拉积分器 (Explicit Euler) 与无状态关键帧拟合在相空间守恒律与反事实扰动下的分岔表现。
 */
let english = $state(false);

type IntegratorMode = "symplectic" | "euler" | "keyframe";
let mode = $state<IntegratorMode>("symplectic");

// Discrete time step Δt (s)
let dt = $state(0.16);
// Physical damping ratio ζ
let damping = $state(0.03);
// External impulse perturbation Δv injected at step 28
let impulse = $state(0.45);

let stepCursor = $state(0);
let isPlaying = $state(true);

const TOTAL_STEPS = 80;
const IMPULSE_STEP = 26;

interface TrajectoryPoint {
	step: number;
	x: number;
	v: number;
	energy: number;
}

const rollout = $derived.by(() => {
	const pts: TrajectoryPoint[] = [];
	let x = 1.0;
	let v = 0.0;

	for (let k = 0; k <= TOTAL_STEPS; k++) {
		// Inject counterfactual external impulse at IMPULSE_STEP
		if (k === IMPULSE_STEP && impulse > 0) {
			if (mode !== "keyframe") {
				v += impulse;
			}
		}

		const energy = 0.5 * (x * x + v * v);
		pts.push({ step: k, x, v, energy });

		if (mode === "symplectic") {
			// Symplectic Euler: update velocity first, then position using new velocity
			const acc = -x - 2 * damping * v;
			v = v + acc * dt;
			x = x + v * dt;
		} else if (mode === "euler") {
			// Explicit Euler: update position using old velocity, injecting artificial energy
			const nextX = x + v * dt;
			const nextV = v + (-x - 2 * damping * v) * dt;
			x = Math.max(-2.6, Math.min(2.6, nextX));
			v = Math.max(-2.6, Math.min(2.6, nextV));
		} else {
			// Stateless keyframe curve: pre-baked cosine that ignores momentum impulse
			const t = (k + 1) * dt;
			const decay = Math.exp(-damping * t);
			x = decay * Math.cos(t);
			v = -decay * Math.sin(t);
		}
	}

	const initialEnergy = pts[0]?.energy ?? 0.5;
	const finalEnergy = pts[pts.length - 1]?.energy ?? 0.5;
	const energyRatio = finalEnergy / Math.max(0.001, initialEnergy);

	// Bifurcation threshold for Explicit Euler: Δt > 2ζ causes eigenvalue |λ| > 1
	const criticalDt = Number((2 * damping).toFixed(2));
	const isExploding = mode === "euler" && dt > 2 * damping + 0.01;
	const isCausalBroken = mode === "keyframe" && impulse > 0.05;

	let regimeZh = "哈密顿相空间守恒 / 稳定吸引子 (Symplectic Manifold)";
	let regimeEn = "Symplectic Manifold / True Attractor";
	if (isExploding) {
		regimeZh = "数值能量发散崩塌 (Artificial Energy Explosion)";
		regimeEn = "Numerical Energy Explosion (|λ| > 1)";
	} else if (isCausalBroken) {
		regimeZh = "反事实干预失效：无内部动量状态 (No Causal Momentum)";
		regimeEn = "Counterfactual Failure: Stateless Keyframe";
	}

	return {
		pts,
		energyRatio,
		criticalDt,
		isExploding,
		isCausalBroken,
		isHealthy: !isExploding && !isCausalBroken,
		regimeZh,
		regimeEn,
	};
});

const currentPt = $derived.by(() => {
	const idx = Math.floor(stepCursor) % rollout.pts.length;
	return rollout.pts[idx] ?? { step: 0, x: 1, v: 0, energy: 0.5 };
});

const phasePathD = $derived.by(() => {
	return rollout.pts
		.map((p, i) => {
			const px = 310 + p.x * 42;
			const py = 95 - p.v * 42;
			return `${i === 0 ? "M" : "L"} ${px.toFixed(1)} ${py.toFixed(1)}`;
		})
		.join(" ");
});

const impulsePt = $derived.by(() => {
	const p = rollout.pts[IMPULSE_STEP] ?? { x: 0, v: 0 };
	return {
		px: 310 + p.x * 42,
		py: 95 - p.v * 42,
	};
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

	if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
		isPlaying = false;
	}

	let rafId = 0;
	const tick = () => {
		if (isPlaying) {
			stepCursor = (stepCursor + 0.35) % rollout.pts.length;
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
			<span class="lens-tag">MONOGRAPH V · PROLOGUE SANDBOX</span>
			<button type="button" class="anim-toggle" onclick={() => (isPlaying = !isPlaying)}>
				{isPlaying
					? english ? "⏸ Pause Rollout" : "⏸ 暂停律动"
					: english ? "▶ Resume Rollout" : "▶ 开启律动"}
			</button>
		</div>
		<h3 class="lens-title">
			{english
				? "Anatomy of a Code World Model: Symplectic Integrators vs. Keyframe Illusions"
				: "显式代码世界模型解剖台：辛几何守恒、欧拉发散与反事实冲击"}
		</h3>
		<p class="lens-subtitle">
			{english
				? "Bret Victor & Hairer (Geometric Numerical Integration): A real code animation is a miniature universe governed by s_{t+1} = f(s_t, a_t, Δt). Swapping one line of update order determines whether energy is conserved or explodes."
				: "Bret Victor 与几何数值积分原理：真正的程序化动画不是死板的 CSS 关键帧，而是由状态转移方程 s_{t+1} = f(s_t, a_t, Δt) 驱动的微型宇宙。仅调换一行速度与位置的更新顺序，就会决定这个世界是守恒还是崩溃。"}
		</p>
	</div>

	<div class="mode-pills" role="group" aria-label="世界演化法则选择">
		<button
			type="button"
			class="mode-pill"
			class:active={mode === "symplectic"}
			onclick={() => (mode = "symplectic")}
		>
			{english ? "① Symplectic Euler (Conserved World)" : "① 辛欧拉积分 (相空间守恒世界模型)"}
		</button>
		<button
			type="button"
			class="mode-pill"
			class:active={mode === "euler"}
			onclick={() => (mode = "euler")}
		>
			{english ? "② Explicit Euler (Spurious Energy)" : "② 显式欧拉积分 (伪能量发散分岔)"}
		</button>
		<button
			type="button"
			class="mode-pill"
			class:active={mode === "keyframe"}
			onclick={() => (mode = "keyframe")}
		>
			{english ? "③ Stateless Keyframe (No Causality)" : "③ 无状态关键帧曲线 (反事实干预失效)"}
		</button>
	</div>

	<div class="controls-grid">
		<div class="ctrl-box">
			<div class="label-row">
				<span>{english ? "Discrete Time Step (Δt)" : "离散时间步长 (Δt)"}</span>
				<strong class="val-pill">{dt.toFixed(2)} s</strong>
			</div>
			<input type="range" min="0.04" max="0.34" step="0.02" bind:value={dt} class="slider" />
			<span class="sub-hint">
				{english
					? `Euler explodes when Δt > 2ζ (${rollout.criticalDt}s)`
					: `分岔阈值：显式欧拉在 Δt > 2ζ (${rollout.criticalDt}s) 时发生能量爆炸`}
			</span>
		</div>

		<div class="ctrl-box">
			<div class="label-row">
				<span>{english ? "Damping Ratio (ζ) & Impulse (a_t)" : "物理阻尼比 (ζ) 与外部冲击干预"}</span>
				<strong class="val-pill">ζ={damping.toFixed(2)} | Δv=+{impulse.toFixed(2)}</strong>
			</div>
			<input type="range" min="0.00" max="0.18" step="0.01" bind:value={damping} class="slider" />
			<span class="sub-hint">
				{english
					? "Golden regime: ζ = 0.00 ~ 0.05 with impulse injected at step 26"
					: "黄金观测区：ζ = 0.00 ~ 0.05，第 26 步自动施加外部动量冲击 Δv"}
			</span>
		</div>
	</div>

	<div class="stage-box" class:stable={rollout.isHealthy}>
		<svg viewBox="0 0 430 190" class="world-svg">
			<!-- Left Panel: Physical Space Observation o_t = g(s_t) -->
			<text x="18" y="22" font-size="10.5" font-weight="700" fill="var(--text-muted, #64748b)">
				{english ? "PHYSICAL RENDER o_t = g(s_t)" : "物理渲染投影 o_t = g(s_t)"}
			</text>
			<line x1="65" y1="36" x2="145" y2="36" stroke="var(--brand-clay, #8a7356)" stroke-width="3" />
			<!-- Pendulum Rod & Bob -->
			<line
				x1="105"
				y1="36"
				x2={105 + currentPt.x * 46}
				y2={36 + Math.sqrt(Math.max(12, 95 * 95 - (currentPt.x * 46) ** 2))}
				stroke="#0284c7"
				stroke-width="2.5"
			/>
			<circle
				cx={105 + currentPt.x * 46}
				cy={36 + Math.sqrt(Math.max(12, 95 * 95 - (currentPt.x * 46) ** 2))}
				r="10"
				fill={rollout.isHealthy ? "#10b981" : "#ea580c"}
			/>
			<!-- Velocity Vector Arrow -->
			<line
				x1={105 + currentPt.x * 46}
				y1={36 + Math.sqrt(Math.max(12, 95 * 95 - (currentPt.x * 46) ** 2))}
				x2={105 + currentPt.x * 46 + currentPt.v * 24}
				y2={36 + Math.sqrt(Math.max(12, 95 * 95 - (currentPt.x * 46) ** 2))}
				stroke="#ea580c"
				stroke-width="2"
			/>
			<text x="18" y="168" font-size="10" font-family="monospace" fill="var(--text-muted, #64748b)">
				step={currentPt.step}/{TOTAL_STEPS} | x={currentPt.x.toFixed(2)} | v={currentPt.v.toFixed(2)}
			</text>

			<!-- Divider -->
			<line x1="205" y1="15" x2="205" y2="175" stroke="rgba(120, 140, 130, 0.25)" stroke-dasharray="3 3" />

			<!-- Right Panel: Phase Space Portrait (x_t, v_t) -->
			<text x="220" y="22" font-size="10.5" font-weight="700" fill="var(--text-muted, #64748b)">
				{english ? "PHASE SPACE PORTRAIT (x_t, v_t)" : "内部状态相空间流形 (x_t, v_t)"}
			</text>
			<line x1="225" y1="95" x2="395" y2="95" stroke="rgba(120, 140, 130, 0.3)" stroke-width="1" />
			<line x1="310" y1="28" x2="310" y2="165" stroke="rgba(120, 140, 130, 0.3)" stroke-width="1" />
			<!-- Reference Unit Conservation Circle -->
			<circle
				cx="310"
				cy="95"
				r="42"
				fill="none"
				stroke="rgba(120, 140, 130, 0.28)"
				stroke-width="1"
				stroke-dasharray="2 2"
			/>
			<!-- Phase Trajectory Path -->
			<path
				d={phasePathD}
				fill="none"
				stroke={rollout.isHealthy ? "#0d9488" : "#ea580c"}
				stroke-width="1.8"
			/>
			<!-- Impulse Injection Marker -->
			<circle cx={impulsePt.px} cy={impulsePt.py} r="3.5" fill="#d97706" />
			<!-- Live State Point in Phase Space -->
			<circle
				cx={310 + currentPt.x * 42}
				cy={95 - currentPt.v * 42}
				r="5.5"
				fill="var(--brand-ink, #1e293b)"
			/>
		</svg>

		<div class="metrics-strip">
			<div class="m-item">
				<span class="k">{english ? "Update Law s_{t+1}" : "状态转移法则 s_{t+1}"}</span>
				<strong class="v zero">
					{mode === "symplectic"
						? "v' = v + a·Δt; x' = x + v'·Δt"
						: mode === "euler"
							? "x' = x + v·Δt; v' = v + a·Δt"
							: "x(t) = exp(-ζt)·cos(t)"}
				</strong>
			</div>
			<div class="m-item">
				<span class="k">{english ? "World Regime" : "相空间动力学状态"}</span>
				<strong class="v" class:ok={rollout.isHealthy}>
					{english ? rollout.regimeEn : rollout.regimeZh}
				</strong>
			</div>
			<div class="m-item">
				<span class="k">{english ? "End/Start Energy Ratio" : "末端/初始机械能比 (E_T / E_0)"}</span>
				<strong class="v">{(rollout.energyRatio * 100).toFixed(0)}%</strong>
			</div>
		</div>
	</div>

	<div class="insight-box">
		<div class="insight-icon">🧭</div>
		<div class="insight-text">
			<strong>
				{english
					? "Why Code Animation Is an Explicit World Model:"
					: "为什么写代码动画就是锻造显式世界模型："}
			</strong>
			{english
				? "A CSS keyframe or pixel-level video predictor only memorizes surface curves—when hit by an external impulse at step 26, it cannot update momentum. Symplectic code integration preserves the geometric invariants of phase space, turning 10 lines of TypeScript into a causally manipulable miniature universe."
				: "无状态关键帧（以及纯表观像素拟合）只记住了表面曲线，当第 26 步遭遇外部动量冲击时无法产生真实的因果响应；而显式欧拉积分则因破坏辛几何面积守恒导致能量无中生有。唯有尊重物理不变量的代码积分器，才能用十行代码构建出可干预、可推演的微型世界模型。"}
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
	margin: 1.5rem 0;
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
	font-size: 1.3rem;
	margin: 0.25rem 0 0.5rem;
	font-weight: 700;
}
.lens-subtitle {
	font-size: 0.86rem;
	line-height: 1.55;
	color: var(--text-muted, #718096);
	margin-bottom: 1rem;
}
.mode-pills {
	display: flex;
	flex-wrap: wrap;
	gap: 0.5rem;
	margin-bottom: 1rem;
}
.mode-pill {
	font-size: 0.78rem;
	font-weight: 600;
	padding: 0.38rem 0.75rem;
	border-radius: 99px;
	border: 1px solid rgba(120, 140, 130, 0.3);
	background: transparent;
	color: inherit;
	cursor: pointer;
	transition: all 0.15s ease;
}
.mode-pill.active {
	background: #0f766e;
	border-color: #0f766e;
	color: #ffffff;
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
	color: var(--brand-clay, #8a7356);
}
.slider {
	width: 100%;
	accent-color: #0f766e;
}
.sub-hint {
	font-size: 0.72rem;
	color: var(--text-muted, #718096);
}
.stage-box {
	background: rgba(255, 255, 255, 0.6);
	border: 1px solid rgba(234, 88, 12, 0.35);
	border-radius: 8px;
	padding: 1rem;
	margin-bottom: 1.25rem;
}
:global(.dark) .stage-box {
	background: rgba(0, 0, 0, 0.25);
	border-color: rgba(234, 88, 12, 0.35);
}
.stage-box.stable {
	border-color: #10b981;
}
.world-svg {
	width: 100%;
	max-height: 195px;
}
.metrics-strip {
	display: grid;
	grid-template-columns: repeat(3, 1fr);
	gap: 0.75rem;
	padding-top: 0.75rem;
	border-top: 1px dashed rgba(120, 140, 130, 0.25);
}
@media (max-width: 600px) {
	.metrics-strip {
		grid-template-columns: 1fr;
	}
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
	font-size: 0.82rem;
	font-family: monospace;
}
.m-item .v.zero {
	color: #0284c7;
}
.m-item .v.ok {
	color: #10b981;
}
.insight-box {
	display: flex;
	gap: 0.75rem;
	padding: 0.85rem 1rem;
	background: rgba(15, 118, 110, 0.08);
	border-left: 3px solid #0f766e;
	border-radius: 4px;
	font-size: 0.84rem;
	line-height: 1.5;
}
.insight-icon {
	font-size: 1.1rem;
}
</style>
