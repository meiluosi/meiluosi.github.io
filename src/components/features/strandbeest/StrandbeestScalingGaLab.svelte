<script lang="ts">
import { onMount } from "svelte";
import {
	playLatchClick,
	playMechanicalTick,
	playSlideTick,
} from "@/utils/haptics";
import {
	type CustomLinkLengths,
	DEFAULT_CRANK,
	polygonPath,
	rodPath,
	solveStrandbeest,
	traceStrandbeest,
} from "@/utils/strandbeest-geometry";

const uid = $props.id();
let english = $state(false);
let activeTab = $state<"scaling" | "ga">("scaling");

// Tab 1: Scaling & Friction Phase Diagram State
let scalingLaw = $state<"thin" | "critical" | "solid">("solid");
let frictionLevel = $state<0 | 1 | 2>(0); // 0: 0 N·m (MuJoCo 30-grid simulation baseline), 1: 0.0005 N·m (L=1 test + extrapolation), 2: 0.002 N·m (L=1 test + extrapolation)
let probeScaleIdx = $state(4); // Index into SCALE_POINTS
let energyDriveMode = $state<"sail" | "motor">("sail");

const SCALE_POINTS = [0.03, 0.1, 0.2, 0.35, 0.6, 1.0, 1.5, 2.2, 3.0] as const;

// Verified simulation data curves from docs/EXPERIMENTS.md & Part 5 report (§3.1-§3.2)
// In solid scaling (p=3), L=0.35 is 0.529, L=0.6 is 0.415 (minimum), L=1.0 is 0.489, L=3.0 is 0.865.
const BASELINE_SIMULATION_CURVES: Record<
	"thin" | "critical" | "solid",
	readonly number[]
> = {
	thin: [2.45, 1.38, 1.02, 0.8, 0.61, 0.489, 0.44, 0.41, 0.39],
	critical: [0.53, 0.51, 0.5, 0.495, 0.49, 0.489, 0.482, 0.478, 0.475],
	solid: [0.92, 0.71, 0.6, 0.529, 0.415, 0.489, 0.585, 0.72, 0.865],
};

// Extrapolation based on the 82% pin friction dissipation finding:
// Measured on L=1.0: 0 -> 0.489, 0.0005 -> 1.04 (2.13x), 0.002 -> 1.27 (2.59x).
// Measured on L=0.03 (solid, 0.002 N·m): 18.4 m/s (locked).
const FRICTION_EXTRAPOLATION: Record<0 | 1 | 2, readonly number[]> = {
	0: [1, 1, 1, 1, 1, 1, 1, 1, 1],
	1: [8.5, 4.2, 2.9, 2.4, 2.18, 2.13, 1.95, 1.82, 1.75],
	2: [20.0, 6.8, 4.3, 3.25, 2.78, 2.59, 2.32, 2.15, 2.02],
};

const currentCurve = $derived(
	SCALE_POINTS.map((L, idx) => {
		const base = BASELINE_SIMULATION_CURVES[scalingLaw][idx] ?? 0.5;
		const mult = FRICTION_EXTRAPOLATION[frictionLevel][idx] ?? 1;
		const v = Number((base * mult).toFixed(3));
		return { L, v, locked: v > 4.17 };
	}),
);

const probedPoint = $derived(currentCurve[probeScaleIdx] ?? currentCurve[5]);

// Tab 2: Genetic Algorithm Historical Specimens (Pre-baked Archetypes from docs/REPORT.md)
type ArchetypeKey =
	| "cheat_hip"
	| "cheat_slide"
	| "cheat_stilt"
	| "cmaes"
	| "ga_blind"
	| "target_shape";
let selectedArchetype = $state<ArchetypeKey>("cmaes");
let crankAngle = $state(205);

interface GaSpecimen {
	titleZh: string;
	titleEn: string;
	badgeZh: string;
	badgeEn: string;
	ruleViolatedZh: string;
	ruleViolatedEn: string;
	crank: number;
	lengths: CustomLinkLengths;
	diagnosisZh: string;
	diagnosisEn: string;
	isCheat: boolean;
	evidenceLevel: string;
}

const SPECIMENS: Record<ArchetypeKey, GaSpecimen> = {
	cheat_hip: {
		titleZh: "作弊表型 1：脚荡向髋关节上方（无髋下约束）",
		titleEn: "Cheat Archetype 1: Foot Flips Above Hip (No Hip Ceiling)",
		badgeZh: "典型作弊机制 · REWARD HACKING",
		badgeEn: "REWARD HACKING ARCHETYPE",
		ruleViolatedZh: "缺少规则 A：未规定足端必须始终位于髋关节下方",
		ruleViolatedEn: "Missing Rule A: Foot must stay below hip pivot",
		crank: 22,
		lengths: {
			GK: 36,
			CK: 46,
			GL: 35,
			KL: 52,
			GM: 34,
			CM: 56,
			LN: 33,
			MN: 42,
			MF: 35,
			NF: 44,
		},
		diagnosisZh:
			"若只奖励抬腿高度而不限制机身上方空间，GA 会把下三角直接翻折到机身上方，刷出极大的虚假抬腿分，但在物理机身上会直接撞毁机身横梁。",
		diagnosisEn:
			"When lift height is rewarded without a ceiling below the hip, GA flips the lower triangle upward, faking huge lift that would collide with the frame.",
		isCheat: true,
		evidenceLevel: "E0 (历史复现记录)",
	},
	cheat_slide: {
		titleZh: "作弊表型 2：脚只滑不抬（100% 触地直线）",
		titleEn: "Cheat Archetype 2: Pure Sliding, Zero Lift",
		badgeZh: "典型作弊机制 · REWARD HACKING",
		badgeEn: "REWARD HACKING ARCHETYPE",
		ruleViolatedZh: "缺少规则 B：未设置最小抬腿高度门槛（≥ 宽度 15%）",
		ruleViolatedEn: "Missing Rule B: Minimum lift height threshold",
		crank: 7.5,
		lengths: {
			GK: 43,
			CK: 49,
			GL: 42,
			KL: 56,
			GM: 42,
			CM: 58,
			LN: 41,
			MN: 38,
			MF: 54,
			NF: 68,
		},
		diagnosisZh:
			"只奖励“触地段平而长”时，GA 会将曲柄行程极度压缩，把轨迹压成一条贴在地面来回蹭的直线。触地占比达到 100%，但根本无法迈步越障。",
		diagnosisEn:
			"Rewarding long stance without requiring lift collapses the path into a back-and-forth ground scrape with 100% stance duty but zero walking capability.",
		isCheat: true,
		evidenceLevel: "E0 (历史复现记录)",
	},
	cheat_stilt: {
		titleZh: "作弊表型 3：高跷畸形腿（按抬腿高度归一化）",
		titleEn: "Cheat Archetype 3: Stilt Distortion (Lift-Normalized)",
		badgeZh: "典型作弊机制 · REWARD HACKING",
		badgeEn: "REWARD HACKING ARCHETYPE",
		ruleViolatedZh: "缺少规则 C：平直度按抬腿高度（而非步幅宽度）归一化",
		ruleViolatedEn: "Missing Rule C: Normalize flatness by width, not lift",
		crank: 19.5,
		lengths: {
			GK: 40,
			CK: 51,
			GL: 39,
			KL: 57,
			GM: 38,
			CM: 63,
			LN: 37,
			MN: 35,
			MF: 56,
			NF: 72,
		},
		diagnosisZh:
			"若将平直度定义为“垂直标准差 / 抬腿高度”，GA 会故意演化出极其夸张的高跷腿，因为分母变大使得底部哪怕颠簸不平也被算法误判为平直。",
		diagnosisEn:
			"Defining flatness relative to lift height causes GA to evolve giant stilt legs, diluting stance roughness by inflating the denominator.",
		isCheat: true,
		evidenceLevel: "E0 (历史复现记录)",
	},
	cmaes: {
		titleZh: "CMA-ES 盲搜解（20 种子中 13 个超过 Jansen 评分）",
		titleEn: "CMA-ES Blind Search (13/20 Seeds Exceeded Jansen Score)",
		badgeZh: "离线优化样本 · 12,000 次评估",
		badgeEn: "OFFLINE OPTIMIZATION · 12K EVALS",
		ruleViolatedZh: "三项反作弊规则全部生效，算法收敛良好",
		ruleViolatedEn: "All 3 anti-cheat rules active, solid convergence",
		crank: 16.0,
		lengths: {
			GK: 42.2,
			CK: 50.8,
			GL: 40.6,
			KL: 55.2,
			GM: 39.8,
			CM: 62.4,
			LN: 39.8,
			MN: 36.2,
			MF: 50.4,
			NF: 66.8,
		},
		diagnosisZh:
			"在补齐三条反作弊规则后，CMA-ES 仅用 GA 16% 的评估预算就让得分中位数达到 0.529（Jansen 自身得分为 0.481），证明此前盲搜受阻主要源于 GA 陷入局部最优，而非目标不兼容。",
		diagnosisEn:
			"With all 3 anti-cheat rules enforced, CMA-ES reached median score 0.529 in 12k evaluations (Jansen score: 0.481), showing standard GA had been trapped in local optima.",
		isCheat: false,
		evidenceLevel: "E0 (离线仿真数据)",
	},
	ga_blind: {
		titleZh: "标准 GA 盲搜典型解（易陷局部最优）",
		titleEn: "Standard GA Blind Search (Local Optima Trap)",
		badgeZh: "离线优化样本 · 75,000 次评估",
		badgeEn: "OFFLINE OPTIMIZATION · 75K EVALS",
		ruleViolatedZh: "三项反作弊规则全部生效，但受限于遗传漂移",
		ruleViolatedEn: "Rules active, but hampered by genetic drift",
		crank: 14.2,
		lengths: {
			GK: 43.1,
			CK: 51.4,
			GL: 41.2,
			KL: 54.0,
			GM: 40.5,
			CM: 60.2,
			LN: 40.4,
			MN: 37.8,
			MF: 47.2,
			NF: 64.0,
		},
		diagnosisZh:
			"标准 GA 在 150 种群 × 500 代下盲搜中位数仅为 0.390（0/20 超过 Jansen）；即使把变异率拉高到 0.5，中位数依然停留在 0.384，无法跳出局部极值盆地。",
		diagnosisEn:
			"Standard GA plateaued at median score 0.390 (0/20 beating Jansen); even raising mutation rate to 0.5 yielded 0.384, stuck in local basins.",
		isCheat: false,
		evidenceLevel: "E0 (离线仿真数据)",
	},
	target_shape: {
		titleZh: "目标形状恢复解（形状 19/20 找回，但数字完全不同）",
		titleEn: "Target Shape Inversion (19/20 Found Shape, 0/20 Numbers)",
		badgeZh: "多对一映射 · 历史反例",
		badgeEn: "MANY-TO-ONE MAPPING · COUNTEREXAMPLE",
		ruleViolatedZh: "以 Jansen 足端形状距离为目标函数",
		ruleViolatedEn: "Targeted Jansen foot trajectory shape distance",
		crank: 14.6,
		lengths: {
			GK: 40.4,
			CK: 48.9,
			GL: 39.2,
			KL: 54.6,
			GM: 38.4,
			CM: 60.5,
			LN: 38.6,
			MN: 35.9,
			MF: 47.9,
			NF: 64.2,
		},
		diagnosisZh:
			"以 Jansen 真实足端轨迹为目标时，19/20 种子成功复现了 D 形曲线（形状距离 < 0.02），但 0/20 找回原作者的 13 个数字——从轨迹反推连杆尺寸是严重的多对一欠定问题。",
		diagnosisEn:
			"Targeting Jansen's trajectory recovered the D-shape in 19/20 seeds, yet 0/20 recovered his numbers, proving shape-to-linkage inversion is highly underdetermined.",
		isCheat: false,
		evidenceLevel: "E1 (双向几何核对)",
	},
};

const activeSpecimen = $derived(SPECIMENS[selectedArchetype]);
const baselineTrace = traceStrandbeest(DEFAULT_CRANK, 180);
const specimenPose = $derived(
	solveStrandbeest(
		crankAngle,
		activeSpecimen.crank,
		activeSpecimen.lengths,
		false,
	),
);
const specimenTrace = $derived(
	traceStrandbeest(activeSpecimen.crank, 180, activeSpecimen.lengths, false),
);

const say = (zh: string, en: string) => (english ? en : zh);

function chartX(idx: number): number {
	return 58 + (idx / (SCALE_POINTS.length - 1)) * 450;
}
function chartY(v: number): number {
	const clamped = Math.min(5.0, Math.max(0.2, v));
	return 195 - ((clamped - 0.2) / 4.8) * 160;
}

const polylinePoints = $derived(
	currentCurve
		.map((pt, idx) => `${chartX(idx).toFixed(1)},${chartY(pt.v).toFixed(1)}`)
		.join(" "),
);

onMount(() => {
	const syncLocale = () => {
		english = document.documentElement.lang.startsWith("en");
	};
	syncLocale();
	document.addEventListener("site-locale-change", syncLocale);
	return () => {
		document.removeEventListener("site-locale-change", syncLocale);
	};
});
</script>

<section class="scaling-ga-lab" aria-label={say("动力学尺度数据与演化样本实验台", "Dynamics Scaling Data & GA Specimen Sandbox")}>
	<div class="lab-top-bar">
		<div class="tab-pills" role="tablist">
			<button
				type="button"
				role="tab"
				aria-selected={activeTab === "scaling"}
				class:active={activeTab === "scaling"}
				onclick={() => {
					activeTab = "scaling";
					playLatchClick();
				}}
			>
				📊 {say("起步风速 × 尺度相图（MuJoCo/K2 离线格点与外推）", "Start-up Wind × Scale Phase Data (MuJoCo/K2 Grids & Extrapolations)")}
			</button>
			<button
				type="button"
				role="tab"
				aria-selected={activeTab === "ga"}
				class:active={activeTab === "ga"}
				onclick={() => {
					activeTab = "ga";
					playLatchClick();
				}}
			>
				🧬 {say("遗传算法作弊与收敛表型重放（历史样本）", "GA Reward-Hacking & Convergence Archetype Replay")}
			</button>
		</div>
		<span class="evidence-pill">
			{activeTab === "scaling"
				? say("数据性质：30 格离线仿真格点 + 敏感性外推（E0/E1）", "Nature: 30-grid offline simulation points + analytical extrapolation (E0/E1)")
				: say("数据性质：历史 20 种子演化典型表型重放（非实时计算）", "Nature: Replay of historical typical phenotypes (Not live computed)")}
		</span>
	</div>

	{#if activeTab === "scaling"}
		<div class="scaling-grid">
			<div class="chart-panel">
				<div class="control-strip">
					<div class="ctrl-group">
						<span class="ctrl-label">{say("质量标度律（离线物理仿真基线）：", "Mass Scaling Law (Offline Simulation Baseline):")}</span>
						{#each [
							["solid", say("实心 p=3 (非单调 V 形，经 30 格完整求解)", "Solid p=3 (Non-monotonic V, 30-grid solved)")],
							["thin", say("薄壁管 p=1 (小尺度斜率 -0.51)", "Thin-wall p=1 (Slope -0.51)")],
							["critical", say("临界 p=2 (尺度无关，斜率 -0.03)", "Critical p=2 (Slope -0.03)")],
						] as [id, label]}
							<button
								type="button"
								class="mini-btn"
								class:active={scalingLaw === id}
								onclick={() => {
									scalingLaw = id as "thin" | "critical" | "solid";
									playMechanicalTick();
								}}
							>
								{label}
							</button>
						{/each}
					</div>

					<div class="ctrl-group">
						<span class="ctrl-label">{say("销轴库仑摩擦工况：", "Pin Friction Regime:")}</span>
						{#each [
							[0, "0 N·m (MuJoCo 理想基准格点)"],
							[1, "0.0005 N·m (L=1 测定 + 动力学外推)"],
							[2, "0.002 N·m (2.59× 敏感性 + 小尺度锁死)"],
						] as [lvl, label]}
							<button
								type="button"
								class="mini-btn"
								class:active={frictionLevel === lvl}
								onclick={() => {
									frictionLevel = lvl as 0 | 1 | 2;
									playMechanicalTick();
								}}
							>
								{label}
							</button>
						{/each}
					</div>
				</div>

				<svg
					viewBox="0 0 540 230"
					class="phase-svg"
					role="img"
					aria-label={say("起步风速随尺度 L 变化的离线仿真与外推曲线", "Start-up wind speed vs scale L offline simulation & extrapolation curve")}
				>
					<line x1="58" y1={chartY(4.17)} x2="512" y2={chartY(4.17)} class="ref-threshold" />
					<text x="64" y={chartY(4.17) - 5} class="threshold-text">
						{say("常规海风上限 4.17 m/s (15 km/h) — 高于此线机体在常态自然风中无法起步", "Wind limit 4.17 m/s (15 km/h) — locked regime above")}
					</text>

					<line x1="58" y1="28" x2="58" y2="195" class="axis-line" />
					<line x1="58" y1="195" x2="512" y2="195" class="axis-line" />

					{#each [0.5, 1.5, 3.0, 4.5] as yTick}
						<line x1="53" y1={chartY(yTick)} x2="512" y2={chartY(yTick)} class="grid-tick" />
						<text x="48" y={chartY(yTick) + 3} text-anchor="end" class="axis-text">{yTick} m/s</text>
					{/each}

					<polyline
						points={polylinePoints}
						class="wind-curve"
						class:extrapolated={frictionLevel > 0}
					/>

					{#each currentCurve as pt, idx}
						{@const cx = chartX(idx)}
						{@const cy = chartY(pt.v)}
						<!-- svelte-ignore a11y_click_events_have_key_events -->
						<!-- svelte-ignore a11y_no_static_element_interactions -->
						<g
							class="data-node"
							onclick={() => {
								probeScaleIdx = idx;
								playSlideTick(1);
							}}
						>
							<circle cx={cx} cy={cy} r="10" fill="transparent" />
							<circle
								cx={cx}
								cy={cy}
								r={probeScaleIdx === idx ? 5.5 : 3.6}
								class:locked={pt.locked}
								class:selected={probeScaleIdx === idx}
								class="node-dot"
							/>
							<text x={cx} y="213" text-anchor="middle" class="axis-text">
								{pt.L}×
							</text>
							{#if probeScaleIdx === idx}
								<text x={cx} y={cy - 10} text-anchor="middle" class="node-val-label">
									{pt.v > 15 ? ">18 m/s" : `${pt.v.toFixed(2)} m/s`}
								</text>
							{/if}
						</g>
					{/each}
				</svg>

				<div class="chart-annotation">
					<span>
						{frictionLevel === 0
							? say("实线：MuJoCo 30 格无摩擦完整多体动力学求解格点 (E0 仿真)", "Solid: MuJoCo 30-grid frictionless multi-body dynamics solved points (E0)")
							: say("虚线：基于 L=1 摩擦敏感性测试与 82% 能量耗散机理的动力学分析外推曲线（非全网格动力学解）", "Dashed: Analytical extrapolation based on L=1 friction tests and 82% energy dissipation mechanism (not full-grid solved)")}
					</span>
				</div>

				<div class="probe-bar">
					<label for={`${uid}-scale-probe`}>
						<span>
							{say("探针检视机体尺度 L：", "Probe organism scale L:")}
							<b>L = {probedPoint.L}×</b>
						</span>
						<input
							id={`${uid}-scale-probe`}
							type="range"
							min="0"
							max={SCALE_POINTS.length - 1}
							step="1"
							bind:value={probeScaleIdx}
							oninput={() => playSlideTick(1)}
						/>
					</label>
					<div class="probe-readout" class:warn={probedPoint.locked}>
						<span>{say("40s 转满一圈起步风速：", "40s Start Wind Speed:")}</span>
						<strong>
							{probedPoint.v > 15
								? `${probedPoint.v.toFixed(1)} m/s (锁死区)`
								: `${probedPoint.v.toFixed(3)} m/s`}
						</strong>
					</div>
				</div>
			</div>

			<div class="scaling-side-cards">
				<div class="side-card">
					<h4>{say("双独立求解器交叉核对（E1 / E2 证据边界）", "Dual-Solver Cross-Verification (E1 / E2 Boundary)")}</h4>
					<p class="side-desc">
						{say(
							"为排除单一引擎（如 MuJoCo 软化接触）的数值假象，项目构建了独立的平面刚体求解器 K2（Moreau 时间步进 + 库仑干摩擦）：",
							"To eliminate numerical artifacts of soft contact, independent solver K2 was built with Moreau time-stepping and dry friction:",
						)}
					</p>
					<div class="compare-rows">
						<div class="c-row">
							<span>{say("Pop (2014) 实验台步幅比对", "Pop (2014) Test Stride")}</span>
							<b>57.08 vs 57.10 mm (0.03% · E2)</b>
						</div>
						<div class="c-row">
							<span>{say("MuJoCo vs K2 每圈步幅", "MuJoCo vs K2 Stride")}</span>
							<b>0.2760 vs 0.2776 m (+0.58% · E1)</b>
						</div>
						<div class="c-row">
							<span>{say("15s 严判据起步风速量级", "15s Start Wind Magnitude")}</span>
							<b>0.692 vs 0.723 m/s (+4.4% · E1)</b>
						</div>
						<div class="c-row">
							<span>{say("驱动扭矩（受接触摩擦模型影响）", "Drive Torque (Model-sensitive)")}</span>
							<b>2.16 vs 2.54 mN·m (+17.6% · E0)</b>
						</div>
					</div>
				</div>

				<div class="side-card">
					<div class="energy-head">
						<h4>{say("能量耗散账目（仿真揭示的真实机制）", "Energy Ledger: Work Dissipation Breakdown")}</h4>
						<div class="energy-tabs">
							<button
								type="button"
								class:active={energyDriveMode === "sail"}
								onclick={() => {
									energyDriveMode = "sail";
									playMechanicalTick();
								}}>{say("4 m/s 风帆驱动", "4 m/s Sail Drive")}</button
							>
							<button
								type="button"
								class:active={energyDriveMode === "motor"}
								onclick={() => {
									energyDriveMode = "motor";
									playMechanicalTick();
								}}>{say("电机恒速驱动", "Motor Drive")}</button
							>
						</div>
					</div>
					{#if energyDriveMode === "sail"}
						<div class="energy-bar">
							<div class="e-seg joint" style="width: 82%">82% {say("销轴摩擦与阻尼", "Joint Friction")}</div>
							<div class="e-seg ground" style="width: 18%">18% {say("地面打滑", "Ground Slip")}</div>
						</div>
						<p class="side-desc">
							{say(
								"风帆驱动下，82% 的输入功被转动副销轴摩擦消耗！这解释了为什么销轴摩擦一旦从 0 升至 0.002 N·m，起步风速便猛增 2.59 倍。",
								"Under sail drive, 82% of work goes into joint friction—explaining why minor pin friction surges start wind by 2.59x.",
							)}
						</p>
					{:else}
						<div class="energy-bar">
							<div class="e-seg ground" style="width: 96.5%">96.5% {say("足端地面打滑摩擦", "Foot-Ground Slip")}</div>
							<div class="e-seg joint" style="width: 3.5%"></div>
						</div>
						<p class="side-desc">
							{say(
								"在电机恒速驱动下，96.5% 输入功耗散在足端与地面的微幅滑移上。简化准静态模型因忽略滑移，导致平均扭矩预测偏低约 8 N·m。",
								"Under motor drive, 96.5% is lost in ground micro-slip. Simplified models ignore slip and underestimate torque by ~8 N·m.",
							)}
						</p>
					{/if}
				</div>
			</div>
		</div>
	{:else}
		<div class="ga-grid">
			<div class="ga-controls-col">
				<div class="side-card">
					<h4>{say("选择演化表型（历史实验记录重放）：", "Select Phenotype (Historical Run Replay):")}</h4>
					<div class="archetype-selector">
						<span class="sub-label">{say("【适应度函数作弊表型 · Reward Hacking】", "[Fitness Function Exploits · Reward Hacking]")}</span>
						<div class="btn-cluster">
							<button
								type="button"
								class="mini-btn warn"
								class:active={selectedArchetype === "cheat_hip"}
								onclick={() => { selectedArchetype = "cheat_hip"; playLatchClick(); }}
							>
								{say("作弊 1：脚荡向髋上", "Exploit 1: Upward Flip")}
							</button>
							<button
								type="button"
								class="mini-btn warn"
								class:active={selectedArchetype === "cheat_slide"}
								onclick={() => { selectedArchetype = "cheat_slide"; playLatchClick(); }}
							>
								{say("作弊 2：只滑不抬", "Exploit 2: Pure Slide")}
							</button>
							<button
								type="button"
								class="mini-btn warn"
								class:active={selectedArchetype === "cheat_stilt"}
								onclick={() => { selectedArchetype = "cheat_stilt"; playLatchClick(); }}
							>
								{say("作弊 3：高跷畸形腿", "Exploit 3: Stilt Dilution")}
							</button>
						</div>

						<span class="sub-label">{say("【正向演化与多对一逆问题对比】", "[Legitimate Optimization & Inverse Mapping]")}</span>
						<div class="btn-cluster">
							<button
								type="button"
								class="mini-btn"
								class:active={selectedArchetype === "cmaes"}
								onclick={() => { selectedArchetype = "cmaes"; playLatchClick(); }}
							>
								{say("CMA-ES 盲搜最优 (13/20 胜 Jansen)", "CMA-ES Optimal (13/20 Beat)")}
							</button>
							<button
								type="button"
								class="mini-btn"
								class:active={selectedArchetype === "ga_blind"}
								onclick={() => { selectedArchetype = "ga_blind"; playLatchClick(); }}
							>
								{say("标准 GA 盲搜 (局部最优陷阱)", "Standard GA (Local Trap)")}
							</button>
							<button
								type="button"
								class="mini-btn"
								class:active={selectedArchetype === "target_shape"}
								onclick={() => { selectedArchetype = "target_shape"; playLatchClick(); }}
							>
								{say("目标形状反推 (形状吻合但数字不同)", "Target Shape Inversion")}
							</button>
						</div>
					</div>

					<div class="specimen-summary" class:cheat={activeSpecimen.isCheat}>
						<div class="spec-head">
							<strong>{say(activeSpecimen.titleZh, activeSpecimen.titleEn)}</strong>
							<span class="spec-badge">{say(activeSpecimen.badgeZh, activeSpecimen.badgeEn)}</span>
						</div>
						<p class="rule-flag"><b>{say("规则边界：", "Rule Boundary: ")}</b>{say(activeSpecimen.ruleViolatedZh, activeSpecimen.ruleViolatedEn)}</p>
						<p>{say(activeSpecimen.diagnosisZh, activeSpecimen.diagnosisEn)}</p>
						<div class="spec-metrics">
							<span>{say("证据级别：", "Evidence Level:")} <b>{activeSpecimen.evidenceLevel}</b></span>
							<span>{say("曲柄半径 OC：", "Crank OC:")} <b>{activeSpecimen.crank}</b></span>
						</div>
					</div>
				</div>
			</div>

			<div class="ga-preview-col">
				<svg
					viewBox="-96 -66 188 174"
					class="ga-svg"
					role="img"
					aria-label={say("演化典型表型轨迹与 Jansen 基准对比", "Phenotype trajectory vs Jansen baseline")}
				>
					<path d={baselineTrace.path} class="jansen-ghost" />
					<path
						d={specimenTrace.path}
						class="specimen-trace"
						class:cheat-trace={activeSpecimen.isCheat}
					/>
					<path
						d={polygonPath(specimenPose, ["G", "K", "L"])}
						class="tri-up"
					/>
					<path
						d={polygonPath(specimenPose, ["M", "N", "F"])}
						class="tri-low"
					/>
					<path d={rodPath(specimenPose)} class="spec-rods" />
					{#if specimenPose.points.O && specimenPose.points.C}
						<line
							x1={specimenPose.points.O.x}
							y1={-specimenPose.points.O.y}
							x2={specimenPose.points.C.x}
							y2={-specimenPose.points.C.y}
							class="spec-crank"
						/>
					{/if}
					{#if specimenPose.points.F}
						<circle
							cx={specimenPose.points.F.x}
							cy={-specimenPose.points.F.y}
							r="2.1"
							class="spec-foot"
						/>
					{/if}
				</svg>
				<div class="ga-legend">
					<span><i class="dot-ghost"></i> {say("浅灰虚线：Jansen 原始轨迹", "Dashed: Jansen baseline")}</span>
					<span><i class="dot-spec" class:cheat-dot={activeSpecimen.isCheat}></i> {say("实线：当前表型重放轨迹", "Solid: Phenotype replay")}</span>
				</div>
				<label class="angle-scrub" for={`${uid}-ga-angle`}>
					<span>{say("手动转动曲柄检视闭链姿态", "Turn crank to inspect posture")} <output>{crankAngle}°</output></span>
					<input
						id={`${uid}-ga-angle`}
						type="range"
						min="0"
						max="360"
						bind:value={crankAngle}
						oninput={() => playSlideTick(1)}
					/>
				</label>
			</div>
		</div>
	{/if}
</section>

<style>
	.scaling-ga-lab{border:1px solid var(--brand-line);border-radius:6px;padding:22px 24px;margin:28px auto 38px;background:color-mix(in srgb,var(--brand-sage) 11%,transparent);color:var(--brand-ink)}
	.lab-top-bar{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:10px;margin-bottom:16px;padding-bottom:12px;border-bottom:1px solid var(--brand-line)}
	.tab-pills{display:flex;flex-wrap:wrap;gap:8px}
	.tab-pills button{font:inherit;font-size:13px;padding:7px 14px;border:1px solid var(--brand-line);border-radius:999px;background:color-mix(in srgb,var(--brand-paper) 75%,transparent);color:var(--brand-ink);cursor:pointer;transition:background .15s,border-color .15s}
	.tab-pills button.active{background:var(--brand-sage);border-color:var(--brand-accent);font-weight:500}
	.evidence-pill{font:11.5px "JetBrains Mono",monospace;color:var(--brand-muted);padding:4px 10px;border:1px solid var(--brand-line);border-radius:999px;background:color-mix(in srgb,var(--brand-paper) 65%,transparent)}
	.scaling-grid{display:grid;grid-template-columns:1.25fr .95fr;gap:20px;align-items:start}
	.chart-panel,.side-card,.ga-preview-col{border:1px solid var(--brand-line);border-radius:5px;padding:14px 16px;background:color-mix(in srgb,var(--brand-paper) 82%,transparent)}
	.control-strip{display:flex;flex-direction:column;gap:8px;margin-bottom:10px}
	.ctrl-group{display:flex;flex-wrap:wrap;align-items:center;gap:6px}
	.ctrl-label{font-size:12px;color:var(--brand-muted);margin-right:4px}
	.mini-btn{font:inherit;font-size:12px;padding:4px 10px;border:1px solid var(--brand-line);border-radius:4px;background:transparent;color:var(--brand-ink);cursor:pointer}
	.mini-btn.active{background:var(--brand-peach);border-color:var(--brand-clay);font-weight:500}
	.mini-btn.warn.active{background:var(--brand-clay);color:var(--brand-paper);border-color:var(--brand-clay)}
	.phase-svg{display:block;width:100%;height:auto;margin:6px 0}
	.ref-threshold{stroke:var(--brand-clay);stroke-width:1;stroke-dasharray:4 3;opacity:.75}
	.threshold-text{font:9.5px "JetBrains Mono",monospace;fill:var(--brand-clay)}
	.axis-line{stroke:var(--brand-ink);stroke-width:1;opacity:.45}
	.grid-tick{stroke:var(--brand-line);stroke-width:.6;stroke-dasharray:2 2}
	.axis-text{font:10px "JetBrains Mono",monospace;fill:var(--brand-muted)}
	.wind-curve{fill:none;stroke:var(--brand-accent);stroke-width:2.2;stroke-linecap:round;stroke-linejoin:round}
	.wind-curve.extrapolated{stroke-dasharray:5 3}
	.data-node{cursor:pointer}
	.node-dot{fill:var(--brand-paper);stroke:var(--brand-accent);stroke-width:2;transition:r .15s}
	.node-dot.locked{stroke:var(--brand-clay);fill:var(--brand-peach)}
	.node-dot.selected{fill:var(--brand-clay);stroke:var(--brand-ink)}
	.node-val-label{font:600 10.5px "JetBrains Mono",monospace;fill:var(--brand-ink)}
	.chart-annotation{font-size:11.5px;color:var(--brand-muted);line-height:1.6;margin:6px 0 10px;padding:6px 8px;border-left:2px solid var(--brand-accent);background:color-mix(in srgb,var(--brand-sage) 12%,transparent)}
	.probe-bar{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:12px;padding-top:10px;border-top:1px solid var(--brand-line);font-size:12.5px}
	.probe-bar label{flex:1;min-width:180px}
	.probe-bar label span{display:flex;justify-content:space-between}
	.probe-bar input[type=range]{width:100%;margin-top:4px;accent-color:var(--brand-clay)}
	.probe-readout{padding:6px 12px;border:1px solid var(--brand-line);border-radius:4px;background:color-mix(in srgb,var(--brand-sage) 18%,transparent)}
	.probe-readout.warn{background:color-mix(in srgb,var(--brand-peach) 35%,transparent);border-color:var(--brand-clay);color:var(--brand-clay)}
	.probe-readout strong{font-family:"JetBrains Mono",monospace;margin-left:4px}
	.scaling-side-cards{display:flex;flex-direction:column;gap:14px}
	.side-card h4{font:500 14.5px/1.4 "Noto Serif SC","Songti SC",Georgia,serif;margin:0 0 8px;color:var(--brand-ink)}
	.side-desc{font-size:12.5px;line-height:1.7;color:var(--brand-body);margin:0 0 10px}
	.compare-rows{display:flex;flex-direction:column;gap:6px;font-size:12px}
	.c-row{display:flex;justify-content:space-between;gap:8px;padding:5px 8px;border-radius:4px;background:color-mix(in srgb,var(--brand-sage) 12%,transparent)}
	.c-row b{font-family:"JetBrains Mono",monospace;color:var(--brand-ink)}
	.energy-head{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:6px;margin-bottom:8px}
	.energy-head h4{margin:0}
	.energy-tabs{display:flex;gap:4px}
	.energy-tabs button{font:inherit;font-size:11px;padding:3px 8px;border:1px solid var(--brand-line);border-radius:4px;background:transparent;cursor:pointer}
	.energy-tabs button.active{background:var(--brand-sage);border-color:var(--brand-accent);font-weight:500}
	.energy-bar{display:flex;height:24px;border-radius:4px;overflow:hidden;border:1px solid var(--brand-line);margin-bottom:8px;font:11px "JetBrains Mono",monospace}
	.e-seg{display:flex;align-items:center;justify-content:center;white-space:nowrap;overflow:hidden;padding:0 6px}
	.e-seg.joint{background:var(--brand-peach);color:var(--brand-ink)}
	.e-seg.ground{background:var(--brand-sage);color:var(--brand-ink)}
	.ga-grid{display:grid;grid-template-columns:1.15fr .85fr;gap:20px;align-items:start}
	.ga-controls-col{display:flex;flex-direction:column;gap:14px}
	.archetype-selector{display:flex;flex-direction:column;gap:6px;margin-bottom:12px}
	.sub-label{font-size:11.5px;font-weight:600;color:var(--brand-muted);margin-top:6px}
	.btn-cluster{display:flex;flex-wrap:wrap;gap:6px}
	.specimen-summary{padding:10px 12px;border-left:3px solid var(--brand-accent);background:color-mix(in srgb,var(--brand-sage) 15%,transparent);border-radius:4px;font-size:12.5px;line-height:1.7}
	.specimen-summary.cheat{border-left-color:var(--brand-clay);background:color-mix(in srgb,var(--brand-peach) 28%,transparent)}
	.spec-head{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:6px;margin-bottom:4px}
	.spec-badge{font:10.5px "JetBrains Mono",monospace;padding:2px 6px;border-radius:3px;background:color-mix(in srgb,var(--brand-paper) 75%,transparent)}
	.rule-flag{font-size:12px;color:var(--brand-clay);margin:2px 0 6px}
	.specimen-summary p{margin:4px 0 8px;color:var(--brand-body)}
	.spec-metrics{display:flex;flex-wrap:wrap;gap:14px;font:11.5px "JetBrains Mono",monospace;color:var(--brand-muted)}
	.spec-metrics b{color:var(--brand-ink)}
	.ga-svg{display:block;width:100%;height:clamp(230px,34vw,300px)}
	.jansen-ghost{fill:none;stroke:var(--brand-muted);stroke-width:.65;stroke-dasharray:1.4 1.4;opacity:.5}
	.specimen-trace{fill:none;stroke:var(--brand-accent);stroke-width:1.35}
	.specimen-trace.cheat-trace{stroke:var(--brand-clay);stroke-dasharray:2 1}
	.tri-up{fill:var(--brand-sage);fill-opacity:.5}
	.tri-low{fill:var(--brand-peach);fill-opacity:.5}
	.spec-rods{fill:none;stroke:var(--brand-ink);stroke-width:1.1;stroke-linecap:round;stroke-linejoin:round}
	.spec-crank{stroke:var(--brand-clay);stroke-width:2;stroke-linecap:round}
	.spec-foot{fill:var(--brand-clay)}
	.ga-legend{display:flex;justify-content:center;gap:14px;font-size:11.5px;color:var(--brand-muted);margin:6px 0 10px}
	.dot-ghost{display:inline-block;width:12px;border-top:1.5px dashed var(--brand-muted);vertical-align:middle;margin-right:4px}
	.dot-spec{display:inline-block;width:12px;border-top:2px solid var(--brand-accent);vertical-align:middle;margin-right:4px}
	.dot-spec.cheat-dot{border-top:2px dashed var(--brand-clay)}
	.angle-scrub{display:block;font-size:12px;color:var(--brand-muted)}
	.angle-scrub span{display:flex;justify-content:space-between}
	.angle-scrub input{width:100%;margin-top:3px;accent-color:var(--brand-clay)}
	@media(max-width:860px){
		.scaling-grid,.ga-grid{grid-template-columns:1fr}
	}
</style>
