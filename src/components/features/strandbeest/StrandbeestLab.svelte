<script lang="ts">
import { onMount } from "svelte";
import {
	playLatchClick,
	playMechanicalTick,
	playSlideTick,
} from "@/utils/haptics";
import {
	type CustomLinkLengths,
	computeFootVelocity,
	computeGaitMetrics,
	DEFAULT_CRANK,
	DEFAULT_LINK_MAP,
	type JointName,
	type LinkId,
	polygonPath,
	resolveLinkSpecs,
	rodPath,
	STRANDBEEST_LINKS,
	STRANDBEEST_PRESETS,
	solveStrandbeest,
	traceStrandbeest,
} from "@/utils/strandbeest-geometry";

let {
	mode = "glance",
	activeStep = undefined,
}: {
	mode?: "glance" | "study";
	activeStep?: number;
} = $props();

const uid = $props.id();
let root: HTMLDivElement;
let svgEl = $state<SVGSVGElement>();
let angle = $state(205);
let crank = $state(DEFAULT_CRANK);
let customLengths = $state<CustomLinkLengths>({});
let activePresetId = $state("jansen");
let step = $state(0);
let playing = $state(false);
let speedMultiplier = $state(1);
let showMirroredPair = $state(false);
let showVelocity = $state(true);
let showAllSliders = $state(false);
let hoveredJoint = $state<JointName | null>(null);
let draggingCrank = $state(false);
let ready = $state(false);
let english = $state(false);
let reduced = $state(false);
let visible = false;
let frame = 0;
let lastTime = 0;

$effect(() => {
	if (
		activeStep !== undefined &&
		activeStep >= 0 &&
		activeStep <= 3 &&
		activeStep !== step
	) {
		step = activeStep;
	}
});

const baseline = traceStrandbeest();
const activeLinks = $derived(resolveLinkSpecs(customLengths));
const pose = $derived(solveStrandbeest(angle, crank, customLengths, false));
const mirrorPose = $derived(
	solveStrandbeest(angle, crank, customLengths, true),
);
const trace = $derived(traceStrandbeest(crank, 360, customLengths, false));
const mirrorTrace = $derived(traceStrandbeest(crank, 360, customLengths, true));
const metrics = $derived(computeGaitMetrics(crank, customLengths));
const velocity = $derived(computeFootVelocity(angle, crank, customLengths));
const full = $derived(mode === "glance" || step >= 2);
const say = (zh: string, en: string) => (english ? en : zh);

const jointDescriptions: Record<JointName, { zh: string; en: string }> = {
	G: {
		zh: "机身固定铰点 G (0, 0)：承受上三角 GKL 与下连杆 GM 的反作用力",
		en: "Fixed frame pivot G (0, 0): anchors upper triangle GKL and rod GM",
	},
	O: {
		zh: "曲柄轴心 O (38, 7.8)：风帆动力输入中心，驱动曲柄 OC 旋转",
		en: "Crankshaft center O (38, 7.8): wind-powered rotary input axis",
	},
	C: {
		zh: "曲柄销 C：唯一做纯圆周运动的驱动点（可在画布中直接按住拖动）",
		en: "Crank pin C: pure circular driver (drag directly on canvas to turn)",
	},
	K: {
		zh: "交点 K：由固定圆 G(41.5) 与运动圆 C(50) 交汇决定，驱动上三角摇摆",
		en: "Intersection K: solved from circle G(41.5) and moving circle C(50)",
	},
	L: {
		zh: "上三角顶点 L：将 K 的摆动传递至外侧连杆 LN",
		en: "Upper triangle apex L: relays oscillation of K to outer rod LN",
	},
	M: {
		zh: "中枢铰点 M：连接固定点 G、曲柄 C 与下三角 MNF 的关键枢纽",
		en: "Central hub M: bridges fixed pivot G, crank C, and lower foot triangle MNF",
	},
	N: {
		zh: "下三角膝点 N：由 L 与 M 共同约束，控制足端抬起与落下时机",
		en: "Knee joint N: constrained by L and M, shaping foot lift timing",
	},
	F: {
		zh: "足端 F：最终输出近似平直触地段与弓形抬腿段的闭合步态轨迹",
		en: "Foot tip F: traces the closed D-shaped walking trajectory",
	},
};

const stepText = $derived(
	[
		say(
			"第一步 · 纯圆周输入：桃色短杆 OC 绕机身固定点 O 旋转，端点 C 只能沿着虚线圆周运动。直接在画布上按住 C 点拖动，观察旋转如何传导。",
			"Step 1 · Circular Input: The terracotta crank OC turns around fixed pivot O, constraining C to a circle. Drag pin C directly on the canvas to feel the transmission.",
		),
		say(
			"第二步 · 双圆交点定乾坤：关节 K 同时连着 G 和 C，两根杆长度固定。以 G、C 为圆心画圆，K 只能落在两圆交点上。模型固定选择同一装配侧。",
			"Step 2 · Circle Intersection: Joint K connects to both G and C with fixed rod lengths. Drawing circles around G and C narrows K to their intersection.",
		),
		say(
			"第三步 · 十杆双三角成腿：依次求解 K → L → M → N → F。注意底部实线（触地推进段）近似水平，而顶部虚线（抬腿回摆段）向上拱起，实现平稳迈步。",
			"Step 3 · Ten Rods Form a Leg: Solving K → L → M → N → F yields a flat bottom stance segment (solid terracotta) and an arched return swing (dashed).",
		),
		say(
			"第四步 · 神圣数字与演化边界：切换下方预设或微调 11 根杆长。若将曲柄拉长到 30，部分角度两圆相离，机构进入“死点”无法闭合。",
			"Step 4 · Holy Numbers & Singularity: Switch presets or tune all 11 rod lengths below. Lengthening the crank to 30 causes circles to separate at unreachable angles.",
		),
	][step],
);

function stopFrame() {
	cancelAnimationFrame(frame);
	frame = 0;
	lastTime = 0;
}

function tick(time: number) {
	frame = 0;
	if (!playing || !visible || document.hidden) {
		lastTime = 0;
		return;
	}
	if (!lastTime) lastTime = time;
	const elapsed = time - lastTime;
	if (elapsed >= 1000 / 60) {
		angle =
			(angle + Math.min(elapsed, 65) * 0.032 * speedMultiplier + 360) % 360;
		lastTime = time;
	}
	frame = requestAnimationFrame(tick);
}

function syncFrame() {
	if (!ready) return;
	if (playing && visible && !document.hidden) {
		if (!frame) frame = requestAnimationFrame(tick);
	} else stopFrame();
}

function toggle() {
	playing = !playing;
	playLatchClick();
	syncFrame();
}

function manual() {
	playing = false;
	stopFrame();
}

function selectStep(index: number) {
	step = index;
	playLatchClick();
	if (index === 3 && activePresetId === "jansen") {
		// Keep current state ready for tuning
	}
}

function applyPreset(presetId: string) {
	const preset = STRANDBEEST_PRESETS.find((p) => p.id === presetId);
	if (!preset) return;
	activePresetId = preset.id;
	crank = preset.crank;
	customLengths = { ...preset.lengths };
	playLatchClick();
}

function updateLinkLength(id: LinkId, value: number) {
	customLengths = { ...customLengths, [id]: value };
	activePresetId = "custom";
	playSlideTick(1);
}

function reset() {
	manual();
	angle = 205;
	crank = DEFAULT_CRANK;
	customLengths = {};
	activePresetId = "jansen";
	playLatchClick();
}

function updateAngleFromPointer(e: PointerEvent) {
	if (!svgEl) return;
	const ctm = svgEl.getScreenCTM();
	if (!ctm) return;
	const pt = svgEl.createSVGPoint();
	pt.x = e.clientX;
	pt.y = e.clientY;
	const svgPt = pt.matrixTransform(ctm.inverse());
	// SVG has y inverted (-y), O is at (38, 7.8) in math coords -> (38, -7.8) in SVG
	const mathX = svgPt.x;
	const mathY = -svgPt.y;
	const dx = mathX - 38;
	const dy = mathY - 7.8;
	let deg = (Math.atan2(dy, dx) * 180) / Math.PI;
	if (deg < 0) deg += 360;
	angle = Math.round(deg);
	playSlideTick(1.2);
}

function handleSvgPointerDown(e: PointerEvent) {
	if (!ready || !svgEl) return;
	manual();
	draggingCrank = true;
	svgEl.setPointerCapture(e.pointerId);
	updateAngleFromPointer(e);
}

function handleSvgPointerMove(e: PointerEvent) {
	if (!draggingCrank) return;
	updateAngleFromPointer(e);
}

function handleSvgPointerUp(e: PointerEvent) {
	if (!draggingCrank) return;
	draggingCrank = false;
	if (svgEl?.hasPointerCapture(e.pointerId)) {
		svgEl.releasePointerCapture(e.pointerId);
	}
}

onMount(() => {
	ready = true;
	const media = window.matchMedia("(prefers-reduced-motion: reduce)");
	reduced = media.matches;
	const locale = () => {
		english = document.documentElement.lang.startsWith("en");
	};
	const motion = () => {
		reduced = media.matches;
		if (reduced) manual();
	};
	locale();
	playing = mode === "glance" && !reduced;
	const observer = new IntersectionObserver(
		([entry]) => {
			visible = entry.isIntersecting;
			syncFrame();
		},
		{ threshold: 0.08 },
	);
	observer.observe(root);
	media.addEventListener("change", motion);
	document.addEventListener("visibilitychange", syncFrame);
	document.addEventListener("site-locale-change", locale);
	return () => {
		stopFrame();
		observer.disconnect();
		media.removeEventListener("change", motion);
		document.removeEventListener("visibilitychange", syncFrame);
		document.removeEventListener("site-locale-change", locale);
	};
});

const svgViewBox = $derived(
	showMirroredPair
		? "-88 -66 252 174"
		: mode === "study" && step === 1
			? "-94 -76 211 183"
			: "-96 -62 184 168",
);

const isFootTouchingGround = $derived(
	Boolean(
		pose.valid &&
			pose.points.F &&
			trace.minY !== null &&
			trace.height !== null &&
			pose.points.F.y <= trace.minY + trace.height * 0.18,
	),
);
</script>

<div class:study={mode === "study"} class="linkage" bind:this={root}>
	{#if mode === "glance"}
		<div class="glance-toolbar">
			<div class="pill-group" role="group" aria-label={say("视图模式", "View mode")}>
				<button
					type="button"
					class:active={!showMirroredPair}
					disabled={!ready}
					onclick={() => {
						showMirroredPair = false;
						playLatchClick();
					}}
				>
					{say("单腿几何剖析", "Single Leg")}
				</button>
				<button
					type="button"
					class:active={showMirroredPair}
					disabled={!ready}
					onclick={() => {
						showMirroredPair = true;
						playLatchClick();
					}}
				>
					{say("双腿对称共轴", "Mirrored Pair")}
				</button>
			</div>
			<div class="pill-group">
				<button
					type="button"
					class:active={showVelocity}
					disabled={!ready}
					onclick={() => {
						showVelocity = !showVelocity;
						playMechanicalTick();
					}}
				>
					{say("速度矢量 & 触地线", "Velocity & Ground")}
				</button>
			</div>
		</div>
	{/if}

	{#if mode === "study"}
		<div class="steps" aria-label={say("拆开看这条腿", "Look inside the leg")}>
			{#each [
				say("1 · 一个圆", "1 · One circle"),
				say("2 · 两个交点", "2 · Two intersections"),
				say("3 · 十杆成腿", "3 · Ten rods"),
				say("4 · 神圣数字调优", "4 · Holy Numbers"),
			] as label, index}
				<button
					type="button"
					class:active={step === index}
					aria-pressed={step === index}
					disabled={!ready}
					onclick={() => selectStep(index)}>{label}</button
				>
			{/each}
		</div>
	{/if}

	<div class="drawing" class:dragging={draggingCrank}>
		<svg
			bind:this={svgEl}
			viewBox={svgViewBox}
			role="img"
			aria-label={say(
				"Jansen 式平面连杆与脚端轨迹（可直接在画布拖动曲柄）",
				"Jansen-style planar linkage and foot path (drag crank directly on canvas)",
			)}
			onpointerdown={handleSvgPointerDown}
			onpointermove={handleSvgPointerMove}
			onpointerup={handleSvgPointerUp}
			onpointercancel={handleSvgPointerUp}
		>
			<title>{say("从圆周运动到脚端轨迹", "From rotation to a foot path")}</title>
			<desc
				>{say(
					"两处固定铰，桃色曲柄带动十根连杆。脚端路径由圆交点逐步计算。",
					"Two fixed pivots and a terracotta crank drive ten rods. The foot path is calculated through successive circle intersections.",
				)}</desc
			>

			<!-- Beach ground contact line when full trajectory is shown -->
			{#if full && trace.minY !== null}
				{@const groundSvgY = -trace.minY + 0.6}
				<line
					x1="-88"
					y1={groundSvgY}
					x2={showMirroredPair ? "158" : "76"}
					y2={groundSvgY}
					class="ground-line"
				/>
				<text x="-85" y={groundSvgY + 5.2} class="ground-label">
					{say("沙滩触地基准面", "Beach ground plane")}
				</text>
			{/if}

			{#if full}<path d={baseline.path} class="baseline" />{/if}
			{#if full}
				<!-- Full trace kept for SSR compatibility, enhanced with split stance/swing styling -->
				<path d={trace.path} class="trace" />
				{#if trace.stancePath}
					<path d={trace.stancePath} class="trace-stance" />
				{/if}
			{/if}

			<!-- Optional mirrored right leg sharing crank pin C -->
			{#if showMirroredPair}
				<path d={mirrorTrace.path} class="trace mirror-trace" />
				<g class="mirror-leg" opacity="0.82">
					<path
						d={polygonPath(mirrorPose, ["G", "K", "L"])}
						class="triangle upper"
					/>
					<path
						d={polygonPath(mirrorPose, ["M", "N", "F"])}
						class="triangle lower"
					/>
					<path d={rodPath(mirrorPose, activeLinks)} class="rods" />
					{#each Object.entries(mirrorPose.points) as [mName, mPt]}
						{#if mPt && mName !== "O" && mName !== "C"}
							<circle
								cx={mPt.x}
								cy={-mPt.y}
								r={mName === "F" ? 1.7 : 1.05}
								class:foot={mName === "F"}
								class="joint"
							/>
							{#if mName === "G"}
								<circle cx={mPt.x} cy={-mPt.y} r="2.8" class="fixed" />
							{/if}
						{/if}
					{/each}
				</g>
			{/if}

			<g opacity={full ? 1 : 0.2}>
				<path d={polygonPath(pose, ["G", "K", "L"])} class="triangle upper" />
				<path d={polygonPath(pose, ["M", "N", "F"])} class="triangle lower" />
				<path d={rodPath(pose, activeLinks)} class="rods" />
			</g>

			{#if pose.points.O}
				<circle
					cx={pose.points.O.x}
					cy={-pose.points.O.y}
					r={crank}
					class="construction"
				/>
			{/if}

			{#if mode === "study" && step === 1 && pose.points.C}
				<circle
					cx="0"
					cy="0"
					r={customLengths.GK ?? DEFAULT_LINK_MAP.GK}
					class="construction circle-g"
				/>
				<circle
					cx={pose.points.C.x}
					cy={-pose.points.C.y}
					r={customLengths.CK ?? DEFAULT_LINK_MAP.CK}
					class="construction circle-c"
				/>
				<path
					d={rodPath(pose, activeLinks.slice(0, 2))}
					class="rods highlight-rods"
				/>
			{/if}

			{#if pose.points.O && pose.points.C}
				<line
					x1={pose.points.O.x}
					y1={-pose.points.O.y}
					x2={pose.points.C.x}
					y2={-pose.points.C.y}
					class="crank"
				/>
				<!-- Interactive grab halo around C -->
				<circle
					cx={pose.points.C.x}
					cy={-pose.points.C.y}
					r="4.2"
					class="crank-grab-halo"
				/>
			{/if}

			<!-- Foot velocity vector & ground contact pulse -->
			{#if full && showVelocity && pose.valid && pose.points.F && velocity}
				{@const fx = pose.points.F.x}
				{@const fy = -pose.points.F.y}
				{@const vx = velocity.vx * 14}
				{@const vy = -velocity.vy * 14}
				{#if isFootTouchingGround}
					<circle cx={fx} cy={fy} r="4.2" class="stance-pulse" />
				{/if}
				<line
					x1={fx}
					y1={fy}
					x2={fx + vx}
					y2={fy + vy}
					class="velocity-vector"
				/>
				<circle cx={fx + vx} cy={fy + vy} r="0.95" class="velocity-tip" />
			{/if}

			{#each Object.entries(pose.points) as [name, point]}
				{#if point}
					<!-- svelte-ignore a11y_no_static_element_interactions -->
					<g
						class="joint-group"
						onpointerenter={() => {
							hoveredJoint = name as JointName;
						}}
						onpointerleave={() => {
							if (hoveredJoint === name) hoveredJoint = null;
						}}
					>
						<circle
							cx={point.x}
							cy={-point.y}
							r="4.5"
							fill="transparent"
							class="joint-hit"
						/>
						<circle
							cx={point.x}
							cy={-point.y}
							r={name === "F" ? 1.8 : name === "C" ? 1.5 : 1.15}
							class:foot={name === "F"}
							class:hovered={hoveredJoint === name}
							class="joint"
							opacity={full ||
							["G", "O", "C", ...(step === 1 ? ["K"] : [])].includes(name)
								? 1
								: 0.2}
						/>
						{#if ["G", "O"].includes(name)}
							<circle cx={point.x} cy={-point.y} r="2.8" class="fixed" />
						{/if}
						{#if (mode === "study" && (["G", "O", "C"].includes(name) || (step === 1 && name === "K") || (step >= 2 && ["F", "K", "L", "M", "N"].includes(name)))) || hoveredJoint === name}
							<text x={point.x + 3.4} y={-point.y - 3.4}>{name}</text>
						{/if}
					</g>
				{/if}
			{/each}
		</svg>

		<div class="canvas-badge">
			<span class="badge-dot" class:stance={isFootTouchingGround}></span>
			{#if hoveredJoint && pose.points[hoveredJoint]}
				{@const pt = pose.points[hoveredJoint]}
				{#if pt}
					<span
						><b>{hoveredJoint}</b> ({pt.x.toFixed(1)}, {pt.y.toFixed(1)}) — {say(
							jointDescriptions[hoveredJoint].zh,
							jointDescriptions[hoveredJoint].en,
						)}</span
					>
				{/if}
			{:else}
				<span
					>{say(
						"提示：可直接在图中按住桃色曲柄 C 旋转，或悬停任意关节查看约束关系",
						"Tip: Drag the terracotta crank pin C directly on the canvas, or hover any joint",
					)}</span
				>
			{/if}
		</div>

		<div class="legend">
			<span class="legend-crank"></span>{say("曲柄 OC", "Crank OC")}
			<span class="legend-stance"></span>{say("触地推进段", "Stance phase")}
			<span class="legend-trace"></span>{say("抬腿回摆段", "Swing phase")}
			{#if showVelocity}
				<span class="legend-vel"></span>{say("瞬时速度矢量", "Foot velocity")}
			{/if}
		</div>
	</div>

	<!-- Real-time Gait Telemetry Bar -->
	{#if full}
		<div class="telemetry-grid" aria-label={say("实时步态指标", "Live gait telemetry")}>
			<div class="metric-card">
				<span class="metric-label">{say("可闭合相位", "Valid Phases")}</span>
				<strong class="metric-val" class:warn={metrics.validSamples < 360}>
					{metrics.validSamples}<small>/360°</small>
				</strong>
			</div>
			<div class="metric-card">
				<span class="metric-label">{say("触地段平直度", "Stance Flatness")}</span>
				<strong class="metric-val">
					{metrics.flatnessScore !== null ? `${metrics.flatnessScore}%` : "—"}
				</strong>
			</div>
			<div class="metric-card">
				<span class="metric-label">{say("单步跨度 × 抬高", "Stride × Lift")}</span>
				<strong class="metric-val">
					{metrics.strideWidth !== null && metrics.liftHeight !== null
						? `${metrics.strideWidth.toFixed(1)} × ${metrics.liftHeight.toFixed(1)}`
						: "—"}
				</strong>
			</div>
			<div class="metric-card">
				<span class="metric-label">{say("触地占空比", "Stance Duty")}</span>
				<strong class="metric-val">
					{metrics.dutyFactor !== null ? `${metrics.dutyFactor}%` : "—"}
				</strong>
			</div>
		</div>
	{/if}

	{#if mode === "study"}<p class="explanation" aria-live="polite">{stepText}</p>{/if}

	<div class="controls">
		<button
			type="button"
			class="play"
			disabled={!ready}
			onclick={toggle}
			aria-label={playing
				? say("暂停连杆运动", "Pause the linkage")
				: say("播放连杆运动", "Play the linkage")}
		>
			<span aria-hidden="true">{playing ? "Ⅱ" : "▷"}</span>
			{playing ? say("暂停", "Pause") : say("播放", "Play")}
		</button>
		<label for={`${uid}-angle`}>
			<span>
				{say("手动转动 / 画布拖拽", "Turn by hand / drag canvas")}
				<output>{Math.round(angle)}°</output>
			</span>
			<input
				id={`${uid}-angle`}
				type="range"
				min="0"
				max="360"
				step="1"
				bind:value={angle}
				oninput={manual}
				disabled={!ready}
			/>
		</label>
		<div class="speed-group" role="group" aria-label={say("倍速", "Speed")}>
			{#each [0.5, 1, 2] as spd}
				<button
					type="button"
					class="speed-btn"
					class:active={speedMultiplier === spd}
					disabled={!ready}
					onclick={() => {
						speedMultiplier = spd;
						playMechanicalTick();
					}}>{spd}×</button
				>
			{/each}
		</div>
		{#if mode === "study"}
			<button type="button" class="reset" disabled={!ready} onclick={reset}>
				{say("恢复初始", "Reset")}
			</button>
		{/if}
	</div>

	{#if mode === "study" && step >= 2}
		<div class="preset-section">
			<div class="preset-header">
				<span>{say("演化比例预设对比：", "Evolutionary Presets:")}</span>
				<div class="preset-pills">
					{#each STRANDBEEST_PRESETS as preset}
						<button
							type="button"
							class="preset-pill"
							class:active={activePresetId === preset.id}
							disabled={!ready}
							onclick={() => applyPreset(preset.id)}
						>
							{say(preset.nameZh, preset.nameEn)}
						</button>
					{/each}
				</div>
			</div>
			{#if activePresetId !== "custom"}
				{@const currentPreset = STRANDBEEST_PRESETS.find(
					(p) => p.id === activePresetId,
				)}
				{#if currentPreset}
					<p class="preset-note">
						{say(currentPreset.noteZh, currentPreset.noteEn)}
					</p>
				{/if}
			{/if}
		</div>

		<div class="length-control">
			<label for={`${uid}-length`}>
				<span>
					{say("曲柄长度 OC", "Crank length OC")}
					<output>{crank.toFixed(1)}</output>
				</span>
				<input
					id={`${uid}-length`}
					type="range"
					min="5"
					max="40"
					step=".5"
					bind:value={crank}
					oninput={() => {
						manual();
						activePresetId = "custom";
						playSlideTick(1);
					}}
					disabled={!ready}
				/>
			</label>
			<div class="sub-length-toggle">
				<p>
					{say(
						"默认 15 · 浅灰虚线保留 Jansen 原始轨迹作为对照",
						"Default 15 · Light dashed path retains Jansen baseline for comparison",
					)}
				</p>
				<button
					type="button"
					class="toggle-all-btn"
					disabled={!ready}
					onclick={() => {
						showAllSliders = !showAllSliders;
						playLatchClick();
					}}
				>
					{showAllSliders
						? say("收起其余 10 根神圣数字连杆 ▲", "Hide 10 rod sliders ▲")
						: say("展开微调其余 10 根神圣数字连杆 ▼", "Tune all 10 Holy Number rods ▼")}
				</button>
			</div>

			{#if showAllSliders || step === 3}
				<div class="holy-sliders-grid">
					{#each STRANDBEEST_LINKS as spec}
						{@const currentVal = customLengths[spec.id] ?? spec.length}
						<label class="rod-slider-item" for={`${uid}-rod-${spec.id}`}>
							<span>
								<b>{spec.id}</b>
								<small>({DEFAULT_LINK_MAP[spec.id]})</small>
								<output>{currentVal.toFixed(1)}</output>
							</span>
							<input
								id={`${uid}-rod-${spec.id}`}
								type="range"
								min={Math.max(15, Math.round(spec.length * 0.65))}
								max={Math.round(spec.length * 1.35)}
								step="0.2"
								value={currentVal}
								disabled={!ready}
								oninput={(e) =>
									updateLinkLength(
										spec.id,
										Number.parseFloat((e.currentTarget as HTMLInputElement).value),
									)}
							/>
						</label>
					{/each}
				</div>
			{/if}
		</div>
		<p class="cycle-state">
			{say("每隔 1° 采样：", "Sampled every 1°: ")}{trace.validSamples} / 360 {say(
				"个相位可闭合。这是数值采样，不是连续可达性的证明。",
				"phases close. This is a numerical sample, not a proof of continuous reachability.",
			)}
		</p>
	{/if}

	<p class:invalid={!pose.valid} class="geometry-status" aria-live="polite">
		{#if !pose.valid}
			{say(
				`当前角度无法闭合：关节 ${pose.failedJoint} 的两个圆没有可用交点。图中只保留已求出的杆件；请转动曲柄或恢复初始长度。`,
				`This angle cannot close: joint ${pose.failedJoint} has no usable circle intersection. Only solved rods remain. Turn the crank or restore the initial length.`,
			)}
		{/if}
	</p>
	{#if reduced}
		<p class="motion-note">
			{say(
				"已遵循系统减少动态效果设置，默认静止；可手动转动或主动播放。",
				"Reduced motion is on. The diagram starts still; turn it by hand or choose Play.",
			)}
		</p>
	{/if}
</div>

<style>
	.linkage{color:var(--brand-ink);min-width:0}
	.glance-toolbar{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:10px;max-width:720px;margin:0 auto 12px}
	.pill-group{display:flex;gap:6px;flex-wrap:wrap}
	.pill-group button{padding:6px 12px;min-height:34px;font-size:12.5px;border-radius:999px}
	.pill-group button.active{background:var(--brand-sage);border-color:var(--brand-accent);font-weight:500}
	.drawing{position:relative;max-width:720px;margin:auto;cursor:grab;user-select:none}
	.drawing.dragging{cursor:grabbing}
	.drawing svg{display:block;width:100%;height:clamp(295px,46vw,470px);overflow:hidden;touch-action:none}
	.drawing text{font:4.5px "JetBrains Mono",monospace;fill:var(--brand-muted);pointer-events:none}
	.ground-line{stroke:var(--brand-muted);stroke-width:.45;stroke-dasharray:2.5 1.5;opacity:.55}
	.ground-label{font-size:3.6px!important;fill:var(--brand-muted);opacity:.75}
	.rods{fill:none;stroke:var(--brand-ink);stroke-width:1.15;stroke-linecap:round;stroke-linejoin:round}
	.highlight-rods{stroke:var(--brand-accent);stroke-width:1.45}
	.triangle{fill-opacity:.55;transition:fill-opacity .2s}
	.upper{fill:var(--brand-sage)}
	.lower{fill:var(--brand-peach)}
	.crank{stroke:var(--brand-clay);stroke-width:2.1;stroke-linecap:round}
	.crank-grab-halo{fill:var(--brand-clay);fill-opacity:.14;stroke:var(--brand-clay);stroke-width:.35;stroke-dasharray:1 1}
	.joint-group{cursor:pointer}
	.joint{fill:var(--brand-paper);stroke:var(--brand-ink);stroke-width:.7;transition:r .15s,stroke-width .15s}
	.joint.hovered{stroke:var(--brand-clay);stroke-width:1.2}
	.joint.foot{fill:var(--brand-clay)}
	.fixed{fill:none;stroke:var(--brand-ink);stroke-width:.5}
	.construction{fill:none;stroke:var(--brand-muted);stroke-width:.55;stroke-dasharray:1 1.6;opacity:.65}
	.circle-g{stroke:var(--brand-accent);stroke-width:.75;opacity:.85}
	.circle-c{stroke:var(--brand-clay);stroke-width:.75;opacity:.85}
	.trace{fill:none;stroke:var(--brand-clay);stroke-width:.75;stroke-dasharray:1.2 1.8;opacity:.72}
	.trace-stance{fill:none;stroke:var(--brand-clay);stroke-width:1.45;stroke-linecap:round;opacity:.95}
	.mirror-trace{stroke:var(--brand-accent);opacity:.45}
	.baseline{fill:none;stroke:var(--brand-muted);stroke-width:.55;stroke-dasharray:1 1;opacity:.32}
	.velocity-vector{stroke:var(--brand-accent);stroke-width:.95;stroke-linecap:round}
	.velocity-tip{fill:var(--brand-accent)}
	.stance-pulse{fill:var(--brand-clay);fill-opacity:.2;stroke:var(--brand-clay);stroke-width:.4}
	.canvas-badge{display:flex;align-items:center;gap:8px;font-size:12px;line-height:1.5;color:var(--brand-muted);background:color-mix(in srgb,var(--brand-sage) 18%,transparent);border:1px solid var(--brand-line);border-radius:4px;padding:6px 11px;margin:8px auto 10px;max-width:660px}
	.badge-dot{width:7px;height:7px;border-radius:50%;background:var(--brand-muted);flex-shrink:0}
	.badge-dot.stance{background:var(--brand-clay);box-shadow:0 0 0 3px color-mix(in srgb,var(--brand-clay) 25%,transparent)}
	.legend{display:flex;flex-wrap:wrap;justify-content:center;align-items:center;gap:8px;font-size:12.5px;color:var(--brand-muted);margin:6px 0 16px}
	.legend-crank{width:16px;border-top:3px solid var(--brand-clay)}
	.legend-stance{width:16px;border-top:2.5px solid var(--brand-clay);margin-left:10px}
	.legend-trace{width:16px;border-top:1.5px dashed var(--brand-clay);margin-left:10px}
	.legend-vel{width:14px;border-top:2px solid var(--brand-accent);margin-left:10px}
	.telemetry-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;max-width:660px;margin:0 auto 14px}
	.metric-card{border:1px solid var(--brand-line);border-radius:4px;padding:7px 10px;background:color-mix(in srgb,var(--brand-paper) 75%,transparent);text-align:center}
	.metric-label{display:block;font-size:11px;color:var(--brand-muted);margin-bottom:2px}
	.metric-val{font:500 14px "JetBrains Mono",monospace;color:var(--brand-ink)}
	.metric-val.warn{color:var(--brand-clay)}
	.metric-val small{font-size:11px;color:var(--brand-muted);margin-left:1px}
	.controls{display:flex;gap:16px;align-items:center;max-width:660px;margin:auto;padding:10px 0}
	.controls label{flex:1;min-width:0}
	label{display:block;font-size:13.5px;color:var(--brand-body)}
	label>span{display:flex;justify-content:space-between;gap:12px}
	output{font-variant-numeric:tabular-nums;font-family:"JetBrains Mono",monospace}
	input[type=range]{width:100%;height:30px;margin:3px 0 0;accent-color:var(--brand-clay);cursor:pointer}
	button{font:inherit;font-size:13.5px;border:1px solid var(--brand-line);border-radius:4px;color:var(--brand-ink);background:transparent;padding:8px 14px;min-height:40px;cursor:pointer;white-space:nowrap;transition:background .15s,border-color .15s}
	button:disabled{cursor:default;opacity:.5}
	button:hover:not(:disabled){background:var(--brand-sage)}
	button:focus-visible,input:focus-visible{outline:2px solid var(--brand-accent);outline-offset:3px}
	.play{min-width:86px;font-weight:500}
	.play span{font-size:14px;margin-right:4px}
	.speed-group{display:flex;gap:4px}
	.speed-btn{padding:6px 8px;min-height:34px;font:12px "JetBrains Mono",monospace}
	.speed-btn.active{background:var(--brand-sage);border-color:var(--brand-accent);font-weight:600}
	.steps{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px}
	.steps button{flex:1;padding:9px 10px;font-size:13px}
	.steps .active{background:var(--brand-sage);border-color:var(--brand-accent);font-weight:500}
	.explanation{font-size:14.5px;line-height:1.85;min-height:4.8em;max-width:650px;margin:10px auto;color:var(--brand-body)}
	.preset-section{max-width:660px;margin:12px auto 0;padding-top:14px;border-top:1px solid var(--brand-line)}
	.preset-header{display:flex;flex-direction:column;gap:8px;font-size:13px;color:var(--brand-muted)}
	.preset-pills{display:flex;flex-wrap:wrap;gap:6px}
	.preset-pill{padding:5px 10px;min-height:32px;font-size:12px;border-radius:999px}
	.preset-pill.active{background:var(--brand-peach);border-color:var(--brand-clay);font-weight:500}
	.preset-note{font-size:12.5px;line-height:1.7;color:var(--brand-muted);margin:8px 0 4px}
	.length-control{max-width:660px;margin:10px auto 0;padding-top:12px;border-top:1px dashed var(--brand-line)}
	.sub-length-toggle{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:8px;margin-top:4px}
	.sub-length-toggle p{font-size:12.5px;color:var(--brand-muted);margin:0}
	.toggle-all-btn{padding:4px 10px;min-height:30px;font-size:12px;color:var(--brand-accent)}
	.holy-sliders-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px 18px;margin-top:12px;padding:12px;border:1px solid var(--brand-line);border-radius:4px;background:color-mix(in srgb,var(--brand-paper) 65%,transparent)}
	.rod-slider-item{font-size:12px}
	.rod-slider-item small{color:var(--brand-muted)}
	.rod-slider-item input[type=range]{height:22px;margin-top:1px}
	.cycle-state,.motion-note{font-size:12.5px;line-height:1.75;color:var(--brand-muted);max-width:660px;margin:8px auto 0}
	.geometry-status{font-size:13px;line-height:1.8;max-width:660px;margin:0 auto}
	.geometry-status.invalid{border-left:2px solid var(--brand-clay);color:var(--brand-clay);padding:8px 12px;margin-top:12px}
	.geometry-status:empty{display:none}
	@media(max-width:600px){
		.telemetry-grid{grid-template-columns:repeat(2,1fr)}
		.holy-sliders-grid{grid-template-columns:1fr}
		.controls{gap:10px;flex-wrap:wrap}
		.controls .reset{width:100%;border:none;text-decoration:underline;text-underline-offset:4px;padding:4px}
		.steps{gap:4px}
		.steps button{font-size:12px;padding:7px 4px}
		.explanation{font-size:13.5px;min-height:6.5em}
		.drawing svg{height:300px}
	}
</style>
