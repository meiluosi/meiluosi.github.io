<script lang="ts">
import { onMount } from "svelte";
import {
	playLatchClick,
	playMechanicalTick,
	playSlideTick,
} from "@/utils/haptics";
import {
	DEFAULT_CRANK,
	polygonPath,
	rodPath,
	solveStrandbeest,
	traceStrandbeest,
} from "@/utils/strandbeest-geometry";
import type { Beast3DController, BeastCameraPreset } from "./create-beast-3d";

const uid = $props.id();
let root: HTMLDivElement;
let canvas3d = $state<HTMLCanvasElement>();
let viewMode = $state<"wind2d" | "stereo3d">("wind2d");
let legPairsCount = $state<1 | 3>(3);
let windSpeed = $state(6.5);
let gustBoost = $state(0);
let crankAngle = $state(205);
let explodeFactor = $state(1.15);
let activeCameraPreset = $state<BeastCameraPreset>("isometric");
let playing = $state(true);
let ready = $state(false);
let english = $state(false);
let reduced = $state(false);
let visible = false;
let webglLoading = $state(false);
let webglError = $state(false);
let controller3d: Beast3DController | null = null;
let frame = 0;
let lastTime = 0;
let beachScroll = $state(0);

const say = (zh: string, en: string) => (english ? en : zh);

const baselineTrace = traceStrandbeest(DEFAULT_CRANK, 180);
const groundFloorY = baselineTrace.minY ?? -84;

const bayOffsets = $derived(
	legPairsCount === 3 ? ([0, 120, 240] as const) : ([0] as const),
);

const effectiveWind = $derived(Math.min(16, windSpeed + gustBoost));

const beaufortLabel = $derived(
	effectiveWind < 0.5
		? say("0 级 · 无风静止", "Force 0 · Calm")
		: effectiveWind < 3.5
			? say("2 级 · 轻风徐行", "Force 2 · Light breeze")
			: effectiveWind < 8
				? say("4 级 · 和风稳步", "Force 4 · Moderate breeze")
				: say("6 级 · 强海风疾行", "Force 6 · Strong sea wind"),
);

// Solve all active leg pairs for the 2D multi-leg walker
const solvedBays = $derived(
	bayOffsets.map((phase, idx) => {
		const a = (crankAngle + phase) % 360;
		const left = solveStrandbeest(a, DEFAULT_CRANK, undefined, false);
		const right = solveStrandbeest(a, DEFAULT_CRANK, undefined, true);
		const leftFootY = left.points.F?.y ?? groundFloorY;
		const rightFootY = right.points.F?.y ?? groundFloorY;
		return {
			idx,
			phase,
			angle: a,
			left,
			right,
			minFootY: Math.min(leftFootY, rightFootY),
			leftTouching: leftFootY <= groundFloorY + 2.8,
			rightTouching: rightFootY <= groundFloorY + 2.8,
		};
	}),
);

// Supporting foot height determines vertical chassis bounce
const lowestActiveFootY = $derived(
	Math.min(...solvedBays.map((b) => b.minFootY)),
);
const chassisBounce = $derived(groundFloorY - lowestActiveFootY);

async function ensure3DController() {
	if (controller3d || webglLoading || !canvas3d) return;
	webglLoading = true;
	webglError = false;
	try {
		const mod = await import("./create-beast-3d");
		if (!canvas3d) return;
		controller3d = mod.createBeast3D({
			canvas: canvas3d,
			initialAngle: crankAngle,
			crank: DEFAULT_CRANK,
			explode: explodeFactor,
		});
		controller3d.setCameraPreset(activeCameraPreset);
	} catch {
		webglError = true;
	} finally {
		webglLoading = false;
	}
}

function switchView(mode: "wind2d" | "stereo3d") {
	viewMode = mode;
	playLatchClick();
	if (mode === "stereo3d") {
		setTimeout(() => {
			ensure3DController();
		}, 0);
	}
}

function triggerGust() {
	gustBoost = 6.5;
	if (!playing) {
		playing = true;
		syncFrame();
	}
	playLatchClick();
}

function setPresetCamera(preset: BeastCameraPreset) {
	activeCameraPreset = preset;
	controller3d?.setCameraPreset(preset);
	playMechanicalTick();
}

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
	const elapsed = Math.min(time - lastTime, 65);
	if (elapsed >= 1000 / 60) {
		if (gustBoost > 0.05) {
			gustBoost = Math.max(0, gustBoost - elapsed * 0.0028);
		} else {
			gustBoost = 0;
		}
		const degDelta = elapsed * 0.0095 * effectiveWind;
		crankAngle = (crankAngle + degDelta) % 360;
		beachScroll = (beachScroll + degDelta * 0.34) % 48;
		if (viewMode === "stereo3d" && controller3d) {
			controller3d.updateState({
				angle: crankAngle,
				crank: DEFAULT_CRANK,
				explode: explodeFactor,
				windSpeed: effectiveWind,
			});
		}
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

$effect(() => {
	if (viewMode === "stereo3d" && controller3d) {
		controller3d.updateState({
			angle: crankAngle,
			crank: DEFAULT_CRANK,
			explode: explodeFactor,
			windSpeed: effectiveWind,
		});
	}
});

onMount(() => {
	ready = true;
	const media = window.matchMedia("(prefers-reduced-motion: reduce)");
	reduced = media.matches;
	playing = !reduced;
	const locale = () => {
		english = document.documentElement.lang.startsWith("en");
	};
	const motion = () => {
		reduced = media.matches;
		if (reduced) {
			playing = false;
			stopFrame();
		}
	};
	locale();
	const observer = new IntersectionObserver(
		([entry]) => {
			visible = entry.isIntersecting;
			syncFrame();
		},
		{ threshold: 0.08 },
	);
	observer.observe(root);
	const handleResize = () => controller3d?.resize();
	window.addEventListener("resize", handleResize);
	media.addEventListener("change", motion);
	document.addEventListener("visibilitychange", syncFrame);
	document.addEventListener("site-locale-change", locale);
	return () => {
		stopFrame();
		observer.disconnect();
		window.removeEventListener("resize", handleResize);
		media.removeEventListener("change", motion);
		document.removeEventListener("visibilitychange", syncFrame);
		document.removeEventListener("site-locale-change", locale);
		controller3d?.dispose();
		controller3d = null;
	};
});
</script>

<div class="beast-stage" bind:this={root}>
	<div class="stage-header">
		<div class="mode-tabs" role="tablist" aria-label={say("整机展示模式", "Whole-beast display mode")}>
			<button
				type="button"
				role="tab"
				aria-selected={viewMode === "wind2d"}
				class:active={viewMode === "wind2d"}
				disabled={!ready}
				onclick={() => switchView("wind2d")}
			>
				🌬️ {say("2D 海滩风力多腿协同", "2D Wind & Multi-Leg Walker")}
			</button>
			<button
				type="button"
				role="tab"
				aria-selected={viewMode === "stereo3d"}
				class:active={viewMode === "stereo3d"}
				disabled={!ready}
				onclick={() => switchView("stereo3d")}
			>
				🧊 {say("3D 立体 PVC 骨架拆解 (Three.js)", "3D PVC Skeleton & Crankshaft")}
			</button>
		</div>

		{#if viewMode === "wind2d"}
			<div class="pair-selector">
				<button
					type="button"
					class:active={legPairsCount === 3}
					disabled={!ready}
					onclick={() => {
						legPairsCount = 3;
						playMechanicalTick();
					}}
				>
					{say("3 对 6 腿 (120° 错相平稳)", "3 Pairs / 6 Legs (120° Phased)")}
				</button>
				<button
					type="button"
					class:active={legPairsCount === 1}
					disabled={!ready}
					onclick={() => {
						legPairsCount = 1;
						playMechanicalTick();
					}}
				>
					{say("1 对 2 腿 (机身颠簸对照)", "1 Pair / 2 Legs (Bounce Demo)")}
				</button>
			</div>
		{:else}
			<div class="pair-selector">
				{#each [
					["isometric", say("立体视角", "Isometric")],
					["side", say("正侧面", "Side")],
					["axial", say("曲柄轴向", "Axial 120°")],
					["top", say("俯视图", "Top")],
				] as [preset, label]}
					<button
						type="button"
						class:active={activeCameraPreset === preset}
						disabled={!ready}
						onclick={() => setPresetCamera(preset as BeastCameraPreset)}
					>
						{label}
					</button>
				{/each}
			</div>
		{/if}
	</div>

	{#if viewMode === "wind2d"}
		<div class="wind-canvas-wrap">
			<svg
				viewBox="-92 -106 260 206"
				role="img"
				aria-label={say(
					"风帆驱动的多腿 Strandbeest 海滩行走仿真",
					"Wind-sail driven multi-leg Strandbeest walking across a beach",
				)}
			>
				<title>{say("海风驱动多腿动力兽行走", "Wind-driven multi-leg Strandbeest")}</title>

				<!-- Wind streamlines in the sky -->
				{#if effectiveWind > 0.3}
					{#each [0, 1, 2, 3] as wIdx}
						{@const wy = -88 + wIdx * 28}
						{@const wx = -85 + ((beachScroll * (1.8 + wIdx * 0.25) + wIdx * 55) % 210)}
						<line
							x1={wx}
							y1={wy}
							x2={wx + 14 + effectiveWind * 1.4}
							y2={wy}
							class="wind-line"
						/>
						<circle
							cx={wx + 16 + effectiveWind * 1.4}
							cy={wy}
							r="0.7"
							class="wind-particle"
						/>
					{/each}
				{/if}

				<!-- Scrolling Beach Ground & Sand Footprints -->
				<line
					x1="-92"
					y1={-groundFloorY + 0.8}
					x2="168"
					y2={-groundFloorY + 0.8}
					class="beach-surface"
				/>
				{#each [0, 1, 2, 3, 4, 5, 6] as sandIdx}
					{@const sx = 160 - ((beachScroll + sandIdx * 36) % 252)}
					<ellipse
						cx={sx}
						cy={-groundFloorY + 3.2}
						rx="3.8"
						ry="1.1"
						class="sand-footprint"
					/>
				{/each}

				<!-- Chassis Group with realistic vertical bounce based on active supporting feet -->
				<g transform={`translate(0, ${-chassisBounce.toFixed(2)})`}>
					<!-- Top Wind Sails / Wings oscillating around central mast (x=38) -->
					{#each solvedBays as bay}
						{@const sway = Math.sin((bay.angle * Math.PI) / 180) * (6 + effectiveWind * 0.65)}
						<g opacity={bay.idx === 0 ? 0.92 : 0.42}>
							<line
								x1="38"
								y1="-7.8"
								x2={38 + sway}
								y2="-52"
								class="mast-pole"
							/>
							<path
								d={`M 38 -16 Q ${48 + sway * 1.3} -34 ${38 + sway} -50 L 38 -24 Z`}
								class="wind-sail"
							/>
						</g>
					{/each}

					<!-- Spine beam connecting G_left (0,0), O (38,-7.8), and G_right (76,0) -->
					<line x1="0" y1="0" x2="76" y2="0" class="chassis-beam" />
					<line x1="0" y1="0" x2="38" y2="-7.8" class="chassis-strut" />
					<line x1="76" y1="0" x2="38" y2="-7.8" class="chassis-strut" />

					<!-- Crank circle -->
					<circle cx="38" cy="-7.8" r={DEFAULT_CRANK} class="crank-orbit" />

					<!-- Render back-to-front leg bays -->
					{#each [...solvedBays].reverse() as bay}
						{@const isFront = bay.idx === 0}
						<g
							class="bay-group"
							opacity={isFront ? 1 : bay.idx === 1 ? 0.52 : 0.3}
						>
							<!-- Left & Right legs for this bay -->
							{#each [bay.left, bay.right] as legPose, sideIdx}
								<path
									d={polygonPath(legPose, ["G", "K", "L"])}
									class={sideIdx === 0 ? "tri-upper-left" : "tri-upper-right"}
								/>
								<path
									d={polygonPath(legPose, ["M", "N", "F"])}
									class="tri-lower"
								/>
								<path d={rodPath(legPose)} class="bay-rods" />
								{#if legPose.points.F}
									<circle
										cx={legPose.points.F.x}
										cy={-legPose.points.F.y}
										r={isFront ? 1.85 : 1.4}
										class="bay-foot"
									/>
								{/if}
							{/each}

							<!-- Crank arm for this phase -->
							{#if bay.left.points.O && bay.left.points.C}
								<line
									x1={bay.left.points.O.x}
									y1={-bay.left.points.O.y}
									x2={bay.left.points.C.x}
									y2={-bay.left.points.C.y}
									class="bay-crank"
								/>
								<circle
									cx={bay.left.points.C.x}
									cy={-bay.left.points.C.y}
									r="1.35"
									class="bay-pin"
								/>
							{/if}
						</g>
					{/each}

					<!-- Fixed frame pivots GL(0,0), O(38,-7.8), GR(76,0) -->
					<circle cx="0" cy="0" r="2.8" class="fixed-ring" />
					<circle cx="38" cy="-7.8" r="2.8" class="fixed-ring" />
					<circle cx="76" cy="0" r="2.8" class="fixed-ring" />
				</g>

				<!-- Ground contact indicators directly on the beach line -->
				{#each solvedBays as bay}
					{#if bay.leftTouching && bay.left.points.F}
						<circle
							cx={bay.left.points.F.x}
							cy={-groundFloorY + 0.8}
							r="2.8"
							class="contact-ring"
						/>
					{/if}
					{#if bay.rightTouching && bay.right.points.F}
						<circle
							cx={bay.right.points.F.x}
							cy={-groundFloorY + 0.8}
							r="2.8"
							class="contact-ring"
						/>
					{/if}
				{/each}
			</svg>

			<div class="wind-telemetry">
				<span>
					<b>{beaufortLabel}</b> · {effectiveWind.toFixed(1)} m/s
				</span>
				<span>
					{say("机身垂直起伏：", "Chassis vertical bounce: ")}
					<b>{Math.abs(chassisBounce).toFixed(1)}</b>
					{legPairsCount === 3
						? say("（三相交错支撑，机身平稳）", "(3-phase support keeps body level)")
						: say("（单对腿缺乏交替支撑，大幅起伏）", "(Single pair bounces heavily)")}
				</span>
			</div>
		</div>
	{:else}
		<div class="stereo-canvas-wrap">
			<canvas
				bind:this={canvas3d}
				class="beast-3d-canvas"
				aria-label={say(
					"Three.js 3D 黄色 PVC 管动力兽立体交互模型（按住拖拽可 360° 旋转）",
					"Three.js 3D interactive PVC tube Strandbeest model (drag to orbit 360°)",
				)}
			></canvas>
			{#if webglLoading}
				<div class="canvas-overlay">
					{say("正在初始化 Three.js 3D 骨架场景…", "Initializing Three.js 3D skeleton…")}
				</div>
			{:else if webglError}
				<div class="canvas-overlay">
					{say(
						"当前环境未启用 WebGL，请切换回 2D 海滩风力视图查看。",
						"WebGL is unavailable in this browser. Switch to 2D Wind Walker view.",
					)}
				</div>
			{/if}
			<div class="wind-telemetry">
				<span>
					{say(
						"交互提示：在画布上按住拖拽可 360° 旋转视角，滚轮缩放；黄色/琥珀色区分左右对称腿",
						"Tip: Drag on canvas to orbit 360°, scroll to zoom; yellow/amber distinguish left & right leg pairs",
					)}
				</span>
			</div>
		</div>
	{/if}

	<div class="stage-controls">
		<button
			type="button"
			class="play-btn"
			disabled={!ready}
			onclick={() => {
				playing = !playing;
				playLatchClick();
				syncFrame();
			}}
		>
			<span aria-hidden="true">{playing ? "Ⅱ" : "▷"}</span>
			{playing ? say("暂停", "Pause") : say("播放", "Play")}
		</button>

		<label class="slider-box" for={`${uid}-wind`}>
			<span>
				{say("海风风速", "Sea Wind Speed")}
				<output>{effectiveWind.toFixed(1)} m/s</output>
			</span>
			<input
				id={`${uid}-wind`}
				type="range"
				min="0"
				max="12"
				step="0.5"
				bind:value={windSpeed}
				disabled={!ready}
				oninput={() => playSlideTick(windSpeed * 0.2)}
			/>
		</label>

		{#if viewMode === "stereo3d"}
			<label class="slider-box" for={`${uid}-explode`}>
				<span>
					{say("Z 轴曲柄层拆解展开", "Z-Axis Layer Explode")}
					<output>{explodeFactor.toFixed(2)}×</output>
				</span>
				<input
					id={`${uid}-explode`}
					type="range"
					min="1"
					max="2.6"
					step="0.05"
					bind:value={explodeFactor}
					disabled={!ready}
					oninput={() => playSlideTick(1)}
				/>
			</label>
		{:else}
			<button
				type="button"
				class="gust-btn"
				disabled={!ready}
				onclick={triggerGust}
			>
				🌬️ {say("吹一阵海风", "Send a Sea Gust")}
			</button>
		{/if}
	</div>
</div>

<style>
	.beast-stage{border:1px solid var(--brand-line);border-radius:6px;padding:22px 24px;margin:28px auto 40px;max-width:820px;background:linear-gradient(180deg,color-mix(in srgb,var(--brand-sage) 14%,transparent),color-mix(in srgb,var(--brand-peach) 8%,transparent))}
	.stage-header{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:10px;margin-bottom:14px}
	.mode-tabs,.pair-selector{display:flex;flex-wrap:wrap;gap:6px}
	.mode-tabs button,.pair-selector button{font:inherit;font-size:12.5px;padding:6px 12px;min-height:34px;border:1px solid var(--brand-line);border-radius:999px;background:color-mix(in srgb,var(--brand-paper) 70%,transparent);color:var(--brand-ink);cursor:pointer;transition:background .15s,border-color .15s}
	.mode-tabs button.active{background:var(--brand-sage);border-color:var(--brand-accent);font-weight:500}
	.pair-selector button.active{background:var(--brand-peach);border-color:var(--brand-clay);font-weight:500}
	.wind-canvas-wrap,.stereo-canvas-wrap{position:relative;border:1px solid var(--brand-line);border-radius:5px;background:color-mix(in srgb,var(--brand-paper) 82%,transparent);overflow:hidden}
	.wind-canvas-wrap svg{display:block;width:100%;height:clamp(270px,42vw,390px)}
	.beast-3d-canvas{display:block;width:100%;height:clamp(290px,44vw,410px);cursor:grab;touch-action:none}
	.beast-3d-canvas:active{cursor:grabbing}
	.canvas-overlay{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:13px;color:var(--brand-muted);background:color-mix(in srgb,var(--brand-paper) 85%,transparent)}
	.wind-line{stroke:var(--brand-accent);stroke-width:.55;stroke-dasharray:4 2;opacity:.45;stroke-linecap:round}
	.wind-particle{fill:var(--brand-accent);opacity:.5}
	.beach-surface{stroke:var(--brand-clay);stroke-width:.7;opacity:.55}
	.sand-footprint{fill:var(--brand-clay);opacity:.22}
	.mast-pole{stroke:var(--brand-ink);stroke-width:.85;stroke-linecap:round}
	.wind-sail{fill:var(--brand-peach);fill-opacity:.58;stroke:var(--brand-clay);stroke-width:.55}
	.chassis-beam{stroke:var(--brand-ink);stroke-width:1.6;stroke-linecap:round}
	.chassis-strut{stroke:var(--brand-ink);stroke-width:1.05;stroke-dasharray:1.5 1;opacity:.7}
	.crank-orbit{fill:none;stroke:var(--brand-muted);stroke-width:.45;stroke-dasharray:1 1.4;opacity:.6}
	.tri-upper-left{fill:var(--brand-sage);fill-opacity:.55}
	.tri-upper-right{fill:var(--brand-peach);fill-opacity:.45}
	.tri-lower{fill:var(--brand-peach);fill-opacity:.58}
	.bay-rods{fill:none;stroke:var(--brand-ink);stroke-width:1.05;stroke-linecap:round;stroke-linejoin:round}
	.bay-crank{stroke:var(--brand-clay);stroke-width:1.65;stroke-linecap:round}
	.bay-pin{fill:var(--brand-clay)}
	.bay-foot{fill:var(--brand-clay);stroke:var(--brand-paper);stroke-width:.45}
	.fixed-ring{fill:var(--brand-paper);stroke:var(--brand-ink);stroke-width:.75}
	.contact-ring{fill:var(--brand-clay);fill-opacity:.28;stroke:var(--brand-clay);stroke-width:.6}
	.wind-telemetry{display:flex;flex-wrap:wrap;justify-content:space-between;gap:8px;padding:8px 14px;border-top:1px solid var(--brand-line);font-size:12px;color:var(--brand-muted);background:color-mix(in srgb,var(--brand-paper) 60%,transparent)}
	.wind-telemetry b{color:var(--brand-ink);font-weight:500}
	.stage-controls{display:flex;flex-wrap:wrap;align-items:center;gap:16px;margin-top:14px}
	.play-btn,.gust-btn{font:inherit;font-size:13px;padding:8px 14px;min-height:38px;border:1px solid var(--brand-line);border-radius:4px;background:transparent;color:var(--brand-ink);cursor:pointer;white-space:nowrap}
	.gust-btn{background:color-mix(in srgb,var(--brand-peach) 35%,transparent);border-color:var(--brand-clay);font-weight:500}
	.play-btn:hover:not(:disabled),.gust-btn:hover:not(:disabled){background:var(--brand-sage)}
	.slider-box{flex:1;min-width:170px;font-size:13px;color:var(--brand-body)}
	.slider-box>span{display:flex;justify-content:space-between;gap:8px}
	.slider-box output{font:12.5px "JetBrains Mono",monospace;color:var(--brand-ink)}
	.slider-box input[type=range]{width:100%;height:26px;margin-top:2px;accent-color:var(--brand-clay);cursor:pointer}
	@media(max-width:600px){
		.beast-stage{padding:14px 12px;margin-left:-8px;margin-right:-8px}
		.stage-controls{gap:10px}
		.wind-telemetry{flex-direction:column;gap:4px}
	}
</style>
