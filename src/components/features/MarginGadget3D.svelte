<script lang="ts">
import { onMount } from "svelte";
import { playLatchClick, playSlideTick } from "@/utils/haptics";

interface Props {
	initialMode?: "saddle" | "astrolabe" | "balance";
}

let { initialMode = "saddle" }: Props = $props();

let english = $state(false);
let mode = $state<"saddle" | "astrolabe" | "balance">("saddle");
let yaw = $state(0.62);
let pitch = $state(0.42);
let dragging = $state(false);
let autoRotate = $state(true);
let param = $state(0.55); // momentum / frequency / exploration c
let marbleT = $state(0);

let canvasEl: HTMLCanvasElement | undefined = $state();
let lastX = 0;
let lastY = 0;

$effect(() => {
	mode = initialMode;
});

function project3D(
	x: number,
	y: number,
	z: number,
	width: number,
	height: number,
	scale: number,
): { sx: number; sy: number; depth: number } {
	// Rotate around Z (yaw) then X (pitch)
	const cosY = Math.cos(yaw);
	const sinY = Math.sin(yaw);
	const x1 = x * cosY - y * sinY;
	const y1 = x * sinY + y * cosY;
	const z1 = z;

	const cosP = Math.cos(pitch);
	const sinP = Math.sin(pitch);
	const y2 = y1 * cosP - z1 * sinP;
	const z2 = y1 * sinP + z1 * cosP;

	const perspective = 4.2 / (4.2 + y2 * 0.45);
	return {
		sx: width / 2 + x1 * scale * perspective,
		sy: height / 2 - z2 * scale * perspective,
		depth: y2,
	};
}

function drawScene(
	ctx: CanvasRenderingContext2D,
	width: number,
	height: number,
) {
	ctx.clearRect(0, 0, width, height);
	const scale = Math.min(width, height) * 0.31;

	if (mode === "saddle") {
		// Draw hyperbolic paraboloid z = 0.55 * (x^2 - y^2)
		const steps = 10;
		ctx.lineWidth = 1.1;
		for (let i = -steps; i <= steps; i += 2) {
			const u = i / steps;
			ctx.beginPath();
			ctx.strokeStyle =
				i === 0 ? "rgba(65, 99, 79, 0.78)" : "rgba(65, 99, 79, 0.28)";
			for (let j = -steps; j <= steps; j++) {
				const v = j / steps;
				const z = 0.52 * (u * u - v * v);
				const p = project3D(u, v, z, width, height, scale);
				if (j === -steps) ctx.moveTo(p.sx, p.sy);
				else ctx.lineTo(p.sx, p.sy);
			}
			ctx.stroke();

			ctx.beginPath();
			ctx.strokeStyle =
				i === 0 ? "rgba(145, 86, 56, 0.65)" : "rgba(65, 99, 79, 0.28)";
			for (let j = -steps; j <= steps; j++) {
				const v = j / steps;
				const z = 0.52 * (v * v - u * u);
				const p = project3D(v, u, z, width, height, scale);
				if (j === -steps) ctx.moveTo(p.sx, p.sy);
				else ctx.lineTo(p.sx, p.sy);
			}
			ctx.stroke();
		}

		// Draw optimizer trajectory escaping saddle point along negative curvature y
		const phase = (marbleT % 180) / 180;
		const mx = 0.75 * Math.exp(-3.2 * phase) * Math.cos(phase * 6 * param);
		const my = (0.06 + 0.92 * phase * phase) * (param >= 0.5 ? 1 : -1);
		const mz = 0.52 * (mx * mx - my * my);

		ctx.beginPath();
		ctx.strokeStyle = "#915638";
		ctx.lineWidth = 2;
		for (let s = 0; s <= 40; s++) {
			const sp = (s / 40) * phase;
			const tx = 0.75 * Math.exp(-3.2 * sp) * Math.cos(sp * 6 * param);
			const ty = (0.06 + 0.92 * sp * sp) * (param >= 0.5 ? 1 : -1);
			const tz = 0.52 * (tx * tx - ty * ty);
			const tp = project3D(tx, ty, tz, width, height, scale);
			if (s === 0) ctx.moveTo(tp.sx, tp.sy);
			else ctx.lineTo(tp.sx, tp.sy);
		}
		ctx.stroke();

		const mp = project3D(mx, my, mz, width, height, scale);
		ctx.beginPath();
		ctx.arc(mp.sx, mp.sy, 6.5, 0, Math.PI * 2);
		ctx.fillStyle = "#f3c7ac";
		ctx.fill();
		ctx.lineWidth = 2;
		ctx.strokeStyle = "#915638";
		ctx.stroke();
	} else if (mode === "astrolabe") {
		// Three nested 3D gimbal rings rotating at harmonic frequencies
		const rings = [
			{ radius: 1.05, speed: 1.0, stroke: "rgba(65, 99, 79, 0.75)", tilt: 0.2 },
			{
				radius: 0.82,
				speed: 2.1 * param + 0.4,
				stroke: "rgba(145, 86, 56, 0.78)",
				tilt: 0.95,
			},
			{
				radius: 0.58,
				speed: 4.3 * param + 0.8,
				stroke: "rgba(37, 60, 52, 0.85)",
				tilt: -0.75,
			},
		];

		const t = marbleT * 0.025;
		for (const ring of rings) {
			ctx.beginPath();
			ctx.strokeStyle = ring.stroke;
			ctx.lineWidth = 1.8;
			const segments = 56;
			const a = t * ring.speed;
			for (let s = 0; s <= segments; s++) {
				const phi = (s / segments) * Math.PI * 2;
				const rx = ring.radius * Math.cos(phi);
				const ry = ring.radius * Math.sin(phi) * Math.cos(ring.tilt + a);
				const rz = ring.radius * Math.sin(phi) * Math.sin(ring.tilt + a);
				const pt = project3D(rx, ry, rz, width, height, scale);
				if (s === 0) ctx.moveTo(pt.sx, pt.sy);
				else ctx.lineTo(pt.sx, pt.sy);
			}
			ctx.stroke();

			// Orbiting token marker on ring
			const mx = ring.radius * Math.cos(a * 1.4);
			const my = ring.radius * Math.sin(a * 1.4) * Math.cos(ring.tilt + a);
			const mz = ring.radius * Math.sin(a * 1.4) * Math.sin(ring.tilt + a);
			const mpt = project3D(mx, my, mz, width, height, scale);
			ctx.beginPath();
			ctx.arc(mpt.sx, mpt.sy, 4.5, 0, Math.PI * 2);
			ctx.fillStyle = "#f3c7ac";
			ctx.fill();
			ctx.strokeStyle = "#915638";
			ctx.stroke();
		}

		const center = project3D(0, 0, 0, width, height, scale);
		ctx.beginPath();
		ctx.arc(center.sx, center.sy, 5, 0, Math.PI * 2);
		ctx.fillStyle = "#41634f";
		ctx.fill();
	} else {
		// 3D UCT Torque Balance Scale
		const tilt = (param - 0.5) * 0.55 + Math.sin(marbleT * 0.04) * 0.05;
		const baseBottom = project3D(0, 0, -0.75, width, height, scale);
		const fulcrum = project3D(0, 0, 0.25, width, height, scale);

		// Pillar
		ctx.beginPath();
		ctx.strokeStyle = "#41634f";
		ctx.lineWidth = 2.5;
		ctx.moveTo(baseBottom.sx, baseBottom.sy);
		ctx.lineTo(fulcrum.sx, fulcrum.sy);
		ctx.stroke();

		// Circular base
		ctx.beginPath();
		ctx.strokeStyle = "rgba(65, 99, 79, 0.45)";
		ctx.lineWidth = 1.4;
		for (let s = 0; s <= 36; s++) {
			const phi = (s / 36) * Math.PI * 2;
			const bp = project3D(
				0.45 * Math.cos(phi),
				0.45 * Math.sin(phi),
				-0.75,
				width,
				height,
				scale,
			);
			if (s === 0) ctx.moveTo(bp.sx, bp.sy);
			else ctx.lineTo(bp.sx, bp.sy);
		}
		ctx.stroke();

		// Beam
		const lx = -0.95 * Math.cos(tilt);
		const lz = 0.25 - 0.95 * Math.sin(tilt);
		const rx = 0.95 * Math.cos(tilt);
		const rz = 0.25 + 0.95 * Math.sin(tilt);

		const leftTip = project3D(lx, 0, lz, width, height, scale);
		const rightTip = project3D(rx, 0, rz, width, height, scale);

		ctx.beginPath();
		ctx.strokeStyle = "#253c34";
		ctx.lineWidth = 2.4;
		ctx.moveTo(leftTip.sx, leftTip.sy);
		ctx.lineTo(rightTip.sx, rightTip.sy);
		ctx.stroke();

		// Suspended pans
		for (const [endX, endZ, isExplore] of [
			[lx, lz, false],
			[rx, rz, true],
		] as [number, number, boolean][]) {
			const topPt = project3D(endX, 0, endZ, width, height, scale);
			const panZ = endZ - 0.52;
			const panCenter = project3D(endX, 0, panZ, width, height, scale);

			ctx.beginPath();
			ctx.strokeStyle = "rgba(65, 99, 79, 0.55)";
			ctx.lineWidth = 1.2;
			ctx.moveTo(topPt.sx, topPt.sy);
			ctx.lineTo(panCenter.sx, panCenter.sy);
			ctx.stroke();

			ctx.beginPath();
			ctx.arc(
				panCenter.sx,
				panCenter.sy,
				isExplore ? 9 + param * 6 : 12,
				0,
				Math.PI * 2,
			);
			ctx.fillStyle = isExplore ? "#f3c7ac" : "#d5e4d8";
			ctx.fill();
			ctx.strokeStyle = isExplore ? "#915638" : "#41634f";
			ctx.lineWidth = 1.8;
			ctx.stroke();
		}
	}
}

function handlePointerDown(e: PointerEvent) {
	dragging = true;
	autoRotate = false;
	lastX = e.clientX;
	lastY = e.clientY;
	e.currentTarget.setPointerCapture(e.pointerId);
}

function handlePointerMove(e: PointerEvent) {
	if (!dragging) return;
	const dx = e.clientX - lastX;
	const dy = e.clientY - lastY;
	lastX = e.clientX;
	lastY = e.clientY;
	yaw += dx * 0.011;
	pitch = Math.max(-1.1, Math.min(1.1, pitch + dy * 0.011));
}

function handlePointerUp() {
	dragging = false;
}

onMount(() => {
	const sync = () => {
		english = document.documentElement.lang.startsWith("en");
	};
	sync();
	document.addEventListener("site-locale-change", sync);

	const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
	if (reduced) autoRotate = false;

	let frameId = 0;
	const renderLoop = () => {
		if (canvasEl) {
			const dpr = Math.min(2, window.devicePixelRatio || 1);
			const rect = canvasEl.getBoundingClientRect();
			const w = Math.max(240, Math.floor(rect.width));
			const h = Math.max(200, Math.floor(rect.height));
			if (canvasEl.width !== w * dpr || canvasEl.height !== h * dpr) {
				canvasEl.width = w * dpr;
				canvasEl.height = h * dpr;
			}
			const ctx = canvasEl.getContext("2d");
			if (ctx) {
				ctx.save();
				ctx.scale(dpr, dpr);
				drawScene(ctx, w, h);
				ctx.restore();
			}
		}
		if (autoRotate && !dragging) {
			yaw += 0.006;
		}
		if (!reduced) {
			marbleT += 1;
		}
		frameId = requestAnimationFrame(renderLoop);
	};
	frameId = requestAnimationFrame(renderLoop);

	return () => {
		cancelAnimationFrame(frameId);
		document.removeEventListener("site-locale-change", sync);
	};
});
</script>

<aside
	class="gadget-3d"
	aria-label={english
		? "Interactive 3D Mathematical Apparatus"
		: "交互式三维数学与机械小摆件"}
>
	<div class="gadget-header">
		<div>
			<span class="kicker">3D APPARATUS · DRAG TO ORBIT</span>
			<h3>
				{mode === "saddle"
					? english
						? "Saddle Landscape & Momentum Escape"
						: "三维鞍点曲面与动量逃逸轨迹"
					: mode === "astrolabe"
						? english
							? "Harmonic RoPE Gimbal Sphere"
							: "三维多频相位旋转浑天仪"
						: english
							? "UCT Exploration–Exploitation Balance"
							: "UCT 探索与利用三维力矩天平"}
			</h3>
		</div>

		<div class="gadget-tabs" role="group" aria-label="Apparatus mode">
			<button
				type="button"
				class:active={mode === "saddle"}
				onclick={() => {
					mode = "saddle";
					marbleT = 0;
					playLatchClick();
				}}
			>
				{english ? "01 Saddle" : "01 鞍点曲面"}
			</button>
			<button
				type="button"
				class:active={mode === "astrolabe"}
				onclick={() => {
					mode = "astrolabe";
					playLatchClick();
				}}
			>
				{english ? "02 Gimbal" : "02 旋转浑天仪"}
			</button>
			<button
				type="button"
				class:active={mode === "balance"}
				onclick={() => {
					mode = "balance";
					playLatchClick();
				}}
			>
				{english ? "03 UCT Scale" : "03 力矩天平"}
			</button>
		</div>
	</div>

	<div class="canvas-stage">
		<canvas
			bind:this={canvasEl}
			onpointerdown={handlePointerDown}
			onpointermove={handlePointerMove}
			onpointerup={handlePointerUp}
			onpointercancel={handlePointerUp}
		></canvas>
		<span class="orbit-hint">
			{english ? "Drag to rotate 3D view ↻" : "按住拖拽可自由旋转三维视角 ↻"}
		</span>
	</div>

	<div class="gadget-controls">
		<label for="gadget-param">
			<span>
				{mode === "saddle"
					? english
						? "Curvature Perturbation"
						: "动量扰动参数"
					: mode === "astrolabe"
						? english
							? "Harmonic Frequency Ratio"
							: "多频谐波比"
						: english
							? "Exploration Torque (c)"
							: "探索力矩权重 (c)"}
			</span>
			<input
				id="gadget-param"
				type="range"
				min="0.1"
				max="1.0"
				step="0.05"
				value={param}
				oninput={(e) => {
					param = Number(e.currentTarget.value);
					if (mode === "saddle") marbleT = 0;
					playSlideTick(1);
				}}
			/>
		</label>
		<button
			type="button"
			class="quiet-btn"
			onclick={() => {
				autoRotate = !autoRotate;
			}}
		>
			{autoRotate
				? english
					? "Pause Orbit"
					: "暂停自转"
				: english
					? "Auto Orbit"
					: "开启自转"}
		</button>
	</div>
</aside>

<style>
	.gadget-3d {
		background: var(--brand-surface);
		border: 1px solid var(--brand-line);
		border-radius: 18px;
		padding: 22px 24px;
		color: var(--brand-ink);
		display: grid;
		gap: 14px;
	}
	.gadget-header {
		display: flex;
		justify-content: space-between;
		align-items: flex-start;
		flex-wrap: wrap;
		gap: 12px;
	}
	.kicker {
		font: 10px var(--font-mono);
		letter-spacing: 0.1em;
		color: var(--brand-accent);
	}
	.gadget-header h3 {
		margin: 4px 0 0;
		font: 400 22px/1.3 var(--font-serif);
	}
	.gadget-tabs {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}
	button {
		font: 11px var(--font-mono);
		padding: 6px 12px;
		border-radius: 999px;
		border: 1px solid var(--brand-line);
		background: var(--brand-paper);
		color: var(--brand-body);
		cursor: pointer;
	}
	button.active {
		background: var(--brand-accent);
		color: var(--brand-paper);
		border-color: var(--brand-accent);
	}
	.canvas-stage {
		position: relative;
		height: 230px;
		background: var(--brand-paper);
		border: 1px solid var(--brand-line);
		border-radius: 14px;
		overflow: hidden;
	}
	canvas {
		width: 100%;
		height: 100%;
		display: block;
		cursor: grab;
		touch-action: none;
	}
	canvas:active {
		cursor: grabbing;
	}
	.orbit-hint {
		position: absolute;
		right: 12px;
		bottom: 10px;
		font: 10px var(--font-mono);
		color: var(--brand-muted);
		pointer-events: none;
	}
	.gadget-controls {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 16px;
		flex-wrap: wrap;
	}
	.gadget-controls label {
		flex: 1;
		min-width: 180px;
		display: flex;
		align-items: center;
		gap: 12px;
		font-size: 12px;
		color: var(--brand-body);
	}
	.gadget-controls input[type="range"] {
		flex: 1;
		accent-color: var(--brand-accent);
	}
</style>
