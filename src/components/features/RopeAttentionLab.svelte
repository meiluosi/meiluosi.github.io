<script lang="ts">
import { onMount } from "svelte";
import {
	computeRopeSnapshot,
	type RopeScalingMode,
} from "@/utils/rope-attention";

let english = $state(false);
let queryPos = $state(18);
let keyPos = $state(11);
let mode = $state<RopeScalingMode>("standard");
let scaleFactor = $state(4);

const snapshot = $derived(
	computeRopeSnapshot(queryPos, keyPos, 100, scaleFactor, mode),
);

function shiftBoth(delta: number) {
	const maxShift = Math.min(64 - queryPos, 64 - keyPos);
	const minShift = Math.max(-queryPos, -keyPos);
	const step =
		delta > 0
			? maxShift >= delta
				? delta
				: minShift
			: Math.max(minShift, delta);
	queryPos += step;
	keyPos += step;
}

function polarX(cx: number, r: number, rad: number): number {
	return cx + r * Math.cos(rad - Math.PI / 2);
}

function polarY(cy: number, r: number, rad: number): number {
	return cy + r * Math.sin(rad - Math.PI / 2);
}

function xForDist(d: number): number {
	return 48 + (Math.min(64, Math.max(0, d)) / 64) * 448;
}

function yForKernel(val: number): number {
	// val in [-0.4, 1.05]
	const clamped = Math.max(-0.4, Math.min(1.05, val));
	return 168 - ((clamped + 0.4) / 1.45) * 136;
}

const decayPath = $derived(
	snapshot.decayCurve
		.map(
			(pt, idx) =>
				`${idx === 0 ? "M" : "L"}${xForDist(pt.distance).toFixed(1)} ${yForKernel(pt.kernelScore).toFixed(1)}`,
		)
		.join(" "),
);

const absDist = $derived(Math.abs(snapshot.relativeDistance));

onMount(() => {
	const sync = () => {
		english = document.documentElement.lang.startsWith("en");
	};
	sync();
	document.addEventListener("site-locale-change", sync);
	return () => {
		document.removeEventListener("site-locale-change", sync);
	};
});
</script>

<section
	class="rope-lab"
	aria-label={english
		? "RoPE Rotary Position Embedding Astrolabe Workbench"
		: "RoPE 旋转位置编码星盘实验台"}
>
	<div class="controls-top">
		<label for="rope-m" class="pos-control">
			<div class="top-row">
				<span>{english ? "Query Token Position (m)" : "Query 词元位置 (m)"}</span>
				<strong>m = {queryPos}</strong>
			</div>
			<input
				id="rope-m"
				type="range"
				min="0"
				max="64"
				step="1"
				value={queryPos}
				oninput={(e) => (queryPos = Number(e.currentTarget.value))}
			/>
		</label>

		<label for="rope-n" class="pos-control">
			<div class="top-row">
				<span>{english ? "Key Token Position (n)" : "Key 词元位置 (n)"}</span>
				<strong>n = {keyPos}</strong>
			</div>
			<input
				id="rope-n"
				type="range"
				min="0"
				max="64"
				step="1"
				value={keyPos}
				oninput={(e) => (keyPos = Number(e.currentTarget.value))}
			/>
		</label>

		<div class="invariance-box">
			<small>
				{english ? "TRANSLATION INVARIANCE TEST" : "相对位置平移不变性检验"}
			</small>
			<button type="button" class="shift-btn" onclick={() => shiftBoth(5)}>
				{english
					? "Shift Both Tokens (m+5, n+5) →"
					: "同时平移两个词元 (m+5, n+5) →"}
			</button>
			<p>
				{english
					? "Watch every dial rotate while Δ = m − n and the attention score stay locked."
					: "观察所有指针同步旋转，但夹角 Δ = m − n 与注意力内积保持严格不变。"}
			</p>
		</div>
	</div>

	<div class="mode-bar">
		<div class="mode-pills" role="group" aria-label="Context scaling mode">
			{#each [
				{
					id: "standard" as RopeScalingMode,
					zh: "标准 RoPE (θ_k = b^{-2k/d})",
					en: "Standard RoPE",
				},
				{
					id: "linear-pi" as RopeScalingMode,
					zh: "线性位置插值 PI (θ_k / s)",
					en: "Linear Position Interpolation (PI)",
				},
				{
					id: "ntk-aware" as RopeScalingMode,
					zh: "NTK-Aware 基底扩展 (高频保真)",
					en: "NTK-Aware Base Scaling",
				},
			] as item}
				<button
					type="button"
					class:active={mode === item.id}
					onclick={() => (mode = item.id)}
				>
					{english ? item.en : item.zh}
				</button>
			{/each}
		</div>

		{#if mode !== "standard"}
			<div class="scale-selector">
				<span>{english ? "Context Scale s:" : "上下文扩展倍率 s:"}</span>
				{#each [2, 4, 8] as s}
					<button
						type="button"
						class="chip"
						class:active={scaleFactor === s}
						onclick={() => (scaleFactor = s)}
					>
						{s}×
					</button>
				{/each}
			</div>
		{/if}
	</div>

	<div class="summary-strip">
		<div class="stat-pill">
			<small>{english ? "RELATIVE DISTANCE Δ = m − n" : "相对词元距离 Δ = m − n"}</small>
			<strong>{snapshot.relativeDistance >= 0 ? `+${snapshot.relativeDistance}` : snapshot.relativeDistance}</strong>
		</div>
		<div class="stat-pill">
			<small>{english ? "MULTI-PLANE ATTENTION KERNEL" : "多频叠加注意力内积核"}</small>
			<strong>{snapshot.aggregateKernel.toFixed(4)}</strong>
		</div>
		<div class="stat-pill">
			<small>{english ? "HIGH-FREQ WAVELENGTH (k=0)" : "最高频通道波长 (k=0)"}</small>
			<strong>{snapshot.planes[0].wavelength.toFixed(1)} tokens</strong>
		</div>
		<div class="stat-pill">
			<small>{english ? "LOW-FREQ WAVELENGTH (k=5)" : "最低频通道波长 (k=5)"}</small>
			<strong>{snapshot.planes[5].wavelength.toFixed(0)} tokens</strong>
		</div>
	</div>

	<div class="astrolabe-grid">
		{#each snapshot.planes as plane}
			<article class="dial-card">
				<header>
					<h4>{english ? plane.labelEn : plane.labelZh}</h4>
					<code>θ={plane.theta.toFixed(4)}</code>
				</header>
				<svg
					viewBox="0 0 130 130"
					class="dial-svg"
					aria-label={english ? plane.labelEn : plane.labelZh}
				>
					<circle cx="65" cy="65" r="48" class="dial-ring" />
					<circle cx="65" cy="65" r="34" class="dial-inner" />
					<line x1="65" y1="12" x2="65" y2="118" class="crosshair" />
					<line x1="12" y1="65" x2="118" y2="65" class="crosshair" />

					<!-- Key vector hand (n·θ_k) -->
					<line
						x1="65"
						y1="65"
						x2={polarX(65, 42, plane.keyAngleRad)}
						y2={polarY(65, 42, plane.keyAngleRad)}
						class="hand-key"
					/>
					<circle
						cx={polarX(65, 42, plane.keyAngleRad)}
						cy={polarY(65, 42, plane.keyAngleRad)}
						r="4"
						class="tip-key"
					/>

					<!-- Query vector hand (m·θ_k) -->
					<line
						x1="65"
						y1="65"
						x2={polarX(65, 46, plane.queryAngleRad)}
						y2={polarY(65, 46, plane.queryAngleRad)}
						class="hand-query"
					/>
					<circle
						cx={polarX(65, 46, plane.queryAngleRad)}
						cy={polarY(65, 46, plane.queryAngleRad)}
						r="4.5"
						class="tip-query"
					/>

					<circle cx="65" cy="65" r="3" class="pivot" />
				</svg>
				<div class="dial-footer">
					<span>cos(Δ·θ<sub>{plane.index}</sub>)</span>
					<strong class:neg={plane.cosineContribution < 0}>
						{plane.cosineContribution.toFixed(3)}
					</strong>
				</div>
			</article>
		{/each}
	</div>

	<div class="decay-panel">
		<header>
			<div>
				<h3>
					{english
						? "Relative Distance Attention Envelope K(Δ) = (1/K) Σ cos(Δ · θ_k)"
						: "多频旋转叠加后的相对距离衰减包络 K(Δ) = (1/K) Σ cos(Δ · θ_k)"}
				</h3>
				<p>
					{english
						? "Notice how constructive interference near Δ=0 creates a natural local attention bias without explicit distance penalties."
						: "尽管每个复平面只是纯粹的周期余弦旋转，不同频率通道在 Δ=0 附近相干增强、远距离相位抵消，自然涌现出远程衰减特性。"}
				</p>
			</div>
			<div class="hand-legend">
				<span><i class="q-dot"></i>Query (m·θ<sub>k</sub>)</span>
				<span><i class="k-dot"></i>Key (n·θ<sub>k</sub>)</span>
			</div>
		</header>

		<svg
			viewBox="0 0 520 198"
			class="decay-svg"
			role="img"
			aria-label={english
				? "Attention kernel score across relative token distance"
				: "相对距离与注意力核得分曲线"}
		>
			<line x1="48" y1={yForKernel(0)} x2="496" y2={yForKernel(0)} class="zero-line" />
			<line x1="48" y1="24" x2="48" y2="168" class="axis" />
			<line x1="48" y1="168" x2="496" y2="168" class="axis" />

			<path d={decayPath} class="decay-curve" fill="none" />

			<line
				x1={xForDist(absDist)}
				y1="24"
				x2={xForDist(absDist)}
				y2="168"
				class="cursor-line"
			/>
			<circle
				cx={xForDist(absDist)}
				cy={yForKernel(snapshot.aggregateKernel)}
				r="6"
				class="cursor-dot"
			/>
			<text x="272" y="190" class="axis-caption">
				{english
					? `Relative Distance |m − n| = ${absDist} tokens`
					: `相对距离 |m − n| = ${absDist} 个词元`}
			</text>
		</svg>
	</div>
</section>

<style>
	.rope-lab {
		border: 1px solid var(--brand-line);
		border-radius: 18px;
		background: var(--brand-surface);
		padding: 26px;
		color: var(--brand-ink);
	}
	.controls-top {
		display: grid;
		grid-template-columns: 1fr 1fr 1.1fr;
		gap: 16px;
		align-items: stretch;
	}
	.pos-control,
	.invariance-box {
		background: var(--brand-paper);
		border: 1px solid var(--brand-line);
		border-radius: 12px;
		padding: 14px 16px;
		display: grid;
		gap: 8px;
	}
	.top-row {
		display: flex;
		justify-content: space-between;
		font-size: 13px;
		font-weight: 500;
	}
	.top-row strong {
		font: 600 14px var(--font-mono);
		color: var(--brand-accent);
	}
	input[type="range"] {
		width: 100%;
		accent-color: var(--brand-accent);
	}
	.invariance-box small {
		font: 9.5px var(--font-mono);
		letter-spacing: 0.08em;
		color: var(--brand-muted);
	}
	.invariance-box p {
		margin: 0;
		font-size: 11px;
		line-height: 1.5;
		color: var(--brand-body);
	}
	button {
		font: 11.5px var(--font-mono);
		border: 1px solid var(--brand-line);
		border-radius: 999px;
		padding: 8px 14px;
		background: var(--brand-paper);
		color: var(--brand-body);
		cursor: pointer;
	}
	button:hover {
		border-color: var(--brand-accent);
		color: var(--brand-ink);
	}
	button.active {
		background: var(--brand-accent);
		color: var(--brand-paper);
		border-color: var(--brand-accent);
	}
	.shift-btn {
		background: var(--brand-peach);
		color: var(--brand-ink);
		font-weight: 500;
	}
	.mode-bar {
		display: flex;
		justify-content: space-between;
		align-items: center;
		flex-wrap: wrap;
		gap: 12px;
		margin: 16px 0;
	}
	.mode-pills,
	.scale-selector {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 8px;
	}
	.scale-selector span {
		font: 11px var(--font-mono);
		color: var(--brand-muted);
	}
	.summary-strip {
		display: grid;
		grid-template-columns: repeat(4, minmax(0, 1fr));
		gap: 12px;
		margin-bottom: 18px;
	}
	.stat-pill {
		background: var(--brand-paper);
		border: 1px solid var(--brand-line);
		border-radius: 12px;
		padding: 12px 14px;
		display: grid;
		gap: 4px;
	}
	.stat-pill small {
		font: 9.5px var(--font-mono);
		color: var(--brand-muted);
	}
	.stat-pill strong {
		font: 400 20px var(--font-serif);
		color: var(--brand-accent);
	}
	.astrolabe-grid {
		display: grid;
		grid-template-columns: repeat(3, minmax(0, 1fr));
		gap: 14px;
		margin-bottom: 18px;
	}
	.dial-card {
		background: var(--brand-paper);
		border: 1px solid var(--brand-line);
		border-radius: 12px;
		padding: 12px 14px;
		display: grid;
		gap: 6px;
	}
	.dial-card header {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
		gap: 6px;
	}
	.dial-card h4 {
		margin: 0;
		font-size: 12px;
		font-weight: 600;
	}
	.dial-card code {
		font: 10px var(--font-mono);
		color: var(--brand-muted);
	}
	.dial-svg {
		width: 100%;
		max-width: 140px;
		height: auto;
		justify-self: center;
	}
	.dial-ring {
		fill: var(--brand-surface);
		stroke: var(--brand-line);
		stroke-width: 1.5;
	}
	.dial-inner {
		fill: none;
		stroke: var(--brand-line);
		stroke-dasharray: 2 3;
	}
	.crosshair {
		stroke: var(--brand-line);
		stroke-width: 0.8;
	}
	.hand-key {
		stroke: var(--brand-clay);
		stroke-width: 2;
		stroke-dasharray: 4 2;
	}
	.tip-key {
		fill: var(--brand-peach);
		stroke: var(--brand-clay);
		stroke-width: 1.5;
	}
	.hand-query {
		stroke: var(--brand-accent);
		stroke-width: 2.5;
	}
	.tip-query {
		fill: var(--brand-accent);
	}
	.pivot {
		fill: var(--brand-ink);
	}
	.dial-footer {
		display: flex;
		justify-content: space-between;
		font: 11px var(--font-mono);
		color: var(--brand-muted);
		border-top: 1px solid var(--brand-line);
		padding-top: 6px;
	}
	.dial-footer strong {
		color: var(--brand-accent);
	}
	.dial-footer strong.neg {
		color: var(--brand-clay);
	}
	.decay-panel {
		background: var(--brand-paper);
		border: 1px solid var(--brand-line);
		border-radius: 14px;
		padding: 16px 18px;
	}
	.decay-panel header {
		display: flex;
		justify-content: space-between;
		align-items: flex-start;
		flex-wrap: wrap;
		gap: 12px;
		margin-bottom: 10px;
	}
	.decay-panel h3 {
		margin: 0 0 4px;
		font-size: 14.5px;
	}
	.decay-panel p {
		margin: 0;
		font-size: 11.5px;
		color: var(--brand-body);
		max-width: 560px;
		line-height: 1.55;
	}
	.hand-legend {
		display: flex;
		gap: 14px;
		font: 11px var(--font-mono);
	}
	.hand-legend span {
		display: flex;
		align-items: center;
		gap: 6px;
	}
	.q-dot {
		width: 10px;
		height: 10px;
		border-radius: 50%;
		background: var(--brand-accent);
	}
	.k-dot {
		width: 10px;
		height: 10px;
		border-radius: 50%;
		background: var(--brand-peach);
		border: 1.5px solid var(--brand-clay);
	}
	.decay-svg {
		width: 100%;
		height: auto;
		display: block;
		background: var(--brand-surface);
		border-radius: 10px;
		border: 1px solid var(--brand-line);
	}
	.axis {
		stroke: var(--brand-line);
		stroke-width: 1.2;
	}
	.zero-line {
		stroke: var(--brand-line);
		stroke-dasharray: 4 4;
	}
	.decay-curve {
		stroke: var(--brand-accent);
		stroke-width: 2.4;
	}
	.cursor-line {
		stroke: var(--brand-clay);
		stroke-dasharray: 3 3;
	}
	.cursor-dot {
		fill: var(--brand-peach);
		stroke: var(--brand-clay);
		stroke-width: 2;
	}
	.axis-caption {
		fill: var(--brand-muted);
		font: 10.5px var(--font-mono);
		text-anchor: middle;
	}
	@media (max-width: 820px) {
		.controls-top,
		.summary-strip,
		.astrolabe-grid {
			grid-template-columns: 1fr;
		}
	}
</style>
