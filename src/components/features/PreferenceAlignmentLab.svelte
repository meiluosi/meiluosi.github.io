<script lang="ts">
import { onMount } from "svelte";
import { computeAlignmentSnapshot } from "@/utils/preference-alignment";

let english = $state(false);
let beta = $state(0.22);
let hackBias = $state(0.55);

const snapshot = $derived(computeAlignmentSnapshot(beta, hackBias));

// SVG coordinate helpers for Goodhart Frontier (KL in [0, 1.4], Reward in [-0.6, 1.8])
function xForKl(kl: number): number {
	const clamped = Math.max(0, Math.min(1.4, kl));
	return 56 + (clamped / 1.4) * 440;
}

function yForReward(r: number): number {
	const clamped = Math.max(-0.5, Math.min(1.8, r));
	return 186 - ((clamped + 0.5) / 2.3) * 150;
}

const truePath = $derived(
	snapshot.frontier
		.map(
			(pt, idx) =>
				`${idx === 0 ? "M" : "L"}${xForKl(pt.klDivergence).toFixed(1)} ${yForReward(pt.expectedTrueReward).toFixed(1)}`,
		)
		.join(" "),
);

const proxyPath = $derived(
	snapshot.frontier
		.map(
			(pt, idx) =>
				`${idx === 0 ? "M" : "L"}${xForKl(pt.klDivergence).toFixed(1)} ${yForReward(pt.expectedProxyReward).toFixed(1)}`,
		)
		.join(" "),
);

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
	class="dpo-lab"
	aria-label={english
		? "DPO & KL-Regularized Preference Alignment Workbench"
		: "DPO 与 KL 锚定偏好对齐实验台"}
>
	<div class="controls-grid">
		<label for="dpo-beta" class="control-item">
			<div class="label-top">
				<span>
					{english
						? "KL Regularization Temperature (β)"
						: "KL 正则化温度系数 (β)"}
				</span>
				<strong>β = {beta.toFixed(2)}</strong>
			</div>
			<input
				id="dpo-beta"
				type="range"
				min="0.03"
				max="0.90"
				step="0.01"
				value={beta}
				oninput={(e) => (beta = Number(e.currentTarget.value))}
			/>
			<small>
				{english
					? "Smaller β pushes π_θ away from π_ref toward maximum proxy reward; larger β anchors to SFT reference."
					: "β 越小，策略越激进地偏离参考模型 π_ref 追逐奖励；β 越大，越紧贴 SFT 基线分布。"}
			</small>
		</label>

		<label for="dpo-hack" class="control-item">
			<div class="label-top">
				<span>
					{english
						? "Proxy Preference Bias / Flattery Weight (α_bias)"
						: "偏好数据偏差 / 迎合奖励权重 (α_bias)"}
				</span>
				<strong>α = {hackBias.toFixed(2)}</strong>
			</div>
			<input
				id="dpo-hack"
				type="range"
				min="0"
				max="1.0"
				step="0.05"
				value={hackBias}
				oninput={(e) => (hackBias = Number(e.currentTarget.value))}
			/>
			<small>
				{english
					? "Simulates noisy human/judge labels that reward verbosity or sycophancy over truthfulness."
					: "模拟偏好标注中对「冗长套路」或「顺从迎合」的系统性高估偏差。"}
			</small>
		</label>
	</div>

	<div class="presets-row">
		<span class="preset-label">{english ? "Quick Regimes:" : "典型对齐工况:"}</span>
		<button
			type="button"
			onclick={() => {
				beta = 0.25;
				hackBias = 0.2;
			}}
		>
			{english ? "Balanced DPO (β=0.25, α=0.20)" : "稳健校准区 (β=0.25, α=0.20)"}
		</button>
		<button
			type="button"
			onclick={() => {
				beta = 0.04;
				hackBias = 0.75;
			}}
		>
			{english
				? "Reward Hacking Collapse (β=0.04, α=0.75)"
				: "奖励黑客/阿谀崩塌 (β=0.04, α=0.75)"}
		</button>
		<button
			type="button"
			onclick={() => {
				beta = 0.8;
				hackBias = 0.6;
			}}
		>
			{english ? "Under-aligned SFT Anchor (β=0.80)" : "强锚定欠对齐 (β=0.80)"}
		</button>
	</div>

	<div class="status-metrics">
		<div class="metric-box">
			<small>{english ? "KL DIVERGENCE D_KL(π_θ || π_ref)" : "策略偏移 D_KL(π_θ || π_ref)"}</small>
			<strong>{snapshot.klDivergence.toFixed(3)} nats</strong>
		</div>
		<div class="metric-box">
			<small>{english ? "EXPECTED TRUE UTILITY" : "真实研究效用 E[R_true]"}</small>
			<strong class:warn={snapshot.overoptimized}>
				{snapshot.expectedTrueReward.toFixed(3)}
			</strong>
		</div>
		<div class="metric-box">
			<small>{english ? "IMPLICIT PROXY REWARD" : "表观代理奖励 E[R_proxy]"}</small>
			<strong>{snapshot.expectedProxyReward.toFixed(3)}</strong>
		</div>
		<div class="metric-box">
			<small>{english ? "REGIME DIAGNOSIS" : "对齐状态诊断"}</small>
			<strong class:warn={snapshot.overoptimized}>
				{snapshot.overoptimized
					? english
						? "Goodhart Over-optimization!"
						: "古德哈特过优化 (阿谀奉承占优)"
					: english
						? "Calibrated Alignment"
						: "KL 锚定有效"}
			</strong>
		</div>
	</div>

	<div class="main-split">
		<div class="chart-card">
			<header>
				<h3>
					{english
						? "Goodhart Frontier: KL Budget vs. True & Proxy Reward"
						: "古德哈特前沿：KL 偏移预算 vs. 表观奖励与真实质量"}
				</h3>
				<p>
					{english
						? "As β decreases (moving right along KL), proxy reward rises monotonically while true utility peaks and then collapses."
						: "当 β 减小（向右增大 KL 散度）时，表观代理奖励单调上升，但真实质量在越过拐点后因迎合偏见而急剧下滑。"}
				</p>
			</header>
			<svg
				viewBox="0 0 530 220"
				role="img"
				aria-label={english
					? "KL divergence versus true and proxy reward"
					: "KL 散度与真实及代理奖励曲线"}
			>
				<line x1="56" y1="186" x2="500" y2="186" class="axis" />
				<line x1="56" y1="28" x2="56" y2="186" class="axis" />
				<text x="278" y="212" class="axis-label">
					{english
						? "KL Divergence D_KL(π_θ || π_ref) → (smaller β)"
						: "KL 散度 D_KL(π_θ || π_ref) → (β 越小越靠右)"}
				</text>
				<text x="16" y="110" class="axis-label" transform="rotate(-90 16 110)">
					{english ? "Expected Reward" : "期望奖励"}
				</text>

				<path d={proxyPath} class="curve-proxy" fill="none" />
				<path d={truePath} class="curve-true" fill="none" />

				<line
					x1={xForKl(snapshot.klDivergence)}
					y1="28"
					x2={xForKl(snapshot.klDivergence)}
					y2="186"
					class="cursor-line"
				/>
				<circle
					cx={xForKl(snapshot.klDivergence)}
					cy={yForReward(snapshot.expectedProxyReward)}
					r="5.5"
					class="dot-proxy"
				/>
				<circle
					cx={xForKl(snapshot.klDivergence)}
					cy={yForReward(snapshot.expectedTrueReward)}
					r="6.5"
					class="dot-true"
				/>
			</svg>
			<div class="legend">
				<span><i class="swatch-true"></i>{english ? "True Utility E[R_true]" : "真实质量 E[R_true]"}</span>
				<span><i class="swatch-proxy"></i>{english ? "Proxy Reward E[R_proxy]" : "代理偏好分 E[R_proxy]"}</span>
			</div>
		</div>

		<div class="candidates-card">
			<header>
				<h3>
					{english
						? "Candidate Generation Distribution: π_ref(y|x) vs. π_θ(y|x)"
						: "候选回答生成概率对比：参考模型 π_ref vs. 对齐模型 π_θ"}
				</h3>
				<p class="formula-note">
					r̂<sub>DPO</sub>(x,y) = β ln(π<sub>θ</sub>(y|x) / π<sub>ref</sub>(y|x))
				</p>
			</header>

			<div class="candidate-list">
				{#each snapshot.candidates as c}
					<article
						class="candidate-item"
						class:hacked={c.id === "sycophantic" && c.policyProb > 0.45}
					>
						<div class="cand-top">
							<strong>{english ? c.labelEn : c.labelZh}</strong>
							<span class="reward-tag">
								r̂ = {c.implicitReward >= 0 ? `+${c.implicitReward.toFixed(2)}` : c.implicitReward.toFixed(2)}
							</span>
						</div>
						<p class="cand-sample">“{english ? c.sampleEn : c.sampleZh}”</p>
						<div class="dual-bars">
							<div class="bar-row">
								<span>π<sub>ref</sub></span>
								<div class="track">
									<i
										class="fill-ref"
										style={`width:${(c.refProb * 100).toFixed(1)}%`}
									></i>
								</div>
								<output>{(c.refProb * 100).toFixed(1)}%</output>
							</div>
							<div class="bar-row">
								<span>π<sub>θ</sub></span>
								<div class="track">
									<i
										class="fill-policy"
										class:fill-warn={c.id === "sycophantic"}
										style={`width:${(c.policyProb * 100).toFixed(1)}%`}
									></i>
								</div>
								<output>{(c.policyProb * 100).toFixed(1)}%</output>
							</div>
						</div>
					</article>
				{/each}
			</div>
		</div>
	</div>
</section>

<style>
	.dpo-lab {
		border: 1px solid var(--brand-line);
		border-radius: 18px;
		background: var(--brand-surface);
		padding: 26px;
		color: var(--brand-ink);
	}
	.controls-grid {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 22px;
	}
	.control-item {
		display: grid;
		gap: 8px;
		background: var(--brand-paper);
		padding: 16px;
		border-radius: 12px;
		border: 1px solid var(--brand-line);
	}
	.label-top {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
		font-size: 13px;
		font-weight: 500;
	}
	.label-top strong {
		font: 600 14px var(--font-mono);
		color: var(--brand-accent);
	}
	.control-item small {
		font-size: 11px;
		line-height: 1.55;
		color: var(--brand-muted);
	}
	input[type="range"] {
		width: 100%;
		accent-color: var(--brand-accent);
	}
	.presets-row {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 8px;
		margin: 16px 0;
	}
	.preset-label {
		font: 11px var(--font-mono);
		color: var(--brand-muted);
	}
	button {
		font: 11px var(--font-mono);
		padding: 7px 13px;
		border-radius: 999px;
		border: 1px solid var(--brand-line);
		background: var(--brand-paper);
		color: var(--brand-body);
		cursor: pointer;
	}
	button:hover {
		border-color: var(--brand-accent);
		color: var(--brand-ink);
	}
	.status-metrics {
		display: grid;
		grid-template-columns: repeat(4, minmax(0, 1fr));
		gap: 12px;
		margin-bottom: 20px;
	}
	.metric-box {
		background: var(--brand-paper);
		border: 1px solid var(--brand-line);
		border-radius: 12px;
		padding: 12px 14px;
		display: grid;
		gap: 4px;
	}
	.metric-box small {
		font: 9.5px var(--font-mono);
		letter-spacing: 0.06em;
		color: var(--brand-muted);
	}
	.metric-box strong {
		font: 400 20px var(--font-serif);
		color: var(--brand-accent);
	}
	.metric-box strong.warn {
		color: var(--brand-clay);
	}
	.main-split {
		display: grid;
		grid-template-columns: 1fr 1.15fr;
		gap: 18px;
		align-items: start;
	}
	.chart-card,
	.candidates-card {
		background: var(--brand-paper);
		border: 1px solid var(--brand-line);
		border-radius: 14px;
		padding: 18px;
	}
	.chart-card h3,
	.candidates-card h3 {
		margin: 0 0 6px;
		font-size: 15px;
		font-weight: 600;
	}
	.chart-card p {
		margin: 0 0 12px;
		font-size: 11.5px;
		line-height: 1.6;
		color: var(--brand-body);
	}
	.formula-note {
		margin: 0 0 12px;
		font: 11.5px var(--font-mono);
		color: var(--brand-accent);
	}
	svg {
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
	.axis-label {
		fill: var(--brand-muted);
		font: 10px var(--font-mono);
		text-anchor: middle;
	}
	.curve-true {
		stroke: var(--brand-accent);
		stroke-width: 2.6;
	}
	.curve-proxy {
		stroke: var(--brand-clay);
		stroke-width: 2;
		stroke-dasharray: 5 4;
	}
	.cursor-line {
		stroke: var(--brand-muted);
		stroke-width: 1;
		stroke-dasharray: 3 3;
	}
	.dot-true {
		fill: var(--brand-accent);
		stroke: var(--brand-paper);
		stroke-width: 2;
	}
	.dot-proxy {
		fill: var(--brand-peach);
		stroke: var(--brand-clay);
		stroke-width: 2;
	}
	.legend {
		display: flex;
		gap: 18px;
		margin-top: 10px;
		font-size: 11px;
		color: var(--brand-body);
	}
	.legend span {
		display: flex;
		align-items: center;
		gap: 6px;
	}
	.swatch-true {
		width: 16px;
		border-top: 2.5px solid var(--brand-accent);
	}
	.swatch-proxy {
		width: 16px;
		border-top: 2px dashed var(--brand-clay);
	}
	.candidate-list {
		display: grid;
		gap: 10px;
	}
	.candidate-item {
		padding: 11px 13px;
		border-radius: 10px;
		background: var(--brand-surface);
		border: 1px solid var(--brand-line);
	}
	.candidate-item.hacked {
		border-color: var(--brand-clay);
		background: color-mix(in srgb, var(--brand-peach) 30%, var(--brand-surface));
	}
	.cand-top {
		display: flex;
		justify-content: space-between;
		align-items: center;
		font-size: 12.5px;
	}
	.reward-tag {
		font: 11px var(--font-mono);
		color: var(--brand-accent);
	}
	.cand-sample {
		margin: 5px 0 8px;
		font-size: 11.5px;
		line-height: 1.55;
		color: var(--brand-body);
	}
	.dual-bars {
		display: grid;
		gap: 4px;
	}
	.bar-row {
		display: grid;
		grid-template-columns: 34px 1fr 44px;
		align-items: center;
		gap: 8px;
		font: 10.5px var(--font-mono);
	}
	.track {
		height: 7px;
		background: var(--brand-paper);
		border-radius: 999px;
		overflow: hidden;
	}
	.fill-ref {
		display: block;
		height: 100%;
		background: var(--brand-muted);
		opacity: 0.55;
	}
	.fill-policy {
		display: block;
		height: 100%;
		background: var(--brand-accent);
		transition: width 0.15s ease;
	}
	.fill-policy.fill-warn {
		background: var(--brand-clay);
	}
	@media (max-width: 820px) {
		.controls-grid,
		.status-metrics,
		.main-split {
			grid-template-columns: 1fr;
		}
	}
</style>
