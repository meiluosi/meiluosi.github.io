<script lang="ts">
import { onMount } from "svelte";
import { playSlideTick } from "@/utils/haptics";
import { computeReactiveProse } from "@/utils/reactive-prose";

let english = $state(false);
let temperature = $state(0.7);
let topP = $state(0.85);
let contextLength = $state(4096);

const state = $derived(computeReactiveProse(temperature, topP, contextLength));

// Drag scrubber helper
let activeParam: "tau" | "topp" | "ctx" | null = null;
let startX = 0;
let startVal = 0;

function handlePointerDown(
	param: "tau" | "topp" | "ctx",
	e: PointerEvent,
): void {
	activeParam = param;
	startX = e.clientX;
	startVal =
		param === "tau" ? temperature : param === "topp" ? topP : contextLength;
	e.currentTarget.setPointerCapture(e.pointerId);
}

function handlePointerMove(e: PointerEvent): void {
	if (!activeParam) return;
	const dx = e.clientX - startX;
	if (activeParam === "tau") {
		const next = Math.max(0.05, Math.min(2.2, startVal + dx * 0.008));
		temperature = Math.round(next * 100) / 100;
		playSlideTick(1);
	} else if (activeParam === "topp") {
		const next = Math.max(0.15, Math.min(1.0, startVal + dx * 0.005));
		topP = Math.round(next * 100) / 100;
		playSlideTick(1);
	} else if (activeParam === "ctx") {
		const next = Math.max(256, Math.min(32768, startVal + dx * 32));
		contextLength = Math.round(next / 128) * 128;
		playSlideTick(1);
	}
}

function handlePointerUp(): void {
	activeParam = null;
}

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

<aside
	class="reactive-prose-block"
	aria-label={english ? "Explorable Reactive Prose" : "动态响应式推演段落"}
>
	<header class="block-header">
		<span class="badge">EXPLORABLE PROSE · BRET VICTOR STYLE</span>
		<small class="drag-hint">
			{english
				? "Drag any peach highlighted number left/right to recalculate prose in real-time ↔"
				: "水平按住拖动段落中高亮的桃色数值，正文与图表将实时联动推演 ↔"}
		</small>
	</header>

	<div class="prose-body">
		{#if !english}
			<p>
				假设模型在预测句首连词时，我们将解码温度设为
				<span
					class="inline-scrubber"
					role="slider"
					tabindex="0"
					aria-label="解码温度 τ"
					aria-valuenow={temperature}
					onpointerdown={(e) => handlePointerDown("tau", e)}
					onpointermove={handlePointerMove}
					onpointerup={handlePointerUp}
				>
					τ = <b>{temperature.toFixed(2)}</b>
				</span>，
				采样核阈值设为
				<span
					class="inline-scrubber"
					role="slider"
					tabindex="0"
					aria-label="Top-P 阈值"
					aria-valuenow={topP}
					onpointerdown={(e) => handlePointerDown("topp", e)}
					onpointermove={handlePointerMove}
					onpointerup={handlePointerUp}
				>
					p = <b>{topP.toFixed(2)}</b>
				</span>。
				{state.proseSummaryZh}
				当前词元预测的信息熵为
				<mark class="metric-mark">{state.entropy.toFixed(2)} nats</mark>
				（困惑度
				<mark class="metric-mark">{state.perplexity.toFixed(1)}</mark>），
				核内保留候选词
				<mark class="metric-mark">{state.activeTokensCount}</mark> 个。
				若在上下文长度达到
				<span
					class="inline-scrubber"
					role="slider"
					tabindex="0"
					aria-label="上下文长度 L"
					aria-valuenow={contextLength}
					onpointerdown={(e) => handlePointerDown("ctx", e)}
					onpointermove={handlePointerMove}
					onpointerup={handlePointerUp}
				>
					L = <b>{contextLength.toLocaleString()}</b>
				</span>
				个词元时持续推断，KV Cache 显存常驻开销约为
				<mark class="metric-mark">{state.kvCacheGiB.toFixed(2)} GiB</mark>。
			</p>
		{:else}
			<p>
				When predicting an introductory conjunction, we set the temperature to
				<span
					class="inline-scrubber"
					role="slider"
					tabindex="0"
					aria-label="Temperature τ"
					aria-valuenow={temperature}
					onpointerdown={(e) => handlePointerDown("tau", e)}
					onpointermove={handlePointerMove}
					onpointerup={handlePointerUp}
				>
					τ = <b>{temperature.toFixed(2)}</b>
				</span>
				with a nucleus threshold of
				<span
					class="inline-scrubber"
					role="slider"
					tabindex="0"
					aria-label="Top-P threshold"
					aria-valuenow={topP}
					onpointerdown={(e) => handlePointerDown("topp", e)}
					onpointermove={handlePointerMove}
					onpointerup={handlePointerUp}
				>
					p = <b>{topP.toFixed(2)}</b>
				</span>.
				{state.proseSummaryEn}
				The predictive entropy evaluates to
				<mark class="metric-mark">{state.entropy.toFixed(2)} nats</mark>
				(perplexity
				<mark class="metric-mark">{state.perplexity.toFixed(1)}</mark>)
				with
				<mark class="metric-mark">{state.activeTokensCount}</mark> tokens inside the nucleus.
				At context length
				<span
					class="inline-scrubber"
					role="slider"
					tabindex="0"
					aria-label="Context length L"
					aria-valuenow={contextLength}
					onpointerdown={(e) => handlePointerDown("ctx", e)}
					onpointermove={handlePointerMove}
					onpointerup={handlePointerUp}
				>
					L = <b>{contextLength.toLocaleString()}</b>
				</span>,
				the resident KV cache memory budget scales to
				<mark class="metric-mark">{state.kvCacheGiB.toFixed(2)} GiB</mark>.
			</p>
		{/if}
	</div>

	<div class="distribution-row" aria-hidden="true">
		<span class="row-label">{english ? "NUCLEUS CANDIDATES:" : "采样核候选词元:"}</span>
		<div class="bars-strip">
			{#each state.tokens as t}
				<div
					class="token-bar-cell"
					class:truncated={!t.inNucleus}
					title={`${english ? t.token : t.tokenZh}: ${(t.prob * 100).toFixed(1)}%`}
				>
					<span class="token-name">{english ? t.token : t.tokenZh}</span>
					<div class="bar-bg">
						<i class="bar-fill" style={`height:${Math.max(4, t.prob * 56)}px`}></i>
					</div>
					<small class="token-pct">{(t.prob * 100).toFixed(0)}%</small>
				</div>
			{/each}
		</div>
	</div>
</aside>

<style>
	.reactive-prose-block {
		background: var(--brand-paper);
		border: 1px solid var(--brand-line);
		border-left: 3px solid var(--brand-accent);
		border-radius: 12px;
		padding: 20px 22px;
		margin: 24px 0;
		color: var(--brand-ink);
		box-shadow: 0 4px 18px rgba(37, 60, 52, 0.04);
	}
	.block-header {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
		flex-wrap: wrap;
		gap: 8px;
		margin-bottom: 12px;
		padding-bottom: 8px;
		border-bottom: 1px dashed var(--brand-line);
	}
	.badge {
		font: 10px var(--font-mono);
		letter-spacing: 0.08em;
		color: var(--brand-accent);
	}
	.drag-hint {
		font-size: 11px;
		color: var(--brand-muted);
	}
	.prose-body p {
		margin: 0;
		font-size: 15px;
		line-height: 2.1;
		color: var(--brand-body);
	}
	.inline-scrubber {
		display: inline-flex;
		align-items: center;
		padding: 1px 7px;
		margin: 0 2px;
		border-radius: 6px;
		background: var(--brand-peach);
		color: var(--brand-ink);
		font: 13.5px var(--font-mono);
		cursor: ew-resize;
		user-select: none;
		touch-action: none;
		border: 1px solid var(--brand-clay);
		box-shadow: 0 1px 3px rgba(145, 86, 56, 0.12);
		transition: background-color 0.15s ease;
	}
	.inline-scrubber:hover,
	.inline-scrubber:active {
		background: color-mix(in srgb, var(--brand-peach) 85%, var(--brand-clay));
	}
	.inline-scrubber b {
		margin-left: 4px;
		font-weight: 600;
	}
	.metric-mark {
		background: transparent;
		color: var(--brand-accent);
		font-weight: 600;
		font-family: var(--font-mono);
		font-size: 14px;
	}
	.distribution-row {
		display: flex;
		align-items: flex-end;
		gap: 16px;
		margin-top: 16px;
		padding-top: 14px;
		border-top: 1px solid var(--brand-line);
	}
	.row-label {
		font: 10.5px var(--font-mono);
		color: var(--brand-muted);
		white-space: nowrap;
		margin-bottom: 6px;
	}
	.bars-strip {
		display: flex;
		gap: 10px;
		flex: 1;
		align-items: flex-end;
	}
	.token-bar-cell {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 4px;
		flex: 1;
		max-width: 72px;
	}
	.token-bar-cell.truncated {
		opacity: 0.35;
	}
	.token-name {
		font: 11px var(--font-mono);
		color: var(--brand-ink);
	}
	.bar-bg {
		width: 100%;
		height: 60px;
		background: var(--brand-surface);
		border: 1px solid var(--brand-line);
		border-radius: 6px;
		display: flex;
		align-items: flex-end;
		overflow: hidden;
	}
	.bar-fill {
		width: 100%;
		background: var(--brand-accent);
		border-radius: 4px 4px 0 0;
		transition: height 0.08s ease;
	}
	.token-pct {
		font: 10px var(--font-mono);
		color: var(--brand-muted);
	}
	@media (max-width: 680px) {
		.distribution-row {
			flex-direction: column;
			align-items: stretch;
		}
	}
</style>
