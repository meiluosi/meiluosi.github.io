<script lang="ts">
import { onMount } from "svelte";
import { playSlideTick } from "@/utils/haptics";
import {
	computeBpeCost,
	computeDpoDynamics,
	computeReactiveProse,
} from "@/utils/reactive-prose";

let { mode = "sampling" } = $props<{
	mode?: "sampling" | "bpe" | "dpo";
}>();

let english = $state(false);

// Mode: sampling
let temperature = $state(0.7);
let topP = $state(0.85);
let contextLength = $state(4096);
const samplingState = $derived(
	computeReactiveProse(temperature, topP, contextLength),
);

// Mode: bpe
let vocabSize = $state(32000);
let textChars = $state(10000);
const bpeState = $derived(computeBpeCost(vocabSize, textChars));

// Mode: dpo
let beta = $state(0.1);
let chosenMargin = $state(1.2);
let rejectMargin = $state(-0.8);
const dpoState = $derived(computeDpoDynamics(beta, chosenMargin, rejectMargin));

// Drag scrubber helper
type ParamKey =
	| "tau"
	| "topp"
	| "ctx"
	| "vocab"
	| "chars"
	| "beta"
	| "chosen"
	| "reject";
let activeParam: ParamKey | null = null;
let startX = 0;
let startVal = 0;

function handlePointerDown(param: ParamKey, e: PointerEvent): void {
	activeParam = param;
	startX = e.clientX;
	if (param === "tau") startVal = temperature;
	else if (param === "topp") startVal = topP;
	else if (param === "ctx") startVal = contextLength;
	else if (param === "vocab") startVal = vocabSize;
	else if (param === "chars") startVal = textChars;
	else if (param === "beta") startVal = beta;
	else if (param === "chosen") startVal = chosenMargin;
	else if (param === "reject") startVal = rejectMargin;

	(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
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
	} else if (activeParam === "vocab") {
		const next = Math.max(4000, Math.min(128000, startVal + dx * 200));
		vocabSize = Math.round(next / 1000) * 1000;
		playSlideTick(1);
	} else if (activeParam === "chars") {
		const next = Math.max(1000, Math.min(50000, startVal + dx * 80));
		textChars = Math.round(next / 500) * 500;
		playSlideTick(1);
	} else if (activeParam === "beta") {
		const next = Math.max(0.02, Math.min(1.0, startVal + dx * 0.003));
		beta = Math.round(next * 100) / 100;
		playSlideTick(1);
	} else if (activeParam === "chosen") {
		const next = Math.max(-3.0, Math.min(5.0, startVal + dx * 0.02));
		chosenMargin = Math.round(next * 100) / 100;
		playSlideTick(1);
	} else if (activeParam === "reject") {
		const next = Math.max(-5.0, Math.min(3.0, startVal + dx * 0.02));
		rejectMargin = Math.round(next * 100) / 100;
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
		<span class="badge">
			{#if mode === "bpe"}
				EXPLORABLE PROSE · BPE COMPRESSION & VOCAB COST
			{:else if mode === "dpo"}
				EXPLORABLE PROSE · DPO IMPLICIT REWARD & KL ANCHOR
			{:else}
				EXPLORABLE PROSE · SAMPLING DYNAMICS & KV CACHE
			{/if}
		</span>
		<small class="drag-hint">
			{english
				? "Drag any peach highlighted number left/right to recalculate in real-time ↔"
				: "水平按住拖动段落中高亮的桃色数值，正文与图表将实时联动推演 ↔"}
		</small>
	</header>

	<div class="prose-body">
		{#if mode === "bpe"}
			{#if !english}
				<p>
					假设我们为模型配置一个
					<span
						class="inline-scrubber"
						role="slider"
						tabindex="0"
						aria-label="词表规模 V"
						aria-valuenow={vocabSize}
						onpointerdown={(e) => handlePointerDown("vocab", e)}
						onpointermove={handlePointerMove}
						onpointerup={handlePointerUp}
					>
						V = <b>{(vocabSize / 1000).toFixed(0)}K</b>
					</span>
					大小的分词词表。在处理一段约
					<span
						class="inline-scrubber"
						role="slider"
						tabindex="0"
						aria-label="输入文本字符数"
						aria-valuenow={textChars}
						onpointerdown={(e) => handlePointerDown("chars", e)}
						onpointermove={handlePointerMove}
						onpointerup={handlePointerUp}
					>
						<b>{textChars.toLocaleString()}</b> 字
					</span>
					的中英混合技术语料时，统计平均压缩比为
					<mark class="metric-mark">{bpeState.compressionRatio.toFixed(2)} 字符/token</mark>，
					分词后产生约
					<mark class="metric-mark">{bpeState.tokensCount.toLocaleString()} 个 token</mark>。
					在隐藏层维度 $d=2048$ 下，该词表对应的 Embedding 矩阵参数量达
					<mark class="metric-mark">{bpeState.embedParamsMillion.toFixed(1)} M</mark>，
					在 FP16 精度下常驻显存开销为
					<mark class="metric-mark">{bpeState.embedMemoryMiB.toFixed(1)} MB</mark>。
					{bpeState.proseSummaryZh}
				</p>
			{:else}
				<p>
					Suppose we configure the tokenizer with a vocabulary size of
					<span
						class="inline-scrubber"
						role="slider"
						tabindex="0"
						aria-label="Vocabulary size V"
						aria-valuenow={vocabSize}
						onpointerdown={(e) => handlePointerDown("vocab", e)}
						onpointermove={handlePointerMove}
						onpointerup={handlePointerUp}
					>
						V = <b>{(vocabSize / 1000).toFixed(0)}K</b>
					</span>.
					For a sample text containing
					<span
						class="inline-scrubber"
						role="slider"
						tabindex="0"
						aria-label="Text length in characters"
						aria-valuenow={textChars}
						onpointerdown={(e) => handlePointerDown("chars", e)}
						onpointermove={handlePointerMove}
						onpointerup={handlePointerUp}
					>
						<b>{textChars.toLocaleString()}</b> chars
					</span>,
					the average compression ratio reaches
					<mark class="metric-mark">{bpeState.compressionRatio.toFixed(2)} chars/token</mark>,
					yielding
					<mark class="metric-mark">{bpeState.tokensCount.toLocaleString()} tokens</mark>.
					With hidden dimension $d=2048$, the embedding weights total
					<mark class="metric-mark">{bpeState.embedParamsMillion.toFixed(1)} M parameters</mark>,
					consuming
					<mark class="metric-mark">{bpeState.embedMemoryMiB.toFixed(1)} MB</mark> in FP16 memory.
					{bpeState.proseSummaryEn}
				</p>
			{/if}

			<div class="stat-pills-row" aria-hidden="true">
				<div class="stat-pill">
					<span class="pill-k">{english ? "TOKENS" : "切分 Token 数"}</span>
					<b class="pill-v">{bpeState.tokensCount.toLocaleString()}</b>
				</div>
				<div class="stat-pill">
					<span class="pill-k">{english ? "COMPRESSION" : "平均压缩比"}</span>
					<b class="pill-v">{bpeState.compressionRatio.toFixed(2)}×</b>
				</div>
				<div class="stat-pill">
					<span class="pill-k">{english ? "EMBED PARAMS" : "Embedding 参数"}</span>
					<b class="pill-v">{bpeState.embedParamsMillion.toFixed(1)} M</b>
				</div>
				<div class="stat-pill">
					<span class="pill-k">{english ? "EMBED MEMORY" : "Embedding 显存"}</span>
					<b class="pill-v">{bpeState.embedMemoryMiB.toFixed(1)} MB</b>
				</div>
			</div>

		{:else if mode === "dpo"}
			{#if !english}
				<p>
					在成对偏好优化中，设 KL 锚定惩罚系数为
					<span
						class="inline-scrubber"
						role="slider"
						tabindex="0"
						aria-label="KL 正则化系数 β"
						aria-valuenow={beta}
						onpointerdown={(e) => handlePointerDown("beta", e)}
						onpointermove={handlePointerMove}
						onpointerup={handlePointerUp}
					>
						β = <b>{beta.toFixed(2)}</b>
					</span>。
					若当前策略模型相对参考模型在胜出样本上的对数概率差为
					<span
						class="inline-scrubber"
						role="slider"
						tabindex="0"
						aria-label="Chosen 对数差"
						aria-valuenow={chosenMargin}
						onpointerdown={(e) => handlePointerDown("chosen", e)}
						onpointermove={handlePointerMove}
						onpointerup={handlePointerUp}
					>
						Δ_θ(y_w) = <b>{chosenMargin > 0 ? `+${chosenMargin.toFixed(2)}` : chosenMargin.toFixed(2)}</b>
					</span>，
					在落败样本上为
					<span
						class="inline-scrubber"
						role="slider"
						tabindex="0"
						aria-label="Rejected 对数差"
						aria-valuenow={rejectMargin}
						onpointerdown={(e) => handlePointerDown("reject", e)}
						onpointermove={handlePointerMove}
						onpointerup={handlePointerUp}
					>
						Δ_θ(y_l) = <b>{rejectMargin > 0 ? `+${rejectMargin.toFixed(2)}` : rejectMargin.toFixed(2)}</b>
					</span>，
					则闭式隐式奖励优势为
					<mark class="metric-mark">Δr = {dpoState.implicitRewardMargin.toFixed(2)}</mark>。
					此时计算得到的 DPO 损失为
					<mark class="metric-mark">L = {dpoState.loss.toFixed(3)}</mark>，
					反向传播梯度权重为
					<mark class="metric-mark">{(dpoState.gradientWeight * 100).toFixed(0)}%</mark>。
					{dpoState.proseSummaryZh}
				</p>
			{:else}
				<p>
					In direct preference optimization, set the KL constraint coefficient to
					<span
						class="inline-scrubber"
						role="slider"
						tabindex="0"
						aria-label="Beta coefficient β"
						aria-valuenow={beta}
						onpointerdown={(e) => handlePointerDown("beta", e)}
						onpointermove={handlePointerMove}
						onpointerup={handlePointerUp}
					>
						β = <b>{beta.toFixed(2)}</b>
					</span>.
					Given policy log-ratio on winning response
					<span
						class="inline-scrubber"
						role="slider"
						tabindex="0"
						aria-label="Chosen log-ratio"
						aria-valuenow={chosenMargin}
						onpointerdown={(e) => handlePointerDown("chosen", e)}
						onpointermove={handlePointerMove}
						onpointerup={handlePointerUp}
					>
						Δ_θ(y_w) = <b>{chosenMargin > 0 ? `+${chosenMargin.toFixed(2)}` : chosenMargin.toFixed(2)}</b>
					</span>
					and losing response
					<span
						class="inline-scrubber"
						role="slider"
						tabindex="0"
						aria-label="Rejected log-ratio"
						aria-valuenow={rejectMargin}
						onpointerdown={(e) => handlePointerDown("reject", e)}
						onpointermove={handlePointerMove}
						onpointerup={handlePointerUp}
					>
						Δ_θ(y_l) = <b>{rejectMargin > 0 ? `+${rejectMargin.toFixed(2)}` : rejectMargin.toFixed(2)}</b>
					</span>,
					the implicit reward margin evaluates to
					<mark class="metric-mark">Δr = {dpoState.implicitRewardMargin.toFixed(2)}</mark>.
					The DPO loss value is
					<mark class="metric-mark">L = {dpoState.loss.toFixed(3)}</mark>
					with gradient scale
					<mark class="metric-mark">{(dpoState.gradientWeight * 100).toFixed(0)}%</mark>.
					{dpoState.proseSummaryEn}
				</p>
			{/if}

			<div class="stat-pills-row" aria-hidden="true">
				<div class="stat-pill">
					<span class="pill-k">{english ? "REWARD MARGIN" : "隐式奖励差 Δr"}</span>
					<b class="pill-v">{dpoState.implicitRewardMargin.toFixed(2)}</b>
				</div>
				<div class="stat-pill">
					<span class="pill-k">{english ? "DPO LOSS" : "DPO 目标损失"}</span>
					<b class="pill-v">{dpoState.loss.toFixed(3)}</b>
				</div>
				<div class="stat-pill">
					<span class="pill-k">{english ? "GRADIENT SCALE" : "更新梯度灵敏度"}</span>
					<b class="pill-v">{(dpoState.gradientWeight * 100).toFixed(0)}%</b>
				</div>
				<div class="stat-pill">
					<span class="pill-k">{english ? "DRIFT ESTIMATE" : "估计 KL 偏离"}</span>
					<b class="pill-v">{dpoState.approxKlDrift.toFixed(2)}</b>
				</div>
			</div>

		{:else}
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
					{samplingState.proseSummaryZh}
					当前词元预测的信息熵为
					<mark class="metric-mark">{samplingState.entropy.toFixed(2)} nats</mark>
					（困惑度
					<mark class="metric-mark">{samplingState.perplexity.toFixed(1)}</mark>），
					核内保留候选词
					<mark class="metric-mark">{samplingState.activeTokensCount}</mark> 个。
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
					<mark class="metric-mark">{samplingState.kvCacheGiB.toFixed(2)} GiB</mark>。
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
					{samplingState.proseSummaryEn}
					The predictive entropy evaluates to
					<mark class="metric-mark">{samplingState.entropy.toFixed(2)} nats</mark>
					(perplexity
					<mark class="metric-mark">{samplingState.perplexity.toFixed(1)}</mark>)
					with
					<mark class="metric-mark">{samplingState.activeTokensCount}</mark> tokens inside the nucleus.
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
					<mark class="metric-mark">{samplingState.kvCacheGiB.toFixed(2)} GiB</mark>.
				</p>
			{/if}

			<div class="distribution-row" aria-hidden="true">
				<span class="row-label">{english ? "NUCLEUS CANDIDATES:" : "采样核候选词元:"}</span>
				<div class="bars-strip">
					{#each samplingState.tokens as t}
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
		{/if}
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
	.stat-pills-row {
		display: flex;
		gap: 10px;
		flex-wrap: wrap;
		margin-top: 14px;
		padding-top: 12px;
		border-top: 1px solid var(--brand-line);
	}
	.stat-pill {
		display: flex;
		flex-direction: column;
		background: var(--brand-surface);
		border: 1px solid var(--brand-line);
		border-radius: 8px;
		padding: 6px 12px;
		min-width: 100px;
		flex: 1;
	}
	.pill-k {
		font: 9.5px var(--font-mono);
		color: var(--brand-muted);
		letter-spacing: 0.05em;
	}
	.pill-v {
		font: 14px var(--font-mono);
		color: var(--brand-ink);
		margin-top: 2px;
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
