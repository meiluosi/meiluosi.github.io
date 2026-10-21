<script lang="ts">
import { onMount } from "svelte";
import {
	playLatchClick,
	playMechanicalTick,
	playShutterTick,
	playSlideTick,
} from "@/utils/haptics";
import {
	DEFAULT_VOCAB,
	type ForwardPassTrace,
	runForwardPass,
} from "@/utils/micro-transformer";

let english = $state(false);
let currentTokens = $state<string[]>(["[BOS]", "the", "cat"]);
let activeStage = $state<number>(3); // Default to Attention stage
let temperature = $state<number>(0.7);
let topP = $state<number>(0.9);
let noiseScale = $state<number>(0.0);
let ablateHead0 = $state<boolean>(false);
let ablateHead1 = $state<boolean>(false);
let hoveredMatrixCell = $state<{
	row: number;
	col: number;
	val: number;
	head: number;
} | null>(null);

const PRESETS = [
	{
		labelZh: "经典语法：猫咪坐在垫子上",
		labelEn: "Syntax: The cat sat on",
		tokens: ["[BOS]", "the", "cat", "sat", "on"],
	},
	{
		labelZh: "哲学思考：AI学习熵与心灵",
		labelEn: "Semantic: AI learns entropy",
		tokens: ["[BOS]", "ai", "learns", "entropy"],
	},
	{
		labelZh: "动作流转：小狗飞速奔跑",
		labelEn: "Action: The dog ran fast",
		tokens: ["[BOS]", "the", "dog", "ran", "fast"],
	},
];

const FORMULAS = {
	stage1:
		"\\( X_t = \\text{Embed}(w_t) + \\text{PE}(t) \\in \\mathbb{R}^{d_{model}} \\) （\\(d_{model} = 16\\) 维稠密特征空间）",
	stage2:
		"\\( Q = X W_Q,\\quad K = X W_K,\\quad V = X W_V \\in \\mathbb{R}^{T \\times d_k} \\) （\\(d_k = 8\\) 每头维度）",
	stage3:
		"\\( A_{ij} = \\text{Softmax}\\left(\\frac{Q_i K_j^T}{\\sqrt{d_k}} + M_{ij}\\right) \\in [0, 1] \\) （上三角被因果掩码 \\(-\\infty\\) 遮蔽，无法偷看未来）",
	stage4:
		"\\( X^{(1)} = \\text{RMSNorm}(X + \\text{MHA}),\\quad X^{(2)} = \\text{RMSNorm}(X^{(1)} + \\text{FFN}(X^{(1)})) \\)",
	stage5:
		"\\( \\text{Logits} = X_{final} W_{unembed} \\in \\mathbb{R}^V,\\quad P(w_{next}) = \\text{Softmax}(\\text{Logits} / T) \\)",
};

function getCellBg(val: number): string {
	const alpha = Math.min(1, Math.abs(val) * 1.2);
	return val >= 0
		? `rgba(92, 122, 107, ${alpha})`
		: `rgba(224, 122, 95, ${alpha})`;
}

function getAttnBg(weight: number, isMasked: boolean): string {
	if (isMasked) return "var(--brand-surface)";
	return `rgba(92, 122, 107, ${Math.max(0.08, weight)})`;
}

function getMiniOpacity(v: number): string {
	return String(Math.min(1, Math.abs(v) + 0.2));
}

const ablatedHeads = $derived([
	...(ablateHead0 ? [0] : []),
	...(ablateHead1 ? [1] : []),
]);

const trace: ForwardPassTrace = $derived(
	runForwardPass(currentTokens, {
		temperature,
		topP,
		noiseScale,
		ablateHeads: ablatedHeads,
	}),
);

function setPreset(tokens: string[]) {
	currentTokens = [...tokens];
	playLatchClick();
}

function appendToken(tok: string) {
	if (currentTokens.length >= 8) {
		currentTokens = currentTokens.slice(1);
	}
	currentTokens = [...currentTokens, tok];
	playMechanicalTick(1.2);
}

function stepForwardAndAppend() {
	const next = trace.sampledNextToken.token;
	playShutterTick();
	if (currentTokens.length >= 8) {
		currentTokens = currentTokens.slice(1);
	}
	currentTokens = [...currentTokens, next];
}

function resetSequence() {
	currentTokens = ["[BOS]", "the"];
	playLatchClick();
}

function setStage(idx: number) {
	activeStage = idx;
	playMechanicalTick(0.9 + idx * 0.1);
}

function onSliderChange() {
	playSlideTick();
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

<div class="transformer-lab" aria-label={english ? "Transformer In-Browser Forward Pass Lab" : "Transformer 端侧单步前向推演实验室"}>
	<!-- Header / Controls Top Bar -->
	<div class="lab-controls">
		<div class="preset-group">
			<span class="control-label">{english ? "Presets:" : "预设语料:"}</span>
			{#each PRESETS as p}
				<button
					type="button"
					class="btn-pill"
					onclick={() => setPreset(p.tokens)}
				>
					{english ? p.labelEn : p.labelZh}
				</button>
			{/each}
			<button type="button" class="btn-pill reset-btn" onclick={resetSequence}>
				{english ? "↺ Reset" : "↺ 重置"}
			</button>
		</div>

		<!-- Token Stream Tape -->
		<div class="token-tape-container">
			<span class="control-label">{english ? "Prompt Tokens:" : "当前序列:"}</span>
			<div class="token-tape">
				{#each currentTokens as tok, i}
					<span class="token-tag" class:last-tok={i === currentTokens.length - 1}>
						<span class="pos-idx">{i}</span>
						<span class="tok-text">{tok}</span>
					</span>
				{/each}
			</div>
			<!-- Appendable vocab drawer -->
			<div class="vocab-quick-add">
				<span class="sub-label">{english ? "+ Append:" : "+ 补词:"}</span>
				{#each DEFAULT_VOCAB as v}
					<button
						type="button"
						class="mini-vocab-btn"
						onclick={() => appendToken(v)}
					>
						{v}
					</button>
				{/each}
			</div>
		</div>
	</div>

	<!-- Step Navigation Pipeline Tabs -->
	<nav class="pipeline-nav">
		<button
			type="button"
			class:active={activeStage === 1}
			onclick={() => setStage(1)}
		>
			<span class="num">1</span>
			<span class="title">{english ? "Embed + Pos" : "嵌入与位置"}</span>
		</button>
		<button
			type="button"
			class:active={activeStage === 2}
			onclick={() => setStage(2)}
		>
			<span class="num">2</span>
			<span class="title">{english ? "QKV Projections" : "QKV 投影"}</span>
		</button>
		<button
			type="button"
			class:active={activeStage === 3}
			onclick={() => setStage(3)}
		>
			<span class="num">3</span>
			<span class="title">{english ? "Causal Attention" : "因果自注意力"}</span>
		</button>
		<button
			type="button"
			class:active={activeStage === 4}
			onclick={() => setStage(4)}
		>
			<span class="num">4</span>
			<span class="title">{english ? "RMSNorm & FFN" : "残差与前馈"}</span>
		</button>
		<button
			type="button"
			class:active={activeStage === 5}
			onclick={() => setStage(5)}
		>
			<span class="num">5</span>
			<span class="title">{english ? "Logits & Sampling" : "概率与采样"}</span>
		</button>
	</nav>

	<!-- Main Stage Display -->
	<div class="stage-canvas">
		<!-- STAGE 1: Embedding -->
		{#if activeStage === 1}
			<div class="stage-panel">
				<div class="panel-header">
					<h3>{english ? "Stage 1: Token Embeddings + Sinusoidal Positional Encoding" : "第 1 步：词嵌入向量与正弦位置编码融合"}</h3>
					<p class="formula-caption">{FORMULAS.stage1}</p>
				</div>
				<div class="vector-strip-view">
					{#each trace.tokens as tok, t}
						<div class="vector-row">
							<span class="tok-label"><strong>pos {t}:</strong> "{tok}"</span>
							<div class="feature-strip">
								{#each trace.embeddings[t] as val, d}
									<div
										class="cell-block"
										style="background-color: {getCellBg(val)}"
										title="pos {t}, dim {d}: {val.toFixed(3)}"
									></div>
								{/each}
							</div>
						</div>
					{/each}
				</div>
				<div class="legend-row">
					<span class="legend-item"><span class="swatch pos"></span> {english ? "Positive weight" : "正向特征权重"}</span>
					<span class="legend-item"><span class="swatch neg"></span> {english ? "Negative weight" : "负向特征权重"}</span>
				</div>
			</div>
		{/if}

		<!-- STAGE 2: QKV Projections -->
		{#if activeStage === 2}
			<div class="stage-panel">
				<div class="panel-header">
					<h3>{english ? "Stage 2: Linear Query, Key, Value Projections" : "第 2 步：Query, Key, Value 线性变换分解"}</h3>
					<p class="formula-caption">{FORMULAS.stage2}</p>
				</div>
				<div class="qkv-comparison-grid">
					{#each trace.heads as head}
						<div class="head-card">
							<div class="head-card-title">
								<h4>{english ? `Head ${head.headIndex}` : `注意力头 ${head.headIndex}`}</h4>
								<span class="badge">{head.isAblated ? (english ? "Ablated" : "已消融") : (english ? "Active" : "正常工作")}</span>
							</div>
							<div class="qkv-subgrid">
								<div class="qkv-col">
									<span class="qkv-title">Q (Query)</span>
									{#each head.q as row, i}
										<div class="qkv-row">
											<span class="tok-nano">{trace.tokens[i]}</span>
											<div class="mini-strip">
												{#each row as v}
													<span class="mini-cell" style="opacity: {getMiniOpacity(v)}"></span>
												{/each}
											</div>
										</div>
									{/each}
								</div>
								<div class="qkv-col">
									<span class="qkv-title">K (Key)</span>
									{#each head.k as row, i}
										<div class="qkv-row">
											<span class="tok-nano">{trace.tokens[i]}</span>
											<div class="mini-strip">
												{#each row as v}
													<span class="mini-cell" style="opacity: {getMiniOpacity(v)}"></span>
												{/each}
											</div>
										</div>
									{/each}
								</div>
								<div class="qkv-col">
									<span class="qkv-title">V (Value)</span>
									{#each head.v as row, i}
										<div class="qkv-row">
											<span class="tok-nano">{trace.tokens[i]}</span>
											<div class="mini-strip">
												{#each row as v}
													<span class="mini-cell" style="opacity: {getMiniOpacity(v)}"></span>
												{/each}
											</div>
										</div>
									{/each}
								</div>
							</div>
						</div>
					{/each}
				</div>
			</div>
		{/if}

		<!-- STAGE 3: Causal Attention Heatmap & Arcs -->
		{#if activeStage === 3}
			<div class="stage-panel">
				<div class="panel-header">
					<h3>{english ? "Stage 3: Causal Scaled Dot-Product Attention Heatmaps" : "第 3 步：因果下三角自注意力点积热力图"}</h3>
					<p class="formula-caption">{FORMULAS.stage3}</p>
				</div>

				<div class="attention-heads-container">
					{#each trace.heads as head}
						<div class="matrix-box" class:ablated={head.isAblated}>
							<div class="matrix-title">
								<span>{english ? `Head ${head.headIndex}: ` : `注意力头 ${head.headIndex}: `}</span>
								<em>{head.headIndex === 0 ? (english ? "Syntax & Local Induction" : "局部语法流转回路") : (english ? "Semantic & Topic Cohesion" : "远距语义连贯回路")}</em>
							</div>

							<!-- Attention Grid -->
							<div class="matrix-grid">
								<div class="grid-header-row">
									<span class="corner-cell">Q \ K</span>
									{#each trace.tokens as tok}
										<span class="k-label">{tok}</span>
									{/each}
								</div>
								{#each head.attnWeights as row, i}
									<div class="grid-row">
										<span class="q-label">{trace.tokens[i]}</span>
										{#each row as weight, j}
											<div
												class="attn-cell"
												class:masked={j > i}
												style="background-color: {getAttnBg(weight, j > i)}"
												onmouseenter={() => {
													hoveredMatrixCell = { row: i, col: j, val: weight, head: head.headIndex };
													playMechanicalTick(0.8 + weight);
												}}
												onmouseleave={() => (hoveredMatrixCell = null)}
											>
												<span class="cell-val">{j > i ? "—" : (weight * 100).toFixed(0)}</span>
											</div>
										{/each}
									</div>
								{/each}
							</div>
						</div>
					{/each}
				</div>

				{#if hoveredMatrixCell}
					<div class="hover-info-toast">
						<span>{english ? "Attention Weight:" : "注意力权重:"}</span>
						<strong>Q("{trace.tokens[hoveredMatrixCell.row]}") ➔ K("{trace.tokens[hoveredMatrixCell.col]}") = {(hoveredMatrixCell.val * 100).toFixed(1)}%</strong>
					</div>
				{/if}
			</div>
		{/if}

		<!-- STAGE 4: RMSNorm & FFN -->
		{#if activeStage === 4}
			<div class="stage-panel">
				<div class="panel-header">
					<h3>{english ? "Stage 4: Residual Addition, RMSNorm, and SwiGLU FFN" : "第 4 步：残差直连、RMSNorm 稳态化与前馈网络非线性映射"}</h3>
					<p class="formula-caption">{FORMULAS.stage4}</p>
				</div>
				<div class="ffn-flow-diagram">
					<div class="flow-step">
						<span class="flow-pill">MHA Output</span>
						<span class="flow-arrow">+</span>
						<span class="flow-pill">Residual 1</span>
						<span class="flow-arrow">➔</span>
						<span class="flow-pill highlight">RMSNorm</span>
					</div>
					<div class="flow-step">
						<span class="flow-arrow">➔</span>
						<span class="flow-pill">SwiGLU FFN (d_ffn = 32)</span>
						<span class="flow-arrow">+</span>
						<span class="flow-pill">Residual 2</span>
						<span class="flow-arrow">➔</span>
						<span class="flow-pill highlight">Final Norm</span>
					</div>
				</div>
				<p class="explainer-prose">
					{english
						? "The feed-forward block projects the 16-dimensional attention representation up to 32 dimensions, applies the SiLU activation function to produce non-linear feature gating, and projects back down to 16 dimensions before the final RMSNorm stabilizes gradient flow."
						: "前馈网络（FFN）将注意力汇聚出的 16 维表征升维至 32 维空间，通过 SiLU 激活函数执行非线性门控过滤，再投影回 16 维，最后经由 RMSNorm 保证各特征通道方差稳定，避免极深层推断中的数值漂移。"}
				</p>
			</div>
		{/if}

		<!-- STAGE 5: Logits, Softmax & Sampling -->
		{#if activeStage === 5}
			<div class="stage-panel">
				<div class="panel-header">
					<h3>{english ? "Stage 5: Unembedding LM Head, Temperature & Top-p Sampling" : "第 5 步：语言模型反投影头、温度缩放与 Top-p 核采样"}</h3>
					<p class="formula-caption">{FORMULAS.stage5}</p>
				</div>

				<div class="sampling-showcase">
					<!-- Top Candidates Distribution Bars -->
					<div class="bars-container">
						<h4>{english ? "Next Token Probabilities:" : "候选下一 Token 概率分布:"}</h4>
						{#each trace.topCandidates as cand}
							<div class="prob-row" class:winner={cand.token === trace.sampledNextToken.token}>
								<span class="cand-name">"{cand.token}"</span>
								<div class="bar-outer">
									<div class="bar-fill" style="width: {(cand.prob * 100).toFixed(1)}%"></div>
								</div>
								<span class="cand-pct">{(cand.prob * 100).toFixed(1)}%</span>
								<span class="cand-logit">logit: {cand.logit.toFixed(2)}</span>
							</div>
						{/each}
					</div>

					<!-- Real-time Metrics Card -->
					<div class="metrics-card">
						<h4>{english ? "Entropy & Prediction:" : "预测结果与信息熵:"}</h4>
						<div class="metric-block">
							<span class="m-label">{english ? "Sampled Next Token:" : "本次抽样预测:"}</span>
							<span class="m-val winner-token">"{trace.sampledNextToken.token}"</span>
						</div>
						<div class="metric-block">
							<span class="m-label">{english ? "Shannon Entropy:" : "香农信息熵:"}</span>
							<span class="m-val">{trace.entropy.toFixed(3)} bits</span>
						</div>
						<div class="metric-block">
							<span class="m-label">{english ? "Active Temperature:" : "当前温度 T:"}</span>
							<span class="m-val">{trace.temperature.toFixed(2)}</span>
						</div>

						<button type="button" class="btn-primary step-btn" onclick={stepForwardAndAppend}>
							{english ? "⚡ Forward Step & Append" : "⚡ 前进一步并吐出 Token"}
						</button>
					</div>
				</div>
			</div>
		{/if}
	</div>

	<!-- Bottom Interpretability & Ablation Sandbox Tray -->
	<div class="interpretability-tray">
		<div class="tray-title">
			<strong>{english ? "Mechanistic Interpretability & Parameter Knobs:" : "机制可解释性与超参数调谐器:"}</strong>
		</div>
		<div class="tray-controls">
			<!-- Temperature Slider -->
			<div class="knob-group">
				<label for="temp-slider">
					{english ? "Temperature T:" : "采样温度 T:"}
					<strong>{temperature.toFixed(2)}</strong>
				</label>
				<input
					id="temp-slider"
					type="range"
					min="0.1"
					max="2.0"
					step="0.05"
					bind:value={temperature}
					oninput={onSliderChange}
				/>
			</div>

			<!-- Top-P Slider -->
			<div class="knob-group">
				<label for="topp-slider">
					{english ? "Top-p Nucleus:" : "核采样 Top-p:"}
					<strong>{topP.toFixed(2)}</strong>
				</label>
				<input
					id="topp-slider"
					type="range"
					min="0.1"
					max="1.0"
					step="0.05"
					bind:value={topP}
					oninput={onSliderChange}
				/>
			</div>

			<!-- Noise Injection Slider -->
			<div class="knob-group">
				<label for="noise-slider">
					{english ? "Weight Noise:" : "权重扰动/噪声:"}
					<strong>{noiseScale.toFixed(2)}</strong>
				</label>
				<input
					id="noise-slider"
					type="range"
					min="0.0"
					max="0.8"
					step="0.05"
					bind:value={noiseScale}
					oninput={onSliderChange}
				/>
			</div>

			<!-- Head Ablation Toggles -->
			<div class="ablation-toggles">
				<span class="ablate-label">{english ? "Ablate Heads:" : "神经回路消融:"}</span>
				<label class="toggle-pill" class:active={ablateHead0}>
					<input type="checkbox" bind:checked={ablateHead0} onchange={() => playLatchClick()} />
					{english ? "Zero Head 0 (Syntax)" : "关闭 Head 0 (语法)"}
				</label>
				<label class="toggle-pill" class:active={ablateHead1}>
					<input type="checkbox" bind:checked={ablateHead1} onchange={() => playLatchClick()} />
					{english ? "Zero Head 1 (Topic)" : "关闭 Head 1 (语义)"}
				</label>
			</div>
		</div>
	</div>
</div>

<style>
	.transformer-lab {
		display: flex;
		flex-direction: column;
		gap: 20px;
		background: var(--brand-paper);
		border: 1px solid var(--brand-line);
		border-radius: 14px;
		padding: 24px;
		color: var(--brand-ink);
		box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
	}

	.lab-controls {
		display: flex;
		flex-direction: column;
		gap: 16px;
		padding-bottom: 16px;
		border-bottom: 1px solid var(--brand-line);
	}

	.preset-group {
		display: flex;
		align-items: center;
		flex-wrap: wrap;
		gap: 8px;
	}

	.control-label {
		font-size: 13px;
		font-weight: 600;
		color: var(--brand-muted);
	}

	.btn-pill {
		font-size: 12px;
		padding: 5px 12px;
		border-radius: 999px;
		border: 1px solid var(--brand-line);
		background: var(--brand-surface);
		color: var(--brand-body);
		cursor: pointer;
		transition: all 0.15s ease;
	}

	.btn-pill:hover {
		border-color: var(--brand-accent);
		color: var(--brand-accent);
	}

	.reset-btn {
		margin-left: auto;
		background: transparent;
	}

	.token-tape-container {
		display: flex;
		flex-direction: column;
		gap: 8px;
	}

	.token-tape {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
		align-items: center;
	}

	.token-tag {
		display: inline-flex;
		align-items: center;
		gap: 4px;
		padding: 4px 10px;
		background: var(--brand-surface);
		border: 1px solid var(--brand-line);
		border-radius: 6px;
		font-family: var(--font-mono, monospace);
		font-size: 13px;
	}

	.token-tag.last-tok {
		border-color: var(--brand-accent);
		background: rgba(92, 122, 107, 0.12);
		font-weight: 600;
	}

	.pos-idx {
		font-size: 10px;
		color: var(--brand-muted);
	}

	.vocab-quick-add {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 4px;
		font-size: 11px;
	}

	.sub-label {
		color: var(--brand-muted);
		margin-right: 4px;
	}

	.mini-vocab-btn {
		font-family: var(--font-mono, monospace);
		font-size: 11px;
		padding: 2px 7px;
		border: 1px solid var(--brand-line);
		background: transparent;
		border-radius: 4px;
		cursor: pointer;
		color: var(--brand-body);
	}

	.mini-vocab-btn:hover {
		border-color: var(--brand-accent);
		color: var(--brand-accent);
		background: var(--brand-surface);
	}

	.pipeline-nav {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
		border-bottom: 1px solid var(--brand-line);
		padding-bottom: 12px;
	}

	.pipeline-nav button {
		display: flex;
		align-items: center;
		gap: 6px;
		padding: 7px 14px;
		border: 1px solid var(--brand-line);
		border-radius: 8px;
		background: transparent;
		color: var(--brand-body);
		font-size: 13px;
		cursor: pointer;
		transition: all 0.15s ease;
	}

	.pipeline-nav button .num {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		width: 18px;
		height: 18px;
		border-radius: 50%;
		background: var(--brand-line);
		font-size: 11px;
		font-weight: 700;
	}

	.pipeline-nav button.active {
		border-color: var(--brand-accent);
		background: var(--brand-surface);
		color: var(--brand-accent);
		font-weight: 600;
	}

	.pipeline-nav button.active .num {
		background: var(--brand-accent);
		color: #fff;
	}

	.stage-canvas {
		min-height: 320px;
	}

	.panel-header h3 {
		margin: 0 0 4px;
		font-size: 18px;
		font-family: var(--font-serif, serif);
	}

	.formula-caption {
		font-size: 13px;
		color: var(--brand-muted);
		margin: 0 0 16px;
	}

	/* Vector Feature Strips */
	.vector-strip-view {
		display: flex;
		flex-direction: column;
		gap: 8px;
		margin-bottom: 14px;
	}

	.vector-row {
		display: flex;
		align-items: center;
		gap: 12px;
	}

	.tok-label {
		font-family: var(--font-mono, monospace);
		font-size: 13px;
		min-width: 100px;
	}

	.feature-strip {
		display: flex;
		gap: 3px;
		flex: 1;
		height: 22px;
		background: var(--brand-surface);
		padding: 2px;
		border-radius: 4px;
	}

	.cell-block {
		flex: 1;
		border-radius: 2px;
		cursor: help;
	}

	.legend-row {
		display: flex;
		gap: 16px;
		font-size: 12px;
		color: var(--brand-muted);
	}

	.swatch {
		display: inline-block;
		width: 12px;
		height: 12px;
		border-radius: 2px;
		vertical-align: middle;
	}

	.swatch.pos {
		background: rgba(92, 122, 107, 0.9);
	}

	.swatch.neg {
		background: rgba(224, 122, 95, 0.9);
	}

	/* QKV Grid */
	.qkv-comparison-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
		gap: 16px;
	}

	.head-card {
		border: 1px solid var(--brand-line);
		border-radius: 8px;
		padding: 14px;
		background: var(--brand-surface);
	}

	.head-card-title {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 12px;
	}

	.head-card-title h4 {
		margin: 0;
		font-size: 15px;
	}

	.badge {
		font-size: 11px;
		padding: 2px 6px;
		border-radius: 4px;
		background: var(--brand-line);
	}

	.qkv-subgrid {
		display: grid;
		grid-template-columns: repeat(3, 1fr);
		gap: 8px;
	}

	.qkv-title {
		font-size: 11px;
		font-weight: 700;
		color: var(--brand-muted);
		display: block;
		margin-bottom: 4px;
		text-align: center;
	}

	.qkv-row {
		display: flex;
		align-items: center;
		gap: 4px;
		margin-bottom: 3px;
	}

	.tok-nano {
		font-size: 10px;
		font-family: var(--font-mono, monospace);
		width: 24px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.mini-strip {
		display: flex;
		gap: 1px;
		flex: 1;
		height: 12px;
	}

	.mini-cell {
		flex: 1;
		background: var(--brand-accent);
		border-radius: 1px;
	}

	/* Attention Matrix Grid */
	.attention-heads-container {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
		gap: 20px;
	}

	.matrix-box {
		border: 1px solid var(--brand-line);
		border-radius: 8px;
		padding: 16px;
		background: var(--brand-surface);
	}

	.matrix-box.ablated {
		opacity: 0.45;
		filter: grayscale(1);
	}

	.matrix-title {
		font-size: 13px;
		margin-bottom: 14px;
	}

	.matrix-title em {
		color: var(--brand-muted);
		font-style: normal;
	}

	.matrix-grid {
		display: flex;
		flex-direction: column;
		gap: 2px;
		overflow-x: auto;
	}

	.grid-header-row,
	.grid-row {
		display: flex;
		gap: 2px;
		align-items: center;
	}

	.corner-cell,
	.q-label {
		width: 50px;
		font-size: 11px;
		font-family: var(--font-mono, monospace);
		text-align: right;
		padding-right: 6px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.k-label {
		width: 38px;
		font-size: 10px;
		font-family: var(--font-mono, monospace);
		text-align: center;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.attn-cell {
		width: 38px;
		height: 32px;
		display: flex;
		align-items: center;
		justify-content: center;
		border-radius: 4px;
		cursor: pointer;
		transition: transform 0.1s;
	}

	.attn-cell:hover {
		transform: scale(1.1);
		outline: 2px solid var(--brand-accent);
		z-index: 2;
	}

	.cell-val {
		font-size: 10px;
		font-family: var(--font-mono, monospace);
		color: var(--brand-ink);
	}

	.attn-cell.masked .cell-val {
		color: var(--brand-muted);
		opacity: 0.3;
	}

	.hover-info-toast {
		margin-top: 14px;
		padding: 8px 14px;
		background: var(--brand-surface);
		border-left: 3px solid var(--brand-accent);
		font-size: 13px;
		display: flex;
		gap: 8px;
	}

	/* FFN Diagram */
	.ffn-flow-diagram {
		display: flex;
		flex-direction: column;
		gap: 12px;
		padding: 18px;
		background: var(--brand-surface);
		border-radius: 8px;
		margin-bottom: 14px;
	}

	.flow-step {
		display: flex;
		align-items: center;
		gap: 8px;
		flex-wrap: wrap;
	}

	.flow-pill {
		padding: 6px 12px;
		border: 1px solid var(--brand-line);
		border-radius: 6px;
		font-size: 12px;
		font-family: var(--font-mono, monospace);
	}

	.flow-pill.highlight {
		border-color: var(--brand-accent);
		background: rgba(92, 122, 107, 0.1);
		color: var(--brand-accent);
		font-weight: 600;
	}

	.flow-arrow {
		color: var(--brand-muted);
		font-weight: 700;
	}

	.explainer-prose {
		font-size: 14px;
		line-height: 1.7;
		color: var(--brand-body);
	}

	/* Sampling Showcase */
	.sampling-showcase {
		display: grid;
		grid-template-columns: 1fr 280px;
		gap: 20px;
	}

	@media (max-width: 768px) {
		.sampling-showcase {
			grid-template-columns: 1fr;
		}
	}

	.bars-container h4,
	.metrics-card h4 {
		margin: 0 0 12px;
		font-size: 14px;
	}

	.prob-row {
		display: flex;
		align-items: center;
		gap: 8px;
		margin-bottom: 8px;
		font-size: 12px;
		font-family: var(--font-mono, monospace);
	}

	.cand-name {
		width: 60px;
		text-align: right;
		font-weight: 600;
	}

	.bar-outer {
		flex: 1;
		height: 14px;
		background: var(--brand-surface);
		border-radius: 4px;
		overflow: hidden;
		border: 1px solid var(--brand-line);
	}

	.bar-fill {
		height: 100%;
		background: var(--brand-accent);
		border-radius: 3px;
	}

	.prob-row.winner .bar-fill {
		background: var(--brand-clay, #e07a5f);
	}

	.cand-pct {
		width: 44px;
		text-align: right;
	}

	.cand-logit {
		font-size: 10px;
		color: var(--brand-muted);
		width: 65px;
	}

	.metrics-card {
		border: 1px solid var(--brand-line);
		border-radius: 8px;
		padding: 16px;
		background: var(--brand-surface);
		display: flex;
		flex-direction: column;
		gap: 10px;
	}

	.metric-block {
		display: flex;
		justify-content: space-between;
		font-size: 13px;
	}

	.m-label {
		color: var(--brand-muted);
	}

	.m-val {
		font-family: var(--font-mono, monospace);
		font-weight: 600;
	}

	.winner-token {
		color: var(--brand-accent);
		font-size: 16px;
	}

	.step-btn {
		margin-top: 10px;
		padding: 10px 16px;
		background: var(--brand-accent);
		color: #fff;
		border: none;
		border-radius: 8px;
		font-size: 13px;
		font-weight: 600;
		cursor: pointer;
		transition: opacity 0.15s ease;
	}

	.step-btn:hover {
		opacity: 0.9;
	}

	/* Interpretability Tray */
	.interpretability-tray {
		border-top: 1px solid var(--brand-line);
		padding-top: 16px;
		display: flex;
		flex-direction: column;
		gap: 12px;
	}

	.tray-title {
		font-size: 13px;
	}

	.tray-controls {
		display: flex;
		flex-wrap: wrap;
		gap: 20px;
		align-items: center;
	}

	.knob-group {
		display: flex;
		flex-direction: column;
		gap: 4px;
		font-size: 12px;
		min-width: 140px;
	}

	.knob-group input[type="range"] {
		accent-color: var(--brand-accent);
	}

	.ablation-toggles {
		display: flex;
		align-items: center;
		flex-wrap: wrap;
		gap: 8px;
	}

	.ablate-label {
		font-size: 12px;
		color: var(--brand-muted);
	}

	.toggle-pill {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		font-size: 12px;
		padding: 4px 10px;
		border: 1px solid var(--brand-line);
		border-radius: 6px;
		background: var(--brand-surface);
		cursor: pointer;
	}

	.toggle-pill.active {
		border-color: var(--brand-clay, #e07a5f);
		color: var(--brand-clay, #e07a5f);
		background: rgba(224, 122, 95, 0.1);
	}
</style>
