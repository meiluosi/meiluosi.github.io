<script lang="ts">
import {
	CANDIDATES,
	FFN_ACTIVATIONS,
	LN_1,
	LN_FINAL,
	TOKENS,
} from "../data/modelData";
import TokenCapsule from "../parts/TokenCapsule.svelte";
import VectorChip from "../parts/VectorChip.svelte";
import { easeInOut, easeOut, fmt, PALETTE, seg } from "../timeline";

interface Props {
	actTime: number; // 0 ~ 20s
	temperature?: number;
	topP?: number;
	english?: boolean;
	heroIdx?: number;
}

const {
	actTime,
	temperature = 0.7,
	topP = 0.9,
	english = false,
	heroIdx = 3,
}: Props = $props();

const activeTemp = $derived(Math.max(0.05, temperature));

// ① FFN 知识激活层 (0 ~ 6.6s，在 6.6s 前完全淡出，给 8.0s 的 Softmax 驻留留出完全空间)
const ffnAlpha = $derived(
	actTime < 5.2
		? seg(actTime, 0.2, 1.2, easeOut)
		: 1 - seg(actTime, 5.2, 6.6, easeOut),
);
const neuronFiringProgress = $derived(seg(actTime, 1.0, 4.5, easeInOut));

// ② Softmax 概率分布与调温 (6.6s 起平滑淡入，7.8s 达到 100% 满状态，在 8.0s 驻留点稳定接管控制台)
const softmaxAlpha = $derived(seg(actTime, 6.6, 7.8, easeOut));
const topPScissorsAlpha = $derived(seg(actTime, 14.0, 16.5, easeOut));

// 计算当前温度下的 Softmax 分布与累积概率
const softmaxData = $derived.by(() => {
	const rawExp = CANDIDATES.map((c) => Math.exp(c.logit / activeTemp));
	const sumExp = rawExp.reduce((a, b) => a + b, 0);
	const probs = rawExp.map((e) => e / sumExp);

	// 计算香农信息熵 (bits)
	const entropy = -probs.reduce(
		(acc, p) => (p > 1e-6 ? acc + p * Math.log2(p) : acc),
		0,
	);

	// 累积概率计算以评估 Top-P 截断
	let cum = 0;
	const cumProbs = probs.map((p) => {
		cum += p;
		return cum;
	});

	// 首个被剪除的索引 (前驱累积概率达到或超过 topP)
	const firstPrunedIdx = cumProbs.findIndex(
		(_, i) => i > 0 && cumProbs[i - 1] >= topP,
	);

	let regimeLabelZh = "黄金平衡区间 (Balanced Sweetspot: T=0.7)";
	let regimeLabelEn = "Golden Balanced Regime (T=0.7)";
	let regimeColor = PALETTE.v;

	if (activeTemp < 0.35) {
		regimeLabelZh = "极低熵贪婪独裁塌缩 (Greedy Argmax: T < 0.35)";
		regimeLabelEn = "Greedy Argmax Collapse (T < 0.35)";
		regimeColor = PALETTE.q;
	} else if (activeTemp > 1.25) {
		regimeLabelZh = "高熵白噪声混乱混沌 (High-Entropy Noise: T > 1.25)";
		regimeLabelEn = "High-Entropy Noise (T > 1.25)";
		regimeColor = PALETTE.pos;
	}

	return {
		probs,
		cumProbs,
		entropy,
		firstPrunedIdx,
		regimeLabelZh,
		regimeLabelEn,
		regimeColor,
	};
});
</script>

<g class="scene-4-container">
	<!-- ════════ 上半区: FFN 隐藏层知识激活神经元 (0 ~ 6.6s) ════════ -->
	{#if actTime < 6.8 && ffnAlpha > 0}
		<g transform="translate(0, -90)" opacity={ffnAlpha}>
			<!-- 背景卡片 -->
			<rect
				x="-480"
				y="-110"
				width="960"
				height="220"
				rx="12"
				fill="rgba(17, 26, 36, 0.95)"
				stroke={PALETTE.ffn}
				stroke-width="1.8"
			/>

			<!-- 标题与闭环公式 -->
			<g transform="translate(0, -80)">
				<text
					x="0"
					y="0"
					text-anchor="middle"
					font-size="14"
					font-weight="800"
					fill={PALETTE.ffn}
				>
					{english
						? "★ MLP / Feed-Forward Network: d_model(4) -> d_ffn(8) -> d_model(4)"
						: "★ MLP 前馈知识层：d_model(4) ➜ d_ffn(8) 升维激活事实常识记忆"}
				</text>
				<g transform="translate(0, 20)">
					<VectorChip
						values={LN_1}
						label="LN₁(h₁)"
						x={0}
						y={0}
						color={PALETTE.muted}
						{english}
					/>
				</g>
			</g>

			<!-- 8 个隐藏神经元横向排布 -->
			<g transform="translate(0, 40)">
				{#each FFN_ACTIVATIONS as neuron, idx}
					{@const nx = (idx - 3.5) * 115}
					{@const fired = neuronFiringProgress * neuron.act}
					{@const isActive = neuron.act > 0.5}

					<g transform={`translate(${nx}, 0)`}>
						<!-- 神经元外圈 -->
						<circle
							cx="0"
							cy="-15"
							r="24"
							fill={isActive ? `rgba(168, 85, 247, ${0.15 + fired * 0.35})` : "rgba(255, 255, 255, 0.03)"}
							stroke={isActive ? PALETTE.ffn : PALETTE.muted}
							stroke-width={isActive ? 2 : 1}
						/>
						<!-- 脉冲波纹 -->
						{#if isActive && fired > 0.4}
							<circle
								cx="0"
								cy="-15"
								r={24 + 10 * fired}
								fill="none"
								stroke={PALETTE.ffn}
								stroke-width="1"
								opacity={1 - fired}
							/>
						{/if}
						<text
							x="0"
							y="-10"
							text-anchor="middle"
							font-size="11"
							font-weight="800"
							fill={isActive ? PALETTE.ink : PALETTE.muted}
						>
							{fmt(fired, 2)}
						</text>
						<text
							x="0"
							y="24"
							text-anchor="middle"
							font-size="10"
							font-weight="700"
							fill={isActive ? PALETTE.ffn : PALETTE.muted}
						>
							{english ? neuron.labelEn : neuron.labelZh}
						</text>
						<text
							x="0"
							y="38"
							text-anchor="middle"
							font-size="8.5"
							font-family="monospace"
							fill={PALETTE.muted}
						>
							N#{neuron.id}
						</text>
					</g>
				{/each}
			</g>
		</g>
	{/if}

	<!-- ════════ 6.6s 以后：Softmax 预测柱状图与调温控制 (基线位于 y = 85) ════════ -->
	{#if softmaxAlpha > 0}
		<g transform="translate(0, 0)" opacity={softmaxAlpha}>
			<!-- 顶部状态栏: 温度与信息熵徽章 (y = -190) -->
			<g transform="translate(0, -190)">
				<rect
					x="-340"
					y="-20"
					width="680"
					height="40"
					rx="8"
					fill="rgba(17, 26, 36, 0.95)"
					stroke={softmaxData.regimeColor}
					stroke-width="1.5"
				/>
				<text
					x="-310"
					y="5"
					font-size="12.5"
					font-weight="800"
					fill={softmaxData.regimeColor}
				>
					{english ? softmaxData.regimeLabelEn : softmaxData.regimeLabelZh}
				</text>
				<text
					x="120"
					y="5"
					font-size="11.5"
					font-family="monospace"
					font-weight="700"
					fill={PALETTE.ink}
				>
					T={activeTemp.toFixed(2)} | H={softmaxData.entropy.toFixed(2)} bit
				</text>
			</g>

			<!-- 概率映射公式与闭环输出 (y = -135) -->
			<text
				x="0"
				y="-145"
				text-anchor="middle"
				font-size="11"
				font-family="monospace"
				fill={PALETTE.muted}
			>
				P(w_i) = exp(z_i / T) / ∑ exp(z_j / T) &nbsp;&nbsp;|&nbsp;&nbsp; Top-P = {topP.toFixed(2)}
			</text>

			<g transform="translate(0, -120)">
				<VectorChip
					values={LN_FINAL}
					label="LN_final"
					x={0}
					y={0}
					color={PALETTE.muted}
					{english}
				/>
			</g>

			<!-- 动态概率柱状图 (基线 y = 85，5 根柱子间距 155px) -->
			<g transform="translate(0, 85)">
				<!-- 水平基线 -->
				<line
					x1="-380"
					y1="0"
					x2="380"
					y2="0"
					stroke={PALETTE.muted}
					stroke-width="1"
					stroke-opacity="0.4"
				/>

				{#each CANDIDATES as item, idx}
					{@const prob = softmaxData.probs[idx]}
					{@const barW = 86}
					{@const barX = (idx - 2) * 155 - barW / 2}
					{@const barH = Math.min(190, prob * 220)}
					{@const isWinner = idx === 0}
					{@const isPrunedByTopP =
						topPScissorsAlpha > 0.5 &&
						softmaxData.firstPrunedIdx !== -1 &&
						idx >= softmaxData.firstPrunedIdx}

					<!-- 柱体 -->
					<rect
						x={barX}
						y={-barH}
						width={barW}
						height={barH}
						rx="6"
						fill={isPrunedByTopP ? "rgba(244, 63, 94, 0.2)" : isWinner ? PALETTE.v : PALETTE.k}
						stroke={isPrunedByTopP ? PALETTE.head1 : isWinner ? PALETTE.v : PALETTE.k}
						stroke-width={isWinner ? 2.5 : 1.2}
						stroke-dasharray={isPrunedByTopP ? "4 2" : "none"}
						opacity={isPrunedByTopP ? 0.45 : 1}
					/>

					<!-- 概率数字 (百分比) -->
					<text
						x={barX + barW / 2}
						y={-barH - 8}
						text-anchor="middle"
						font-size="12"
						font-weight="800"
						font-family="monospace"
						fill={isPrunedByTopP ? PALETTE.muted : isWinner ? PALETTE.hero : PALETTE.ink}
					>
						{(prob * 100).toFixed(1)}%
					</text>

					<!-- 候选词徽章 (采用 TokenCapsule 原语) -->
					<TokenCapsule
						text={item.word}
						isHero={isWinner}
						x={barX + barW / 2}
						y={32}
						width={64}
						height={40}
						selectable={false}
						showPos={false}
						showId={false}
						tag={`logit: ${fmt(item.logit, 1)}`}
						color={isWinner ? PALETTE.hero : PALETTE.muted}
						{english}
					/>

					<!-- Top-P 剪除标记 -->
					{#if isPrunedByTopP}
						<g transform={`translate(${barX + barW / 2}, ${-barH / 2})`} opacity={topPScissorsAlpha}>
							<rect
								x="-32"
								y="-10"
								width="64"
								height="20"
								rx="4"
								fill="#7f1d1d"
								stroke="#f87171"
								stroke-width="1"
							/>
							<text
								x="0"
								y="4"
								text-anchor="middle"
								font-size="8.5"
								font-weight="800"
								fill="#fef2f2"
							>
								{english ? "✂️ Pruned" : "✂️ 剪除"}
							</text>
						</g>
					{/if}
				{/each}

				<!-- Top-P 累积截断分割线 -->
				{#if topPScissorsAlpha > 0 && softmaxData.firstPrunedIdx !== -1}
					{@const cutX = (softmaxData.firstPrunedIdx - 2.5) * 155}
					<g opacity={topPScissorsAlpha}>
						<line
							x1={cutX}
							y1="35"
							x2={cutX}
							y2="-205"
							stroke={PALETTE.hero}
							stroke-width="2"
							stroke-dasharray="6 4"
						/>
						<!-- 截断线顶部标签 -->
						<g transform={`translate(${cutX}, -195)`}>
							<rect
								x="-60"
								y="-12"
								width="120"
								height="24"
								rx="4"
								fill="rgba(17, 26, 36, 0.95)"
								stroke={PALETTE.hero}
								stroke-width="1.2"
							/>
							<text
								x="0"
								y="4"
								text-anchor="middle"
								font-size="9"
								font-weight="800"
								fill={PALETTE.hero}
							>
								{english
									? `✂️ Top-P Cut (${(topP * 100).toFixed(0)}%)`
									: `✂️ Top-P 截断线 (${(topP * 100).toFixed(0)}%)`}
							</text>
						</g>
					</g>
				{/if}
			</g>
		</g>
	{/if}
</g>

<style>
	.scene-4-container {
		user-select: none;
	}
</style>
