<script lang="ts">
import {
	computeFullAttentionPipeline,
	HEAD1,
	HEAD2,
	TOKENS,
	X_MATRIX,
} from "../data/modelData";
import MatrixGrid from "../parts/MatrixGrid.svelte";
import TokenCapsule from "../parts/TokenCapsule.svelte";
import VectorChip from "../parts/VectorChip.svelte";
import { easeInOut, easeOut, lerp, PALETTE, seg, spring } from "../timeline";

interface Props {
	actTime: number; // 0 ~ 20s (极速高能节拍)
	english?: boolean;
	heroIdx?: number;
	onSelectHero?: (idx: number) => void;
}

const { actTime, english = false, heroIdx = 3, onSelectHero }: Props = $props();

const activeHero = $derived(Math.max(0, Math.min(TOKENS.length - 1, heroIdx)));
const heroToken = $derived(TOKENS[activeHero]);

// 闭环动态注意力推导
const fullPipeline = $derived(computeFullAttentionPipeline(activeHero));

// ① 输入向量与点积撞击 (0 ~ 6s)
const introAlpha = $derived(seg(actTime, 0.2, 1.5, easeOut));
const dotProductStream = $derived(seg(actTime, 1.5, 5.0, easeInOut));
const blindspotWarningAlpha = $derived(seg(actTime, 3.5, 5.5, easeOut));

// ② 多头注意力分立 (6 ~ 13s)
const forkProgress = $derived(
	actTime >= 6.0
		? spring(actTime, 6.0, { mass: 1, stiffness: 90, damping: 15 })
		: 0,
);
const head1Alpha = $derived(seg(actTime, 7.5, 9.0, easeOut));
const head2Alpha = $derived(seg(actTime, 8.5, 10.0, easeOut));

// ③ QKV 三向探针展开 (12 ~ 20s)
const qkvAlpha = $derived(seg(actTime, 11.5, 13.0, easeOut));
const qkvUnpack = $derived(seg(actTime, 12.0, 16.0, easeOut));
const pulsePhase = $derived((actTime * 3) % (2 * Math.PI));

// 交互悬停行状态
let hoveredWqRow = $state<number | null>(null);
</script>

<g class="scene-2-container" opacity={introAlpha}>
	<!-- ════════ 顶部跑道: 概念徽章与单头盲区警告 (y = -240) ════════ -->
	<g transform="translate(0, -240)">
		<rect
			x="-320"
			y="-20"
			width="640"
			height="40"
			rx="8"
			fill="rgba(17, 26, 36, 0.9)"
			stroke={PALETTE.muted}
			stroke-width="1.2"
		/>
		<text
			x="0"
			y="5"
			text-anchor="middle"
			font-size="14"
			font-weight="700"
			fill={PALETTE.ink}
			letter-spacing="0.04em"
		>
			{english
				? "ACT II: Multi-Head Projection Array (X -> Head 1 & Head 2 Fork)"
				: "第二幕：多头注意力分立与 QKV 几何投影阵列"}
		</text>

		{#if blindspotWarningAlpha > 0}
			<g transform="translate(0, 32)" opacity={blindspotWarningAlpha}>
				<rect
					x="-260"
					y="-12"
					width="520"
					height="24"
					rx="6"
					fill="rgba(244, 63, 94, 0.15)"
					stroke={PALETTE.head1}
					stroke-width="1"
				/>
				<text
					x="0"
					y="4"
					text-anchor="middle"
					font-size="11"
					font-weight="700"
					fill={PALETTE.head1}
				>
					{english
						? "⚠️ Single-head blindspot: Cannot attend to syntax and topic simultaneously!"
						: "⚠️ 单头视野盲区：单一注意力无法兼顾句法修饰与主题语义，必须多头分化！"}
				</text>
			</g>
		{/if}
	</g>

	<!-- ════════ 左侧: 主角词元与向量 X[hero] (x = -390) ════════ -->
	<g transform="translate(-390, 0)">
		<TokenCapsule
			text={heroToken.text}
			pos={heroToken.pos}
			id={heroToken.id}
			isHero={true}
			x={0}
			y={-50}
			width={100}
			height={48}
			selectable={true}
			showPos={true}
			showId={true}
			{english}
			onClick={() => onSelectHero?.((activeHero + 1) % TOKENS.length)}
		/>

		<!-- X[hero] 输入向量 Chip -->
		<VectorChip
			values={X_MATRIX[activeHero]}
			label={`X[${heroToken.text}]`}
			x={0}
			y={25}
			color={PALETTE.hero}
			{english}
		/>

		<text
			x="0"
			y="60"
			text-anchor="middle"
			font-size="9"
			font-family="monospace"
			fill={PALETTE.muted}
		>
			d_model = 4
		</text>

		<!-- 点积粒子流向矩阵 -->
		{#if dotProductStream > 0}
			<g opacity={1 - forkProgress}>
				<line
					x1="55"
					y1="-10"
					x2={lerp(55, 140, dotProductStream)}
					y2="-10"
					stroke={PALETTE.hero}
					stroke-width="3"
					stroke-linecap="round"
					stroke-dasharray="6 4"
				/>
			</g>
		{/if}
	</g>

	<!-- ════════ 中央: 投影矩阵门 W_proj (x = -230) ════════ -->
	<g transform="translate(-230, 0)">
		<MatrixGrid
			matrix={HEAD1.WQ}
			title="W_Q (4 × 2)"
			x={0}
			y={0}
			cellWidth={34}
			cellHeight={22}
			digits={1}
			color={PALETTE.q}
			hoveredRow={hoveredWqRow}
			onRowHover={(r) => (hoveredWqRow = r)}
			{english}
		/>

		<!-- 分流贝塞尔连接曲线 -->
		{#if forkProgress > 0}
			<path
				d={`M 50 0 C 85 0, 95 ${lerp(0, -110, forkProgress)}, 135 -110`}
				fill="none"
				stroke={PALETTE.head1}
				stroke-width="2.5"
				stroke-linecap="round"
				opacity={forkProgress}
			/>
			<path
				d={`M 50 0 C 85 0, 95 ${lerp(0, 110, forkProgress)}, 135 110`}
				fill="none"
				stroke={PALETTE.head2}
				stroke-width="2.5"
				stroke-linecap="round"
				opacity={forkProgress}
			/>
		{/if}
	</g>

	<!-- ════════ 右侧上半区: Head 1 句法修饰头 (y = -110) ════════ -->
	{#if head1Alpha > 0}
		<g transform="translate(0, -110)" opacity={head1Alpha}>
			<!-- 背景卡片 -->
			<rect
				x="-70"
				y="-65"
				width="560"
				height="130"
				rx="10"
				fill="rgba(244, 63, 94, 0.06)"
				stroke={PALETTE.head1}
				stroke-width="1.5"
			/>

			<!-- 头标题与定位目标 -->
			<g transform="translate(-55, -42)">
				<rect
					x="0"
					y="-12"
					width="170"
					height="24"
					rx="5"
					fill="rgba(244, 63, 94, 0.2)"
					stroke={PALETTE.head1}
					stroke-width="1"
				/>
				<text
					x="85"
					y="4"
					text-anchor="middle"
					font-size="11.5"
					font-weight="800"
					fill={PALETTE.head1}
				>
					{english ? HEAD1.nameEn : HEAD1.nameZh}
				</text>
				<text
					x="185"
					y="4"
					font-size="11"
					font-weight="700"
					fill={PALETTE.ink}
				>
					{english
						? `Target: [${fullPipeline.h1.focusTarget}]`
						: `聚焦目标：【${fullPipeline.h1.focusTarget}】`}
				</text>
			</g>

			<!-- Q1, K1, V1 三向探针 -->
			{#if qkvAlpha > 0}
				<g transform="translate(145, 15)" opacity={qkvAlpha}>
					<!-- Q1 探针 -->
					<VectorChip
						values={fullPipeline.h1.q}
						label="Q₁"
						x={lerp(0, -140, qkvUnpack)}
						y={0}
						color={PALETTE.q}
						{english}
					/>

					<!-- K1 探针 -->
					<VectorChip
						values={HEAD1.K[activeHero]}
						label={`K₁[${heroToken.text}]`}
						x={lerp(0, 15, qkvUnpack)}
						y={0}
						color={PALETTE.k}
						{english}
					/>

					<!-- V1 探针 -->
					<VectorChip
						values={HEAD1.V[activeHero]}
						label={`V₁[${heroToken.text}]`}
						x={lerp(0, 170, qkvUnpack)}
						y={0}
						color={PALETTE.v}
						{english}
					/>
				</g>
			{/if}
		</g>
	{/if}

	<!-- ════════ 右侧下半区: Head 2 语义主语头 (y = +110) ════════ -->
	{#if head2Alpha > 0}
		<g transform="translate(0, 110)" opacity={head2Alpha}>
			<!-- 背景卡片 -->
			<rect
				x="-70"
				y="-65"
				width="560"
				height="130"
				rx="10"
				fill="rgba(14, 165, 233, 0.06)"
				stroke={PALETTE.head2}
				stroke-width="1.5"
			/>

			<!-- 头标题与定位目标 -->
			<g transform="translate(-55, -42)">
				<rect
					x="0"
					y="-12"
					width="170"
					height="24"
					rx="5"
					fill="rgba(14, 165, 233, 0.2)"
					stroke={PALETTE.head2}
					stroke-width="1"
				/>
				<text
					x="85"
					y="4"
					text-anchor="middle"
					font-size="11.5"
					font-weight="800"
					fill={PALETTE.head2}
				>
					{english ? HEAD2.nameEn : HEAD2.nameZh}
				</text>
				<text
					x="185"
					y="4"
					font-size="11"
					font-weight="700"
					fill={PALETTE.ink}
				>
					{english
						? `Target: [${fullPipeline.h2.focusTarget}]`
						: `聚焦目标：【${fullPipeline.h2.focusTarget}】`}
				</text>
			</g>

			<!-- Q2, K2, V2 三向探针 -->
			{#if qkvAlpha > 0}
				<g transform="translate(145, 15)" opacity={qkvAlpha}>
					<!-- Q2 探针 -->
					<VectorChip
						values={fullPipeline.h2.q}
						label="Q₂"
						x={lerp(0, -140, qkvUnpack)}
						y={0}
						color={PALETTE.q}
						{english}
					/>

					<!-- K2 探针 -->
					<VectorChip
						values={HEAD2.K[activeHero]}
						label={`K₂[${heroToken.text}]`}
						x={lerp(0, 15, qkvUnpack)}
						y={0}
						color={PALETTE.k}
						{english}
					/>

					<!-- V2 探针 -->
					<VectorChip
						values={HEAD2.V[activeHero]}
						label={`V₂[${heroToken.text}]`}
						x={lerp(0, 170, qkvUnpack)}
						y={0}
						color={PALETTE.v}
						{english}
					/>
				</g>
			{/if}
		</g>
	{/if}
</g>
