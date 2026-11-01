<script lang="ts">
import {
	KV_CACHE_TABLE,
	NEW_TOKEN,
	TOKENS,
	WINNER_PROB,
} from "../data/modelData";
import AttentionArc from "../parts/AttentionArc.svelte";
import TokenCapsule from "../parts/TokenCapsule.svelte";
import VectorChip from "../parts/VectorChip.svelte";
import {
	easeInOut,
	easeOut,
	fmt,
	lerp,
	PALETTE,
	seg,
	spring,
} from "../timeline";

interface Props {
	actTime: number; // 0 ~ 12s
	english?: boolean;
	heroIdx?: number;
}

const { actTime, english = false, heroIdx = 3 }: Props = $props();

// ① 轮盘赌与胜出新词结晶 (0 ~ 6s) - 使用物理谐振弹簧
const introAlpha = $derived(seg(actTime, 0.2, 1.2, easeOut));
const tokenCrystallize = $derived(
	actTime >= 1.0
		? spring(actTime, 1.0, { mass: 1, stiffness: 100, damping: 14 })
		: 0,
);
const sentenceAssembleAlpha = $derived(seg(actTime, 3.5, 5.5, easeOut));

// ② ★ KV Cache 显存常驻表与 O(1) 追加 (5 ~ 12s)
const kvCacheAlpha = $derived(seg(actTime, 5.0, 7.0, easeOut));
const appendHighlight = $derived(seg(actTime, 6.5, 8.5, easeInOut));
const loopFlowAlpha = $derived(seg(actTime, 8.5, 11.5, easeOut));

const badgeScale = $derived(lerp(0.4, 1.0, Math.min(1.2, tokenCrystallize)));
</script>

<g class="scene-5-container" opacity={introAlpha}>
	<!-- ════════ 顶部跑道: 组装全句流水线 (y = -220) ════════ -->
	<g transform="translate(0, -220)" opacity={sentenceAssembleAlpha}>
		<rect
			x="-400"
			y="-28"
			width="800"
			height="56"
			rx="10"
			fill="rgba(17, 26, 36, 0.95)"
			stroke={PALETTE.v}
			stroke-width="1.5"
		/>
		<!-- 词元组装序列 -->
		<g transform="translate(-250, 0)">
			{#each TOKENS as t, i}
				<TokenCapsule
					text={t.text}
					pos={t.pos}
					id={t.id}
					x={i * 105}
					y={0}
					width={80}
					height={38}
					isHero={false}
					selectable={false}
					showPos={false}
					showId={false}
					{english}
				/>
				<text
					x={i * 105 + 50}
					y={4}
					text-anchor="middle"
					font-size="14"
					fill={PALETTE.muted}
				>
					➜
				</text>
			{/each}

			<!-- 胜出追加词元 【考】 -->
			<TokenCapsule
				text={NEW_TOKEN.text}
				pos={NEW_TOKEN.pos}
				id={NEW_TOKEN.id}
				x={4 * 105}
				y={0}
				width={86}
				height={42}
				isHero={true}
				selectable={false}
				showPos={false}
				showId={false}
				color={PALETTE.hero}
				{english}
			/>
		</g>
	</g>

	<!-- ════════ 左侧区: 新词元【考】结晶胜出徽章 (x = -320, y = -25) ════════ -->
	<g transform="translate(-320, -25)">
		<g transform={`scale(${badgeScale})`}>
			<TokenCapsule
				text={NEW_TOKEN.text}
				pos={NEW_TOKEN.pos}
				id={NEW_TOKEN.id}
				isHero={true}
				x={0}
				y={-10}
				width={110}
				height={64}
				selectable={false}
				showPos={true}
				showId={true}
				{english}
			/>

			<text
				x="0"
				y="62"
				text-anchor="middle"
				font-size="12"
				font-weight="800"
				fill={PALETTE.v}
			>
				{english ? "Winner Token: [考]" : "轮盘赌胜出：【考】"}
			</text>
			<text
				x="0"
				y="80"
				text-anchor="middle"
				font-size="10"
				font-family="monospace"
				fill={PALETTE.muted}
			>
				P = {(WINNER_PROB * 100).toFixed(1)}% (Top-P Renormalized)
			</text>
		</g>

		<!-- 自回归因果流动反馈弧线 (采用 AttentionArc) -->
		{#if loopFlowAlpha > 0}
			<g opacity={loopFlowAlpha}>
				<AttentionArc
					sourceX={50}
					sourceY={10}
					targetX={-40}
					targetY={120}
					weight={0.9}
					color={PALETTE.v}
					curvature={60}
					label={english ? "Next Input" : "下一轮输入"}
					showBadge={true}
					badgeOffset={14}
					{english}
				/>
				<text
					x="0"
					y="160"
					text-anchor="middle"
					font-size="10"
					font-weight="800"
					fill={PALETTE.v}
				>
					{english
						? "Autoregressive Loop: [考] is next input"
						: "自回归因果闭环：【考】成为下一轮输入"}
				</text>
			</g>
		{/if}
	</g>

	<!-- ════════ 右侧区: ★ KV Cache 显存常驻表 (x = 110, y = -25) ════════ -->
	{#if kvCacheAlpha > 0}
		<g transform="translate(110, -25)" opacity={kvCacheAlpha}>
			<!-- 表格卡片底板 -->
			<rect
				x="-280"
				y="-125"
				width="560"
				height="250"
				rx="12"
				fill="rgba(17, 26, 36, 0.95)"
				stroke={PALETTE.cache}
				stroke-width="1.8"
			/>

			<!-- 表格标题 -->
			<g transform="translate(0, -98)">
				<text
					x="0"
					y="0"
					text-anchor="middle"
					font-size="13"
					font-weight="800"
					fill={PALETTE.cache}
				>
					{english
						? "★ KV Cache Memory Table (O(1) Append)"
						: "★ KV Cache 显存常驻表：历史常驻，O(1) 瞬时追加"}
				</text>
				<text
					x="0"
					y="16"
					text-anchor="middle"
					font-size="9"
					font-family="monospace"
					fill={PALETTE.muted}
				>
					{english
						? "Previous tokens are never recomputed! Only [考] adds new (K, V)."
						: "无需重复计算前驱词元！仅需计算新词【考】并压入缓存尾部"}
				</text>
			</g>

			<!-- 5 行 KV Cache 记录 (采用 VectorChip 原语) -->
			<g transform="translate(0, -50)">
				{#each KV_CACHE_TABLE as row, idx}
					{@const ry = idx * 34}
					{@const isAppended = row.isNew}
					{@const rowAlpha = isAppended ? appendHighlight : 1}

					<g transform={`translate(0, ${ry})`} opacity={rowAlpha}>
						<!-- 斑马纹底色与新插入高亮 -->
						<rect
							x="-260"
							y="-12"
							width="520"
							height="28"
							rx="4"
							fill={isAppended
								? "rgba(16, 185, 129, 0.18)"
								: idx % 2 === 0
									? "rgba(255, 255, 255, 0.02)"
									: "rgba(255, 255, 255, 0.04)"}
							stroke={isAppended ? PALETTE.cache : "none"}
							stroke-width={isAppended ? 1.5 : 0}
						/>

						<!-- 位置与词元 -->
						<text
							x="-235"
							y="4"
							font-size="10"
							font-family="monospace"
							font-weight="700"
							fill={isAppended ? PALETTE.hero : PALETTE.ink}
						>
							p{row.pos}
						</text>
						<text
							x="-195"
							y="4"
							font-size="12"
							font-weight={isAppended ? 900 : 700}
							fill={isAppended ? PALETTE.hero : PALETTE.ink}
						>
							{row.token}
						</text>

						<!-- K 向量 Chip -->
						<VectorChip
							values={row.k}
							label="K"
							x={-85}
							y={0}
							scale={0.85}
							color={PALETTE.k}
							{english}
						/>

						<!-- V 向量 Chip -->
						<VectorChip
							values={row.v}
							label="V"
							x={85}
							y={0}
							scale={0.85}
							color={PALETTE.v}
							{english}
						/>

						<!-- 状态徽章 -->
						<g transform="translate(210, 0)">
							{#if isAppended}
								<rect
									x="-32"
									y="-9"
									width="64"
									height="18"
									rx="3"
									fill="rgba(16, 185, 129, 0.3)"
									stroke={PALETTE.cache}
									stroke-width="1"
								/>
								<text
									x="0"
									y="3.5"
									text-anchor="middle"
									font-size="8.5"
									font-weight="800"
									fill={PALETTE.cache}
								>
									{english ? "+O(1) NEW" : "+O(1) 新增"}
								</text>
							{:else}
								<text
									x="0"
									y="3"
									text-anchor="middle"
									font-size="8"
									font-family="monospace"
									fill={PALETTE.muted}
								>
									{english ? "Cached" : "常驻"}
								</text>
							{/if}
						</g>
					</g>
				{/each}
			</g>
		</g>
	{/if}
</g>
