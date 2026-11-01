<script lang="ts">
import {
	computeAttention,
	D_K,
	empiricalDotVariance,
	FUTURE_TOKEN,
	HEAD1,
	HEAD2,
	SQRT_DK,
	TOKENS,
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
	actTime: number; // 0 ~ 20s
	english?: boolean;
	heroIdx?: number;
	onSelectHero?: (idx: number) => void;
}

const { actTime, english = false, heroIdx = 3, onSelectHero }: Props = $props();

const activeHero = $derived(Math.max(0, Math.min(TOKENS.length - 1, heroIdx)));
const heroToken = $derived(TOKENS[activeHero]);

// 闭环动态注意力计算 (对选中的主角词元实时计算两头分布)
const h1Attn = $derived(computeAttention(0, activeHero));
const h2Attn = $derived(computeAttention(1, activeHero));

// Bret Victor 探索态交互：悬停追踪 (Provenance Tracing)
let hoveredKeyIdx = $state<number | null>(null);
let isZHovered = $state(false);

// ① 雷达扫描与除以根号 d_k (0 ~ 6s)
const introAlpha = $derived(seg(actTime, 0.2, 1.2, easeOut));
const radarSweep = $derived(seg(actTime, 1.2, 5.0, easeInOut));
const sqrtDkBadgeAlpha = $derived(seg(actTime, 2.5, 5.5, easeOut));

// 经验方差闭环派生
const varRaw = $derived(empiricalDotVariance(D_K, false).variance);
const varScaled = $derived(empiricalDotVariance(D_K, true).variance);

// ② 因果掩码防作弊铁幕 (6 ~ 13s)
const futureTokenAlpha = $derived(seg(actTime, 6.0, 7.5, easeOut));
const cheatRayProgress = $derived(seg(actTime, 7.0, 9.0, easeInOut));
const maskShieldDrop = $derived(
	actTime >= 8.5
		? spring(actTime, 8.5, { mass: 1, stiffness: 110, damping: 13 })
		: 0,
);
const maskLockedAlpha = $derived(seg(actTime, 10.5, 13.0, easeOut));

// ③ ★ Value 向量加权汇聚生成 Z (12 ~ 20s)
const valueGatherAlpha = $derived(seg(actTime, 12.5, 14.5, easeOut));
const gatherParticles = $derived(seg(actTime, 13.5, 17.5, easeInOut));

// 词元在跑道上的 X 坐标基准
function getTokenX(idx: number): number {
	return (idx - 1.5) * 160;
}

const heroX = $derived(getTokenX(activeHero));
const maskWallX = $derived(heroX + 80);
</script>

<g class="scene-3-container" opacity={introAlpha}>
	<!-- ════════ 顶部状态栏: 根号 d_k 抑噪与掩码结果提示 (y = -245) ════════ -->
	<g transform="translate(0, -245)">
		{#if sqrtDkBadgeAlpha > 0}
			<g transform="translate(-130, 0)" opacity={sqrtDkBadgeAlpha}>
				<rect
					x="-195"
					y="-16"
					width="390"
					height="32"
					rx="6"
					fill="rgba(17, 26, 36, 0.95)"
					stroke={PALETTE.hero}
					stroke-width="1.2"
				/>
				<text
					x="0"
					y="4"
					text-anchor="middle"
					font-size="10.5"
					font-weight="700"
					fill={PALETTE.ink}
				>
					{english
						? `Variance suppression: ÷ √d_k (${fmt(SQRT_DK, 2)}) Var ${fmt(varRaw, 1)} -> ${fmt(varScaled, 1)}`
						: `⚡ 缩放抑噪：除以 √d_k = ${fmt(SQRT_DK, 2)}，将方差由 ${fmt(varRaw, 1)} 压回 ${fmt(varScaled, 1)}，阻断 Softmax 饱和`}
				</text>
			</g>
		{/if}

		{#if maskLockedAlpha > 0}
			<g transform="translate(230, 0)" opacity={maskLockedAlpha}>
				<rect
					x="-120"
					y="-16"
					width="240"
					height="32"
					rx="6"
					fill="rgba(127, 29, 29, 0.5)"
					stroke="#ef4444"
					stroke-width="1.2"
				/>
				<text
					x="0"
					y="4"
					text-anchor="middle"
					font-size="10"
					font-weight="700"
					fill="#fca5a5"
				>
					{english
						? "Softmax(-∞) = 0.00 (Fully Insulated)"
						: "Softmax(-∞) = 0.00 (彻底绝缘)"}
				</text>
			</g>
		{/if}
	</g>

	<!-- ════════ 次顶跑道: 4 离散词元 + 因果掩码铁幕 (y = -170) ════════ -->
	<g transform="translate(0, -170)">
		{#each TOKENS as tok, idx}
			{@const tx = getTokenX(idx)}
			{@const isHero = idx === activeHero}
			{@const isMaskedOut = idx > activeHero}
			{@const isHighlighted =
				hoveredKeyIdx === idx || (isZHovered && h1Attn.weights[idx] > 0.05)}

			<TokenCapsule
				text={tok.text}
				pos={tok.pos}
				id={tok.id}
				{isHero}
				isHovered={isHighlighted}
				x={tx}
				y={0}
				width={105}
				height={50}
				color={isMaskedOut ? PALETTE.mask : undefined}
				opacity={isMaskedOut ? 0.45 : 1.0}
				selectable={true}
				showPos={true}
				showId={true}
				{english}
				onClick={() => onSelectHero?.(idx)}
				onHover={(h) => (hoveredKeyIdx = h ? idx : null)}
			/>

			<!-- Head 1 与 Head 2 权重指示徽章 -->
			{#if actTime >= 4.0 && !isMaskedOut}
				<g transform="translate({tx}, 36)" opacity={seg(actTime, 4.0, 5.5, easeOut)}>
					<!-- Head 1 徽章 -->
					<rect
						x="-44"
						y="-8"
						width="42"
						height="16"
						rx="3"
						fill="rgba(244, 63, 94, 0.2)"
						stroke={PALETTE.head1}
						stroke-width="0.8"
					/>
					<text
						x="-23"
						y="3.5"
						text-anchor="middle"
						font-size="8"
						font-family="monospace"
						font-weight="700"
						fill={PALETTE.head1}
					>
						{fmt(h1Attn.weights[idx] * 100, 0)}%
					</text>

					<!-- Head 2 徽章 -->
					<rect
						x="2"
						y="-8"
						width="42"
						height="16"
						rx="3"
						fill="rgba(14, 165, 233, 0.2)"
						stroke={PALETTE.head2}
						stroke-width="0.8"
					/>
					<text
						x="23"
						y="3.5"
						text-anchor="middle"
						font-size="8"
						font-family="monospace"
						font-weight="700"
						fill={PALETTE.head2}
					>
						{fmt(h2Attn.weights[idx] * 100, 0)}%
					</text>
				</g>
			{/if}
		{/each}

		<!-- 因果掩码激光铁幕 (Causal Mask Wall: 动态立在当前主角右侧) -->
		{#if maskShieldDrop > 0}
			<g
				transform={`translate(${maskWallX}, ${lerp(-100, 0, maskShieldDrop)})`}
				opacity={maskShieldDrop}
			>
				<line
					x1="0"
					y1="-60"
					x2="0"
					y2="60"
					stroke="#ef4444"
					stroke-width="3"
					stroke-linecap="round"
				/>
				<!-- 护盾警告符 -->
				<rect
					x="-38"
					y="-14"
					width="76"
					height="28"
					rx="5"
					fill="#7f1d1d"
					stroke="#f87171"
					stroke-width="1.2"
				/>
				<text
					x="0"
					y="4"
					text-anchor="middle"
					font-size="10.5"
					font-weight="900"
					fill="#fef2f2"
				>
					{english ? "-∞ Mask" : "-∞ 掩码"}
				</text>
			</g>
		{/if}

		<!-- 未来词元 【吗】(右侧作弊者，严格隔离) -->
		{#if futureTokenAlpha > 0}
			<g transform="translate(380, 0)" opacity={futureTokenAlpha}>
				<TokenCapsule
					text={FUTURE_TOKEN.text}
					pos={FUTURE_TOKEN.pos}
					id={FUTURE_TOKEN.id}
					x={0}
					y={0}
					width={88}
					height={48}
					color={PALETTE.head1}
					selectable={false}
					showPos={true}
					showId={true}
					{english}
				/>
			</g>
		{/if}
	</g>

	<!-- ════════ 中部: 动态贝塞尔注意力拉拽流线 (y = -140 ~ -20) ════════ -->
	{#if actTime >= 2.0}
		<g transform="translate(0, 0)">
			<!-- 遍历所有未被因果掩码遮蔽的 Keys (j <= activeHero) -->
			{#each TOKENS as _, j}
				{#if j <= activeHero}
					{@const targetX = getTokenX(j)}
					{@const w1 = h1Attn.weights[j]}
					{@const w2 = h2Attn.weights[j]}

					<!-- Head 1 玫瑰粉弧线 -->
					{#if w1 > 0.02}
						<AttentionArc
							sourceX={heroX}
							sourceY={-140}
							{targetX}
							targetY={-140}
							weight={w1}
							rawScore={h1Attn.scores[j]}
							color={PALETTE.head1}
							curvature={-70 - (activeHero - j) * 8}
							progress={Math.min(1, radarSweep * 1.5)}
							isHovered={hoveredKeyIdx === j}
							isFocus={w1 === Math.max(...h1Attn.weights)}
							label={`H1: ${fmt(w1 * 100, 0)}%`}
							badgeOffset={-8}
							{english}
							onHover={(h) => (hoveredKeyIdx = h ? j : null)}
						/>
					{/if}

					<!-- Head 2 湛蓝弧线 -->
					{#if w2 > 0.02 && j !== activeHero}
						<AttentionArc
							sourceX={heroX}
							sourceY={-140}
							{targetX}
							targetY={-140}
							weight={w2}
							rawScore={h2Attn.scores[j]}
							color={PALETTE.head2}
							curvature={50 + (activeHero - j) * 6}
							progress={Math.min(1, radarSweep * 1.5)}
							isHovered={hoveredKeyIdx === j}
							isFocus={w2 === Math.max(...h2Attn.weights)}
							label={`H2: ${fmt(w2 * 100, 0)}%`}
							badgeOffset={10}
							{english}
							onHover={(h) => (hoveredKeyIdx = h ? j : null)}
						/>
					{/if}
				{/if}
			{/each}

			<!-- 作弊射线被铁幕反弹阻断 (从 '吗' x=380 射向掩码) -->
			{#if cheatRayProgress > 0 && actTime < 12.0}
				<line
					x1="380"
					y1="-140"
					x2={lerp(380, maskWallX, cheatRayProgress)}
					y2="-140"
					stroke="#f87171"
					stroke-width="2.5"
					stroke-dasharray="3 3"
				/>
			{/if}
		</g>
	{/if}

	<!-- ════════ 下部: ★ Value 向量加权汇聚生成语境向量 Z (y = 150) ════════ -->
	{#if valueGatherAlpha > 0}
		<g transform="translate(0, 150)" opacity={valueGatherAlpha}>
			<!-- 背景卡片 -->
			<rect
				x="-480"
				y="-65"
				width="960"
				height="130"
				rx="12"
				fill="rgba(17, 26, 36, 0.95)"
				stroke={isZHovered ? PALETTE.hero : PALETTE.v}
				stroke-width={isZHovered ? 2 : 1.5}
			/>

			<!-- 汇聚公式标题 -->
			<g transform="translate(-450, -40)">
				<rect
					x="0"
					y="-11"
					width="280"
					height="22"
					rx="4"
					fill="rgba(74, 222, 128, 0.15)"
					stroke={PALETTE.v}
					stroke-width="1"
				/>
				<text
					x="140"
					y="4"
					text-anchor="middle"
					font-size="11"
					font-weight="800"
					fill={PALETTE.v}
				>
					{english
						? "★ Bret Victor Provenance: Z = ∑ α_i V_i"
						: "★ 因果溯源汇聚：Z = ∑ α_i V_i"}
				</text>
			</g>

			<!-- 汇聚粒子流向 -->
			{#if gatherParticles > 0}
				<g opacity={gatherParticles}>
					<path
						d={`M -200 -20 C -100 -20, -50 10, ${lerp(-100, 160, gatherParticles)} 10`}
						fill="none"
						stroke={PALETTE.head1}
						stroke-width="2"
						stroke-dasharray="4 4"
					/>
					<path
						d={`M -200 35 C -100 35, -50 10, ${lerp(-100, 160, gatherParticles)} 10`}
						fill="none"
						stroke={PALETTE.head2}
						stroke-width="2"
						stroke-dasharray="4 4"
					/>
				</g>
			{/if}

			<!-- 左侧: 双头加权计算公式说明 -->
			<g transform="translate(-450, -2)">
				<text
					x="0"
					y="0"
					font-size="11"
					font-family="monospace"
					font-weight="700"
					fill={PALETTE.head1}
				>
					{english ? "Head 1: " : "Head 1: "}Z₁ = {h1Attn.weights
						.map((w, j) => (j <= activeHero ? `${fmt(w, 2)}×V[${TOKENS[j].text}]` : ""))
						.filter(Boolean)
						.join(" + ")}
				</text>
				<text
					x="0"
					y="24"
					font-size="11"
					font-family="monospace"
					font-weight="700"
					fill={PALETTE.head2}
				>
					{english ? "Head 2: " : "Head 2: "}Z₂ = {h2Attn.weights
						.map((w, j) => (j <= activeHero ? `${fmt(w, 2)}×V[${TOKENS[j].text}]` : ""))
						.filter(Boolean)
						.join(" + ")}
				</text>
				<text
					x="0"
					y="46"
					font-size="9.5"
					fill={PALETTE.muted}
				>
					{english
						? "💡 Hover Z to highlight source V vectors; Click tokens above to dynamically swap Hero!"
						: "💡 悬停 Z 查看加权源向量；点击上方词元动态更换主角 Query 并重算！"}
				</text>
			</g>

			<!-- 右侧: Z 语境向量晶体展示 (使用 VectorChip 原语) -->
			<g transform="translate(290, 5)">
				<VectorChip
					values={h1Attn.Z}
					label="Z₁"
					x={-100}
					y={0}
					color={PALETTE.head1}
					isHovered={isZHovered}
					{english}
					onHover={(h) => (isZHovered = h)}
				/>
				<VectorChip
					values={h2Attn.Z}
					label="Z₂"
					x={20}
					y={0}
					color={PALETTE.head2}
					isHovered={isZHovered}
					{english}
					onHover={(h) => (isZHovered = h)}
				/>
				<VectorChip
					values={[...h1Attn.Z, ...h2Attn.Z]}
					label="Concat(Z)"
					x={145}
					y={0}
					color={PALETTE.hero}
					isHovered={isZHovered}
					{english}
					onHover={(h) => (isZHovered = h)}
				/>
			</g>
		</g>
	{/if}
</g>
