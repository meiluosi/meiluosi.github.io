<script lang="ts">
import { EMBED_MATRIX, POS_MATRIX, TOKENS, X_MATRIX } from "../data/modelData";
import TokenCapsule from "../parts/TokenCapsule.svelte";
import VectorChip from "../parts/VectorChip.svelte";
import { easeOut, fmt, lerp, PALETTE, seg, spring } from "../timeline";

interface Props {
	actTime: number; // 0 ~ 18s (极速高能节拍)
	english?: boolean;
	heroIdx?: number;
	onSelectHero?: (idx: number) => void;
}

const { actTime, english = false, heroIdx = 3, onSelectHero }: Props = $props();

const activeHero = $derived(Math.max(0, Math.min(TOKENS.length - 1, heroIdx)));
const heroToken = $derived(TOKENS[activeHero]);

// ① 打字机节奏 (0 ~ 3.5s)
const promptBoxAlpha = $derived(seg(actTime, 0.2, 1.2, easeOut));
const charCount = $derived(
	Math.floor(lerp(0, 11, seg(actTime, 0.8, 3.5, easeOut))),
);
const fullPrompt = "人工智能能够思考吗？";
const typedPrompt = $derived(fullPrompt.slice(0, charCount));

// ② BPE 树切分与词元展开 (3.5 ~ 9s) - 采用解析解物理弹簧
const bpeAlpha = $derived(seg(actTime, 3.5, 5.0, easeOut));
const tokensSpread = $derived(
	actTime >= 4.5
		? spring(actTime, 4.5, { mass: 1, stiffness: 85, damping: 14 })
		: 0,
);

// ③ 词表几何流形与距离 (8 ~ 13s)
const manifoldAlpha = $derived(seg(actTime, 8.0, 10.5, easeOut));

// ④ 位置编码多频叠加与 X = E + P (11 ~ 18s)
const posBadgeAlpha = $derived(seg(actTime, 11.0, 13.0, easeOut));
const posAdditionProgress = $derived(seg(actTime, 13.5, 16.5, easeOut));
</script>

<g class="scene-1-container">
	<!-- ════════ 顶部跑道: Prompt 输入与 BPE 切分树 (y = -190) ════════ -->
	<g transform="translate(0, -190)" opacity={promptBoxAlpha}>
		<rect
			x="-240"
			y="-22"
			width="480"
			height="44"
			rx="8"
			fill="rgba(17, 26, 36, 0.9)"
			stroke={PALETTE.muted}
			stroke-width="1.2"
			stroke-dasharray={actTime < 4 ? "4 3" : "none"}
		/>
		<text
			x="0"
			y="5"
			text-anchor="middle"
			font-size="16"
			font-weight="700"
			fill={PALETTE.ink}
			letter-spacing="0.06em"
		>
			“ {typedPrompt}{actTime < 4 && actTime % 0.5 < 0.25 ? "▌" : ""} ”
		</text>
		<text
			x="0"
			y="36"
			text-anchor="middle"
			font-size="10.5"
			font-family="monospace"
			fill={PALETTE.muted}
		>
			{english
				? "BPE Tokenizer: Greedily merging subwords -> Discrete Token Sequence"
				: "【BPE 贪心分词】字节对贪心合并树：将连续字串解构为离散词元序列"}
		</text>
	</g>

	<!-- ════════ 中部跑道: 4 个离散词元展开 (y = -40) ════════ -->
	{#if actTime >= 3.5}
		<g transform="translate(0, -40)" opacity={bpeAlpha}>
			{#each TOKENS as token, idx}
				{@const targetX = (idx - 1.5) * 170}
				{@const curX = lerp(0, targetX, tokensSpread)}
				{@const isHero = token.pos === activeHero}

				<TokenCapsule
					text={token.text}
					pos={token.pos}
					id={token.id}
					{isHero}
					x={curX}
					y={0}
					width={110}
					height={52}
					selectable={true}
					showPos={actTime >= 11}
					showId={true}
					{english}
					onClick={() => onSelectHero?.(token.pos)}
				/>
			{/each}

			<!-- 交互提示：点击切换主角词元 -->
			<text
				x="0"
				y="55"
				text-anchor="middle"
				font-size="9.5"
				font-family="monospace"
				fill={PALETTE.hero}
				opacity={Math.min(1, actTime - 5.5)}
			>
				{english
					? "👆 Click any token to select as Hero Query (Bret Victor Explorable)"
					: "👆 点击任意词元切换主角 Query（Bret Victor 可探索模型）"}
			</text>
		</g>
	{/if}

	<!-- ════════ 底部左侧: 2D 降维流形余弦距离 (x = -325, y = +140) ════════ -->
	{#if actTime >= 8}
		<g transform="translate(-325, 140)" opacity={manifoldAlpha}>
			<rect
				x="-105"
				y="-55"
				width="210"
				height="115"
				rx="8"
				fill={PALETTE.panel}
				stroke={PALETTE.faint}
				stroke-width="1"
			/>
			<text
				x="0"
				y="-38"
				text-anchor="middle"
				font-size="10.5"
				font-weight="700"
				fill={PALETTE.embed}
			>
				{english
					? "Cosine Distance Manifold (2D PCA)"
					: "高维空间流形余弦距离 (PCA 2D)"}
			</text>

			<!-- 语义距离点散布 -->
			<circle cx="20" cy="-5" r="4" fill={PALETTE.hero} />
			<text x="30" y="-3" font-size="9" font-weight="700" fill={PALETTE.hero}>
				{heroToken.text}
			</text>

			<circle cx="35" cy="-12" r="3" fill={PALETTE.v} />
			<text x="45" y="-10" font-size="8.5" fill={PALETTE.v}>{english ? "think (0.91)" : "想 (0.91)"}</text>

			<circle cx="-50" cy="15" r="3" fill={PALETTE.muted} />
			<text x="-40" y="22" font-size="8.5" fill={PALETTE.muted}>{english ? "AI (0.12)" : "人工 (0.12)"}</text>

			<!-- 距离引线 -->
			<line
				x1="20"
				y1="-5"
				x2="35"
				y2="-12"
				stroke={PALETTE.hero}
				stroke-width="1"
				stroke-dasharray="2 2"
			/>
			<text x="0" y="45" text-anchor="middle" font-size="9" fill={PALETTE.muted}>
				{english
					? "Similar tokens naturally cluster in vector space"
					: "语义相近词在高维向量空间自发聚类"}
			</text>
		</g>
	{/if}

	<!-- ════════ 底部右侧: 多频正交位置编码与 X = E + P (x = +110, y = +140) ════════ -->
	{#if actTime >= 10}
		<g transform="translate(110, 140)" opacity={posBadgeAlpha}>
			<!-- E[hero] 词嵌入 -->
			<VectorChip
				values={EMBED_MATRIX[activeHero]}
				label={`E["${heroToken.text}"]`}
				x={-140}
				y={0}
				color={PALETTE.embed}
				{english}
			/>

			<text
				x="-40"
				y="6"
				text-anchor="middle"
				font-size="16"
				font-weight="900"
				fill={PALETTE.muted}
			>
				+
			</text>

			<!-- P[hero] 位置编码 -->
			<VectorChip
				values={POS_MATRIX[activeHero]}
				label={`P[pos=${activeHero}]`}
				x={60}
				y={0}
				color={PALETTE.pos}
				{english}
			/>

			<!-- 等号与最终 X 向量 -->
			{#if actTime >= 13}
				<text
					x="160"
					y="6"
					text-anchor="middle"
					font-size="16"
					font-weight="900"
					fill={PALETTE.muted}
					opacity={posAdditionProgress}
				>
					=
				</text>

				<g opacity={posAdditionProgress}>
					<VectorChip
						values={X_MATRIX[activeHero]}
						label={`X["${heroToken.text}"]`}
						x={260}
						y={0}
						color={PALETTE.hero}
						{english}
					/>
				</g>
			{/if}
		</g>
	{/if}
</g>
