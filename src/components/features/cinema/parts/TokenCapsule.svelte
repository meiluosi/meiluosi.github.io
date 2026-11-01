<script lang="ts">
import { PALETTE } from "../timeline";

interface Props {
	text: string;
	pos?: number;
	id?: number;
	tag?: string;
	isHero?: boolean;
	isHovered?: boolean;
	selectable?: boolean;
	x?: number;
	y?: number;
	width?: number;
	height?: number;
	rx?: number;
	color?: string;
	scale?: number;
	opacity?: number;
	showPos?: boolean;
	showId?: boolean;
	english?: boolean;
	onClick?: () => void;
	onHover?: (hovered: boolean) => void;
}

const {
	text,
	pos,
	id,
	tag,
	isHero = false,
	isHovered = false,
	selectable = true,
	x = 0,
	y = 0,
	width = 110,
	height = 52,
	rx = 8,
	color,
	scale = 1,
	opacity = 1,
	showPos = true,
	showId = true,
	english = false,
	onClick,
	onHover,
}: Props = $props();

let internalHover = $state(false);
const activeHover = $derived(isHovered || internalHover);

const strokeColor = $derived(
	isHero ? PALETTE.hero : activeHover ? PALETTE.ink : (color ?? PALETTE.muted),
);

const fillColor = $derived(
	isHero
		? "rgba(251, 191, 36, 0.16)"
		: activeHover
			? "rgba(255, 255, 255, 0.12)"
			: PALETTE.panel,
);

function handleMouseEnter() {
	internalHover = true;
	onHover?.(true);
}

function handleMouseLeave() {
	internalHover = false;
	onHover?.(false);
}

function handleClick(e: MouseEvent | KeyboardEvent) {
	if (!selectable) return;
	if (e instanceof KeyboardEvent && e.key !== "Enter" && e.key !== " ") {
		return;
	}
	e.preventDefault();
	onClick?.();
}
</script>

<g
	class="token-capsule"
	transform="translate({x}, {y}) scale({scale * (activeHover && selectable ? 1.04 : 1)})"
	{opacity}
	role={selectable ? "button" : undefined}
	tabindex={selectable ? 0 : undefined}
	aria-label={english ? `Token ${text} at position ${pos}` : `词元 ${text} 位置 ${pos}`}
	style:cursor={selectable ? "pointer" : "default"}
	style:transition="transform 0.2s cubic-bezier(0.16, 1, 0.3, 1), opacity 0.2s ease"
	onmouseenter={handleMouseEnter}
	onmouseleave={handleMouseLeave}
	onclick={handleClick}
	onkeydown={handleClick}
>
	<!-- 主体胶囊框 -->
	<rect
		x={-width / 2}
		y={-height / 2}
		{width}
		{height}
		{rx}
		fill={fillColor}
		stroke={strokeColor}
		stroke-width={isHero ? 2.5 : activeHover ? 2 : 1.2}
		filter={isHero ? "url(#hero-glow)" : undefined}
	/>

	<!-- 交互微悬停光环 -->
	{#if activeHover && selectable}
		<rect
			x={-width / 2 - 3}
			y={-height / 2 - 3}
			width={width + 6}
			height={height + 6}
			rx={rx + 2}
			fill="none"
			stroke={strokeColor}
			stroke-width="1"
			stroke-opacity="0.4"
			stroke-dasharray="4 2"
		/>
	{/if}

	<!-- 英雄皇冠与 Query 标记 (Hero Crown) -->
	{#if isHero}
		<g transform="translate(0, {-height / 2 - 14})">
			<!-- 皇冠微图标 -->
			<path
				d="M -11 0 L -14 -9 L -5 -4 L 0 -11 L 5 -4 L 14 -9 L 11 0 Z"
				fill={PALETTE.hero}
			/>
			<circle cx="-14" cy="-10" r="1.5" fill="#fff" />
			<circle cx="0" cy="-12" r="1.8" fill="#fff" />
			<circle cx="14" cy="-10" r="1.5" fill="#fff" />
			<!-- HERO / QUERY 徽标 -->
			<text
				x="0"
				y="-16"
				text-anchor="middle"
				font-size="9"
				font-weight="900"
				letter-spacing="0.08em"
				fill={PALETTE.hero}
			>
				{english ? "HERO · QUERY" : "主角 · QUERY"}
			</text>
		</g>
	{/if}

	<!-- 词元文本 -->
	<text
		x="0"
		y={showId || showPos || tag ? 2 : 5}
		text-anchor="middle"
		font-size={isHero ? 17 : 15}
		font-weight={isHero ? 800 : 600}
		fill={isHero ? PALETTE.hero : PALETTE.ink}
		letter-spacing="0.04em"
	>
		{text}
	</text>

	<!-- 位置徽章 (Pos Tag) -->
	{#if showPos && pos !== undefined}
		<g transform="translate({-width / 2 + 18}, {-height / 2 + 10})">
			<rect
				x="-12"
				y="-6"
				width="24"
				height="12"
				rx="3"
				fill={isHero ? "rgba(251, 191, 36, 0.25)" : "rgba(192, 132, 252, 0.2)"}
			/>
			<text
				x="0"
				y="3"
				text-anchor="middle"
				font-size="8.5"
				font-family="monospace"
				font-weight="700"
				fill={isHero ? PALETTE.hero : PALETTE.pos}
			>
				p{pos}
			</text>
		</g>
	{/if}

	<!-- 可选语法词性标签 (POS Tag) -->
	{#if tag}
		<g transform="translate({width / 2 - 18}, {-height / 2 + 10})">
			<rect
				x="-12"
				y="-6"
				width="24"
				height="12"
				rx="3"
				fill="rgba(56, 189, 248, 0.15)"
			/>
			<text
				x="0"
				y="3"
				text-anchor="middle"
				font-size="8"
				font-weight="700"
				fill={PALETTE.k}
			>
				{tag}
			</text>
		</g>
	{/if}

	<!-- Token ID 底部微注 -->
	{#if showId && id !== undefined}
		<text
			x="0"
			y={height / 2 - 7}
			text-anchor="middle"
			font-size="9.5"
			font-family="monospace"
			fill={isHero ? "rgba(251, 191, 36, 0.85)" : PALETTE.muted}
		>
			ID:{id}
		</text>
	{/if}
</g>
