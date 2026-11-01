<script lang="ts">
import { fmt, PALETTE } from "../timeline";

interface Props {
	values: number[];
	label?: string;
	dimLabels?: string[];
	x?: number;
	y?: number;
	digits?: number;
	color?: string;
	isHovered?: boolean;
	highlightIdx?: number | null;
	orientation?: "horizontal" | "vertical";
	showBrackets?: boolean;
	scale?: number;
	opacity?: number;
	english?: boolean;
	onHover?: (hovered: boolean) => void;
	onClick?: () => void;
}

const {
	values,
	label,
	dimLabels,
	x = 0,
	y = 0,
	digits = 2,
	color,
	isHovered = false,
	highlightIdx = null,
	orientation = "horizontal",
	showBrackets = true,
	scale = 1,
	opacity = 1,
	english = false,
	onHover,
	onClick,
}: Props = $props();

let internalHover = $state(false);
const activeHover = $derived(isHovered || internalHover);

const accentColor = $derived(
	color ?? (activeHover ? PALETTE.hero : PALETTE.muted),
);

const cellWidth = 44;
const cellHeight = 22;
const isHoriz = $derived(orientation === "horizontal");

const totalWidth = $derived(
	isHoriz
		? values.length * cellWidth + (label ? 60 : 20)
		: cellWidth + (label ? 60 : 30),
);

const totalHeight = $derived(
	isHoriz ? 36 : values.length * cellHeight + (label ? 30 : 20),
);

function getSignColor(v: number): string {
	if (Math.abs(v) < 1e-6) return PALETTE.muted;
	return v > 0 ? "#38bdf8" : "#fb7185";
}

function handleMouseEnter() {
	internalHover = true;
	onHover?.(true);
}

function handleMouseLeave() {
	internalHover = false;
	onHover?.(false);
}
</script>

<g
	class="vector-chip"
	transform="translate({x}, {y}) scale({scale * (activeHover ? 1.03 : 1)})"
	{opacity}
	role={onClick ? "button" : "group"}
	tabindex={onClick ? 0 : undefined}
	aria-label={english ? `Vector ${label ?? ""}` : `向量 ${label ?? ""}`}
	style:cursor={onClick ? "pointer" : "default"}
	style:transition="transform 0.18s cubic-bezier(0.16, 1, 0.3, 1)"
	onmouseenter={handleMouseEnter}
	onmouseleave={handleMouseLeave}
	onclick={onClick}
>
	<!-- 背景底板 -->
	<rect
		x={-totalWidth / 2}
		y={-totalHeight / 2}
		width={totalWidth}
		height={totalHeight}
		rx="6"
		fill="rgba(17, 26, 36, 0.88)"
		stroke={accentColor}
		stroke-width={activeHover ? 1.8 : 1}
	/>

	<!-- 标签前缀 -->
	{#if label}
		<text
			x={isHoriz ? -totalWidth / 2 + 30 : 0}
			y={isHoriz ? 4 : -totalHeight / 2 + 15}
			text-anchor="middle"
			font-size="11"
			font-weight="700"
			fill={accentColor}
			letter-spacing="0.04em"
		>
			{label}
		</text>
	{/if}

	<!-- 向量元素容器 -->
	<g
		transform={isHoriz
			? `translate(${label ? 18 : 0}, 0)`
			: `translate(0, ${label ? 12 : 0})`}
	>
		<!-- 左括弧 -->
		{#if showBrackets}
			<text
				x={isHoriz ? -(values.length * cellWidth) / 2 - 8 : -cellWidth / 2 - 6}
				y={isHoriz ? 4 : 0}
				text-anchor="middle"
				font-size="16"
				font-family="monospace"
				fill={PALETTE.muted}
			>
				[
			</text>
		{/if}

		<!-- 数值单元格 -->
		{#each values as val, i}
			{@const posX = isHoriz ? (i - (values.length - 1) / 2) * cellWidth : 0}
			{@const posY = isHoriz ? 0 : (i - (values.length - 1) / 2) * cellHeight}
			{@const isHigh = highlightIdx === i}

			<g transform="translate({posX}, {posY})">
				{#if isHigh}
					<rect
						x={-cellWidth / 2 + 2}
						y={-cellHeight / 2 + 2}
						width={cellWidth - 4}
						height={cellHeight - 4}
						rx="3"
						fill="rgba(251, 191, 36, 0.2)"
						stroke={PALETTE.hero}
						stroke-width="1"
					/>
				{/if}

				<!-- 维度注记 (微注) -->
				{#if dimLabels && dimLabels[i]}
					<text
						x="0"
						y={-cellHeight / 2 - 2}
						text-anchor="middle"
						font-size="8"
						font-family="monospace"
						fill={PALETTE.muted}
					>
						{dimLabels[i]}
					</text>
				{/if}

				<text
					x="0"
					y="4"
					text-anchor="middle"
					font-size="11"
					font-family="monospace"
					font-weight={isHigh ? 700 : 500}
					fill={isHigh ? PALETTE.hero : getSignColor(val)}
				>
					{fmt(val, digits)}
				</text>
			</g>
		{/each}

		<!-- 右括弧 -->
		{#if showBrackets}
			<text
				x={isHoriz ? (values.length * cellWidth) / 2 + 8 : cellWidth / 2 + 6}
				y={isHoriz ? 4 : 0}
				text-anchor="middle"
				font-size="16"
				font-family="monospace"
				fill={PALETTE.muted}
			>
				]
			</text>
		{/if}
	</g>
</g>
