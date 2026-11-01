<script lang="ts">
import { fmt, PALETTE } from "../timeline";

interface Props {
	sourceX: number;
	sourceY: number;
	targetX: number;
	targetY: number;
	weight: number;
	rawScore?: number;
	color?: string;
	curvature?: number;
	progress?: number;
	isHovered?: boolean;
	isFocus?: boolean;
	showBadge?: boolean;
	badgeOffset?: number;
	label?: string;
	english?: boolean;
	onClick?: () => void;
	onHover?: (hovered: boolean) => void;
}

const {
	sourceX,
	sourceY,
	targetX,
	targetY,
	weight,
	rawScore,
	color,
	curvature = -60,
	progress = 1,
	isHovered = false,
	isFocus = false,
	showBadge = true,
	badgeOffset = 0,
	label,
	english = false,
	onClick,
	onHover,
}: Props = $props();

let internalHover = $state(false);
const activeHover = $derived(isHovered || internalHover);

const strokeColor = $derived(
	color ?? (isFocus ? PALETTE.hero : "rgba(255, 255, 255, 0.4)"),
);

// 依据权重自适应线宽与透明度
const strokeWidth = $derived(
	activeHover ? 3.5 + weight * 4 : Math.max(1.2, 1.0 + weight * 5),
);

const strokeOpacity = $derived(
	activeHover
		? 1.0
		: Math.max(0.18, Math.min(0.95, 0.25 + weight * 0.75)) * progress,
);

// 贝塞尔控制点计算 (保证无论跨度大小，弧度曲率稳定适中)
const span = $derived(Math.abs(targetX - sourceX));
const adaptiveCurvature = $derived(
	curvature * Math.max(0.6, Math.min(1.4, span / 200)),
);
const ctrlX = $derived((sourceX + targetX) / 2);
const ctrlY = $derived(Math.min(sourceY, targetY) + adaptiveCurvature);

// 二阶贝塞尔在 t=0.5 处的顶点坐标 B(0.5) = 0.25*P0 + 0.5*C + 0.25*P2
const apexX = $derived(0.25 * sourceX + 0.5 * ctrlX + 0.25 * targetX);
const apexY = $derived(0.25 * sourceY + 0.5 * ctrlY + 0.25 * targetY);

const pathD = $derived(
	`M ${sourceX} ${sourceY} Q ${ctrlX} ${ctrlY} ${targetX} ${targetY}`,
);

const displayLabel = $derived(label ?? `${fmt(weight * 100, 1)}%`);

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
	class="attention-arc"
	role={onClick ? "button" : "group"}
	tabindex={onClick ? 0 : undefined}
	aria-label={english
		? `Attention flow: weight ${displayLabel}`
		: `注意力连线：权重 ${displayLabel}`}
	style:cursor={onClick || onHover ? "pointer" : "default"}
	onmouseenter={handleMouseEnter}
	onmouseleave={handleMouseLeave}
	onclick={onClick}
>
	<!-- 宽透明命中热区 (Hit Target) -->
	<path
		d={pathD}
		fill="none"
		stroke="transparent"
		stroke-width="24"
		style:cursor="pointer"
	/>

	<!-- 背景辉光线 (当高亮或 Focus 时) -->
	{#if (activeHover || isFocus) && progress > 0}
		<path
			d={pathD}
			fill="none"
			stroke={strokeColor}
			stroke-width={strokeWidth + 4}
			stroke-opacity="0.3"
			filter="url(#hero-glow)"
			stroke-linecap="round"
		/>
	{/if}

	<!-- 核心流动曲线 -->
	{#if progress > 0}
		<path
			d={pathD}
			fill="none"
			stroke={strokeColor}
			stroke-width={strokeWidth}
			stroke-opacity={strokeOpacity}
			stroke-dasharray={weight > 0.05 ? "6 3" : "none"}
			stroke-linecap="round"
			style:transition="stroke-width 0.2s ease, stroke-opacity 0.2s ease"
		/>
	{/if}

	<!-- 目标末端汇聚端点圆环 -->
	<circle
		cx={targetX}
		cy={targetY}
		r={activeHover ? 4.5 : Math.max(2, 1.5 + weight * 3)}
		fill={strokeColor}
		fill-opacity={strokeOpacity}
	/>

	<!-- 顶点百分比徽章 (Percentage Badge) -->
	{#if showBadge && progress > 0.3}
		<g
			transform="translate({apexX}, {apexY + badgeOffset}) scale({activeHover ? 1.12 : 1})"
			style:transition="transform 0.18s cubic-bezier(0.16, 1, 0.3, 1)"
		>
			<rect
				x="-24"
				y="-10"
				width="48"
				height="20"
				rx="5"
				fill="rgba(11, 17, 24, 0.92)"
				stroke={strokeColor}
				stroke-width={isFocus || activeHover ? 1.6 : 0.9}
			/>
			<text
				x="0"
				y="3.5"
				text-anchor="middle"
				font-size="9"
				font-family="monospace"
				font-weight={isFocus || activeHover ? 700 : 500}
				fill={isFocus || activeHover ? strokeColor : PALETTE.ink}
			>
				{displayLabel}
			</text>

			<!-- 悬停微浮窗：显示 raw dot score -->
			{#if activeHover && rawScore !== undefined}
				<g transform="translate(0, -18)">
					<rect
						x="-32"
						y="-9"
						width="64"
						height="16"
						rx="3"
						fill="rgba(17, 26, 36, 0.96)"
						stroke={PALETTE.muted}
						stroke-width="0.8"
					/>
					<text
						x="0"
						y="3"
						text-anchor="middle"
						font-size="8"
						font-family="monospace"
						fill={PALETTE.ink}
					>
						score: {fmt(rawScore, 2)}
					</text>
				</g>
			{/if}
		</g>
	{/if}
</g>
