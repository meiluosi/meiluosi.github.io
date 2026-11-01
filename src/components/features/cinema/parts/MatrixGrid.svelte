<script lang="ts">
import { fmt, PALETTE } from "../timeline";

interface Props {
	matrix: number[][];
	rowLabels?: string[];
	colLabels?: string[];
	title?: string;
	x?: number;
	y?: number;
	cellWidth?: number;
	cellHeight?: number;
	digits?: number;
	color?: string;
	hoveredRow?: number | null;
	hoveredCol?: number | null;
	maskUpperTriangle?: boolean;
	scale?: number;
	opacity?: number;
	english?: boolean;
	onCellHover?: (row: number | null, col: number | null) => void;
	onRowHover?: (row: number | null) => void;
}

const {
	matrix,
	rowLabels,
	colLabels,
	title,
	x = 0,
	y = 0,
	cellWidth = 46,
	cellHeight = 24,
	digits = 2,
	color,
	hoveredRow = null,
	hoveredCol = null,
	maskUpperTriangle = false,
	scale = 1,
	opacity = 1,
	english = false,
	onCellHover,
	onRowHover,
}: Props = $props();

const numRows = $derived(matrix.length);
const numCols = $derived(matrix[0]?.length ?? 0);

let localHoveredRow = $state<number | null>(null);
let localHoveredCol = $state<number | null>(null);

const activeRow = $derived(hoveredRow ?? localHoveredRow);
const activeCol = $derived(hoveredCol ?? localHoveredCol);

const accentColor = $derived(color ?? PALETTE.ink);

const gridWidth = $derived(numCols * cellWidth);
const gridHeight = $derived(numRows * cellHeight);

const rowLabelWidth = $derived(rowLabels ? 48 : 0);
const colLabelHeight = $derived(colLabels ? 20 : 0);
const titleHeight = $derived(title ? 22 : 0);

function getSignColor(v: number): string {
	if (!Number.isFinite(v)) return PALETTE.mask;
	if (Math.abs(v) < 1e-6) return PALETTE.muted;
	return v > 0 ? "#38bdf8" : "#fb7185";
}

function handleCellEnter(r: number, c: number) {
	localHoveredRow = r;
	localHoveredCol = c;
	onCellHover?.(r, c);
	onRowHover?.(r);
}

function handleCellLeave() {
	localHoveredRow = null;
	localHoveredCol = null;
	onCellHover?.(null, null);
	onRowHover?.(null);
}
</script>

<g
	class="matrix-grid"
	transform="translate({x}, {y}) scale({scale})"
	{opacity}
	role="table"
	aria-label={title ?? (english ? "Matrix grid" : "矩阵网格")}
	style:transition="transform 0.2s cubic-bezier(0.16, 1, 0.3, 1), opacity 0.2s ease"
>
	<!-- 标题栏 -->
	{#if title}
		<text
			x={rowLabelWidth / 2}
			y={-gridHeight / 2 - colLabelHeight - 8}
			text-anchor="middle"
			font-size="12"
			font-weight="700"
			fill={accentColor}
			letter-spacing="0.04em"
		>
			{title}
		</text>
	{/if}

	<!-- 列标题 (Top column headers) -->
	{#if colLabels}
		<g transform="translate({rowLabelWidth / 2}, {-gridHeight / 2 - 8})">
			{#each colLabels as cl, c}
				{@const cx = (c - (numCols - 1) / 2) * cellWidth}
				<text
					x={cx}
					y="0"
					text-anchor="middle"
					font-size="9"
					font-family="monospace"
					font-weight={activeCol === c ? 700 : 500}
					fill={activeCol === c ? PALETTE.hero : PALETTE.muted}
				>
					{cl}
				</text>
			{/each}
		</g>
	{/if}

	<!-- 行标题 (Left row headers) -->
	{#if rowLabels}
		<g transform="translate({-gridWidth / 2 - 14}, 0)">
			{#each rowLabels as rl, r}
				{@const ry = (r - (numRows - 1) / 2) * cellHeight}
				<text
					x="0"
					y={ry + 4}
					text-anchor="end"
					font-size="10"
					font-family="monospace"
					font-weight={activeRow === r ? 700 : 500}
					fill={activeRow === r ? PALETTE.hero : PALETTE.muted}
				>
					{rl}
				</text>
			{/each}
		</g>
	{/if}

	<!-- 核心网格容器 (Center at rowLabelWidth / 2, 0) -->
	<g transform="translate({rowLabelWidth / 2}, 0)">
		<!-- 矩阵左大括号 '[' -->
		<path
			d="M {-gridWidth / 2 - 4} {-gridHeight / 2 - 4} L {-gridWidth / 2 - 10} {-gridHeight / 2 - 4} L {-gridWidth / 2 - 10} {gridHeight / 2 + 4} L {-gridWidth / 2 - 4} {gridHeight / 2 + 4}"
			fill="none"
			stroke={accentColor}
			stroke-width="1.8"
		/>

		<!-- 矩阵右大括号 ']' -->
		<path
			d="M {gridWidth / 2 + 4} {-gridHeight / 2 - 4} L {gridWidth / 2 + 10} {-gridHeight / 2 - 4} L {gridWidth / 2 + 10} {gridHeight / 2 + 4} L {gridWidth / 2 + 4} {gridHeight / 2 + 4}"
			fill="none"
			stroke={accentColor}
			stroke-width="1.8"
		/>

		<!-- 激活行的高亮光带 -->
		{#if activeRow !== null && activeRow >= 0 && activeRow < numRows}
			<rect
				x={-gridWidth / 2}
				y={(activeRow - (numRows - 1) / 2) * cellHeight - cellHeight / 2}
				width={gridWidth}
				height={cellHeight}
				rx="3"
				fill="rgba(251, 191, 36, 0.15)"
				stroke={PALETTE.hero}
				stroke-width="1"
				pointer-events="none"
			/>
		{/if}

		<!-- 激活列的高亮光带 -->
		{#if activeCol !== null && activeCol >= 0 && activeCol < numCols}
			<rect
				x={(activeCol - (numCols - 1) / 2) * cellWidth - cellWidth / 2}
				y={-gridHeight / 2}
				width={cellWidth}
				height={gridHeight}
				rx="3"
				fill="rgba(56, 189, 248, 0.12)"
				stroke={PALETTE.k}
				stroke-width="1"
				pointer-events="none"
			/>
		{/if}

		<!-- 各单元格遍历 -->
		{#each matrix as row, r}
			{#each row as val, c}
				{@const cx = (c - (numCols - 1) / 2) * cellWidth}
				{@const cy = (r - (numRows - 1) / 2) * cellHeight}
				{@const isMasked = maskUpperTriangle && c > r}
				{@const isCellActive = activeRow === r && activeCol === c}

				<g
					transform="translate({cx}, {cy})"
					role="cell"
					tabindex={0}
					style:cursor="pointer"
					onmouseenter={() => handleCellEnter(r, c)}
					onmouseleave={handleCellLeave}
				>
					<!-- 单元格底色 -->
					<rect
						x={-cellWidth / 2 + 1}
						y={-cellHeight / 2 + 1}
						width={cellWidth - 2}
						height={cellHeight - 2}
						rx="2"
						fill={isCellActive
							? "rgba(255, 255, 255, 0.16)"
							: isMasked
								? "rgba(239, 68, 68, 0.12)"
								: "rgba(17, 26, 36, 0.7)"}
						stroke={isCellActive ? PALETTE.ink : "rgba(255, 255, 255, 0.04)"}
						stroke-width="1"
					/>

					{#if isMasked}
						<!-- 掩码标记: 因果锁或 -∞ -->
						<text
							x="0"
							y="3"
							text-anchor="middle"
							font-size="9"
							font-family="monospace"
							font-weight="700"
							fill={PALETTE.mask}
						>
							-∞
						</text>
					{:else}
						<!-- 实际数值 -->
						<text
							x="0"
							y="4"
							text-anchor="middle"
							font-size="10"
							font-family="monospace"
							font-weight={isCellActive ? 700 : 500}
							fill={isCellActive ? PALETTE.ink : getSignColor(val)}
						>
							{Number.isFinite(val) ? fmt(val, digits) : "-∞"}
						</text>
					{/if}
				</g>
			{/each}
		{/each}
	</g>
</g>
