<script lang="ts">
	import { onMount } from "svelte";

	export interface GraphNode {
		id: string;
		label: string;
		type: "tag" | "category" | "post";
		url?: string;
		count?: number;
	}

	export interface GraphEdge {
		source: string;
		target: string;
	}

	interface Props {
		nodes: GraphNode[];
		edges: GraphEdge[];
		class?: string;
	}

	let { nodes, edges, class: className = "" }: Props = $props();

	let canvas: HTMLCanvasElement | undefined = $state();
	let container: HTMLDivElement | undefined = $state();
	let tooltip = $state<{
		visible: boolean;
		x: number;
		y: number;
		label: string;
		type: string;
		count?: number;
	}>({ visible: false, x: 0, y: 0, label: "", type: "" });

	// 力导向布局状态
	let positions: { x: number; y: number; vx: number; vy: number }[] = [];
	let width = $state(800);
	let height = $state(600);
	let animationId = 0;
	let isDragging = false;
	let dragNode = -1;
	let dragOffsetX = 0;
	let dragOffsetY = 0;

	// 节点颜色
	const typeColors: Record<string, string> = {
		category: "#f59e0b", // amber
		tag: "#3b82f6", // blue
		post: "#10b981", // emerald
	};

	const typeSizes: Record<string, number> = {
		category: 14,
		tag: 10,
		post: 6,
	};

	function getNodeRadius(node: GraphNode): number {
		const base = typeSizes[node.type] || 6;
		if (node.type === "tag" && node.count) {
			return Math.min(base + Math.log2(node.count + 1) * 3, 20);
		}
		return base;
	}

	function initLayout() {
		const cx = width / 2;
		const cy = height / 2;

		positions = nodes.map((node, i) => {
			// 按类型分组初始位置
			let angle: number;
			let radius: number;
			if (node.type === "category") {
				angle = ((nodes.filter((n) => n.type === "category").indexOf(node) / Math.max(1, nodes.filter((n) => n.type === "category").length)) * Math.PI * 2);
				radius = Math.min(width, height) * 0.15;
			} else if (node.type === "tag") {
				angle = ((nodes.filter((n) => n.type === "tag").indexOf(node) / Math.max(1, nodes.filter((n) => n.type === "tag").length)) * Math.PI * 2);
				radius = Math.min(width, height) * 0.3;
			} else {
				angle = Math.random() * Math.PI * 2;
				radius = Math.random() * Math.min(width, height) * 0.35;
			}
			return {
				x: cx + Math.cos(angle) * radius + (Math.random() - 0.5) * 20,
				y: cy + Math.sin(angle) * radius + (Math.random() - 0.5) * 20,
				vx: 0,
				vy: 0,
			};
		});
	}

	function simulate() {
		const alpha = 0.5;
		const centerX = width / 2;
		const centerY = height / 2;

		for (let i = 0; i < nodes.length; i++) {
			const node = nodes[i];
			const pos = positions[i];

			// 中心引力
			pos.vx += (centerX - pos.x) * 0.001;
			pos.vy += (centerY - pos.y) * 0.001;

			// 节点间斥力
			for (let j = i + 1; j < nodes.length; j++) {
				const other = positions[j];
				const dx = pos.x - other.x;
				const dy = pos.y - other.y;
				const dist = Math.max(1, Math.sqrt(dx * dx + dy * dy));
				const force = 500 / (dist * dist);
				const fx = (dx / dist) * force;
				const fy = (dy / dist) * force;
				pos.vx += fx;
				pos.vy += fy;
				other.vx -= fx;
				other.vy -= fy;
			}

			// 边引力
			for (const edge of edges) {
				let sourceIdx = -1;
				let targetIdx = -1;
				if (edge.source === node.id) {
					sourceIdx = i;
					targetIdx = nodes.findIndex((n) => n.id === edge.target);
				} else if (edge.target === node.id) {
					sourceIdx = nodes.findIndex((n) => n.id === edge.source);
					targetIdx = i;
				}
				if (sourceIdx >= 0 && targetIdx >= 0) {
					const other = positions[targetIdx];
					const dx = pos.x - other.x;
					const dy = pos.y - other.y;
					const dist = Math.max(1, Math.sqrt(dx * dx + dy * dy));
					const force = (dist - 80) * 0.01;
					const fx = (dx / dist) * force;
					const fy = (dy / dist) * force;
					pos.vx -= fx;
					pos.vy -= fy;
				}
			}
		}

		// 更新位置
		for (let i = 0; i < nodes.length; i++) {
			if (isDragging && dragNode === i) continue;
			const pos = positions[i];
			pos.vx *= 0.9; // 阻尼
			pos.vy *= 0.9;
			pos.x += pos.vx * alpha;
			pos.y += pos.vy * alpha;
			// 边界限制
			pos.x = Math.max(20, Math.min(width - 20, pos.x));
			pos.y = Math.max(20, Math.min(height - 20, pos.y));
		}
	}

	function render(ctx: CanvasRenderingContext2D) {
		ctx.clearRect(0, 0, width, height);

		// 绘制边
		ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--card-border").trim() || "rgba(156,163,175,0.2)";
		ctx.lineWidth = 0.5;
		for (const edge of edges) {
			const si = nodes.findIndex((n) => n.id === edge.source);
			const ti = nodes.findIndex((n) => n.id === edge.target);
			if (si < 0 || ti < 0) continue;
			ctx.beginPath();
			ctx.moveTo(positions[si].x, positions[si].y);
			ctx.lineTo(positions[ti].x, positions[ti].y);
			ctx.stroke();
		}

		// 绘制节点
		for (let i = 0; i < nodes.length; i++) {
			const node = nodes[i];
			const pos = positions[i];
			const r = getNodeRadius(node);
			const color = typeColors[node.type] || "#6b7280";

			// 光晕
			ctx.beginPath();
			ctx.arc(pos.x, pos.y, r + 4, 0, Math.PI * 2);
			ctx.fillStyle = color + "1a";
			ctx.fill();

			// 主体
			ctx.beginPath();
			ctx.arc(pos.x, pos.y, r, 0, Math.PI * 2);
			ctx.fillStyle = color;
			ctx.fill();
			ctx.strokeStyle = "#fff";
			ctx.lineWidth = 1.5;
			ctx.stroke();

			// 标签（仅 tag 和 category 显示）
			if (node.type !== "post" || r > 8) {
				ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--text-color").trim() || "#374151";
				ctx.font = `${Math.max(10, r * 0.9)}px system-ui, sans-serif`;
				ctx.textAlign = "center";
				ctx.textBaseline = "middle";
				const label = node.type === "post"
					? (node.label.length > 8 ? node.label.slice(0, 7) + "…" : node.label)
					: node.label;
				ctx.fillText(label, pos.x, pos.y + r + 12);
			}
		}
	}

	function animate() {
		const ctx = canvas?.getContext("2d");
		if (!ctx) return;
		simulate();
		render(ctx);
		animationId = requestAnimationFrame(animate);
	}

	function handleResize() {
		if (!container) return;
		width = container.clientWidth;
		height = container.clientHeight;
		if (canvas) {
			canvas.width = width * devicePixelRatio;
			canvas.height = height * devicePixelRatio;
			canvas.style.width = `${width}px`;
			canvas.style.height = `${height}px`;
			const ctx = canvas.getContext("2d");
			if (ctx) ctx.scale(devicePixelRatio, devicePixelRatio);
		}
	}

	function getNodeAt(x: number, y: number): number {
		for (let i = nodes.length - 1; i >= 0; i--) {
			const pos = positions[i];
			const r = getNodeRadius(nodes[i]) + 6;
			const dx = pos.x - x;
			const dy = pos.y - y;
			if (dx * dx + dy * dy < r * r) return i;
		}
		return -1;
	}

	function handleMouseMove(e: MouseEvent) {
		if (!canvas) return;
		const rect = canvas.getBoundingClientRect();
		const mx = e.clientX - rect.left;
		const my = e.clientY - rect.top;

		if (isDragging && dragNode >= 0) {
			positions[dragNode].x = mx - dragOffsetX;
			positions[dragNode].y = my - dragOffsetY;
			positions[dragNode].vx = 0;
			positions[dragNode].vy = 0;
			return;
		}

		const idx = getNodeAt(mx, my);
		if (idx >= 0) {
			const node = nodes[idx];
			tooltip = {
				visible: true,
				x: e.clientX + 12,
				y: e.clientY - 8,
				label: node.label,
				type: node.type,
				count: node.count,
			};
			canvas.style.cursor = "pointer";
		} else {
			tooltip = { ...tooltip, visible: false };
			canvas.style.cursor = "grab";
		}
	}

	function handleMouseDown(e: MouseEvent) {
		if (!canvas) return;
		const rect = canvas.getBoundingClientRect();
		const mx = e.clientX - rect.left;
		const my = e.clientY - rect.top;
		const idx = getNodeAt(mx, my);
		if (idx >= 0) {
			isDragging = true;
			dragNode = idx;
			dragOffsetX = mx - positions[idx].x;
			dragOffsetY = my - positions[idx].y;
		}
	}

	function handleMouseUp(e: MouseEvent) {
		if (isDragging && dragNode >= 0) {
			// 如果没怎么拖动，视为点击
			const node = nodes[dragNode];
			if (node.url) {
				window.location.href = node.url;
			}
		}
		isDragging = false;
		dragNode = -1;
	}

	function handleMouseLeave() {
		tooltip = { ...tooltip, visible: false };
	}

	onMount(() => {
		initLayout();
		handleResize();
		animate();

		const observer = new ResizeObserver(handleResize);
		if (container) observer.observe(container);

		window.addEventListener("resize", handleResize);
		window.addEventListener("mouseup", handleMouseUp);

		return () => {
			cancelAnimationFrame(animationId);
			observer.disconnect();
			window.removeEventListener("resize", handleResize);
			window.removeEventListener("mouseup", handleMouseUp);
		};
	});
</script>

<div class="knowledge-graph {className}" bind:this={container}>
	<canvas
		bind:this={canvas}
		onmousemove={handleMouseMove}
		onmousedown={handleMouseDown}
		onmouseleave={handleMouseLeave}
		role="img"
		aria-label="知识图谱可视化"
	></canvas>

	<!-- 图例 -->
	<div class="graph-legend">
		<div class="legend-item">
			<span class="legend-dot" style="background: #f59e0b;"></span>
			<span>分类</span>
		</div>
		<div class="legend-item">
			<span class="legend-dot" style="background: #3b82f6;"></span>
			<span>标签</span>
		</div>
		<div class="legend-item">
			<span class="legend-dot" style="background: #10b981;"></span>
			<span>文章</span>
		</div>
		<div class="legend-hint">🖱 拖拽节点 · 点击跳转</div>
	</div>

	<!-- 悬浮提示 -->
	{#if tooltip.visible}
		<div
			class="graph-tooltip"
			style="left: {tooltip.x}px; top: {tooltip.y}px;"
		>
			<div class="tooltip-label">{tooltip.label}</div>
			<div class="tooltip-type">
				{tooltip.type === "tag" ? "标签" : tooltip.type === "category" ? "分类" : "文章"}
				{#if tooltip.count}
					<span class="tooltip-count">{tooltip.count} 篇</span>
				{/if}
			</div>
		</div>
	{/if}
</div>

<style>
	.knowledge-graph {
		position: relative;
		width: 100%;
		height: 600px;
		border-radius: 12px;
		overflow: hidden;
		background: var(--card-bg, #fff);
		border: 1px solid var(--card-border, #e5e7eb);
	}

	:global(.dark) .knowledge-graph {
		background: var(--card-bg, #1f2937);
		border-color: var(--card-border, #374151);
	}

	canvas {
		display: block;
		width: 100%;
		height: 100%;
		cursor: grab;
	}

	canvas:active {
		cursor: grabbing;
	}

	.graph-legend {
		position: absolute;
		top: 12px;
		left: 12px;
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 12px;
		padding: 8px 14px;
		border-radius: 8px;
		background: rgba(255, 255, 255, 0.85);
		backdrop-filter: blur(8px);
		border: 1px solid var(--card-border, #e5e7eb);
		font-size: 12px;
		pointer-events: none;
	}

	:global(.dark) .graph-legend {
		background: rgba(31, 41, 55, 0.85);
	}

	.legend-item {
		display: flex;
		align-items: center;
		gap: 4px;
	}

	.legend-dot {
		width: 10px;
		height: 10px;
		border-radius: 50%;
	}

	.legend-hint {
		color: var(--text-muted, #9ca3af);
		font-size: 11px;
		margin-left: 4px;
	}

	.graph-tooltip {
		position: fixed;
		pointer-events: none;
		z-index: 1000;
		padding: 6px 12px;
		border-radius: 8px;
		background: rgba(0, 0, 0, 0.85);
		color: #fff;
		font-size: 13px;
		line-height: 1.4;
		white-space: nowrap;
	}

	.tooltip-type {
		font-size: 11px;
		opacity: 0.7;
		display: flex;
		align-items: center;
		gap: 6px;
	}

	.tooltip-count {
		opacity: 0.8;
	}

	@media (max-width: 640px) {
		.knowledge-graph {
			height: 450px;
		}

		.graph-legend {
			top: 8px;
			left: 8px;
			gap: 8px;
			padding: 6px 10px;
			font-size: 11px;
		}
	}
</style>