<script lang="ts">
import { onMount } from "svelte";
import {
	type GraphEdge,
	type GraphNode,
	MATURITY_DEFINITIONS,
	type MaturityLevel,
} from "@/utils/digital-garden";
import {
	playLatchClick,
	playMechanicalTick,
	playShutterTick,
} from "@/utils/haptics";

interface Props {
	initialNodes: GraphNode[];
	initialEdges: GraphEdge[];
}

let { initialNodes, initialEdges }: Props = $props();

let english = $state(false);
let canvasEl = $state<HTMLCanvasElement | null>(null);
let activeFilter = $state<MaturityLevel | "all">("all");
let searchQuery = $state("");
let selectedNode = $state<GraphNode | null>(null);
let hoveredNode = $state<GraphNode | null>(null);

// Transform / Pan / Zoom state
let transform = $state({ x: 0, y: 0, k: 1 });
let isDraggingCanvas = false;
let dragStart = { x: 0, y: 0 };
let draggedNode: GraphNode | null = null;

// Dynamic simulation nodes
let simNodes = $state<GraphNode[]>([]);
let simEdges = $state<GraphEdge[]>([]);

const activeMaturityCount = $derived(() => {
	const counts: Record<string, number> = { all: simNodes.length };
	for (const n of simNodes) {
		counts[n.maturity] = (counts[n.maturity] || 0) + 1;
	}
	return counts;
});

// Connected neighbors for the selected node
const connectedNeighbors = $derived(() => {
	if (!selectedNode) return [];
	const id = selectedNode.id;
	const neighborIds = new Set<string>();
	for (const e of simEdges) {
		if (e.source === id) neighborIds.add(e.target);
		if (e.target === id) neighborIds.add(e.source);
	}
	return simNodes.filter((n) => neighborIds.has(n.id));
});

function initSimulation() {
	if (!canvasEl) return;
	const rect = canvasEl.getBoundingClientRect();
	const width = rect.width || 800;
	const height = rect.height || 600;

	transform.x = width / 2;
	transform.y = height / 2;

	// Copy initial data
	simNodes = initialNodes.map((n, i) => {
		const angle = (i / initialNodes.length) * 2 * Math.PI;
		const radius = 120 + (i % 4) * 55;
		return {
			...n,
			x: Math.cos(angle) * radius,
			y: Math.sin(angle) * radius,
			vx: (Math.random() - 0.5) * 2,
			vy: (Math.random() - 0.5) * 2,
		};
	});
	simEdges = [...initialEdges];
}

let animFrameId: number;
function tickSimulation() {
	if (!canvasEl) return;
	const ctx = canvasEl.getContext("2d");
	if (!ctx) return;

	const rect = canvasEl.getBoundingClientRect();
	const width = canvasEl.width;
	const height = canvasEl.height;

	// Spring forces & Repulsion
	const damping = 0.88;
	const repulsion = 1800;
	const springLen = 70;
	const springK = 0.04;

	// 1. Repulsion between all pairs
	for (let i = 0; i < simNodes.length; i++) {
		const n1 = simNodes[i];
		for (let j = i + 1; j < simNodes.length; j++) {
			const n2 = simNodes[j];
			const dx = (n2.x || 0) - (n1.x || 0);
			const dy = (n2.y || 0) - (n1.y || 0);
			const distSq = dx * dx + dy * dy + 100;
			const dist = Math.sqrt(distSq);
			const force = repulsion / distSq;
			const fx = (dx / dist) * force;
			const fy = (dy / dist) * force;

			n1.vx = (n1.vx || 0) - fx;
			n1.vy = (n1.vy || 0) - fy;
			n2.vx = (n2.vx || 0) + fx;
			n2.vy = (n2.vy || 0) + fy;
		}

		// Centering gravity
		const cDist = Math.hypot(n1.x || 0, n1.y || 0);
		n1.vx = (n1.vx || 0) - (n1.x || 0) * 0.0008;
		n1.vy = (n1.vy || 0) - (n1.y || 0) * 0.0008;
	}

	// 2. Spring attraction along edges
	const nodeMap = new Map(simNodes.map((n) => [n.id, n]));
	for (const edge of simEdges) {
		const s = nodeMap.get(edge.source);
		const t = nodeMap.get(edge.target);
		if (!s || !t) continue;
		const dx = (t.x || 0) - (s.x || 0);
		const dy = (t.y || 0) - (s.y || 0);
		const dist = Math.hypot(dx, dy) || 1;
		const delta = dist - springLen;
		const force = delta * springK * edge.weight;
		const fx = (dx / dist) * force;
		const fy = (dy / dist) * force;

		s.vx = (s.vx || 0) + fx;
		s.vy = (s.vy || 0) + fy;
		t.vx = (t.vx || 0) - fx;
		t.vy = (t.vy || 0) - fy;
	}

	// 3. Update positions
	for (const n of simNodes) {
		if (n === draggedNode) continue;
		n.vx = (n.vx || 0) * damping;
		n.vy = (n.vy || 0) * damping;
		n.x = (n.x || 0) + n.vx;
		n.y = (n.y || 0) + n.vy;
	}

	// 4. Render Canvas
	ctx.clearRect(0, 0, width, height);
	ctx.save();
	ctx.translate(transform.x, transform.y);
	ctx.scale(transform.k, transform.k);

	// Draw edges
	const qLower = searchQuery.trim().toLowerCase();
	for (const e of simEdges) {
		const s = nodeMap.get(e.source);
		const t = nodeMap.get(e.target);
		if (!s || !t) continue;

		const isConnectedToHover =
			hoveredNode && (s.id === hoveredNode.id || t.id === hoveredNode.id);
		const isConnectedToSelected =
			selectedNode && (s.id === selectedNode.id || t.id === selectedNode.id);

		ctx.beginPath();
		ctx.moveTo(s.x || 0, s.y || 0);
		ctx.lineTo(t.x || 0, t.y || 0);

		if (isConnectedToHover || isConnectedToSelected) {
			ctx.strokeStyle = "rgba(92, 122, 107, 0.8)";
			ctx.lineWidth = 2 / transform.k;
		} else {
			ctx.strokeStyle = "rgba(160, 160, 160, 0.2)";
			ctx.lineWidth = 1 / transform.k;
		}
		ctx.stroke();
	}

	// Draw nodes
	for (const n of simNodes) {
		const isHovered = hoveredNode?.id === n.id;
		const isSelected = selectedNode?.id === n.id;
		const matchesSearch =
			qLower.length === 0 ||
			n.label.toLowerCase().includes(qLower) ||
			Boolean(n.tags?.some((t) => t.toLowerCase().includes(qLower)));
		const matchesFilter = activeFilter === "all" || n.maturity === activeFilter;

		const alpha = matchesFilter && matchesSearch ? 1 : 0.18;
		const meta =
			MATURITY_DEFINITIONS[n.maturity] || MATURITY_DEFINITIONS.budding;
		const radius =
			(n.kind === "hub" ? 14 : n.kind === "lab" ? 11 : 8) *
			(isHovered ? 1.3 : 1);

		ctx.save();
		ctx.globalAlpha = alpha;

		// Glow or ring
		if (isHovered || isSelected) {
			ctx.beginPath();
			ctx.arc(n.x || 0, n.y || 0, radius + 5, 0, Math.PI * 2);
			ctx.fillStyle = "rgba(92, 122, 107, 0.25)";
			ctx.fill();
		}

		// Node circle
		ctx.beginPath();
		ctx.arc(n.x || 0, n.y || 0, radius, 0, Math.PI * 2);
		ctx.fillStyle = meta.color;
		ctx.fill();
		ctx.strokeStyle = "#ffffff";
		ctx.lineWidth = 1.5 / transform.k;
		ctx.stroke();

		// Label text
		if (transform.k > 0.6 || isHovered || isSelected || n.kind === "hub") {
			ctx.font = `${n.kind === "hub" ? "bold 13px" : "11px"} -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`;
			ctx.fillStyle =
				isHovered || isSelected ? "#1b2a26" : "rgba(40, 50, 45, 0.85)";
			ctx.textAlign = "center";
			const textY = (n.y || 0) + radius + 14;
			const displayLabel = english && n.labelEn ? n.labelEn : n.label;
			ctx.fillText(displayLabel, n.x || 0, textY);
		}

		ctx.restore();
	}

	ctx.restore();
	animFrameId = requestAnimationFrame(tickSimulation);
}

function getNodeAt(x: number, y: number): GraphNode | null {
	// Invert transform
	const wx = (x - transform.x) / transform.k;
	const wy = (y - transform.y) / transform.k;

	for (let i = simNodes.length - 1; i >= 0; i--) {
		const n = simNodes[i];
		const r = n.kind === "hub" ? 16 : 12;
		const dist = Math.hypot((n.x || 0) - wx, (n.y || 0) - wy);
		if (dist <= r) return n;
	}
	return null;
}

function handleMouseDown(e: MouseEvent) {
	if (!canvasEl) return;
	const rect = canvasEl.getBoundingClientRect();
	const mx = e.clientX - rect.left;
	const my = e.clientY - rect.top;

	const hit = getNodeAt(mx, my);
	if (hit) {
		draggedNode = hit;
		selectedNode = hit;
		playLatchClick();
	} else {
		isDraggingCanvas = true;
		dragStart = { x: e.clientX - transform.x, y: e.clientY - transform.y };
	}
}

function handleMouseMove(e: MouseEvent) {
	if (!canvasEl) return;
	const rect = canvasEl.getBoundingClientRect();
	const mx = e.clientX - rect.left;
	const my = e.clientY - rect.top;

	if (draggedNode) {
		draggedNode.x = (mx - transform.x) / transform.k;
		draggedNode.y = (my - transform.y) / transform.k;
		draggedNode.vx = 0;
		draggedNode.vy = 0;
		return;
	}

	if (isDraggingCanvas) {
		transform.x = e.clientX - dragStart.x;
		transform.y = e.clientY - dragStart.y;
		return;
	}

	// Check hover
	const hit = getNodeAt(mx, my);
	if (hit !== hoveredNode) {
		hoveredNode = hit;
		if (hit) {
			playMechanicalTick(0.85);
		}
	}
}

function handleMouseUp() {
	draggedNode = null;
	isDraggingCanvas = false;
}

function handleWheel(e: WheelEvent) {
	e.preventDefault();
	if (!canvasEl) return;
	const rect = canvasEl.getBoundingClientRect();
	const mx = e.clientX - rect.left;
	const my = e.clientY - rect.top;

	const zoomFactor = e.deltaY < 0 ? 1.08 : 0.92;
	const newK = Math.max(0.4, Math.min(3.5, transform.k * zoomFactor));

	transform.x = mx - ((mx - transform.x) / transform.k) * newK;
	transform.y = my - ((my - transform.y) / transform.k) * newK;
	transform.k = newK;
}

function resetView() {
	if (!canvasEl) return;
	const rect = canvasEl.getBoundingClientRect();
	transform.x = rect.width / 2;
	transform.y = rect.height / 2;
	transform.k = 1.0;
	playShutterTick();
}

function setFilter(filter: MaturityLevel | "all") {
	activeFilter = filter;
	playLatchClick();
}

function focusNode(node: GraphNode) {
	selectedNode = node;
	transform.x = (canvasEl?.width || 800) / 2 - (node.x || 0) * transform.k;
	transform.y = (canvasEl?.height || 600) / 2 - (node.y || 0) * transform.k;
	playLatchClick();
}

function closeDrawer() {
	selectedNode = null;
	playLatchClick();
}

onMount(() => {
	const sync = () => {
		english = document.documentElement.lang.startsWith("en");
	};
	sync();
	document.addEventListener("site-locale-change", sync);

	const handleResize = () => {
		if (canvasEl) {
			const rect = canvasEl.getBoundingClientRect();
			canvasEl.width = rect.width * window.devicePixelRatio;
			canvasEl.height = rect.height * window.devicePixelRatio;
			const ctx = canvasEl.getContext("2d");
			if (ctx) ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
		}
	};

	handleResize();
	initSimulation();
	animFrameId = requestAnimationFrame(tickSimulation);
	window.addEventListener("resize", handleResize);

	return () => {
		cancelAnimationFrame(animFrameId);
		window.removeEventListener("resize", handleResize);
		document.removeEventListener("site-locale-change", sync);
	};
});
</script>

<div class="knowledge-cosmos-wrapper">
	<!-- Top Controls Toolbar -->
	<div class="cosmos-toolbar">
		<div class="filter-strip">
			<button
				type="button"
				class="filter-pill"
				class:active={activeFilter === "all"}
				onclick={() => setFilter("all")}
			>
				{english ? "All Nodes" : "全部节点"}
				<span class="count">{simNodes.length}</span>
			</button>
			{#each Object.entries(MATURITY_DEFINITIONS) as [lvl, meta]}
				<button
					type="button"
					class="filter-pill"
					class:active={activeFilter === lvl}
					style="--pill-color: {meta.color}"
					onclick={() => setFilter(lvl as MaturityLevel)}
				>
					<span class="icon">{meta.icon}</span>
					{english ? meta.labelEn : meta.labelZh}
				</button>
			{/each}
		</div>

		<div class="search-and-reset">
			<input
				type="search"
				class="cosmos-search-input"
				placeholder={english ? "Search concept or note…" : "搜索知识概念或笔记…"}
				bind:value={searchQuery}
			/>
			<button type="button" class="btn-icon" onclick={resetView} title={english ? "Reset View" : "重置视图"}>
				⟲
			</button>
		</div>
	</div>

	<!-- Canvas Graph Arena -->
	<div class="canvas-container">
		<canvas
			bind:this={canvasEl}
			onmousedown={handleMouseDown}
			onmousemove={handleMouseMove}
			onmouseup={handleMouseUp}
			onwheel={handleWheel}
		></canvas>

		<!-- Watermark / Hint -->
		<div class="canvas-hint">
			<span>{english ? "Drag to pan · Scroll to zoom · Click node to inspect" : "按住拖拽漫游 · 滚轮缩放 · 点击节点展开卡片"}</span>
		</div>

		<!-- Andy Matuschak Sliding Drawer -->
		{#if selectedNode}
			<aside class="sliding-pane-drawer">
				<div class="drawer-header">
					<div class="drawer-badge" style="background: {MATURITY_DEFINITIONS[selectedNode.maturity]?.badgeBg || 'transparent'}; color: {MATURITY_DEFINITIONS[selectedNode.maturity]?.color || 'inherit'}">
						<span>{MATURITY_DEFINITIONS[selectedNode.maturity]?.icon}</span>
						<strong>{english ? MATURITY_DEFINITIONS[selectedNode.maturity]?.labelEn : MATURITY_DEFINITIONS[selectedNode.maturity]?.labelZh}</strong>
					</div>
					<button type="button" class="btn-close" onclick={closeDrawer} aria-label="Close">✕</button>
				</div>

				<h3 class="drawer-title">{english && selectedNode.labelEn ? selectedNode.labelEn : selectedNode.label}</h3>

				<div class="drawer-meta">
					<span class="meta-tag">{selectedNode.category}</span>
					<span class="meta-tag">{selectedNode.kind.toUpperCase()}</span>
				</div>

				{#if selectedNode.excerpt}
					<p class="drawer-excerpt">{selectedNode.excerpt}</p>
				{/if}

				{#if selectedNode.tags && selectedNode.tags.length > 0}
					<div class="drawer-tags">
						{#each selectedNode.tags as t}
							<span class="tag-pill">{t}</span>
						{/each}
					</div>
				{/if}

				<!-- Connected Nodes -->
				<div class="neighbors-section">
					<h4>{english ? "Connected Thoughts & Experiments:" : "关联思想与实验:"}</h4>
					<div class="neighbor-list">
						{#each connectedNeighbors() as nbr}
							<button type="button" class="neighbor-pill" onclick={() => focusNode(nbr)}>
								<span>{MATURITY_DEFINITIONS[nbr.maturity]?.icon}</span>
								<span>{english && nbr.labelEn ? nbr.labelEn : nbr.label}</span>
							</button>
						{/each}
					</div>
				</div>

				<div class="drawer-actions">
					<a href={selectedNode.href} class="btn-action-primary">
						{english ? "Open Note / Experiment ↗" : "打开完整笔记 / 实验 ↗"}
					</a>
				</div>
			</aside>
		{/if}
	</div>
</div>

<style>
	.knowledge-cosmos-wrapper {
		display: flex;
		flex-direction: column;
		gap: 14px;
		background: var(--brand-paper);
		border: 1px solid var(--brand-line);
		border-radius: 14px;
		overflow: hidden;
		box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
	}

	.cosmos-toolbar {
		display: flex;
		justify-content: space-between;
		align-items: center;
		flex-wrap: wrap;
		gap: 12px;
		padding: 16px 20px;
		border-bottom: 1px solid var(--brand-line);
		background: var(--brand-surface);
	}

	.filter-strip {
		display: flex;
		align-items: center;
		flex-wrap: wrap;
		gap: 6px;
	}

	.filter-pill {
		display: inline-flex;
		align-items: center;
		gap: 5px;
		padding: 6px 12px;
		border-radius: 999px;
		border: 1px solid var(--brand-line);
		background: transparent;
		color: var(--brand-body);
		font-size: 12px;
		cursor: pointer;
		transition: all 0.15s ease;
	}

	.filter-pill:hover,
	.filter-pill.active {
		border-color: var(--pill-color, var(--brand-accent));
		color: var(--pill-color, var(--brand-accent));
		background: rgba(92, 122, 107, 0.08);
	}

	.filter-pill.active {
		font-weight: 600;
	}

	.count {
		font-size: 10px;
		opacity: 0.7;
	}

	.search-and-reset {
		display: flex;
		align-items: center;
		gap: 8px;
	}

	.cosmos-search-input {
		padding: 6px 12px;
		border-radius: 8px;
		border: 1px solid var(--brand-line);
		background: var(--brand-paper);
		color: var(--brand-ink);
		font-size: 12px;
		outline: none;
		min-width: 180px;
	}

	.cosmos-search-input:focus {
		border-color: var(--brand-accent);
	}

	.btn-icon {
		padding: 6px 10px;
		border-radius: 8px;
		border: 1px solid var(--brand-line);
		background: transparent;
		color: var(--brand-body);
		cursor: pointer;
		font-size: 13px;
	}

	.btn-icon:hover {
		border-color: var(--brand-accent);
		color: var(--brand-accent);
	}

	.canvas-container {
		position: relative;
		width: 100%;
		height: 640px;
		background: radial-gradient(circle at center, rgba(92, 122, 107, 0.03) 0%, transparent 70%);
	}

	canvas {
		width: 100%;
		height: 100%;
		display: block;
		cursor: grab;
	}

	canvas:active {
		cursor: grabbing;
	}

	.canvas-hint {
		position: absolute;
		bottom: 12px;
		left: 16px;
		font-size: 11px;
		color: var(--brand-muted);
		pointer-events: none;
		background: rgba(255, 255, 255, 0.7);
		backdrop-filter: blur(4px);
		padding: 4px 8px;
		border-radius: 6px;
	}

	/* Andy Matuschak Sliding Drawer */
	.sliding-pane-drawer {
		position: absolute;
		top: 14px;
		right: 14px;
		bottom: 14px;
		width: 340px;
		max-width: calc(100% - 28px);
		background: var(--brand-paper);
		border: 1px solid var(--brand-line);
		border-radius: 12px;
		box-shadow: -4px 0 24px rgba(0, 0, 0, 0.08);
		padding: 20px;
		display: flex;
		flex-direction: column;
		gap: 14px;
		overflow-y: auto;
		z-index: 10;
		animation: slideIn 0.2s cubic-bezier(0.16, 1, 0.3, 1);
	}

	@keyframes slideIn {
		from {
			transform: translateX(30px);
			opacity: 0;
		}
		to {
			transform: translateX(0);
			opacity: 1;
		}
	}

	.drawer-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
	}

	.drawer-badge {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		padding: 4px 10px;
		border-radius: 6px;
		font-size: 12px;
	}

	.btn-close {
		border: none;
		background: transparent;
		color: var(--brand-muted);
		cursor: pointer;
		font-size: 14px;
		padding: 4px;
	}

	.btn-close:hover {
		color: var(--brand-ink);
	}

	.drawer-title {
		margin: 0;
		font-size: 18px;
		font-family: var(--font-serif, serif);
		line-height: 1.4;
	}

	.drawer-meta {
		display: flex;
		gap: 6px;
	}

	.meta-tag {
		font-size: 11px;
		color: var(--brand-muted);
		background: var(--brand-surface);
		border: 1px solid var(--brand-line);
		padding: 2px 7px;
		border-radius: 4px;
	}

	.drawer-excerpt {
		font-size: 13px;
		line-height: 1.7;
		color: var(--brand-body);
		margin: 0;
	}

	.drawer-tags {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}

	.tag-pill {
		font-size: 11px;
		color: var(--brand-muted);
		border: 1px solid var(--brand-line);
		padding: 2px 8px;
		border-radius: 999px;
	}

	.neighbors-section h4 {
		margin: 0 0 8px;
		font-size: 12px;
		color: var(--brand-muted);
	}

	.neighbor-list {
		display: flex;
		flex-direction: column;
		gap: 6px;
	}

	.neighbor-pill {
		display: flex;
		align-items: center;
		gap: 8px;
		padding: 6px 10px;
		border: 1px solid var(--brand-line);
		border-radius: 6px;
		background: var(--brand-surface);
		color: var(--brand-body);
		font-size: 12px;
		cursor: pointer;
		text-align: left;
		transition: all 0.15s ease;
	}

	.neighbor-pill:hover {
		border-color: var(--brand-accent);
		color: var(--brand-accent);
	}

	.drawer-actions {
		margin-top: auto;
		padding-top: 14px;
		border-top: 1px solid var(--brand-line);
	}

	.btn-action-primary {
		display: block;
		text-align: center;
		padding: 10px 16px;
		background: var(--brand-accent);
		color: #fff;
		border-radius: 8px;
		font-size: 13px;
		font-weight: 600;
		text-decoration: none;
		transition: opacity 0.15s ease;
	}

	.btn-action-primary:hover {
		opacity: 0.9;
	}
</style>
