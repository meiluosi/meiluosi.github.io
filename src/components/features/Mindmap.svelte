<script lang="ts">
import { onMount } from "svelte";
import type { ConceptGraph, ConceptNode } from "@/utils/article-concept-graph";
import {
	playLatchClick,
	playMechanicalTick,
	playShutterTick,
} from "@/utils/haptics";

interface Props {
	graph: ConceptGraph;
	english?: boolean;
}

let { graph, english = false }: Props = $props();

// View and filter states
let viewMode = $state<"network" | "tree">("network");
let depth = $state(2);
let branch = $state("root");
let selected = $state("root");
let hovered = $state<string | null>(null);
let query = $state("");
let zoom = $state(1);
let compact = $state(false);

// Network drag state
let draggingNodeId = $state<string | null>(null);
let dragOffset = { x: 0, y: 0 };
let nodeCustomPositions = $state<Map<string, { x: number; y: number }>>(
	new Map(),
);

let svgEl = $state<SVGSVGElement | null>(null);

onMount(() => {
	const media = window.matchMedia("(max-width: 640px)");
	const update = () => {
		compact = media.matches;
		if (compact && viewMode === "network") {
			// On very small screens, tree mode is often more easily scrollable, but user can still switch
		}
	};
	update();
	media.addEventListener("change", update);
	return () => media.removeEventListener("change", update);
});

// Compute node hierarchy levels
const levels = $derived.by(() => {
	const map = new Map<string, number>([["root", 0]]);
	for (const node of graph.nodes) {
		map.set(node.id, node.parent ? (map.get(node.parent) ?? 0) + 1 : 0);
	}
	return map;
});

const maxDepth = $derived(Math.max(1, ...levels.values()));

// Visible nodes based on branch, depth, search query
const visibleNodeIds = $derived.by(() => {
	const visible = new Set(
		graph.nodes
			.filter((n) => (levels.get(n.id) ?? 0) <= depth)
			.map((n) => n.id),
	);

	if (branch !== "root") {
		for (const id of [...visible]) {
			let current = graph.nodes.find((n) => n.id === id);
			while (current && current.id !== branch && current.parent) {
				current = graph.nodes.find((n) => n.id === current?.parent);
			}
			if (id !== "root" && current?.id !== branch) {
				visible.delete(id);
			}
		}
	}

	const q = query.trim().toLowerCase();
	if (q) {
		visible.clear();
		visible.add("root");
		for (const n of graph.nodes) {
			if (n.label.toLowerCase().includes(q)) {
				let current: ConceptNode | undefined = n;
				while (current) {
					visible.add(current.id);
					current = graph.nodes.find((p) => p.id === current?.parent);
				}
			}
		}
	}

	return visible;
});

// Active selected node and its neighbors
const activeNode = $derived(
	graph.nodes.find((n) => n.id === selected) ?? graph.nodes[0],
);

// Cross-links matching the active/hovered node
const activeCrossLinks = $derived.by(() => {
	const targetId = hovered ?? selected;
	return graph.links.filter((l) => l.from === targetId || l.to === targetId);
});

// Highlighting set: when a node is hovered, compute connected nodes
const highlightedIds = $derived.by(() => {
	const targetId = hovered;
	if (!targetId) return null;

	const set = new Set<string>([targetId]);
	const targetNode = graph.nodes.find((n) => n.id === targetId);
	if (targetNode?.parent) set.add(targetNode.parent);

	// Children
	for (const n of graph.nodes) {
		if (n.parent === targetId) set.add(n.id);
	}

	// Cross-links
	for (const l of graph.links) {
		if (l.from === targetId) set.add(l.to);
		if (l.to === targetId) set.add(l.from);
	}

	return set;
});

// --- TREE VIEW LAYOUT COMPUTATION ---
const treeLayout = $derived.by(() => {
	const visible = visibleNodeIds;
	const nodeH = 46;
	const gapY = compact ? 56 : 60;
	const stepX = compact ? 180 : 250;

	let row = 0;
	const positions = new Map<string, { x: number; y: number }>();

	function visit(id: string) {
		const children = graph.nodes
			.filter((n) => n.parent === id && visible.has(n.id))
			.sort(
				(a, b) => Number(a.kind === "concept") - Number(b.kind === "concept"),
			);

		const y = 50 + row++ * gapY;
		const lvl = levels.get(id) ?? 0;
		positions.set(id, {
			x: (compact ? 16 : 28) + lvl * stepX,
			y,
		});

		for (const child of children) {
			visit(child.id);
		}
	}

	visit("root");

	const maxLvl = Math.max(0, ...[...visible].map((id) => levels.get(id) ?? 0));

	return {
		nodes: graph.nodes
			.filter((n) => visible.has(n.id))
			.map((n) => ({ ...n, ...(positions.get(n.id) ?? { x: 0, y: 0 }) })),
		positions,
		width: (compact ? 240 : 290) + maxLvl * stepX,
		height: Math.max(220, row * gapY + 40),
	};
});

// --- NETWORK CONSTELLATION LAYOUT COMPUTATION ---
const networkLayout = $derived.by(() => {
	const visible = visibleNodeIds;
	const width = compact ? 620 : 880;
	const height = compact ? 560 : 660;
	const cx = width / 2;
	const cy = height / 2;

	const positions = new Map<string, { x: number; y: number }>();

	// Center root
	positions.set("root", { x: cx, y: cy });

	// Group sections and their concepts
	const sections = graph.nodes.filter(
		(n) => n.parent === "root" && visible.has(n.id),
	);
	const numSections = Math.max(1, sections.length);
	const sectionRadius = compact ? 140 : 210;

	sections.forEach((sec, idx) => {
		const angle = (idx / numSections) * 2 * Math.PI - Math.PI / 2;
		const sx = cx + Math.cos(angle) * sectionRadius;
		const sy = cy + Math.sin(angle) * sectionRadius;
		positions.set(sec.id, { x: sx, y: sy });

		// Place concepts around each section
		const concepts = graph.nodes.filter(
			(n) => n.parent === sec.id && visible.has(n.id),
		);
		const numConcepts = concepts.length;
		const conceptSpread = Math.min(Math.PI * 0.9, 0.35 + numConcepts * 0.18);
		const conceptDist = compact ? 80 : 110;

		concepts.forEach((concept, cIdx) => {
			const startAngle = angle - conceptSpread / 2;
			const stepAngle = numConcepts > 1 ? conceptSpread / (numConcepts - 1) : 0;
			const cAngle = numConcepts === 1 ? angle : startAngle + cIdx * stepAngle;
			const cxPos = sx + Math.cos(cAngle) * conceptDist;
			const cyPos = sy + Math.sin(cAngle) * conceptDist;
			positions.set(concept.id, { x: cxPos, y: cyPos });
		});
	});

	// Apply any user custom dragged positions
	for (const [id, pos] of nodeCustomPositions) {
		if (positions.has(id)) {
			positions.set(id, pos);
		}
	}

	return {
		width,
		height,
		positions,
		nodes: graph.nodes
			.filter((n) => visible.has(n.id))
			.map((n) => ({ ...n, ...(positions.get(n.id) ?? { x: cx, y: cy }) })),
	};
});

function wrapLabel(label: string, maxLen = 22): string[] {
	if (label.length <= maxLen) return [label];
	const first = label.slice(0, maxLen);
	const rest = label.slice(maxLen);
	return [first, rest.length > maxLen ? `${rest.slice(0, maxLen - 1)}…` : rest];
}

// Drag handlers for network mode
function handlePointerDown(nodeId: string, e: PointerEvent) {
	if (viewMode !== "network") return;
	e.preventDefault();
	(e.target as Element).setPointerCapture(e.pointerId);
	draggingNodeId = nodeId;
	const currentPos = networkLayout.positions.get(nodeId) || { x: 0, y: 0 };
	dragOffset = {
		x: e.clientX - currentPos.x * zoom,
		y: e.clientY - currentPos.y * zoom,
	};
	playLatchClick();
}

function handlePointerMove(e: PointerEvent) {
	if (!draggingNodeId || viewMode !== "network") return;
	const nextX = (e.clientX - dragOffset.x) / zoom;
	const nextY = (e.clientY - dragOffset.y) / zoom;
	nodeCustomPositions.set(draggingNodeId, { x: nextX, y: nextY });
	nodeCustomPositions = new Map(nodeCustomPositions);
}

function handlePointerUp(e: PointerEvent) {
	if (draggingNodeId) {
		try {
			(e.target as Element).releasePointerCapture(e.pointerId);
		} catch {
			// ignore
		}
		draggingNodeId = null;
	}
}

function selectNode(id: string) {
	selected = id;
	playMechanicalTick();
}

function resetAll() {
	zoom = 1;
	depth = 2;
	query = "";
	selected = "root";
	branch = "root";
	hovered = null;
	nodeCustomPositions = new Map();
	playShutterTick();
}

function downloadSVG() {
	if (!svgEl) return;
	const copy = svgEl.cloneNode(true) as SVGSVGElement;
	const curLayout = viewMode === "network" ? networkLayout : treeLayout;
	copy.setAttribute("xmlns", "http://www.w3.org/2000/svg");
	copy.setAttribute("width", String(curLayout.width));
	copy.setAttribute("height", String(curLayout.height));
	const blob = new Blob([new XMLSerializer().serializeToString(copy)], {
		type: "image/svg+xml;charset=utf-8",
	});
	const url = URL.createObjectURL(blob);
	const a = document.createElement("a");
	a.href = url;
	a.download = `${graph.nodes[0]?.label || "article"}-concept-network.svg`;
	a.click();
	URL.revokeObjectURL(url);
}
</script>

{#if graph.nodes.length > 1}
<section
	id="article-concept-network"
	class="concept-map"
	aria-label={english ? "Article concept network" : "文章概念网络"}
	data-pagefind-ignore
>
	<div class="header-row">
		<div class="title-group">
			<h2>{english ? "Article Concept Network" : "文章概念网络"}</h2>
			<p class="intro">
				{english
					? "Syntactic tree and cross-section semantic connections extracted from the article."
					: "可视化呈现全文核心思想脉络。实线表述章节从属，虚线连结跨章节高频复现的孪生概念。"}
			</p>
		</div>

		<!-- View Switcher Toggle -->
		<div class="mode-toggle">
			<button
				type="button"
				class="mode-btn"
				class:active={viewMode === "network"}
				onclick={() => {
					viewMode = "network";
					playShutterTick();
				}}
			>
				🕸️ {english ? "Network" : "概念星系"}
			</button>
			<button
				type="button"
				class="mode-btn"
				class:active={viewMode === "tree"}
				onclick={() => {
					viewMode = "tree";
					playShutterTick();
				}}
			>
				🌿 {english ? "Mindmap" : "层级导图"}
			</button>
		</div>
	</div>

	<!-- Interactive Toolbar -->
	<div class="toolbar">
		<label>
			<span class="label-text">{english ? "Branch" : "分支"}</span>
			<select bind:value={branch} onchange={() => playMechanicalTick()}>
				<option value="root">{english ? "All sections" : "全部章节"}</option>
				{#each graph.nodes.filter((n) => n.parent === "root" && n.kind === "section") as n}
					<option value={n.id}>{n.label}</option>
				{/each}
			</select>
		</label>

		<label>
			<span class="label-text">{english ? "Depth" : "层级"}</span>
			<select bind:value={depth} onchange={() => playMechanicalTick()}>
				{#each Array.from({ length: maxDepth }, (_, i) => i + 1) as d}
					<option value={d}>{d}</option>
				{/each}
			</select>
		</label>

		<label class="search">
			<span class="label-text">{english ? "Find" : "查找"}</span>
			<input
				bind:value={query}
				placeholder={english ? "Filter terms…" : "检索概念或术语…"}
			/>
		</label>

		<div class="btn-group">
			<button
				type="button"
				onclick={() => (zoom = Math.max(0.4, zoom - 0.2))}
				aria-label={english ? "Zoom out" : "缩小"}
				title={english ? "Zoom out" : "缩小"}
			>−</button>
			<button
				type="button"
				onclick={() => (zoom = Math.min(2.5, zoom + 0.2))}
				aria-label={english ? "Zoom in" : "放大"}
				title={english ? "Zoom in" : "放大"}
			>＋</button>
			<button type="button" onclick={resetAll}>{english ? "Reset" : "重置"}</button>
			<button type="button" class="btn-export" onclick={downloadSVG}>
				{english ? "Export SVG" : "导出 SVG"}
			</button>
		</div>
	</div>

	<p class="count">
		<span class="badge-count">
			{visibleNodeIds.size} / {graph.nodes.length}
		</span>
		{english
			? "nodes active · drag nodes to explore constellation or click to inspect text context"
			: "个活跃概念节点 · 可自由拖拽星丛或点击查看原文引用"}
	</p>

	<!-- SVG Interactive Canvas Arena -->
	<div
		class="canvas-frame"
		role="region"
		aria-label={english ? "Interactive concept graph" : "可交互概念网络图"}
		onpointermove={handlePointerMove}
		onpointerup={handlePointerUp}
		onpointercancel={handlePointerUp}
	>
		{#if viewMode === "network"}
			<!-- 1. NETWORK CONSTELLATION VIEW -->
			<svg
				bind:this={svgEl}
				width={networkLayout.width * zoom}
				height={networkLayout.height * zoom}
				viewBox={`0 0 ${networkLayout.width} ${networkLayout.height}`}
				class="network-svg"
			>
				<defs>
					<!-- Subtle grid pattern -->
					<pattern id="net-grid" width="30" height="30" patternUnits="userSpaceOnUse">
						<circle cx="15" cy="15" r="1" fill="var(--brand-line, #cad7cd)" opacity="0.4" />
					</pattern>
					<!-- Soft radial glow -->
					<filter id="node-glow" x="-40%" y="-40%" width="180%" height="180%">
						<feGaussianBlur stdDeviation="3.5" result="blur" />
						<feComposite in="SourceGraphic" in2="blur" operator="over" />
					</filter>
				</defs>

				<rect width={networkLayout.width} height={networkLayout.height} fill="url(#net-grid)" />

				<!-- Background orbits around root -->
				<circle
					cx={networkLayout.width / 2}
					cy={networkLayout.height / 2}
					r={compact ? 140 : 210}
					fill="none"
					stroke="var(--brand-line, #cad7cd)"
					stroke-width="1"
					stroke-dasharray="4 6"
					opacity="0.6"
				/>

				<!-- 1.1 Structural parent-child links -->
				{#each networkLayout.nodes.filter((n) => n.parent) as node}
					{@const p = networkLayout.positions.get(node.parent!)!}
					{@const isLit =
						highlightedIds === null ||
						(highlightedIds.has(node.id) && highlightedIds.has(node.parent!))}
					<line
						x1={p.x}
						y1={p.y}
						x2={node.x}
						y2={node.y}
						stroke={isLit ? "var(--brand-accent, #406d5a)" : "var(--brand-line, #ccd9d0)"}
						stroke-width={isLit && highlightedIds ? 2.2 : 1.2}
						opacity={isLit ? (highlightedIds ? 0.95 : 0.6) : 0.15}
						stroke-linecap="round"
					/>
				{/each}

				<!-- 1.2 Cross-section recurring semantic bridges -->
				{#each graph.links as link}
					{@const a = networkLayout.positions.get(link.from)}
					{@const b = networkLayout.positions.get(link.to)}
					{#if a && b && visibleNodeIds.has(link.from) && visibleNodeIds.has(link.to)}
						{@const isSelectedLink =
							selected === link.from ||
							selected === link.to ||
							hovered === link.from ||
							hovered === link.to}
						{@const midX = (a.x + b.x) / 2}
						{@const midY = (a.y + b.y) / 2}
						<!-- Curved arch bridging the two concepts -->
						<path
							d={`M ${a.x} ${a.y} Q ${midX} ${midY - 25} ${b.x} ${b.y}`}
							fill="none"
							stroke={isSelectedLink ? "#df7a52" : "#c2886a"}
							stroke-width={isSelectedLink ? 2.8 : 1.4}
							stroke-dasharray={isSelectedLink ? "none" : "4 4"}
							opacity={isSelectedLink ? 1 : 0.45}
						>
							<title>{link.label}</title>
						</path>
					{/if}
				{/each}

				<!-- 1.3 Interactive Nodes -->
				{#each networkLayout.nodes as node}
					{@const isSelected = selected === node.id}
					{@const isHovered = hovered === node.id}
					{@const isDimmed =
						highlightedIds !== null && !highlightedIds.has(node.id)}
					{@const isArticle = node.kind === "article"}
					{@const isSection = node.kind === "section"}
					{@const radius = isArticle ? 24 : isSection ? 17 : 11}
					<g
						transform={`translate(${node.x}, ${node.y})`}
						class="network-node-group"
						class:selected={isSelected}
						class:dimmed={isDimmed}
						onpointerdown={(e) => handlePointerDown(node.id, e)}
						onmouseenter={() => (hovered = node.id)}
						onmouseleave={() => (hovered = null)}
						onclick={() => selectNode(node.id)}
						onkeydown={(e) => {
							if (e.key === "Enter" || e.key === " ") {
								e.preventDefault();
								selectNode(node.id);
							}
						}}
						role="button"
						tabindex="0"
						aria-label={node.label}
						aria-pressed={selected === node.id}
					>
						<!-- Outer pulse halo when selected or hovered -->
						{#if isSelected || isHovered}
							<circle
								r={radius + 8}
								fill={isArticle
									? "rgba(64, 109, 90, 0.2)"
									: isSection
										? "rgba(80, 130, 110, 0.25)"
										: "rgba(223, 122, 82, 0.25)"}
								filter="url(#node-glow)"
							/>
						{/if}

						<!-- Central celestial sphere -->
						<circle
							r={radius}
							fill={isArticle
								? "var(--brand-accent, #33614e)"
								: isSection
									? "#4a7865"
									: "#e88f68"}
							stroke="#ffffff"
							stroke-width={isSelected ? 3 : 1.8}
							box-shadow="0 2px 8px rgba(0,0,0,0.15)"
						/>

						<!-- Core icon/letter -->
						<text
							text-anchor="middle"
							dominant-baseline="central"
							fill="#ffffff"
							font-size={isArticle ? "13" : isSection ? "11" : "9"}
							font-weight="bold"
							pointer-events="none"
						>
							{isArticle ? "📖" : isSection ? "§" : "•"}
						</text>

						<!-- Label pill -->
						<g transform={`translate(0, ${radius + 12})`}>
							<rect
								x={-Math.min(90, node.label.length * 6.5 + 10)}
								y="-9"
								width={Math.min(180, node.label.length * 13 + 20)}
								height="18"
								rx="9"
								fill="var(--brand-surface, #ffffff)"
								stroke={isSelected ? "var(--brand-accent, #33614e)" : "var(--brand-line, #cad7cd)"}
								stroke-width={isSelected ? 1.8 : 1}
								opacity={isDimmed ? 0.3 : 0.95}
							/>
							<text
								text-anchor="middle"
								dominant-baseline="central"
								fill="var(--brand-ink, #203e32)"
								font-size="11"
								font-family="Inter, -apple-system, sans-serif"
								font-weight={isSelected || isSection ? "600" : "400"}
							>
								{wrapLabel(node.label, 14)[0]}
							</text>
						</g>
						<title>{node.label}</title>
					</g>
				{/each}
			</svg>
		{:else}
			<!-- 2. STREAMLINED CAPSULE TREE VIEW -->
			<svg
				bind:this={svgEl}
				width={treeLayout.width * zoom}
				height={treeLayout.height * zoom}
				viewBox={`0 0 ${treeLayout.width} ${treeLayout.height}`}
				class="tree-svg"
			>
				<!-- Connecting smooth cubic splines -->
				{#each treeLayout.nodes.filter((n) => n.parent) as node}
					{@const p = treeLayout.positions.get(node.parent!)!}
					{@const isSelectedPath =
						selected === node.id || selected === node.parent}
					<path
						d={`M ${p.x + (compact ? 170 : 210)} ${p.y} C ${p.x + (compact ? 170 : 210) + 35} ${p.y}, ${node.x - 30} ${node.y}, ${node.x} ${node.y}`}
						fill="none"
						stroke={isSelectedPath ? "var(--brand-accent, #32634e)" : "var(--brand-line, #b4c7bc)"}
						stroke-width={isSelectedPath ? 2.4 : 1.4}
					/>
				{/each}

				<!-- Cross-section recurring links -->
				{#each graph.links as link}
					{@const a = treeLayout.positions.get(link.from)}
					{@const b = treeLayout.positions.get(link.to)}
					{#if a && b && visibleNodeIds.has(link.from) && visibleNodeIds.has(link.to)}
						{@const isSelectedLink =
							selected === link.from || selected === link.to}
						{@const widthCapsule = compact ? 170 : 210}
						<path
							d={`M ${a.x + widthCapsule - 6} ${a.y} C ${a.x + widthCapsule + 32} ${a.y}, ${b.x + widthCapsule + 32} ${b.y}, ${b.x + widthCapsule - 6} ${b.y}`}
							fill="none"
							stroke="#b07857"
							stroke-width={isSelectedLink ? 2.8 : 1.2}
							stroke-dasharray="4 4"
						>
							<title>{link.label}</title>
						</path>
					{/if}
				{/each}

				<!-- Capsule Tree Nodes -->
				{#each treeLayout.nodes as node}
					{@const isSelected = selected === node.id}
					{@const isConcept = node.kind === "concept"}
					{@const isArticle = node.kind === "article"}
					{@const capW = compact ? 170 : 210}
					<g
						transform={`translate(${node.x}, ${node.y - 21})`}
						class="tree-node-group"
						class:selected={isSelected}
						role="button"
						tabindex="0"
						aria-label={node.label}
						aria-pressed={selected === node.id}
						onclick={() => selectNode(node.id)}
						onkeydown={(e) => {
							if (e.key === "Enter" || e.key === " ") {
								e.preventDefault();
								selectNode(node.id);
							}
						}}
						onmouseenter={() => (hovered = node.id)}
						onmouseleave={() => (hovered = null)}
					>
						<rect
							width={capW}
							height="42"
							rx="12"
							fill={isConcept
								? "var(--brand-surface, #ffffff)"
								: isArticle
									? "var(--brand-paper, #f3f7f4)"
									: "var(--brand-paper, #f6f8f5)"}
							stroke={isSelected
								? "var(--brand-accent, #32634e)"
								: isConcept
									? "rgba(180, 110, 75, 0.45)"
									: "var(--brand-line, #cad7cd)"}
							stroke-width={isSelected ? 2.4 : 1.2}
						/>

						<!-- Left color indicator strip -->
						<rect
							x="2"
							y="2"
							width="5"
							height="38"
							rx="2.5"
							fill={isArticle
								? "var(--brand-accent, #32634e)"
								: isConcept
									? "#df7a52"
									: "#5b8a76"}
						/>

						<!-- Kind Icon -->
						<text
							x="16"
							y="25"
							font-size="12"
						>
							{isArticle ? "📖" : isConcept ? "💡" : "📑"}
						</text>

						<!-- Label text -->
						<text
							x="36"
							y="25"
							fill="var(--brand-ink, #203e32)"
							font-size="12"
							font-family="Inter, -apple-system, sans-serif"
							font-weight={isSelected || isArticle ? "600" : "450"}
						>
							{wrapLabel(node.label, compact ? 13 : 17)[0]}
						</text>
						<title>{node.label}</title>
					</g>
				{/each}
			</svg>
		{/if}
	</div>

	<!-- Bottom Inspector: Context Quote & Article Navigation -->
	<div class="detail-card" aria-live="polite">
		<div class="detail-header">
			<span class="detail-badge" class:concept={activeNode.kind === "concept"}>
				{activeNode.kind === "article"
					? english ? "Article Nucleus" : "文章核心"
					: activeNode.kind === "section"
						? english ? "Section Chapter" : "所属章节"
						: english ? "Key Concept" : "关键术语概念"}
			</span>
			<h3 class="detail-title">{activeNode.label}</h3>
		</div>

		{#if activeNode.evidence}
			<blockquote class="detail-quote">
				<span class="quote-mark">“</span>
				<p>{activeNode.evidence}</p>
			</blockquote>
		{/if}

		<!-- Associated cross-section recurrences -->
		{#if activeCrossLinks.length > 0}
			<div class="cross-recurrence-strip">
				<span class="strip-label">
					🔗 {english ? "Cross-section recurrences:" : "跨章节关联脉络:"}
				</span>
				{#each activeCrossLinks as l}
					<span class="recurrence-pill">{l.label}</span>
				{/each}
			</div>
		{/if}

		<div class="detail-footer">
			<a
				href={activeNode.slug ? `#${activeNode.slug}` : "#post-container"}
				class="btn-jump-context"
			>
				{english ? "Locate in article body ↗" : "直达正文对应章节 ↗"}
			</a>
		</div>
	</div>
</section>
{/if}

<style>
.concept-map {
	margin: 2.25rem 0;
	padding: 1.5rem;
	border: 1px solid var(--brand-line, #cad7cd);
	border-radius: 18px;
	background: var(--brand-paper, #f5f7f2);
	color: var(--brand-ink, #243d32);
	min-width: 0;
	box-shadow: 0 4px 20px -4px rgba(0, 0, 0, 0.05);
}

.header-row {
	display: flex;
	justify-content: space-between;
	align-items: flex-start;
	flex-wrap: wrap;
	gap: 12px;
	margin-bottom: 0.75rem;
}

.title-group h2 {
	font-size: 1.35rem;
	margin: 0 0 0.35rem;
	font-family: var(--font-serif, Georgia, serif);
}

.intro {
	font-size: 0.88rem;
	line-height: 1.6;
	color: var(--brand-muted, #5f756b);
	margin: 0;
}

/* View Mode Switcher */
.mode-toggle {
	display: inline-flex;
	padding: 2px;
	background: var(--brand-surface, #ffffff);
	border: 1px solid var(--brand-line, #cad7cd);
	border-radius: 999px;
}

.mode-btn {
	padding: 5px 12px;
	border: none;
	background: transparent;
	font-size: 11.5px;
	font-weight: 500;
	color: var(--brand-muted, #5f756b);
	cursor: pointer;
	border-radius: 999px;
	transition: all 0.15s ease;
	white-space: nowrap;
}

.mode-btn.active {
	background: var(--brand-accent, #33614e);
	color: #ffffff;
	box-shadow: 0 1px 4px rgba(0, 0, 0, 0.12);
}

/* Toolbar */
.toolbar {
	display: flex;
	gap: 0.65rem;
	align-items: center;
	flex-wrap: wrap;
	margin: 1rem 0;
	padding: 0.6rem 0.85rem;
	background: var(--brand-surface, #ffffff);
	border: 1px solid var(--brand-line, #cad7cd);
	border-radius: 10px;
}

.toolbar label {
	display: flex;
	align-items: center;
	gap: 0.4rem;
	font-size: 12px;
	color: var(--brand-muted, #5f756b);
	white-space: nowrap;
}

.toolbar select,
.toolbar input {
	border: 1px solid var(--brand-line, #cad7cd);
	border-radius: 6px;
	padding: 0.35rem 0.6rem;
	background: var(--brand-paper, #fbfcf9);
	color: var(--brand-ink, #203e32);
	font-size: 12px;
	outline: none;
}

.toolbar select:focus,
.toolbar input:focus {
	border-color: var(--brand-accent, #33614e);
}

.toolbar select {
	max-width: min(200px, 50vw);
}

.search input {
	width: 140px;
}

.btn-group {
	display: flex;
	align-items: center;
	gap: 4px;
	margin-left: auto;
}

.btn-group button {
	border: 1px solid var(--brand-line, #cad7cd);
	border-radius: 6px;
	padding: 0.35rem 0.65rem;
	background: var(--brand-paper, #ffffff);
	color: var(--brand-ink, #203e32);
	font-size: 12px;
	cursor: pointer;
	transition: all 0.15s ease;
}

.btn-group button:hover {
	border-color: var(--brand-accent, #33614e);
	color: var(--brand-accent, #33614e);
}

.btn-group .btn-export {
	font-weight: 500;
	background: rgba(51, 97, 78, 0.08);
	color: var(--brand-accent, #33614e);
}

.count {
	font-size: 0.84rem;
	color: var(--brand-muted, #5f756b);
	margin: 0.4rem 0 0.8rem;
	display: flex;
	align-items: center;
	gap: 6px;
}

.badge-count {
	font-family: var(--font-mono, monospace);
	font-weight: 600;
	color: var(--brand-accent, #33614e);
}

/* Canvas Arena */
.canvas-frame {
	overflow: auto;
	max-height: 580px;
	border: 1px solid var(--brand-line, #cad7cd);
	border-radius: 12px;
	background: var(--brand-surface, #fcfdfa);
	position: relative;
	box-shadow: inset 0 1px 6px rgba(0, 0, 0, 0.03);
}

.canvas-frame svg {
	display: block;
	max-width: none;
}

.network-node-group,
.tree-node-group {
	cursor: pointer;
	transition: opacity 0.2s ease;
	touch-action: none;
}

.network-node-group.dimmed {
	opacity: 0.18;
}

.network-node-group:hover,
.tree-node-group:hover {
	filter: drop-shadow(0 2px 6px rgba(0, 0, 0, 0.15));
}

/* Detail Card at Bottom */
.detail-card {
	margin-top: 1.25rem;
	padding: 1.1rem 1.25rem;
	background: var(--brand-surface, #ffffff);
	border: 1px solid var(--brand-line, #cad7cd);
	border-radius: 12px;
	display: flex;
	flex-direction: column;
	gap: 8px;
}

.detail-header {
	display: flex;
	align-items: center;
	gap: 8px;
}

.detail-badge {
	font-size: 10.5px;
	font-weight: 600;
	text-transform: uppercase;
	letter-spacing: 0.05em;
	padding: 2px 7px;
	border-radius: 4px;
	background: rgba(51, 97, 78, 0.12);
	color: var(--brand-accent, #33614e);
}

.detail-badge.concept {
	background: rgba(223, 122, 82, 0.15);
	color: #bf5b30;
}

.detail-title {
	margin: 0;
	font-size: 1.05rem;
	color: var(--brand-ink, #1b3329);
	font-family: var(--font-serif, Georgia, serif);
}

.detail-quote {
	margin: 4px 0;
	padding: 0.5rem 0.9rem;
	background: var(--brand-paper, #f7f9f5);
	border-left: 3px solid var(--brand-accent, #33614e);
	border-radius: 0 6px 6px 0;
	display: flex;
	gap: 6px;
}

.quote-mark {
	font-size: 1.6rem;
	line-height: 1;
	color: var(--brand-accent, #33614e);
	opacity: 0.4;
	font-family: Georgia, serif;
}

.detail-quote p {
	margin: 0;
	font-size: 0.88rem;
	line-height: 1.65;
	color: var(--brand-body, #2e4a3d);
}

.cross-recurrence-strip {
	display: flex;
	align-items: center;
	gap: 6px;
	flex-wrap: wrap;
	font-size: 11.5px;
}

.strip-label {
	color: var(--brand-muted, #5f756b);
}

.recurrence-pill {
	background: rgba(194, 136, 106, 0.18);
	color: #9c502b;
	padding: 1px 7px;
	border-radius: 4px;
	font-size: 11px;
}

.detail-footer {
	margin-top: 4px;
	display: flex;
	justify-content: flex-end;
}

.btn-jump-context {
	font-size: 12px;
	font-weight: 600;
	color: var(--brand-accent, #33614e);
	text-decoration: none;
	display: inline-flex;
	align-items: center;
	gap: 4px;
	transition: transform 0.15s ease;
}

.btn-jump-context:hover {
	text-decoration: underline;
	transform: translateX(2px);
}

@media (max-width: 600px) {
	.concept-map {
		padding: 1rem;
	}
	.btn-group {
		margin-left: 0;
	}
	.canvas-frame {
		max-height: 440px;
	}
}

/* Dark Mode Theme Adaptation */
:global(.dark) .concept-map {
	--brand-paper: #141816;
	--brand-surface: #1a201d;
	--brand-line: rgba(120, 160, 140, 0.22);
	--brand-ink: #f1f5f9;
	--brand-body: #cbd5e1;
	--brand-muted: #94a3b8;
	--brand-accent: #5ba887;
	background: #141816;
	border-color: rgba(120, 160, 140, 0.22);
	color: #e2e8f0;
	box-shadow: 0 4px 20px -4px rgba(0, 0, 0, 0.45);
}

:global(.dark) .title-group h2 {
	color: #f1f5f9;
}

:global(.dark) .intro {
	color: #94a3b8;
}

:global(.dark) .mode-toggle {
	background: #1a201d;
	border-color: rgba(120, 160, 140, 0.25);
}

:global(.dark) .mode-btn {
	color: #94a3b8;
}

:global(.dark) .mode-btn.active {
	background: var(--brand-accent, #5ba887);
	color: #ffffff;
	box-shadow: 0 1px 4px rgba(0, 0, 0, 0.3);
}

:global(.dark) .toolbar {
	background: #1a201d;
	border-color: rgba(120, 160, 140, 0.25);
}

:global(.dark) .toolbar label {
	color: #94a3b8;
}

:global(.dark) .toolbar select,
:global(.dark) .toolbar input {
	background: #111513;
	border-color: rgba(120, 160, 140, 0.28);
	color: #f1f5f9;
}

:global(.dark) .toolbar select:focus,
:global(.dark) .toolbar input:focus {
	border-color: var(--brand-accent, #5ba887);
}

:global(.dark) .btn-group button {
	background: #111513;
	border-color: rgba(120, 160, 140, 0.28);
	color: #e2e8f0;
}

:global(.dark) .btn-group button:hover {
	border-color: var(--brand-accent, #5ba887);
	color: #ffffff;
	background: #222b27;
}

:global(.dark) .btn-group .btn-export {
	background: rgba(91, 168, 135, 0.16);
	color: #72bf9f;
	border-color: rgba(91, 168, 135, 0.32);
}

:global(.dark) .count {
	color: #94a3b8;
}

:global(.dark) .badge-count {
	color: #72bf9f;
}

:global(.dark) .canvas-frame {
	background: #0f1412;
	border-color: rgba(120, 160, 140, 0.22);
	box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.45);
}

:global(.dark) #net-grid circle {
	fill: rgba(140, 185, 160, 0.2);
}

:global(.dark) .network-svg > circle {
	stroke: rgba(140, 185, 160, 0.18);
}

:global(.dark) .network-svg line {
	stroke: rgba(120, 160, 140, 0.35);
}

:global(.dark) .tree-svg path {
	stroke: rgba(120, 160, 140, 0.35);
}

:global(.dark) .network-node-group g rect {
	fill: #1a201d;
	stroke: rgba(120, 160, 140, 0.35);
}

:global(.dark) .network-node-group.selected g rect {
	stroke: var(--brand-accent, #5ba887);
}

:global(.dark) .network-node-group g text {
	fill: #f1f5f9;
}

:global(.dark) .tree-node-group rect[rx="12"] {
	fill: #1a201d;
	stroke: rgba(120, 160, 140, 0.32);
}

:global(.dark) .tree-node-group.selected rect[rx="12"] {
	stroke: var(--brand-accent, #5ba887);
	fill: #222b26;
}

:global(.dark) .tree-node-group text {
	fill: #f1f5f9;
}

:global(.dark) .detail-card {
	background: #1a201d;
	border-color: rgba(120, 160, 140, 0.25);
}

:global(.dark) .detail-title {
	color: #f1f5f9;
}

:global(.dark) .detail-quote {
	background: rgba(255, 255, 255, 0.04);
	border-left-color: var(--brand-accent, #5ba887);
}

:global(.dark) .quote-mark {
	color: #72bf9f;
	opacity: 0.5;
}

:global(.dark) .detail-quote p {
	color: #cbd5e1;
}

:global(.dark) .detail-badge {
	background: rgba(91, 168, 135, 0.2);
	color: #72bf9f;
}

:global(.dark) .detail-badge.concept {
	background: rgba(223, 122, 82, 0.22);
	color: #f3a886;
}

:global(.dark) .strip-label {
	color: #94a3b8;
}

:global(.dark) .recurrence-pill {
	background: rgba(223, 122, 82, 0.2);
	color: #f3a886;
}

:global(.dark) .btn-jump-context {
	color: #72bf9f;
}
</style>
