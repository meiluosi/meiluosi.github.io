<script lang="ts">
	import { onMount } from "svelte";
	import type { ConceptGraph } from "@/utils/article-concept-graph";
	let { graph, english = false }: { graph: ConceptGraph; english?: boolean } = $props();
	let depth = $state(2);
	let branch = $state("root");
	let selected = $state("root");
	let query = $state("");
	let zoom = $state(1);
	let svg: SVGSVGElement;
	let compact = $state(false);
	const nodeWidth = $derived(compact ? 196 : 218);
	onMount(() => {
		const media = window.matchMedia("(max-width: 640px)");
		const update = () => { compact = media.matches; };
		update(); media.addEventListener("change", update);
		return () => media.removeEventListener("change", update);
	});
	const levels = $derived.by(() => {
		const map = new Map<string, number>([["root", 0]]);
		for (const node of graph.nodes) map.set(node.id, node.parent ? (map.get(node.parent) ?? 0) + 1 : 0);
		return map;
	});
	const maxDepth = $derived(Math.max(1, ...levels.values()));
	const layout = $derived.by(() => {
		const visible = new Set(graph.nodes.filter(n => (levels.get(n.id) ?? 0) <= depth).map(n => n.id));
		if (branch !== "root") {
			for (const id of [...visible]) {
				let current = graph.nodes.find(n => n.id === id);
				while (current && current.id !== branch && current.parent) current = graph.nodes.find(n => n.id === current?.parent);
				if (id !== "root" && current?.id !== branch) visible.delete(id);
			}
		}
		if (query.trim()) {
			visible.clear(); visible.add("root");
			for (const n of graph.nodes.filter(n => n.label.toLocaleLowerCase().includes(query.toLocaleLowerCase().trim()))) {
				let current: typeof n | undefined = n;
				while (current) { visible.add(current.id); current = graph.nodes.find(p => p.id === current?.parent); }
			}
		}
		let row = 0;
		const positions = new Map<string, { x: number; y: number }>();
		function visit(id: string) {
			const children = graph.nodes.filter(n => n.parent === id && visible.has(n.id)).sort((a,b) => Number(a.kind === "concept") - Number(b.kind === "concept"));
			const y = 45 + row++ * 86;
			positions.set(id, { x: (compact ? 12 : 24) + (levels.get(id) ?? 0) * (compact ? 18 : 270), y });
			for (const child of children) visit(child.id);
		}
		visit("root");
		return { nodes: graph.nodes.filter(n => visible.has(n.id)).map(n => ({ ...n, ...positions.get(n.id)! })), positions, height: Math.max(160, row * 86 + 30), width: (compact ? 224 : 280) + Math.max(...[...visible].map(id => levels.get(id) ?? 0)) * (compact ? 18 : 270) };
	});
	const active = $derived(graph.nodes.find(n => n.id === selected) ?? graph.nodes[0]);
	function wrap(label: string): string[] {
		const lines: string[] = []; let line = ""; let width = 0;
		for (const char of label) {
			const size = /[^\x00-\x7f]/.test(char) ? 2 : 1;
			if (width + size > (compact ? 24 : 27)) { lines.push(line); line = ""; width = 0; }
			line += char; width += size;
		}
		if (line) lines.push(line);
		return [lines[0] || "", (lines[1] || "") + (lines.length > 2 ? "…" : "")];
	}
	function download() {
		const copy = svg.cloneNode(true) as SVGSVGElement;
		copy.setAttribute("xmlns", "http://www.w3.org/2000/svg");
		copy.setAttribute("width", String(layout.width)); copy.setAttribute("height", String(layout.height));
		const url = URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(copy)], { type: "image/svg+xml" }));
		const a = document.createElement("a"); a.href = url; a.download = "article-concepts.svg"; a.click(); URL.revokeObjectURL(url);
	}
</script>

{#if graph.nodes.length > 1}
<section id="article-concept-network" class="concept-map" aria-label={english ? "Article concept network" : "文章概念网络"} data-pagefind-ignore>
	<h2>{english ? "Article concept network" : "文章概念网络"}</h2>
	<p class="intro">{english ? "Explore sections and explicit terms from the text. Dashed lines connect repeated concepts, not causal claims." : "从章节进入正文概念。实线表示所属层级，虚线连接跨章节重复出现的概念，不代表因果关系。"}</p>
	<div class="toolbar">
		<label>{english ? "Branch" : "分支"} <select bind:value={branch}><option value="root">{english ? "All sections" : "全部章节"}</option>{#each graph.nodes.filter(n => n.parent === "root" && n.kind === "section") as n}<option value={n.id}>{n.label}</option>{/each}</select></label>
		<label>{english ? "Depth" : "层级"} <select bind:value={depth}>{#each Array.from({length:maxDepth}, (_, i) => i + 1) as d}<option value={d}>{d}</option>{/each}</select></label>
		<label class="search">{english ? "Find" : "查找"} <input bind:value={query} placeholder={english ? "Concept or section" : "概念或章节"} /></label>
		<button onclick={() => zoom = Math.max(.5, zoom - .25)} aria-label={english ? "Zoom out" : "缩小"}>−</button>
		<button onclick={() => zoom = Math.min(2, zoom + .25)} aria-label={english ? "Zoom in" : "放大"}>＋</button>
		<button onclick={() => {zoom = 1; depth = 2; query = ""; selected = "root"; branch = "root";}}>{english ? "Reset" : "重置"}</button>
		<button onclick={download}>{english ? "Save SVG" : "保存 SVG"}</button>
	</div>
	<p class="count">{layout.nodes.length} / {graph.nodes.length} {english ? "nodes · marked terms and named concepts from the text" : "个节点 · 正文标记术语与关键词概念"}</p>
	<div class="canvas" tabindex="0" role="region" aria-label={english ? "Scrollable concept graph" : "可滚动概念图"}>
		<svg bind:this={svg} width={layout.width * zoom} height={layout.height * zoom} viewBox={`0 0 ${layout.width} ${layout.height}`} aria-label={english ? "Section hierarchy and shared concepts" : "章节层级与共同概念"}>
			<title>{graph.nodes[0].label}</title>
			{#each layout.nodes.filter(n => n.parent) as node}
				{@const p = layout.positions.get(node.parent!)!}
				<path d={compact ? `M ${p.x + 7} ${p.y + 31} V ${node.y} H ${node.x}` : `M ${p.x + 218} ${p.y} C ${p.x + 244} ${p.y}, ${node.x - 24} ${node.y}, ${node.x} ${node.y}`} fill="none" stroke="#9baea5" stroke-width="1.5" />
			{/each}
			{#each graph.links.filter(l => layout.positions.has(l.from) && layout.positions.has(l.to)) as link}
				{@const a = layout.positions.get(link.from)!}{@const b = layout.positions.get(link.to)!}
				<path d={`M ${a.x + nodeWidth - 8} ${a.y} C ${a.x + nodeWidth + 20} ${a.y}, ${b.x + nodeWidth + 20} ${b.y}, ${b.x + nodeWidth - 8} ${b.y}`} fill="none" stroke="#b07857" stroke-width={selected === link.from || selected === link.to ? 3 : 1} stroke-dasharray="5 5"><title>{link.label}</title></path>
			{/each}
			{#each layout.nodes as node}
				<g transform={`translate(${node.x}, ${node.y - 31})`}>
					<a href={node.slug ? `#${node.slug}` : "#post-container"} onclick={(e) => {e.preventDefault(); selected = node.id;}} aria-label={node.label}>
						<rect width={nodeWidth} height="62" rx="10" fill={node.kind === "concept" ? "#fff6e9" : node.kind === "article" ? "#d9e9df" : "#f1f6f2"} stroke={selected === node.id ? "#32634e" : "#b1c3b8"} stroke-width={selected === node.id ? 3 : 1} />
						<text x="12" y="25" fill="#203e32" font-size="13" font-family="sans-serif">{#each wrap(node.label) as line, i}<tspan x="12" dy={i ? 19 : 0}>{line}</tspan>{/each}</text>
						<title>{node.label}</title>
					</a>
				</g>
			{/each}
		</svg>
	</div>
	<div class="detail" aria-live="polite"><strong>{active.label}</strong><p>{active.evidence}</p><a href={active.slug ? `#${active.slug}` : "#post-container"}>{english ? "Read in context ↗" : "回到原文 ↗"}</a></div>
</section>
{/if}

<style>
.concept-map{margin:2rem 0;padding:1.25rem;border:1px solid var(--brand-line,#cad7cd);border-radius:16px;background:var(--brand-paper,#f5f7f2);color:var(--brand-ink,#243d32);min-width:0}.concept-map h2{font-size:1.35rem;margin:0 0 .75rem}.intro,.count,.detail p{font-size:.9rem;line-height:1.7}.toolbar{display:flex;gap:.5rem;align-items:center;flex-wrap:wrap;margin:1rem 0}.toolbar label{max-width:100%;min-width:0;white-space:nowrap;display:flex;align-items:center;gap:.4rem}.toolbar button,.toolbar select,.toolbar input{border:1px solid #afc3b5;border-radius:6px;padding:.4rem;background:var(--brand-paper,#fff);color:inherit;min-height:38px}.toolbar select{min-width:0;max-width:min(240px,58vw)}.search input{width:160px;max-width:100%}.canvas{overflow:auto;max-height:560px;border:1px solid #cad7cd;border-radius:10px;background:#fbfcf9}.canvas svg{max-width:none;display:block}.detail{padding-top:1rem;overflow-wrap:anywhere}.detail a{text-decoration:underline}.toolbar button:focus-visible,.canvas:focus-visible{outline:3px solid #b07857;outline-offset:3px}@media(max-width:480px){.concept-map{padding:.75rem}.canvas{max-height:420px}}
</style>
