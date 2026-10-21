<script lang="ts">
import { onMount } from "svelte";
import {
	type LabEntry,
	MATURITY_DEFINITIONS,
	type MaturityLevel,
} from "@/utils/digital-garden";
import {
	playLatchClick,
	playMechanicalTick,
	playSlideTick,
} from "@/utils/haptics";

export interface PostNode {
	slug: string;
	title: string;
	category?: string;
	tags?: string[];
	maturity?: MaturityLevel;
	order?: number;
	sharedTags?: string[];
}

interface Props {
	currentPost: {
		slug: string;
		title: string;
		category?: string;
		tags?: string[];
		maturity?: MaturityLevel;
	};
	relatedArticles?: Array<{
		slug: string;
		title: string;
		sharedTags: string[];
		category?: string;
	}>;
	seriesArticles?: Array<{
		slug: string;
		title: string;
		order: number;
	}>;
	connectedLabs?: LabEntry[];
	lang?: "zh" | "en";
}

let {
	currentPost,
	relatedArticles = [],
	seriesArticles = [],
	connectedLabs = [],
	lang = "zh",
}: Props = $props();

let english = $state(lang === "en");
let activeFilter = $state<"all" | "related" | "series" | "labs">("all");
let activeDrawerItem = $state<{
	id: string;
	title: string;
	kind: "post" | "series" | "lab";
	category: string;
	maturity: MaturityLevel;
	href: string;
	reason?: string;
	tags?: string[];
} | null>(null);

onMount(() => {
	const root = document.documentElement;
	english = root.lang?.startsWith("en") ?? lang === "en";

	const handleLocale = (e: Event) => {
		const detail = (e as CustomEvent<{ lang: string }>).detail;
		if (detail?.lang) {
			english = detail.lang.startsWith("en");
		}
	};
	window.addEventListener("site-locale-change", handleLocale);

	const handleKeydown = (e: KeyboardEvent) => {
		if (e.key === "Escape" && activeDrawerItem) {
			closeDrawer();
		}
	};
	window.addEventListener("keydown", handleKeydown);

	return () => {
		window.removeEventListener("site-locale-change", handleLocale);
		window.removeEventListener("keydown", handleKeydown);
	};
});

function openDrawer(item: {
	id: string;
	title: string;
	kind: "post" | "series" | "lab";
	category: string;
	maturity: MaturityLevel;
	href: string;
	reason?: string;
	tags?: string[];
}) {
	activeDrawerItem = item;
	playSlideTick(1.2);
}

function closeDrawer() {
	activeDrawerItem = null;
	playLatchClick();
}

const neighbors = $derived.by(() => {
	const list: Array<{
		id: string;
		title: string;
		kind: "post" | "series" | "lab";
		category: string;
		maturity: MaturityLevel;
		href: string;
		reason?: string;
		tags?: string[];
	}> = [];

	// Series items (exclude current)
	for (const s of seriesArticles) {
		if (s.slug !== currentPost.slug) {
			list.push({
				id: `series-${s.slug}`,
				title: s.title,
				kind: "series",
				category: currentPost.category || "Series",
				maturity: "evergreen",
				href: `/posts/${s.slug}/`,
				reason: english ? `Part ${s.order} of series` : `系列第 ${s.order} 篇`,
			});
		}
	}

	// Related posts
	for (const r of relatedArticles) {
		if (
			r.slug !== currentPost.slug &&
			!list.some((it) => it.href.includes(r.slug))
		) {
			list.push({
				id: `rel-${r.slug}`,
				title: r.title,
				kind: "post",
				category: r.category || "Article",
				maturity: "budding",
				href: `/posts/${r.slug}/`,
				reason:
					r.sharedTags.length > 0
						? english
							? `Shared: ${r.sharedTags.join(", ")}`
							: `共同标签: ${r.sharedTags.join(" · ")}`
						: english
							? "Shared topic"
							: "同主题笔记",
				tags: r.sharedTags,
			});
		}
	}

	// Connected Labs
	for (const l of connectedLabs) {
		list.push({
			id: l.id,
			title: english ? l.titleEn : l.title,
			kind: "lab",
			category: l.category,
			maturity: "lab",
			href: l.href,
			reason: english ? "Interactive simulation lab" : "配套交互计算实验室",
			tags: l.tags,
		});
	}

	return list;
});

const filteredNeighbors = $derived(
	neighbors.filter((n) => {
		if (activeFilter === "all") return true;
		if (activeFilter === "related") return n.kind === "post";
		if (activeFilter === "series") return n.kind === "series";
		if (activeFilter === "labs") return n.kind === "lab";
		return true;
	}),
);
</script>

<section class="local-graph-section onload-animation" aria-label={english ? "Local Knowledge Neighborhood" : "局部知识邻域与滑动栈"}>
	<div class="section-header">
		<div class="header-left">
			<span class="header-icon">🌐</span>
			<h3 class="header-title">
				{english ? "Local Knowledge Neighborhood" : "本篇知识星丛与滑动侧栈"}
			</h3>
		</div>
		<div class="filter-tabs">
			<button
				type="button"
				class="filter-pill"
				class:active={activeFilter === "all"}
				onclick={() => {
					activeFilter = "all";
					playMechanicalTick(1.0);
				}}
			>
				{english ? "All" : "全部"} ({neighbors.length})
			</button>
			{#if seriesArticles.length > 1}
				<button
					type="button"
					class="filter-pill"
					class:active={activeFilter === "series"}
					onclick={() => {
						activeFilter = "series";
						playMechanicalTick(1.05);
					}}
				>
					{english ? "Series" : "系列篇章"}
				</button>
			{/if}
			{#if relatedArticles.length > 0}
				<button
					type="button"
					class="filter-pill"
					class:active={activeFilter === "related"}
					onclick={() => {
						activeFilter = "related";
						playMechanicalTick(1.1);
					}}
				>
					{english ? "Related" : "相关思考"}
				</button>
			{/if}
			{#if connectedLabs.length > 0}
				<button
					type="button"
					class="filter-pill"
					class:active={activeFilter === "labs"}
					onclick={() => {
						activeFilter = "labs";
						playMechanicalTick(1.15);
					}}
				>
					{english ? "Labs" : "计算沙盒"}
				</button>
			{/if}
		</div>
	</div>

	<!-- Interactive Subgraph Visual View -->
	<div class="graph-radar-container">
		<div class="center-post-badge">
			<span class="hub-pulse"></span>
			<span class="hub-label">{currentPost.title}</span>
			<span class="hub-tag">{english ? "Current Focus" : "当前主轴"}</span>
		</div>

		<div class="neighbor-grid">
			{#each filteredNeighbors as item}
				{@const maturityMeta = MATURITY_DEFINITIONS[item.maturity]}
				<button
					type="button"
					class="neighbor-card"
					class:is-lab={item.kind === "lab"}
					class:is-series={item.kind === "series"}
					onclick={() => openDrawer(item)}
				>
					<div class="card-top">
						<span class="kind-tag">
							{#if item.kind === "lab"}
								⚙️ {english ? "Lab" : "实验"}
							{:else if item.kind === "series"}
								📚 {english ? "Series" : "系列"}
							{:else}
								📝 {english ? "Note" : "笔记"}
							{/if}
						</span>
						<span
							class="maturity-mini"
							style="color: {maturityMeta.color}; background: {maturityMeta.badgeBg}"
							title={english ? maturityMeta.labelEn : maturityMeta.labelZh}
						>
							{maturityMeta.icon}
						</span>
					</div>
					<h4 class="card-title">{item.title}</h4>
					{#if item.reason}
						<span class="card-reason">{item.reason}</span>
					{/if}
					<span class="peek-hint">{english ? "Peek in slide drawer ➔" : "点击滑动展开侧栈 ➔"}</span>
				</button>
			{/each}
		</div>
	</div>

	<!-- Andy Matuschak Sliding Stacked Peek Drawer -->
	{#if activeDrawerItem}
		{@const mat = MATURITY_DEFINITIONS[activeDrawerItem.maturity]}
		<div class="drawer-backdrop" onclick={closeDrawer} aria-hidden="true"></div>
		<aside
			class="sliding-pane"
			role="dialog"
			aria-modal="true"
			aria-label={activeDrawerItem.title}
		>
			<div class="pane-header">
				<div class="pane-meta-top">
					<span class="pane-kind-pill">
						{#if activeDrawerItem.kind === "lab"}
							⚙️ {english ? "Interactive Lab" : "交互计算实验室"}
						{:else if activeDrawerItem.kind === "series"}
							📚 {english ? "Series Continuation" : "系列篇章关联"}
						{:else}
							📝 {english ? "Digital Garden Note" : "数字花园知识笔记"}
						{/if}
					</span>
					<button
						type="button"
						class="pane-close-btn"
						onclick={closeDrawer}
						aria-label={english ? "Close drawer" : "关闭抽屉"}
					>
						✕
					</button>
				</div>
				<h3 class="pane-title">{activeDrawerItem.title}</h3>
			</div>

			<div class="pane-body">
				<!-- Maturity & Category Row -->
				<div class="pane-detail-row">
					<div class="detail-badge" style="color: {mat.color}; background: {mat.badgeBg}; border: 1px solid {mat.color}">
						<span>{mat.icon}</span>
						<span>{english ? mat.labelEn : mat.labelZh}</span>
					</div>
					<span class="pane-cat-badge">{activeDrawerItem.category}</span>
				</div>

				<!-- Connection Rationale -->
				{#if activeDrawerItem.reason}
					<div class="pane-callout">
						<strong>{english ? "Link Context:" : "邻域拓扑关联:"}</strong>
						<p>{activeDrawerItem.reason}</p>
					</div>
				{/if}

				<!-- Tag Chips -->
				{#if activeDrawerItem.tags && activeDrawerItem.tags.length > 0}
					<div class="pane-tags-section">
						<span class="tags-title">{english ? "Associated Concepts:" : "关联概念领域:"}</span>
						<div class="tag-chips">
							{#each activeDrawerItem.tags as t}
								<span class="chip">#{t}</span>
							{/each}
						</div>
					</div>
				{/if}

				<!-- Description / Andy Matuschak Philosophy Prose -->
				<p class="pane-desc">
					{#if activeDrawerItem.kind === "lab"}
						{english
							? "This interactive sandbox allows you to inspect model activations, perturb hyper-parameters, and ablate attention circuits directly inside the browser with zero remote dependencies."
							: "该交互实验室支持在浏览器内实时执行前向传播、注入神经元微扰并探查因果注意力流，无需任何服务端运行时环境。"}
					{:else}
						{english
							? "Following Andy Matuschak's evergreen note principles, this note is interlinked to establish a dense semantic mesh across deep learning, algorithmic game theory, and causal reasoning."
							: "遵循 Andy Matuschak 的常青笔记网格体系，此笔记与当前篇章构成互补论证回路，共同支撑从理论推导到实验验证的思考闭环。"}
					{/if}
				</p>
			</div>

			<div class="pane-footer">
				<a
					href={activeDrawerItem.href}
					class="pane-cta-btn"
					onclick={() => playMechanicalTick(1.2)}
				>
					{#if activeDrawerItem.kind === "lab"}
						⚡ {english ? "Launch Interactive Lab" : "启动交互式实验室"} ➔
					{:else}
						📖 {english ? "Read Full Note" : "前往阅读完整笔记"} ➔
					{/if}
				</a>
			</div>
		</aside>
	{/if}
</section>

<style>
	.local-graph-section {
		position: relative;
		margin-top: 2rem;
		margin-bottom: 2rem;
		padding: 1.25rem 1.5rem;
		border-radius: 14px;
		background: var(--brand-paper);
		border: 1px solid var(--brand-line);
		box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
	}

	.section-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		flex-wrap: wrap;
		gap: 0.75rem;
		margin-bottom: 1.25rem;
		padding-bottom: 0.75rem;
		border-bottom: 1px solid var(--brand-line);
	}

	.header-left {
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}

	.header-icon {
		font-size: 1.25rem;
	}

	.header-title {
		margin: 0;
		font-size: 1.05rem;
		font-weight: 600;
		color: var(--brand-ink);
	}

	.filter-tabs {
		display: flex;
		gap: 0.35rem;
		flex-wrap: wrap;
	}

	.filter-pill {
		font-size: 11.5px;
		padding: 3px 9px;
		border-radius: 999px;
		border: 1px solid var(--brand-line);
		background: var(--brand-surface);
		color: var(--brand-muted);
		cursor: pointer;
		transition: all 0.15s ease;
	}

	.filter-pill:hover,
	.filter-pill.active {
		border-color: var(--brand-accent);
		color: var(--brand-accent);
		background: rgba(92, 122, 107, 0.1);
	}

	/* Radar & Cards */
	.graph-radar-container {
		display: flex;
		flex-direction: column;
		gap: 1rem;
	}

	.center-post-badge {
		position: relative;
		display: flex;
		align-items: center;
		gap: 0.5rem;
		padding: 0.5rem 0.85rem;
		background: var(--brand-surface);
		border: 1px dashed var(--brand-accent);
		border-radius: 8px;
		font-size: 12.5px;
	}

	.hub-pulse {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		background: var(--brand-accent);
		box-shadow: 0 0 0 3px rgba(92, 122, 107, 0.25);
		animation: pulse 2s infinite ease-in-out;
	}

	@keyframes pulse {
		0%, 100% { transform: scale(1); opacity: 1; }
		50% { transform: scale(1.4); opacity: 0.6; }
	}

	.hub-label {
		font-weight: 600;
		color: var(--brand-ink);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		flex: 1;
	}

	.hub-tag {
		font-size: 10px;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		padding: 2px 6px;
		border-radius: 4px;
		background: rgba(92, 122, 107, 0.14);
		color: var(--brand-accent);
		font-weight: 600;
	}

	.neighbor-grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
		gap: 0.85rem;
	}

	.neighbor-card {
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		text-align: left;
		padding: 0.75rem 0.85rem;
		border-radius: 8px;
		background: var(--brand-surface);
		border: 1px solid var(--brand-line);
		cursor: pointer;
		transition: all 0.2s ease;
		color: inherit;
		font: inherit;
	}

	.neighbor-card:hover {
		transform: translateY(-2px);
		border-color: var(--brand-accent);
		box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06);
	}

	.neighbor-card.is-lab {
		border-color: rgba(131, 103, 199, 0.4);
	}

	.neighbor-card.is-lab:hover {
		border-color: #8367c7;
	}

	.card-top {
		display: flex;
		align-items: center;
		justify-content: space-between;
		width: 100%;
		margin-bottom: 0.35rem;
	}

	.kind-tag {
		font-size: 10.5px;
		color: var(--brand-muted);
		font-family: var(--font-mono, monospace);
	}

	.maturity-mini {
		font-size: 11px;
		padding: 1px 4px;
		border-radius: 4px;
	}

	.card-title {
		margin: 0 0 0.35rem;
		font-size: 13px;
		font-weight: 600;
		line-height: 1.35;
		color: var(--brand-ink);
		display: -webkit-box;
		-webkit-line-clamp: 2;
		-webkit-box-orient: vertical;
		overflow: hidden;
	}

	.card-reason {
		font-size: 11px;
		color: var(--brand-muted);
		margin-bottom: 0.5rem;
		line-height: 1.3;
	}

	.peek-hint {
		margin-top: auto;
		font-size: 11px;
		color: var(--brand-accent);
		font-weight: 500;
	}

	/* Andy Matuschak Sliding Pane */
	.drawer-backdrop {
		position: fixed;
		inset: 0;
		background: rgba(0, 0, 0, 0.35);
		backdrop-filter: blur(2px);
		z-index: 998;
		animation: fadeIn 0.2s ease;
	}

	@keyframes fadeIn {
		from { opacity: 0; }
		to { opacity: 1; }
	}

	.sliding-pane {
		position: fixed;
		top: 0;
		right: 0;
		bottom: 0;
		width: min(440px, 92vw);
		background: var(--brand-paper);
		border-left: 1px solid var(--brand-line);
		box-shadow: -8px 0 32px rgba(0, 0, 0, 0.16);
		z-index: 999;
		display: flex;
		flex-direction: column;
		animation: slideIn 0.25s cubic-bezier(0.16, 1, 0.3, 1);
	}

	@keyframes slideIn {
		from { transform: translateX(100%); }
		to { transform: translateX(0); }
	}

	.pane-header {
		padding: 1.25rem 1.5rem 1rem;
		border-bottom: 1px solid var(--brand-line);
	}

	.pane-meta-top {
		display: flex;
		align-items: center;
		justify-content: space-between;
		margin-bottom: 0.5rem;
	}

	.pane-kind-pill {
		font-size: 11px;
		color: var(--brand-muted);
		font-family: var(--font-mono, monospace);
	}

	.pane-close-btn {
		background: transparent;
		border: 1px solid var(--brand-line);
		border-radius: 6px;
		width: 28px;
		height: 28px;
		display: flex;
		align-items: center;
		justify-content: center;
		font-size: 13px;
		color: var(--brand-muted);
		cursor: pointer;
		transition: all 0.15s ease;
	}

	.pane-close-btn:hover {
		color: var(--brand-ink);
		border-color: var(--brand-accent);
	}

	.pane-title {
		margin: 0;
		font-size: 1.25rem;
		line-height: 1.35;
		color: var(--brand-ink);
		font-weight: 700;
	}

	.pane-body {
		padding: 1.25rem 1.5rem;
		flex: 1;
		overflow-y: auto;
		display: flex;
		flex-direction: column;
		gap: 1.15rem;
	}

	.pane-detail-row {
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}

	.detail-badge {
		display: inline-flex;
		align-items: center;
		gap: 0.3rem;
		padding: 0.2rem 0.6rem;
		border-radius: 6px;
		font-size: 11.5px;
		font-weight: 600;
	}

	.pane-cat-badge {
		font-size: 11px;
		padding: 2px 7px;
		border-radius: 4px;
		background: var(--brand-surface);
		border: 1px solid var(--brand-line);
		color: var(--brand-muted);
	}

	.pane-callout {
		padding: 0.75rem 0.9rem;
		background: var(--brand-surface);
		border-left: 3px solid var(--brand-accent);
		border-radius: 4px;
		font-size: 12px;
	}

	.pane-callout strong {
		color: var(--brand-accent);
		display: block;
		margin-bottom: 2px;
	}

	.pane-callout p {
		margin: 0;
		color: var(--brand-body);
	}

	.pane-tags-section {
		display: flex;
		flex-direction: column;
		gap: 0.4rem;
	}

	.tags-title {
		font-size: 11px;
		color: var(--brand-muted);
	}

	.tag-chips {
		display: flex;
		flex-wrap: wrap;
		gap: 0.35rem;
	}

	.chip {
		font-size: 11px;
		padding: 2px 6px;
		border-radius: 4px;
		background: var(--brand-surface);
		border: 1px solid var(--brand-line);
		color: var(--brand-body);
		font-family: var(--font-mono, monospace);
	}

	.pane-desc {
		margin: 0;
		font-size: 13px;
		line-height: 1.6;
		color: var(--brand-body);
	}

	.pane-footer {
		padding: 1rem 1.5rem;
		border-top: 1px solid var(--brand-line);
		background: var(--brand-surface);
	}

	.pane-cta-btn {
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 0.5rem;
		width: 100%;
		padding: 0.75rem 1rem;
		border-radius: 8px;
		background: var(--brand-accent);
		color: #ffffff;
		font-weight: 600;
		font-size: 13.5px;
		text-decoration: none;
		transition: opacity 0.15s ease;
	}

	.pane-cta-btn:hover {
		opacity: 0.9;
	}
</style>
