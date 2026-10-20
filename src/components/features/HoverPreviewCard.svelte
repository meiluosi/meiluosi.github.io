<script lang="ts">
import { onMount } from "svelte";

interface PreviewData {
	id: string;
	title: string;
	titleEn?: string;
	href: string;
	description: string;
	category?: string;
	tags: string[];
	kind: string;
	lang: string;
}

let english = $state(false);
let visible = $state(false);
let activeData = $state<PreviewData | null>(null);
let cardX = $state(0);
let cardY = $state(0);

let itemsCache: PreviewData[] | null = null;
let hoverTimer: ReturnType<typeof setTimeout> | undefined;
let leaveTimer: ReturnType<typeof setTimeout> | undefined;

async function loadIndex(): Promise<PreviewData[]> {
	if (itemsCache) return itemsCache;
	try {
		const res = await fetch("/api/palette.json");
		if (res.ok) {
			const data = (await res.json()) as PreviewData[];
			itemsCache = data;
			return data;
		}
	} catch {}
	return [];
}

function normalizePath(urlStr: string): string {
	try {
		const u = new URL(urlStr, window.location.origin);
		return u.pathname.replace(/\/+$/, "") || "/";
	} catch {
		return urlStr.replace(/\/+$/, "") || "/";
	}
}

function handleLinkHover(target: HTMLAnchorElement): void {
	if (leaveTimer) clearTimeout(leaveTimer);
	if (hoverTimer) clearTimeout(hoverTimer);

	const href = target.getAttribute("href");
	if (!href || href.startsWith("#") || href.startsWith("mailto:")) return;

	// Only trigger for internal content links
	const isInternal =
		href.startsWith("/posts/") ||
		href.startsWith("/en/posts/") ||
		href.startsWith("/work/") ||
		href.startsWith("/lab/") ||
		href.startsWith("/notes/");
	if (!isInternal) return;

	const targetPath = normalizePath(href);
	// Don't preview current page
	if (targetPath === normalizePath(window.location.pathname)) return;

	hoverTimer = setTimeout(async () => {
		const items = await loadIndex();
		const match = items.find((item) => normalizePath(item.href) === targetPath);
		if (!match) return;

		activeData = match;
		const rect = target.getBoundingClientRect();
		const cardWidth = 320;
		const viewportWidth = window.innerWidth;

		// Center card under the link, clamp inside viewport
		let left = rect.left + rect.width / 2 - cardWidth / 2;
		left = Math.max(16, Math.min(viewportWidth - cardWidth - 16, left));

		let top = rect.bottom + 8;
		// If card would overflow bottom, display above
		if (top + 180 > window.innerHeight) {
			top = Math.max(16, rect.top - 190);
		}

		cardX = left;
		cardY = top;
		visible = true;
	}, 220);
}

function handleLinkLeave(): void {
	if (hoverTimer) clearTimeout(hoverTimer);
	leaveTimer = setTimeout(() => {
		visible = false;
	}, 260);
}

function handleCardEnter(): void {
	if (leaveTimer) clearTimeout(leaveTimer);
}

function handleCardLeave(): void {
	visible = false;
}

onMount(() => {
	const sync = () => {
		english = document.documentElement.lang.startsWith("en");
	};
	sync();
	document.addEventListener("site-locale-change", sync);

	const onPointerOver = (e: MouseEvent) => {
		const target = (e.target as Element | null)?.closest<HTMLAnchorElement>(
			".markdown-content a, #swup-container a",
		);
		if (target) {
			handleLinkHover(target);
		}
	};

	const onPointerOut = (e: MouseEvent) => {
		const target = (e.target as Element | null)?.closest<HTMLAnchorElement>(
			".markdown-content a, #swup-container a",
		);
		if (target) {
			handleLinkLeave();
		}
	};

	document.addEventListener("mouseover", onPointerOver);
	document.addEventListener("mouseout", onPointerOut);

	return () => {
		document.removeEventListener("site-locale-change", sync);
		document.removeEventListener("mouseover", onPointerOver);
		document.removeEventListener("mouseout", onPointerOut);
	};
});
</script>

{#if visible && activeData}
	<div
		class="hover-preview-card"
		style={`left: ${cardX}px; top: ${cardY}px;`}
		role="tooltip"
		onmouseenter={handleCardEnter}
		onmouseleave={handleCardLeave}
	>
		<div class="card-meta">
			<span class="kind-tag">
				{activeData.kind === "post"
					? english
						? "ARTICLE"
						: "技术文章"
					: activeData.kind === "en-post"
						? "ENGLISH OVERVIEW"
						: english
							? "RESEARCH PAGE"
							: "研究页面"}
			</span>
			{#if activeData.category}
				<span class="category-tag">{activeData.category}</span>
			{/if}
		</div>
		<h4 class="card-title">
			{english ? activeData.titleEn ?? activeData.title : activeData.title}
		</h4>
		{#if activeData.description}
			<p class="card-desc">{activeData.description}</p>
		{/if}
		{#if activeData.tags && activeData.tags.length > 0}
			<div class="card-tags">
				{#each activeData.tags.slice(0, 3) as tag}
					<span>#{tag}</span>
				{/each}
				<span class="cta-hint">{english ? "Open ↗" : "点击打开 ↗"}</span>
			</div>
		{/if}
	</div>
{/if}

<style>
	.hover-preview-card {
		position: fixed;
		z-index: 120;
		width: 320px;
		background: var(--brand-paper);
		border: 1px solid var(--brand-accent);
		border-radius: 12px;
		padding: 14px 16px;
		box-shadow: 0 16px 38px rgba(37, 60, 52, 0.16);
		color: var(--brand-ink);
		pointer-events: auto;
		animation: cardFadeIn 0.15s ease-out;
	}
	@keyframes cardFadeIn {
		from {
			opacity: 0;
			transform: translateY(4px);
		}
		to {
			opacity: 1;
			transform: translateY(0);
		}
	}
	.card-meta {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 6px;
	}
	.kind-tag {
		font: 9.5px var(--font-mono);
		letter-spacing: 0.08em;
		color: var(--brand-accent);
	}
	.category-tag {
		font: 10px var(--font-mono);
		color: var(--brand-muted);
	}
	.card-title {
		margin: 0 0 6px;
		font: 500 15px/1.35 var(--font-serif);
		color: var(--brand-ink);
	}
	.card-desc {
		margin: 0 0 10px;
		font-size: 11.5px;
		line-height: 1.6;
		color: var(--brand-body);
		display: -webkit-box;
		-webkit-line-clamp: 3;
		-webkit-box-orient: vertical;
		overflow: hidden;
	}
	.card-tags {
		display: flex;
		align-items: center;
		gap: 6px;
		font: 10px var(--font-mono);
		color: var(--brand-muted);
		border-top: 1px solid var(--brand-line);
		padding-top: 8px;
	}
	.cta-hint {
		margin-left: auto;
		color: var(--brand-accent);
		font-weight: 500;
	}
</style>
