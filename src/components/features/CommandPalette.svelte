<script lang="ts">
import { onMount } from "svelte";

type PaletteKind = "post" | "en-post" | "page";
interface PaletteItem {
	id: string;
	title: string;
	titleEn?: string;
	href: string;
	tags: string[];
	category: string;
	description: string;
	kind: PaletteKind;
	lang: "zh" | "en";
}
type Status = "idle" | "loading" | "ready" | "error";

const MAX_RESULTS = 8;

let dialogEl: HTMLDialogElement | undefined = $state<HTMLDialogElement>();
let inputEl: HTMLInputElement | undefined = $state<HTMLInputElement>();
let items = $state<PaletteItem[]>([]);
let query = $state("");
let activeIndex = $state(0);
let english = $state(false);
let status = $state<Status>("idle");
let isOpen = $state(false);
let loading = false;
// Element that opened the palette; focus returns here on close.
let returnFocus: HTMLElement | null = null;

const copy = $derived(
	english
		? {
				label: "Search pages and articles",
				placeholder: "Search pages, notes, and tags…",
				title: "Search",
				loading: "Loading the index…",
				empty: "No matches. Try another keyword, or open all notes.",
				error: "The search index could not load.",
				retry: "Retry",
				fallback: "Browse all notes",
				hint: "↑↓ move · Enter open · Esc close",
				listLabel: "Search results",
				close: "Close search",
				page: "Page",
				post: "Article",
				enPost: "English overview",
			}
		: {
				label: "搜索页面与文章",
				placeholder: "搜索页面、笔记与标签…",
				title: "搜索",
				loading: "正在载入索引…",
				empty: "没有匹配的结果，换个关键词，或打开全部笔记。",
				error: "搜索索引载入失败。",
				retry: "重试",
				fallback: "打开全部笔记",
				hint: "↑↓ 移动 · Enter 打开 · Esc 关闭",
				listLabel: "搜索结果",
				close: "关闭搜索",
				page: "页面",
				post: "文章",
				enPost: "英文概览",
			},
);

async function load(): Promise<void> {
	if (items.length > 0 || loading) return;
	loading = true;
	status = "loading";
	try {
		const response = await fetch("/api/palette.json");
		if (!response.ok) throw new Error(`HTTP ${response.status}`);
		const data: unknown = await response.json();
		if (!Array.isArray(data)) throw new Error("Unexpected palette payload");
		items = data as PaletteItem[];
		status = "ready";
	} catch {
		// A failed fetch keeps status "error"; the next open retries because
		// items stays empty.
		status = "error";
	} finally {
		loading = false;
	}
}

function displayTitle(item: PaletteItem): string {
	return english ? (item.titleEn ?? item.title) : item.title;
}

function metaLabel(item: PaletteItem): string {
	const kind =
		item.kind === "page"
			? copy.page
			: item.kind === "en-post"
				? copy.enPost
				: copy.post;
	const tags = item.tags
		.slice(0, 2)
		.map((tag) => `#${tag}`)
		.join(" ");
	return [item.category || kind, tags].filter(Boolean).join(" · ");
}

// 0 means "no match"; higher is better. Every term must hit something.
function matchScore(item: PaletteItem, terms: string[]): number {
	const title = item.title.toLowerCase();
	const titleEn = (item.titleEn ?? "").toLowerCase();
	const tags = item.tags.join(" ").toLowerCase();
	const category = item.category.toLowerCase();
	const description = item.description.toLowerCase();
	let best = 0;
	for (const term of terms) {
		let score = 0;
		if (title.startsWith(term) || titleEn.startsWith(term)) score = 4;
		else if (title.includes(term) || titleEn.includes(term)) score = 3;
		else if (tags.includes(term) || category.includes(term)) score = 2;
		else if (description.includes(term)) score = 1;
		if (!score) return 0;
		if (score > best) best = score;
	}
	return best;
}

const results = $derived.by(() => {
	const rawQuery = query.trim();
	const terms = rawQuery
		.toLowerCase()
		.split(/\s+/)
		.filter((term) => term.length > 0);
	// A Chinese query still searches Chinese titles in English mode.
	const hasCjk = /[\u3400-\u9fff]/.test(rawQuery);
	// Empty query: the pages are the palette's default destinations.
	if (terms.length === 0) {
		return items.filter((item) => item.kind === "page").slice(0, MAX_RESULTS);
	}
	const ranked: {
		item: PaletteItem;
		score: number;
		langRank: number;
		kindRank: number;
	}[] = [];
	for (const item of items) {
		// In English mode the translated entry replaces its Chinese source,
		// so the two do not show up as duplicate rows.
		if (english && !hasCjk && item.kind === "post" && item.titleEn) continue;
		const score = matchScore(item, terms);
		if (!score) continue;
		ranked.push({
			item,
			score,
			langRank: (english ? item.lang === "en" : item.lang === "zh") ? 0 : 1,
			kindRank: item.kind === "page" ? 0 : 1,
		});
	}
	ranked.sort(
		(a, b) =>
			b.score - a.score || a.langRank - b.langRank || a.kindRank - b.kindRank,
	);
	return ranked.slice(0, MAX_RESULTS).map((entry) => entry.item);
});

function open(): void {
	const element = dialogEl;
	if (!element || element.open) return;
	returnFocus =
		document.activeElement instanceof HTMLElement
			? document.activeElement
			: null;
	query = "";
	activeIndex = 0;
	status = items.length > 0 ? "ready" : "idle";
	element.showModal();
	isOpen = true;
	void load();
	requestAnimationFrame(() => inputEl?.focus());
}

function close(): void {
	if (dialogEl?.open) dialogEl.close();
}

function toggle(): void {
	if (isOpen) close();
	else open();
}

function handleClose(): void {
	isOpen = false;
	const target = returnFocus;
	returnFocus = null;
	if (target && target !== document.body && target.isConnected) {
		target.focus();
	}
}

// Backdrop clicks land on the dialog element itself; clicks inside the shell
// are stopped there by the browser's event target, so compare coordinates.
function handleDialogClick(event: MouseEvent): void {
	const element = dialogEl;
	if (!element || event.target !== element) return;
	const rect = element.getBoundingClientRect();
	const outside =
		event.clientX < rect.left ||
		event.clientX > rect.right ||
		event.clientY < rect.top ||
		event.clientY > rect.bottom;
	if (outside) close();
}

function go(item: PaletteItem): void {
	location.assign(item.href);
}

function handleInputKeydown(event: KeyboardEvent): void {
	// Do not hijack Enter/arrows while an IME composition is running.
	if (event.isComposing || event.keyCode === 229) return;
	const count = results.length;
	if (event.key === "ArrowDown" || event.key === "ArrowUp") {
		event.preventDefault();
		if (!count) return;
		const step = event.key === "ArrowDown" ? 1 : -1;
		activeIndex = (activeIndex + step + count) % count;
		return;
	}
	if (event.key === "Home") {
		event.preventDefault();
		activeIndex = 0;
		return;
	}
	if (event.key === "End") {
		event.preventDefault();
		activeIndex = Math.max(0, count - 1);
		return;
	}
	if (event.key === "Enter") {
		event.preventDefault();
		const item = results[activeIndex];
		if (item) go(item);
	}
}

onMount(() => {
	const onKeydown = (event: KeyboardEvent): void => {
		if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
			event.preventDefault();
			toggle();
			return;
		}
		if (event.key !== "/" || event.metaKey || event.ctrlKey || event.altKey) {
			return;
		}
		const target = event.target;
		const editing =
			target instanceof HTMLElement &&
			(target.closest("input, textarea, select") !== null ||
				target.isContentEditable);
		if (editing || isOpen) return;
		event.preventDefault();
		open();
	};
	const onPaletteOpen = (): void => open();
	const onLocaleChange = (): void => {
		english = document.documentElement.lang.startsWith("en");
	};

	english = document.documentElement.lang.startsWith("en");
	document.addEventListener("keydown", onKeydown);
	window.addEventListener("palette:open", onPaletteOpen);
	document.addEventListener("site-locale-change", onLocaleChange);

	return () => {
		document.removeEventListener("keydown", onKeydown);
		window.removeEventListener("palette:open", onPaletteOpen);
		document.removeEventListener("site-locale-change", onLocaleChange);
	};
});

// Keep the highlighted row inside the list and valid after filtering.
$effect(() => {
	const count = results.length;
	if (activeIndex >= count) activeIndex = 0;
});
$effect(() => {
	if (!isOpen || !results[activeIndex]) return;
	document
		.getElementById(`palette-option-${activeIndex}`)
		?.scrollIntoView({ block: "nearest" });
});
</script>

<dialog
	bind:this={dialogEl}
	class="palette"
	aria-labelledby="palette-title"
	onclick={handleDialogClick}
	onclose={handleClose}
>
	<div class="palette-shell">
		<h2 id="palette-title" class="palette-sr-only">{copy.title}</h2>
		<div class="palette-field">
			<svg class="palette-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true">
				<circle cx="11" cy="11" r="7"></circle>
				<path d="m20 20-3.6-3.6" stroke-linecap="round"></path>
			</svg>
			<input
				bind:this={inputEl}
				bind:value={query}
				type="text"
				role="combobox"
				aria-expanded={isOpen && status === "ready" && results.length > 0}
				aria-controls="palette-list"
				aria-autocomplete="list"
				aria-activedescendant={results[activeIndex] ? `palette-option-${activeIndex}` : undefined}
				aria-label={copy.label}
				placeholder={copy.placeholder}
				autocomplete="off"
				spellcheck="false"
				oninput={() => {
					activeIndex = 0;
				}}
				onkeydown={handleInputKeydown}
			/>
			<button type="button" class="palette-close" onclick={close} aria-label={copy.close}>
				<kbd>Esc</kbd>
			</button>
		</div>

		{#if status === "error"}
			<div class="palette-note" role="status">
				<p>{copy.error}</p>
				<p class="palette-note-actions">
					<button type="button" onclick={() => void load()}>{copy.retry}</button>
					<a href="/notes/">{copy.fallback}</a>
				</p>
			</div>
		{:else if status === "loading" || status === "idle"}
			<p class="palette-note" role="status">{copy.loading}</p>
		{:else if results.length === 0}
			<div class="palette-note" role="status">
				<p>{copy.empty}</p>
				<p class="palette-note-actions"><a href="/notes/">{copy.fallback}</a></p>
			</div>
		{:else}
			<ul id="palette-list" class="palette-list" role="listbox" aria-label={copy.listLabel}>
				{#each results as item, index (item.id)}
					<!-- Keyboard users move the active option with the combobox input, which keeps focus. -->
					<!-- svelte-ignore a11y_click_events_have_key_events -->
					<li
						id={`palette-option-${index}`}
						class="palette-option"
						role="option"
						aria-selected={index === activeIndex}
						onclick={() => go(item)}
						onmouseenter={() => {
							activeIndex = index;
						}}
					>
						<span class="palette-option-title">{displayTitle(item)}</span>
						<span class="palette-option-meta">{metaLabel(item)}</span>
					</li>
				{/each}
			</ul>
		{/if}

		<div class="palette-foot" aria-hidden="true">
			<span>{copy.hint}</span>
		</div>
	</div>
</dialog>

<style>
	.palette {
		position: fixed;
		top: 15vh;
		left: 50%;
		translate: -50% 0;
		margin: 0;
		width: min(560px, 92vw);
		max-height: 70vh;
		display: none;
		padding: 0;
		border: 1px solid var(--brand-line);
		border-radius: 14px;
		background: var(--brand-paper);
		color: var(--brand-ink);
		box-shadow: 0 24px 70px rgb(37 60 52 / 22%);
		overflow: hidden;
	}

	.palette[open] {
		display: flex;
		flex-direction: column;
	}

	.palette::backdrop {
		background: rgb(25 40 34 / 28%);
	}

	.palette-sr-only {
		position: absolute;
		width: 1px;
		height: 1px;
		margin: -1px;
		padding: 0;
		overflow: hidden;
		clip-path: inset(50%);
		white-space: nowrap;
		border: 0;
	}

	.palette-field {
		display: flex;
		align-items: center;
		gap: 10px;
		padding: 12px 14px;
		border-bottom: 1px solid var(--brand-line);
	}

	.palette-field:focus-within {
		outline: 2px solid var(--brand-accent);
		outline-offset: -2px;
	}

	.palette-icon {
		width: 18px;
		height: 18px;
		flex-shrink: 0;
		color: var(--brand-muted);
	}

	.palette-field input {
		flex: 1;
		min-width: 0;
		border: 0;
		background: transparent;
		color: var(--brand-ink);
		font: inherit;
		font-size: 16px;
		padding: 4px 0;
		outline: none;
	}

	.palette-field input::placeholder {
		color: var(--brand-muted);
		opacity: 1;
	}

	.palette-close {
		flex-shrink: 0;
		border: 1px solid var(--brand-line);
		border-radius: 7px;
		background: var(--brand-surface);
		color: var(--brand-body);
		font: 11px/1.4 ui-monospace, SFMono-Regular, monospace;
		padding: 5px 8px;
		cursor: pointer;
	}

	.palette-close:focus-visible {
		outline: 2px solid var(--brand-accent);
		outline-offset: 2px;
	}

	.palette-list {
		margin: 0;
		padding: 6px;
		list-style: none;
		overflow-y: auto;
		overscroll-behavior: contain;
		max-height: 52vh;
	}

	.palette-option {
		display: grid;
		gap: 3px;
		padding: 9px 12px;
		border-radius: 9px;
		cursor: pointer;
	}

	.palette-option:hover {
		background: var(--brand-surface);
	}

	.palette-option[aria-selected="true"] {
		background: var(--brand-surface);
		box-shadow: inset 2px 0 0 var(--brand-accent);
	}

	.palette-option-title {
		font-size: 14px;
		line-height: 1.5;
		color: var(--brand-ink);
		overflow-wrap: anywhere;
	}

	.palette-option-meta {
		font-size: 11px;
		line-height: 1.5;
		color: var(--brand-muted);
		overflow-wrap: anywhere;
	}

	.palette-note {
		padding: 20px 18px;
		font-size: 13px;
		line-height: 1.9;
		color: var(--brand-body);
	}

	.palette-note p {
		margin: 0;
	}

	.palette-note-actions {
		display: flex;
		gap: 14px;
		margin-top: 10px;
		align-items: center;
	}

	.palette-note-actions a,
	.palette-note-actions button {
		border: 0;
		background: transparent;
		padding: 0;
		color: var(--brand-accent);
		font: inherit;
		font-size: 13px;
		text-decoration: underline;
		text-underline-offset: 3px;
		cursor: pointer;
	}

	.palette-foot {
		display: flex;
		justify-content: space-between;
		gap: 12px;
		padding: 8px 14px;
		border-top: 1px solid var(--brand-line);
		color: var(--brand-muted);
		font-size: 11px;
	}

	@media (max-width: 640px) {
		.palette {
			top: 8vh;
			width: 94vw;
			max-height: 80vh;
		}

		.palette-foot {
			display: none;
		}
	}

	@media (prefers-reduced-motion: reduce) {
		.palette-option {
			transition: none;
		}
	}
</style>
