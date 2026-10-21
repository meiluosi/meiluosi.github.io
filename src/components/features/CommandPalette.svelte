<script lang="ts">
import { onMount } from "svelte";
import { LAB_ENTRIES } from "@/utils/digital-garden";
import {
	isHapticsEnabled,
	playLatchClick,
	playMechanicalTick,
	playShutterTick,
	playSlideTick,
	setHapticsEnabled,
} from "@/utils/haptics";

type PaletteKind = "post" | "en-post" | "page" | "action";
interface PaletteItem {
	id: string;
	title: string;
	titleEn?: string;
	href?: string;
	tags: string[];
	category: string;
	categoryEn?: string;
	description: string;
	descriptionEn?: string;
	kind: PaletteKind;
	lang?: "zh" | "en";
	icon?: string;
	shortcut?: string;
	handler?: () => void;
}
type Status = "idle" | "loading" | "ready" | "error";

const MAX_RESULTS = 10;

let dialogEl: HTMLDialogElement | undefined = $state<HTMLDialogElement>();
let inputEl: HTMLInputElement | undefined = $state<HTMLInputElement>();
let items = $state<PaletteItem[]>([]);
let query = $state("");
let activeIndex = $state(0);
let activeTab = $state<"all" | "action" | "lab" | "post">("all");
let english = $state(false);
let status = $state<Status>("idle");
let isOpen = $state(false);
let loading = false;
// Element that opened the palette; focus returns here on close.
let returnFocus: HTMLElement | null = null;

const builtinActions = $derived.by<PaletteItem[]>(() => {
	const soundOn = typeof window !== "undefined" ? isHapticsEnabled() : true;
	return [
		{
			id: "action-sound",
			title: soundOn ? "静音全站触觉与机械音效" : "开启全站触觉与机械音效",
			titleEn: soundOn
				? "Mute All Sound & Mechanical Haptics"
				: "Enable Sound & Mechanical Haptics",
			category: "快速控制",
			categoryEn: "Quick Control",
			tags: ["sound", "audio", "haptics", "音效", "声音", "静音"],
			description: "切换全站按键、齿轮与快门声效反馈",
			descriptionEn: "Toggle procedural click and friction audio feedback",
			kind: "action",
			icon: soundOn ? "🔊" : "🔇",
			shortcut: "> sound",
			handler: () => {
				const next = !isHapticsEnabled();
				setHapticsEnabled(next);
				if (next) playMechanicalTick(1.2);
				close();
			},
		},
		{
			id: "action-lang",
			title: english
				? "切换至中文语言模式 (zh-CN)"
				: "Switch to English Mode (en)",
			titleEn: english ? "Switch to Chinese (zh-CN)" : "Switch to English (en)",
			category: "快速控制",
			categoryEn: "Quick Control",
			tags: ["lang", "language", "chinese", "english", "双语", "中英"],
			description: "无缝切换全站主界面双语模式",
			descriptionEn: "Seamlessly toggle site-wide UI bilingual mode",
			kind: "action",
			icon: "🌐",
			shortcut: "> lang",
			handler: () => {
				const cur = window.location.pathname;
				let target = cur;
				if (english && cur.startsWith("/en/")) {
					target = cur.replace(/^\/en/, "") || "/";
				} else if (!english && !cur.startsWith("/en/")) {
					target = `/en${cur}`;
				}
				window.location.assign(target);
				close();
			},
		},
		{
			id: "action-random-note",
			title: "随机漫游：抽取一篇常青思想笔记",
			titleEn: "I'm Feeling Lucky: Jump to Random Note",
			category: "探索发现",
			categoryEn: "Exploration",
			tags: ["random", "lucky", "note", "随机", "漫游", "手气不错"],
			description: "在数字花园中偶遇一篇深度思考",
			descriptionEn: "Serendipitously explore a note in the digital garden",
			kind: "action",
			icon: "🎲",
			shortcut: "> random",
			handler: () => {
				const notes = items.filter(
					(it) => it.kind === "post" || it.kind === "en-post",
				);
				if (notes.length > 0) {
					const pick = notes[Math.floor(Math.random() * notes.length)];
					if (pick.href) window.location.assign(pick.href);
				}
				close();
			},
		},
		{
			id: "action-graph-3d",
			title: "打开全站 3D 知识星云天球",
			titleEn: "Open 3D Knowledge Cosmos & Orbits",
			category: "全站图谱",
			categoryEn: "Cosmos",
			tags: ["graph", "3d", "cosmos", "知识图谱", "星空", "天球"],
			description: "以三维引力轨道探索全站笔记与实验的拓扑联结",
			descriptionEn: "Explore interlinked notes in 3D celestial orbital space",
			kind: "action",
			icon: "🪐",
			shortcut: "> graph",
			handler: () => {
				window.location.assign("/graph/");
				close();
			},
		},
		// Labs
		...LAB_ENTRIES.map((lab) => ({
			id: `action-${lab.id}`,
			title: `实验沙盒：${lab.title}`,
			titleEn: `Lab: ${lab.titleEn}`,
			category: "交互实验室",
			categoryEn: "Interactive Labs",
			tags: ["lab", "experiment", "沙盒", "仿真", ...lab.tags],
			description: `直接启动可计算沙盒：${lab.category}`,
			descriptionEn: `Launch interactive sandbox: ${lab.category}`,
			kind: "action" as const,
			icon: "⚡",
			shortcut: `> ${lab.id.replace("lab-", "")}`,
			handler: () => {
				window.location.assign(lab.href);
				close();
			},
		})),
	];
});

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

interface HighlightSegment {
	text: string;
	match: boolean;
}

function highlightText(text: string, rawQuery: string): HighlightSegment[] {
	const terms = rawQuery
		.trim()
		.split(/\s+/)
		.filter((t) => t.length > 0)
		.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
	if (terms.length === 0 || !text) return [{ text, match: false }];
	const regex = new RegExp(`(${terms.join("|")})`, "gi");
	const parts = text.split(regex);
	const lowerTerms = new Set(
		rawQuery.trim().toLowerCase().split(/\s+/).filter(Boolean),
	);
	return parts
		.filter((part) => part.length > 0)
		.map((part) => ({
			text: part,
			match: lowerTerms.has(part.toLowerCase()),
		}));
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
	const shortcut = (item.shortcut ?? "").toLowerCase();
	let best = 0;
	for (const term of terms) {
		let score = 0;
		if (
			shortcut &&
			(shortcut.includes(term) ||
				shortcut.replace(">", "").trim().startsWith(term))
		) {
			score = 6;
		} else if (title.startsWith(term) || titleEn.startsWith(term)) {
			score = 5;
		} else if (title.includes(term) || titleEn.includes(term)) {
			score = 3;
		} else if (tags.includes(term) || category.includes(term)) {
			score = 2;
		} else if (description.includes(term)) {
			score = 1;
		}
		if (!score) return 0;
		if (score > best) best = score;
	}
	return best;
}

const results = $derived.by(() => {
	const rawQuery = query.trim();
	const isCommandMode = rawQuery.startsWith(">") || rawQuery.startsWith(":");
	const cleanQuery = isCommandMode ? rawQuery.slice(1).trim() : rawQuery;
	const terms = cleanQuery
		.toLowerCase()
		.split(/\s+/)
		.filter((term) => term.length > 0);
	const hasCjk = /[\u3400-\u9fff]/.test(cleanQuery);

	let pool: PaletteItem[] = [];
	if (isCommandMode || activeTab === "action") {
		pool = builtinActions;
	} else if (activeTab === "lab") {
		pool = builtinActions.filter((a) => a.id.startsWith("action-lab"));
	} else if (activeTab === "post") {
		pool = items.filter((it) => it.kind === "post" || it.kind === "en-post");
	} else {
		// All
		pool = [...builtinActions, ...items];
	}

	// Empty query: default destinations
	if (terms.length === 0) {
		if (isCommandMode || activeTab === "action") {
			return builtinActions.slice(0, MAX_RESULTS);
		}
		if (activeTab === "lab") {
			return builtinActions
				.filter((a) => a.id.startsWith("action-lab"))
				.slice(0, MAX_RESULTS);
		}
		if (activeTab === "post") {
			return items.filter((it) => it.kind === "post").slice(0, MAX_RESULTS);
		}
		// Default show quick actions + top pages
		const defaultActions = builtinActions.slice(0, 4);
		const defaultPages = items
			.filter((item) => item.kind === "page")
			.slice(0, 4);
		return [...defaultActions, ...defaultPages];
	}

	const ranked: {
		item: PaletteItem;
		score: number;
		langRank: number;
		kindRank: number;
	}[] = [];

	for (const item of pool) {
		if (english && !hasCjk && item.kind === "post" && item.titleEn) continue;
		const score = matchScore(item, terms);
		if (!score) continue;
		ranked.push({
			item,
			score,
			langRank: (english ? item.lang === "en" : item.lang === "zh") ? 0 : 1,
			kindRank: item.kind === "action" ? 0 : item.kind === "page" ? 1 : 2,
		});
	}

	ranked.sort(
		(a, b) =>
			b.score - a.score || a.kindRank - b.kindRank || a.langRank - b.langRank,
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
	playShutterTick();
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
	if (item.handler) {
		item.handler();
		return;
	}
	if (item.href) {
		playLatchClick();
		location.assign(item.href);
	}
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
		playMechanicalTick(1.15);
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

	const win = window as Window & { __paletteOpenPending?: boolean };
	if (win.__paletteOpenPending) {
		win.__paletteOpenPending = false;
		open();
	}

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
				placeholder={english ? "Search notes or type > for actions (e.g. > sound, > lab)…" : "搜索笔记，或输入 > 执行指令 (如 > 音效, > 实验)…"}
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

		<!-- Quick Filter Tabs -->
		<div class="palette-tabs" role="tablist">
			<button
				type="button"
				class="tab-btn"
				class:active={activeTab === "all"}
				onclick={() => {
					activeTab = "all";
					playMechanicalTick(1.0);
				}}
			>
				{english ? "All" : "全部"}
			</button>
			<button
				type="button"
				class="tab-btn"
				class:active={activeTab === "action"}
				onclick={() => {
					activeTab = "action";
					playMechanicalTick(1.05);
				}}
			>
				⚡ {english ? "Actions" : "快捷指令"}
			</button>
			<button
				type="button"
				class="tab-btn"
				class:active={activeTab === "lab"}
				onclick={() => {
					activeTab = "lab";
					playMechanicalTick(1.1);
				}}
			>
				⚙️ {english ? "Labs" : "计算沙盒"}
			</button>
			<button
				type="button"
				class="tab-btn"
				class:active={activeTab === "post"}
				onclick={() => {
					activeTab = "post";
					playMechanicalTick(1.15);
				}}
			>
				📄 {english ? "Notes" : "文章笔记"}
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
						class:is-action={item.kind === "action"}
						role="option"
						aria-selected={index === activeIndex}
						onclick={() => go(item)}
						onmouseenter={() => {
							activeIndex = index;
						}}
					>
						{#if item.icon}
							<span class="palette-option-icon" aria-hidden="true">{item.icon}</span>
						{/if}
						<div class="palette-option-main">
							<span class="palette-option-title">
								{#each highlightText(displayTitle(item), query) as seg}
									{#if seg.match}<mark class="palette-mark">{seg.text}</mark>{:else}{seg.text}{/if}
								{/each}
							</span>
							<span class="palette-option-meta">
								{#if item.kind === "action"}
									<span class="action-desc">{english ? (item.descriptionEn ?? item.description) : item.description}</span>
								{:else}
									{#each highlightText(metaLabel(item), query) as seg}
										{#if seg.match}<mark class="palette-mark">{seg.text}</mark>{:else}{seg.text}{/if}
									{/each}
								{/if}
							</span>
						</div>
						{#if item.shortcut}
							<kbd class="palette-shortcut-badge">{item.shortcut}</kbd>
						{/if}
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

	.palette-tabs {
		display: flex;
		align-items: center;
		gap: 6px;
		padding: 8px 14px;
		background: var(--brand-surface);
		border-bottom: 1px solid var(--brand-line);
		overflow-x: auto;
	}

	.tab-btn {
		font-size: 11.5px;
		padding: 3px 9px;
		border-radius: 999px;
		border: 1px solid var(--brand-line);
		background: var(--brand-paper);
		color: var(--brand-muted);
		cursor: pointer;
		white-space: nowrap;
		transition: all 0.15s ease;
	}

	.tab-btn:hover,
	.tab-btn.active {
		border-color: var(--brand-accent);
		color: var(--brand-accent);
		background: rgba(92, 122, 107, 0.1);
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
		display: flex;
		align-items: center;
		gap: 10px;
		padding: 9px 12px;
		border-radius: 9px;
		cursor: pointer;
	}

	.palette-option-icon {
		font-size: 16px;
		flex-shrink: 0;
	}

	.palette-option-main {
		display: flex;
		flex-direction: column;
		gap: 2px;
		flex: 1;
		min-width: 0;
	}

	.palette-shortcut-badge {
		font-family: var(--font-mono, monospace);
		font-size: 10.5px;
		padding: 2px 6px;
		border-radius: 4px;
		background: var(--brand-paper);
		border: 1px solid var(--brand-line);
		color: var(--brand-muted);
		flex-shrink: 0;
	}

	.action-desc {
		color: var(--brand-body);
		font-size: 11.5px;
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

	.palette-mark {
		background: var(--brand-peach);
		color: var(--brand-ink);
		padding: 0 2px;
		border-radius: 3px;
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
