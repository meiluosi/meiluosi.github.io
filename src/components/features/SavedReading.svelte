<script lang="ts">
	import { onMount } from "svelte";
	import { readSaved, toggleSaved } from "@/utils/reading-storage";
	import type { ResearchNote } from "@/utils/research-content";
	let { notes }: { notes: ResearchNote[] } = $props();
	let english = $state(false);
	let ids = $state<string[]>([]);
	let loaded = $state(false);
	let error = $state(false);
	let removed = $state<string | null>(null);
	const saved = $derived(ids.flatMap((id) => { const note = notes.find((entry) => entry.id === id); return note ? [note] : []; }));
	const missing = $derived(ids.length - saved.length);
	function sync() { english = document.documentElement.lang.startsWith("en"); try { ids = readSaved(); error = false; } catch { error = true; } loaded = true; }
	function remove(id: string) { try { toggleSaved(id); removed = id; } catch { error = true; } }
	function undo() { if (!removed) return; try { if (!readSaved().includes(removed)) toggleSaved(removed); removed = null; } catch { error = true; } }
	onMount(() => { sync(); window.addEventListener("storage", sync); window.addEventListener("reading-saved-change", sync); document.addEventListener("site-locale-change", sync); return () => { window.removeEventListener("storage", sync); window.removeEventListener("reading-saved-change", sync); document.removeEventListener("site-locale-change", sync); }; });
</script>
<section class="saved-reading" aria-label={english ? "Your saved notes" : "你收藏的笔记"}>
	<div class="saved-status" role="status">{#if error}<p>{english ? "Browser storage is unavailable. Saved notes could not be read or updated." : "浏览器存储不可用，无法读取或更新收藏。"}</p>{:else if !loaded}<p>{english ? "Loading saved notes…" : "正在读取收藏……"}</p>{:else}<span>{saved.length} {english ? "saved notes" : "篇收藏"}</span>{/if}{#if removed}<span>{english ? "Removed from saved reading." : "已移出收藏。"} <button type="button" onclick={undo}>{english ? "Undo" : "撤销"}</button></span>{/if}</div>
	{#if loaded && !error && saved.length === 0}<div class="empty"><span aria-hidden="true">＋</span><h2>{english ? "A place to pick up the thread." : "给下一次阅读，留个位置。"}</h2><p>{english ? "Use “Save for later” at the top of a note. Your list will be waiting here in this browser." : "在文章顶部点击「留待下次阅读」，下次来到这里，就能继续沿着这个问题读下去。"}</p><a href="/paths/">{english ? "Find a reading path" : "挑一条阅读路线"} ↗</a></div>{/if}
	<ul>{#each saved as note (note.id)}<li><a href={english ? note.hrefEn : note.href}><h2>{english ? note.titleEn : note.title}</h2><p>{english ? note.descriptionEn : note.description}</p><span>{english ? "Continue reading" : "继续阅读"} ↗</span></a><button type="button" onclick={() => remove(note.id)} aria-label={english ? `Remove ${note.titleEn} from saved reading` : `从收藏移除《${note.title}》`}>{english ? "Remove" : "移除"}</button></li>{/each}</ul>
	{#if missing}<p class="missing">{english ? `${missing} saved note(s) are no longer available.` : `${missing} 篇已收藏的笔记目前不可访问。`}</p>{/if}
</section>
<style>.saved-status{display:flex;justify-content:space-between;gap:14px;flex-wrap:wrap;font:11px/1.8 Inter,sans-serif;color:var(--brand-muted);min-height:32px}.saved-status button{margin-left:8px;color:var(--brand-accent);text-decoration:underline}.empty{border:1px solid var(--brand-line);background:var(--brand-surface);border-radius:16px;padding:45px 30px;margin:16px 0}.empty>span{font:42px Georgia;color:var(--brand-accent)}.empty h2{font-size:26px;font-weight:400}.empty p{font-size:13px;line-height:2;color:var(--brand-body);max-width:540px}a{color:var(--brand-accent);text-decoration:none}.empty a{display:inline-block;font-size:12px;margin-top:20px}ul{list-style:none;padding:0;margin:16px 0}li{display:flex;justify-content:space-between;align-items:center;gap:20px;padding:28px 0;border-bottom:1px solid var(--brand-line)}li>a{flex:1}li h2{font:400 24px/1.5 Georgia,"Songti SC",serif;color:var(--brand-ink);margin:0 0 12px}li p{font:13px/1.9 Inter,sans-serif;color:var(--brand-body)}li span{font-size:11px}button{background:none;border:0;color:var(--brand-muted);font:11px Inter,sans-serif;cursor:pointer}li>button{padding:12px 8px;flex-shrink:0}.missing{font-size:12px;color:var(--brand-muted)}@media(max-width:600px){li{align-items:flex-start}li h2{font-size:20px}.empty{padding:28px}.empty h2{font-size:23px}}</style>
