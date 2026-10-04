<script lang="ts">
	import { onMount } from "svelte";
	import { readSaved, toggleSaved } from "@/utils/reading-storage";
	let { sourceId, initialEnglish = false }: { sourceId: string; initialEnglish?: boolean } = $props();
	let english = $state(initialEnglish);
	let saved = $state(false);
	let available = $state(false);
	let error = $state(false);
	function sync() { english = document.documentElement.lang.startsWith("en"); try { saved = readSaved().includes(sourceId); available = true; } catch { available = false; error = true; } }
	function toggle() { try { saved = toggleSaved(sourceId); error = false; } catch { error = true; } }
	onMount(() => { sync(); window.addEventListener("storage", sync); window.addEventListener("reading-saved-change", sync); document.addEventListener("site-locale-change", sync); return () => { window.removeEventListener("storage", sync); window.removeEventListener("reading-saved-change", sync); document.removeEventListener("site-locale-change", sync); }; });
</script>
<div class="save-note"><button type="button" disabled={!available} aria-pressed={saved} onclick={toggle}>{saved ? "✓" : "+"} {saved ? (english ? "Saved for later" : "已加入阅读收藏") : (english ? "Save for later" : "留待下次阅读")}</button><a href="/reading/">{english ? "Saved reading" : "阅读收藏"} ↗</a><span role="status">{error ? (english ? "Browser storage is unavailable." : "浏览器存储不可用，暂时无法保存。") : saved ? (english ? "Saved on this browser." : "已保存在当前浏览器。") : ""}</span></div>
<style>.save-note{display:flex;align-items:center;flex-wrap:wrap;gap:12px 18px;padding:12px 0 22px;font:11px/1.8 Inter,sans-serif;color:var(--brand-muted)}button{padding:8px 14px;border:1px solid var(--brand-line);border-radius:24px;background:var(--brand-surface);color:var(--brand-accent);font:inherit;cursor:pointer}button[aria-pressed=true]{background:var(--brand-peach);color:var(--brand-ink)}button:disabled{opacity:.5;cursor:default}a{color:var(--brand-accent);text-decoration:none}span{font-size:10px}</style>
