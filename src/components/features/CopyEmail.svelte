<script lang="ts">
	import { onMount } from "svelte";
	let { email }: { email: string } = $props();
	let english = $state(false);
	let status = $state<"idle" | "copied" | "failed">("idle");
	onMount(() => {
		const sync = () => { english = document.documentElement.lang.startsWith("en"); };
		sync();
		document.addEventListener("site-locale-change", sync);
		return () => document.removeEventListener("site-locale-change", sync);
	});
	async function copy() {
		try { await navigator.clipboard.writeText(email); status = "copied"; }
		catch { status = "failed"; }
	}
</script>

<div class="copy-row">
	<button type="button" onclick={copy}>{english ? "Copy email address" : "复制邮箱地址"} <span aria-hidden="true">⧉</span></button>
	<span role="status" aria-live="polite">{status === "copied" ? (english ? "Copied. Ready to paste." : "已复制，可以粘贴了。") : status === "failed" ? (english ? "Please select and copy the address above." : "请长按或选中上方邮箱地址复制。") : ""}</span>
</div>

<style>
	.copy-row{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-top:15px;min-height:35px;font-size:11px;color:var(--brand-body)}button{display:inline-flex;gap:15px;border:1px solid var(--brand-line);border-radius:22px;background:var(--brand-surface);color:var(--brand-accent);padding:10px 16px;cursor:pointer;font:inherit}button:hover{border-color:var(--brand-accent)}button:focus-visible{outline:2px solid var(--brand-accent);outline-offset:3px}
</style>
