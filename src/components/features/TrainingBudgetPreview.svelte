<script lang="ts">
import { onMount } from "svelte";
let size = $state(30);
let english = $state(false);
const stateGiB = $derived((size * 1e6 * 16) / 1024 ** 3);
const t = (zh: string, en: string): string => (english ? en : zh);
onMount(() => {
	const sync = () => {
		english = document.documentElement.lang.startsWith("en");
	};
	sync();
	document.addEventListener("site-locale-change", sync);
	return () => document.removeEventListener("site-locale-change", sync);
});
</script>

<div class="budget-preview">
	<fieldset><legend>{t("先试一个预算：模型变大，内存呢？", "Try a budget: how does memory scale?")}</legend>
		{#each [30, 100, 500] as scale}<label class:active={size === scale}><input type="radio" name="home-model-scale" value={scale} bind:group={size} />{scale === 500 ? "0.5B" : `${scale}M`}</label>{/each}
	</fieldset>
	<p class="budget-output" aria-live="polite"><strong>{stateGiB.toFixed(2)} <small>GiB</small></strong><span>{t("仅 FP32 权重、梯度和 Adam 状态", "FP32 weights, gradients, and Adam states only")}</span></p>
	<p class="budget-boundary">{t("估算下界 = 参数量 × 16 bytes；激活、参考模型与系统内存另计。放得下权重，还不能证明能完成训练。", "Lower bound = parameters × 16 bytes; activations, reference model, and system memory are extra. Fitting weights does not establish training feasibility.")}</p>
</div>

<style>
	.budget-preview{border-top:1px solid var(--brand-line);padding:20px 0 0;margin-top:24px;max-width:450px}fieldset{padding:0;margin:0;border:0;display:flex;gap:8px;flex-wrap:wrap}legend{font-size:12px;color:var(--brand-body);padding:0;margin-bottom:12px;line-height:1.8}label{display:inline-flex;gap:6px;align-items:center;border:1px solid var(--brand-line);border-radius:18px;padding:7px 12px;font:11px monospace;cursor:pointer;background:var(--brand-paper);color:var(--brand-accent)}label.active{background:var(--brand-peach);border-color:var(--brand-clay)}input{accent-color:var(--brand-accent);margin:0}label:has(input:focus-visible){outline:2px solid var(--brand-accent);outline-offset:3px}.budget-output{display:flex;align-items:center;gap:15px;margin:18px 0 10px}.budget-output strong{white-space:nowrap;font:28px/1.2 Georgia,serif;color:var(--brand-ink)}.budget-output small{font:11px monospace}.budget-output>span{font-size:11px;line-height:1.6;color:var(--brand-body)}.budget-boundary{font-size:11px;line-height:1.8;color:var(--brand-muted);margin:0}
</style>
