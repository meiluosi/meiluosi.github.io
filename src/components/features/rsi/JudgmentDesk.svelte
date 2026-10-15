<script lang="ts">
	import { onMount } from "svelte";
	import { evaluateRepair, REPAIR_CARDS } from "@/utils/rsi-experiments";
	import "./notebook.css";
	const uid = $props.id();
	let missCost = $state(1);
	let english = $state(false);
	let ready = $state(false);
	const say = (zh: string, en: string) => english ? en : zh;
	const broad = $derived(evaluateRepair(4, missCost));
	const narrow = $derived(evaluateRepair(8, missCost));
	const verdict = $derived(broad.cost === narrow.cost
		? say("按你给出的代价，两种做法打平。它们留下的工作和遗漏，仍然不一样。", "The two approaches tie under your chosen costs. They still leave different work and different omissions.")
		: broad.cost < narrow.cost
			? say("按你给出的代价，“多检查一些”更合适。它多检查三本完好的书，却没有漏掉需要修补的那一本。", "Under your chosen costs, “Inspect more” is preferable. It checks three intact books unnecessarily but misses none that need repair.")
			: say("按你给出的代价，“只查明显的”更合适。它省下三次多余检查，但漏掉了书 E。这笔账没有替你决定遗漏是否可以接受。", "Under your chosen costs, “Inspect obvious cases” is preferable. It avoids three unnecessary checks but misses book E. The calculation does not decide whether that omission is acceptable."));
	onMount(() => {
		ready = true;
		const locale = () => { english = document.documentElement.lang.startsWith("en"); };
		locale(); document.addEventListener("site-locale-change", locale);
		return () => document.removeEventListener("site-locale-change", locale);
	});
</script>

<div class="rsi-interactive">
	<p class="rsi-prompt">{say("把八本书送到修补台之前，先决定愿意承担哪一种差错。", "Before sending eight books to a repair desk, decide which kind of mistake matters more.")}</p>
	<label class="cost-control" for={`${uid}-cost`}><span>{say("漏掉一本需要修补的书，计多少份代价？", "How much should a missed repair cost?")} <output>{missCost}</output></span><input id={`${uid}-cost`} type="range" min="1" max="5" step="1" bind:value={missCost} disabled={!ready} /></label>
	<p class="rsi-note">{say("多检查一本完好的书 = 1 份代价。下面的数字是手工构造的教学记录，不是模型预测或图书馆实测。", "One unnecessary check = 1 cost unit. The records below are hand-authored teaching examples, not model predictions or library measurements.")}</p>
	<div class="comparison">
		{#each [{ title: say("多检查一些", "Inspect more"), criterion: say("线索分 ≥ 4 就检查", "Inspect when clue score ≥ 4"), value: broad }, { title: say("只查明显的", "Inspect obvious cases"), criterion: say("线索分 ≥ 8 才检查", "Inspect when clue score ≥ 8"), value: narrow }] as policy}
			<div class="policy"><h3>{policy.title}</h3><p class="criterion">{policy.criterion}</p><dl><div><dt>{say("多余检查", "Extra checks")}</dt><dd>{policy.value.extraChecks}</dd></div><div><dt>{say("漏掉修补", "Missed repairs")}</dt><dd>{policy.value.missedRepairs}</dd></div><div class="total"><dt>{say("按你的标准计算", "Cost by your criteria")}</dt><dd>{policy.value.cost}</dd></div></dl><p class="formula">{policy.value.extraChecks} × 1 + {policy.value.missedRepairs} × {missCost} = {policy.value.cost}</p></div>
		{/each}
	</div>
	<div class="rsi-result" aria-live="polite" aria-atomic="true"><p>{verdict}</p><p>{say("你改变了“更好”的含义，没有改变任何一本书。", "You changed what “better” means, without changing a single book.")}</p></div>
	<details><summary>{say("翻开八张记录，核对这笔账", "Open the eight records and check the calculation")}</summary><p class="rsi-note">{say("线索分是本例预设的分拣依据：它可能提示破损，也可能误导。实际状态同样由本例预设。计价只含多余检查和漏检，未含所有工作的完整成本。", "The clue score is a preset sorting cue: it may suggest damage or mislead. Actual condition is also preset. The calculation includes only extra checks and missed repairs, not the full cost of all work.")}</p><div class="records"><table><thead><tr><th scope="col">{say("书", "Book")}</th><th scope="col">{say("线索分", "Clue")}</th><th scope="col">{say("实际状态", "Condition")}</th></tr></thead><tbody>{#each REPAIR_CARDS as card}<tr><th scope="row">{card.id}</th><td>{card.clue}</td><td>{card.needsRepair ? say("需要修补", "Needs repair") : say("完好", "Intact")}</td></tr>{/each}</tbody></table></div></details>
</div>

<style>
	.cost-control{display:block;margin-top:25px;font-size:16px;line-height:1.8}.cost-control>span{display:flex;gap:15px;justify-content:space-between;align-items:start}.cost-control output{font:25px Georgia,serif;color:var(--brand-clay);padding:0 8px}.cost-control input{width:100%;height:38px;margin:8px 0 0;accent-color:var(--brand-clay);cursor:pointer}.comparison{display:grid;grid-template-columns:1fr 1fr;gap:30px;margin:28px 0}.policy{border-top:1px solid var(--brand-line);padding-top:12px;min-width:0}.policy h3{font:400 23px/1.5 Georgia,"Songti SC",serif;margin:7px 0}.policy .criterion{font-size:13px;margin:6px 0 20px;color:var(--brand-muted)}dl{margin:0}dl>div{display:flex;justify-content:space-between;gap:15px;font-size:14px;line-height:1.8;margin:9px 0}dd{margin:0;font-variant-numeric:tabular-nums}.total{border-top:1px solid var(--brand-line);padding-top:12px}.total dd{font-size:22px;line-height:1.1;color:var(--brand-clay)}.policy .formula{font:13px/1.8 "JetBrains Mono",monospace;color:var(--brand-muted);margin:14px 0 0}details{margin-top:25px;font-size:14px;line-height:1.8}summary{cursor:pointer;padding:10px 0;color:var(--brand-accent)}.records{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:14px;text-align:left}th,td{padding:10px 12px;border-bottom:1px solid var(--brand-line)}th{font-weight:500}@media(max-width:520px){.comparison{grid-template-columns:1fr;gap:20px}.policy h3{font-size:22px}.cost-control{font-size:16px}th,td{padding:9px 6px}}
</style>
