<script lang="ts">
	import { onMount } from "svelte";
	import { selectCards, type SelectionRule, type CardScene } from "@/utils/rsi-experiments";
	import "./notebook.css";
	const uid = $props.id();
	let rule = $state<SelectionRule>("color");
	let scene = $state<CardScene>("familiar");
	let english = $state(false);
	let ready = $state(false);
	const say = (zh: string, en: string) => english ? en : zh;
	const cards = $derived(selectCards(rule, scene));
	const correct = $derived(cards.filter((card) => card.correct).length);
	const explanation = $derived(scene === "familiar"
		? say("在这一排卡片里，颜色甲、奇数位置和三角形恰好重合。三种规则都能全对；只看这个结果，还分不出它们依据了什么。", "In this row, color A, odd positions, and triangles coincide. All three rules get everything right. This result alone cannot tell us what each rule relies on.")
		: scene === "task"
			? say("题目现在要求圆形，旧规则却没有改变，三种规则都会全错。能抵抗某种外观变化，不等于会理解一个新目标。", "The task now asks for circles, but none of the rules has changed. All three get every decision wrong. Resistance to an appearance change does not imply an ability to understand a new goal.")
			: rule === "shape"
				? say("形状规则在这次变化中仍然有效，因为目标仍是三角形。这个小范围的成功只支持这一项判断，不能据此推断一般智能。", "The shape rule still works because the target is still triangles. This narrow success supports this particular claim; it does not establish general intelligence.")
				: (scene === "color" && rule === "color") || (scene === "position" && rule === "position")
					? say("原来好用的线索失效了。目标没变，改变的只是配色或排列。失败让规则依赖的东西显露出来。", "The useful-looking cue has broken. The target is unchanged; only color or order changed. Failure reveals what the rule depends on.")
					: say("这次变化还没有触及所选规则依赖的线索，所以它仍然全对。试试另一种变化，再判断这个成功有多稳固。", "This change leaves the chosen rule’s cue intact, so it still gets everything right. Try another change before deciding how robust that success is."));
	onMount(() => {
		ready = true;
		const locale = () => { english = document.documentElement.lang.startsWith("en"); };
		locale(); document.addEventListener("site-locale-change", locale);
		return () => document.removeEventListener("site-locale-change", locale);
	});
</script>

<div class="rsi-interactive">
	<div class="lens-controls">
		<label for={`${uid}-rule`}><span>{say("先选一条规则", "Choose a rule")}</span><select id={`${uid}-rule`} bind:value={rule} disabled={!ready}><option value="color">{say("选颜色甲", "Select color A")}</option><option value="position">{say("选第 1、3、5 个位置", "Select positions 1, 3, 5")}</option><option value="shape">{say("选三角形", "Select triangles")}</option></select></label>
		<label for={`${uid}-scene`}><span>{say("再换一种情境", "Change the situation")}</span><select id={`${uid}-scene`} bind:value={scene} disabled={!ready}><option value="familiar">{say("熟悉的卡片", "Familiar cards")}</option><option value="color">{say("只交换配色", "Swap only the colors")}</option><option value="position">{say("只打乱位置", "Reorder only the cards")}</option><option value="task">{say("新题目：改选圆形", "New task: select circles")}</option></select></label>
	</div>
	<p class="target">{say("这次的题目：", "The task: ")}<strong>{scene === "task" ? say("选出所有圆形", "select every circle") : say("选出所有三角形", "select every triangle")}</strong></p>
	<ol class="samples">
		{#each cards as card}
			<li class:selected={card.selected} class:wrong={!card.correct}>
				<span class="number">{card.position}</span>
				<svg viewBox="0 0 100 80" role="img" aria-label={card.triangle ? say("三角形", "Triangle") : say("圆形", "Circle")} class:color-a={card.colorA}>
					{#if card.triangle}<polygon points="50,10 88,70 12,70" />{:else}<circle cx="50" cy="40" r="29" />{/if}
				</svg>
				<span class="color-label">{card.colorA ? say("颜色甲", "Color A") : say("颜色乙", "Color B")}</span>
				<span class="decision">{card.selected ? say("选入", "Selected") : say("略过", "Skipped")} <span role="img" aria-label={card.correct ? say("判断正确", "Correct decision") : say("判断错误", "Incorrect decision")}>{card.correct ? "✓" : "×"}</span></span>
			</li>
		{/each}
	</ol>
	<div class="rsi-result" aria-live="polite" aria-atomic="true"><strong>{correct} / 6 {say("个选入或略过的判断符合题意", "selection decisions match the task")}</strong><p>{explanation}</p></div>
	<p class="rsi-note">{say("所有结果均由你选的固定规则即时计算。这里没有训练、识别模型或 AGI 评分；卡片只用于看清一种迁移与一种失效。", "Every result is computed from the fixed rule you select. No training, recognition model, or AGI score is involved; these cards illustrate a particular transfer and a particular failure.")}</p>
</div>

<style>
	.lens-controls{display:grid;grid-template-columns:1fr 1fr;gap:20px}.lens-controls label{display:flex;flex-direction:column;gap:9px;font-size:15px;min-width:0}.lens-controls select{width:100%;min-width:0;max-width:100%}.target{margin:28px 0 16px}.target strong{font-weight:500}.samples{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px;list-style:none;padding:0;margin:0}.samples li{display:flex;flex-direction:column;align-items:center;padding:12px 5px 14px;border:1px solid var(--brand-line);border-radius:6px;min-width:0;background:var(--brand-paper)}.samples li.selected{background:color-mix(in srgb,var(--brand-sage) 48%,var(--brand-paper))}.samples li.wrong{border-bottom:3px solid var(--brand-clay);padding-bottom:12px}.number{font:12px "JetBrains Mono",monospace;color:var(--brand-muted);align-self:flex-start;padding-left:7px}.samples svg{width:100%;height:80px;fill:var(--brand-clay)}.samples svg.color-a{fill:var(--brand-accent)}.color-label{font-size:12px;color:var(--brand-muted)}.decision{font-size:13px;margin-top:10px;line-height:1.6;text-align:center}.decision>span{font-size:16px;margin-left:4px}.rsi-note{margin-bottom:0}@media(max-width:680px){.samples{grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.samples svg{max-width:100px}}@media(max-width:520px){.lens-controls{grid-template-columns:1fr;gap:16px}.samples svg{height:66px}.decision{font-size:12px}.decision>span{margin-left:2px}}
</style>
