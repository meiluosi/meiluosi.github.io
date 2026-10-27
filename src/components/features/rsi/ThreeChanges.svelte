<script lang="ts">
import { onMount } from "svelte";
import "./notebook.css";

let chosen = $state(0);
let english = $state(false);
let ready = $state(false);
const say = (zh: string, en: string) => (english ? en : zh);
const steps = $derived([
	{
		name: say("多想几步", "Search a little longer"),
		changed: say("搜索统计", "Search statistics"),
		description: say(
			"沿着同一套搜索规则再走几步，更新访问次数与回报估计。眼前这道题可能做得更好；这次操作没有修改模型参数，也没有改写搜索算法。",
			"Run the same search rules for longer, updating visit counts and return estimates. This problem may be solved better; this operation does not change model parameters or rewrite the search algorithm.",
		),
		question: say(
			"换一道题后，留下的是哪些知识？又有哪些统计需要重新积累？",
			"On a new problem, what knowledge remains, and which statistics must be gathered again?",
		),
		href: "/lab/mcts/",
		link: say(
			"在搜索树里多走几步",
			"Take a few more steps through a search tree",
		),
	},
	{
		name: say("从反馈中学习", "Learn from feedback"),
		changed: say("模型参数", "Model parameters"),
		description: say(
			"用数据和反馈更新参数，改变模型之后的预测或决策。它可以从经验中学会一项技能；这里采用的训练规则与评价方式仍由人预先设定。",
			"Update parameters using data and feedback, changing later predictions or decisions. A model can acquire a skill from experience; the training rule and evaluation procedure here are still specified by people.",
		),
		question: say(
			"它学到了可迁移的方法，还是数据中恰好好用的线索？",
			"Has it acquired a transferable method, or a cue that happens to work in the data?",
		),
		href: "/lab/model-pipeline/?stage=pretrain",
		link: say(
			"理解训练目标与参数更新",
			"Understand training objectives and parameter updates",
		),
	},
	{
		name: say("改变改进的方法", "Change how improvement works"),
		changed: say(
			"产生、筛选与采用改进的过程",
			"How improvements are proposed, tested, and adopted",
		),
		description: say(
			"让系统提出并检验对工具、搜索策略或学习流程的修改，改变它下一次尝试改进的方式。要进一步讨论 RSI，还要检验这些修改是否增强后续产生有效改进的能力，而不只看当前任务得分。",
			"Let a system propose and test changes to its tools, search strategy, or learning process, altering how it next attempts improvement. To investigate RSI, we must also test whether those changes enhance its subsequent ability to produce useful improvements, beyond its score on the current task.",
		),
		question: say(
			"下一轮是否真的更会改进？比较时，预算、任务与评价标准保持了什么？",
			"Is the next iteration actually better at improving? What remains comparable in its budget, tasks, and evaluation?",
		),
		href: "/work/#causal-inference",
		link: say("查看因果评估研究", "Causal evaluation research"),
	},
]);
const current = $derived(steps[chosen]);
onMount(() => {
	ready = true;
	const locale = () => {
		english = document.documentElement.lang.startsWith("en");
	};
	locale();
	document.addEventListener("site-locale-change", locale);
	return () => document.removeEventListener("site-locale-change", locale);
});
</script>

<div class="rsi-interactive">
	<p class="rsi-prompt">{say("先选一种改变，看看它发生在哪里。", "Choose a change. Where does it take place?")}</p>
	<div class="rsi-choices" role="group" aria-label={say("三种改变", "Three kinds of change")}>
		{#each steps as item, index}<button type="button" disabled={!ready} aria-pressed={chosen === index} onclick={() => chosen = index}>{item.name}</button>{/each}
	</div>
	<section class="change-map" aria-label={say("改变的位置", "Where the change happens")}>
		{#each [say("一次尝试", "An attempt"), say("会留下的参数", "Parameters that persist"), say("如何产生下一次改进", "How the next improvement is made")] as name, index}
			<div class:marked={index === chosen}><span class="map-number">0{index + 1}</span><strong>{name}</strong><span class="mark">{index === chosen ? say("正在改变这里", "This is changing") : "·"}</span></div>
		{/each}
	</section>
	<div class="rsi-result" aria-live="polite" aria-atomic="true"><strong>{current.changed}</strong><p>{current.description}</p><p class="question">{current.question}</p><a href={current.href}>{current.link} ↗</a></div>
	<p class="rsi-note">{say("这是概念图，三个位置可以同时变化，并非一条自动通向 AGI 的等级阶梯。元学习关注学习方法如何获得或调整；方法改进与递归自我改进需要分别给出证据。", "This is a concept map. The three locations can change together; they are not a ladder that automatically leads to AGI. Meta-learning studies how learning methods are acquired or adapted. Method improvement and recursive self-improvement need distinct evidence.")}</p>
</div>

<style>
	.change-map{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));margin:28px 0 8px;gap:12px}.change-map>div{border-top:1px solid var(--brand-line);padding-top:14px;display:flex;flex-direction:column;gap:12px;min-width:0}.change-map>div.marked{border-top:3px solid var(--brand-clay);padding-top:12px}.map-number{font:13px "JetBrains Mono",monospace;color:var(--brand-muted)}.change-map strong{font-size:16px;line-height:1.7;font-weight:500}.mark{font-size:13px;color:var(--brand-clay);min-height:1.7em}.question{font-family:Georgia,"Songti SC",serif}.rsi-result a{display:inline-block;font-size:14px;margin-top:8px}.rsi-note{margin-top:22px}@media(max-width:520px){.change-map{grid-template-columns:1fr;gap:14px}.change-map>div{display:grid;grid-template-columns:28px 1fr;gap:8px 12px;padding-top:12px}.change-map .mark{grid-column:2}.change-map strong{font-size:16px}.change-map>div.marked{padding-top:10px}}
</style>
