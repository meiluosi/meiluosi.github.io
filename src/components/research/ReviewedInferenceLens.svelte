<script lang="ts">
import semanticChecks from "../../../docs/intelligence-history/round-four/meaning/method-check/results.json";
let { mode }: { mode: "history" | "semantics" | "report" } = $props();
let english = $state(false);
$effect(() => {
	const sync = () => {
		english = document.documentElement.lang.toLowerCase().startsWith("en");
	};
	sync();
	const observer = new MutationObserver(sync);
	observer.observe(document.documentElement, {
		attributes: true,
		attributeFilter: ["lang"],
	});
	return () => observer.disconnect();
});
const text = (zh: string, en: string) => (english ? en : zh);
let history = $state("1100");
let support = $state(true);
let reset = $state(false);
const trace = $derived.by(() => {
	const states = [0];
	for (const h of history) states.push((states.at(-1)! + Number(h)) / 2);
	return states;
});
const current = $derived(reset ? 0 : trace.at(-1)!);
const readout = $derived(current + Number(support));
const points = $derived(
	trace.map((x, i) => `${40 + i * 120},${150 - x * 120}`).join(" "),
);
let anchors = $state("none");
const candidates = $derived(
	semanticChecks.mappings.filter(
		(mapping) =>
			anchors === "none" ||
			(anchors === "one" && mapping.a === "圆") ||
			(anchors === "two" && mapping.a === "圆" && mapping.b === "方"),
	),
);
let stimulus = $state(true);
let reportRequired = $state(true);
const report = $derived(Number(stimulus && reportRequired));
const signalA = $derived(Number(stimulus));
const signalB = $derived(Number(stimulus && reportRequired));
const article = $derived(mode === "history" ? "boundaries" : "meaning");
</script>
<section class="reviewed-lens" data-reviewed-lens={mode}>
	<header><span class="kicker">{text("研究中的有限反例", "FINITE RESEARCH COUNTEREXAMPLE")}</span><h3>{mode === "history" ? text("同样移除工具，为什么会有不同结果？", "Why can the same tool removal produce different results?") : mode === "semantics" ? text("锚点排除歧义，排除了多少？", "How much ambiguity can anchors remove?") : text("没有报告，候选信号为何仍可能不同？", "Why can candidate signals differ without a report?")}</h3>
	<p>{mode === "history" ? text("Clark与Chalmers的工具问题、Smith等的发展研究提示我们区分当前支持与经历。下面只检验一个已定义的状态模型。", "Questions raised by Clark and Chalmers and developmental work by Smith et al. motivate separating current support from history. This is a defined state model only.") : mode === "semantics" ? text("Searle、Harnad与概念角色观点对理解有不同要求。这个模型只问：哪些符号—对象映射仍与约束相容？", "Searle, Harnad and conceptual-role accounts set different requirements for understanding. This model asks only which symbol–object mappings satisfy the constraints.") : text("Block区分访问与体验；GNW关注意识访问。这个反例把报告要求纳入模型，检查同一报告读数能否唯一确定信号依赖。", "Block distinguishes access from experience; GNW studies conscious access. This counterexample tests whether identical reports uniquely determine signal dependencies.")}</p></header>
	{#if mode === "history"}
		<div class="controls"><label>{text("四步训练输入 H", "Four training inputs H")}<select data-history bind:value={history}>{#each ["0000", "1111", "1100", "0011"] as item}<option value={item}>{item}</option>{/each}</select></label><label><input type="checkbox" data-support bind:checked={support} />{text("测试时提供支持 U=1", "Test support U=1")}</label><label><input type="checkbox" data-reset bind:checked={reset} />{text("测试前将状态重置为0", "Reset state to zero before testing")}</label></div>
		<svg viewBox="0 0 560 190" role="img" aria-label={text("训练状态轨迹，横轴为0至4步；重置仅作用于之后的测试", "Training state over steps 0–4; reset applies only to the subsequent test")}><path d="M40 22V150H525" fill="none" stroke="currentColor" opacity=".35"/><polyline {points} fill="none" stroke="var(--brand-accent)" stroke-width="3"/>{#each trace as x, i}<circle cx={40+i*120} cy={150-x*120} r="5" fill="var(--brand-accent)"/><text x={40+i*120} y="175" text-anchor="middle">{i}</text>{/each}<text x="20" y="35">1</text><text x="20" y="150">0</text></svg>
		<p class="formula">x₀=0 · xₜ₊₁=(xₜ+Hₜ)/2 · Y=x₄+U</p><div class="metrics" aria-live="polite"><p>{text("测试状态", "Test state")}<strong data-state>{current}</strong></p><p>{text("测试读数 Y", "Test readout Y")}<strong data-readout>{readout}</strong></p></div>
		<div class="insight-box">{text("把1100改为0011：输入次数相同，状态却不同；重置后差异消失。这里历史完全由当前状态中介。读数没有人类准确率、时间或认知边界的含义，不能据此裁决延展心智。", "Change 1100 to 0011: equal input counts, different states. Reset removes that difference. History is fully mediated by current state here. The readout is not human accuracy, time, or a cognitive boundary, and does not settle extended cognition.")}</div>
	{:else if mode === "semantics"}
		<div class="controls"><label>{text("加入的映射约束", "Mapping constraints")}<select data-anchors bind:value={anchors}><option value="none">{text("无锚点", "No anchors")}</option><option value="one">a → {text("圆", "circle")}</option><option value="two">a → {text("圆", "circle")} · b → {text("方", "square")}</option><option value="conflict">a → {text("圆且方（矛盾）", "circle AND square (conflict)")}</option></select></label></div>
		<p class="formula">{text("三符号与三对象构成双射；各符号独立等概率，序列abca在六种映射下的概率均为1/81。", "Three symbols map bijectively to three objects. Symbols are independent and equiprobable; sequence abca has probability 1/81 under all six mappings.")}</p>
		<div class="metrics" aria-live="polite"><p>{text("仍相容的映射", "Compatible mappings")}<strong data-candidates>{candidates.length}</strong></p></div><ul class="mappings">{#each candidates as mapping}<li>{Object.entries(mapping).map(([symbol, object]) => `${symbol} → ${english ? ({ 圆: "circle", 方: "square", 三角: "triangle" }[object]) : object}`).join(" · ")}</li>{/each}</ul>
		<div class="insight-box">{candidates.length === 0 ? text("零候选表示约束矛盾，不表示零理解。", "Zero candidates means inconsistent constraints, not zero understanding.") : text("候选从6到2再到1，只证明这个有限任务的映射被约束。", "The reduction from 6 to 2 to 1 shows only that this finite mapping task becomes constrained.")} {text("抽象概念可有内部角色；符号接地与操作者／整体是否理解仍须另行论证。", "Abstract concepts can have internal roles; grounding and whether an operator or a whole system understands require separate arguments.")}</div>
	{:else}
		<div class="controls"><label><input type="checkbox" data-stimulus bind:checked={stimulus} />{text("刺激条件 S=1", "Stimulus condition S=1")}</label><label><input type="checkbox" data-report-required bind:checked={reportRequired} />{text("要求报告 Q=1", "Report required Q=1")}</label></div>
		<svg viewBox="0 0 600 140" role="img" aria-label={text("模型A的信号只依赖S，模型B的信号和两者报告依赖S与Q", "Signal A depends on S; signal B and both reports depend on S and Q")}><g fill="none" stroke="currentColor" opacity=".4"><path d="M125 50H265 M125 50L265 105 M125 110L265 105"/><rect x="265" y="18" width="300" height="47" rx="8"/><rect x="265" y="82" width="300" height="47" rx="8"/></g><g fill="currentColor" font-size="19"><text x="25" y="55">S = {Number(stimulus)}</text><text x="25" y="115">Q = {Number(reportRequired)}</text><text x="285" y="49">N_A = S = {signalA}</text><text x="285" y="113">N_B = R = S ∧ Q = {signalB}</text></g></svg>
		<div class="metrics" aria-live="polite"><p>{text("两模型报告 R", "Report R in both models")}<strong data-report>{report}</strong></p><p>N_A<strong data-signal-a>{signalA}</strong></p><p>N_B<strong data-signal-b>{signalB}</strong></p></div>
		<div class="insight-box">{text("保留刺激并取消报告要求：两模型都不报告，信号却不同。S是刺激条件、N是候选信号，均不是体验真值。这个依赖关系反例没有测量意识，也没有计算IIT的Φ；GNW与IIT的具体预测需要分别检验。", "Keep the stimulus and remove the report requirement: neither model reports, but their signals differ. S is a stimulus condition and N a candidate signal, neither a ground truth for experience. This counterexample neither measures consciousness nor computes IIT’s Φ; GNW and IIT predictions require separate tests.")}</div>
	{/if}
	<nav><a href={`/research/intelligence/${article}/#${mode === "history" ? "b3" : mode === "semantics" ? "m2" : "m3"}`}>{text("结论、反方与原文定位", "Conclusion, objections and sources")} →</a><a href={`/research/intelligence/downloads/${article}-protocol.md`}>{text("模型协议", "Model protocol")}</a><a href={`/research/intelligence/downloads/${article}-results.json`}>{text("精确核验", "Exact checks")}</a></nav>
</section>
<style>
	.reviewed-lens{border:1px solid var(--brand-line);border-radius:16px;padding:24px;margin:22px 0;background:var(--brand-surface);color:var(--brand-ink);line-height:1.8}.kicker{font:11px monospace;letter-spacing:.1em;color:var(--brand-muted)}h3{font-size:22px;line-height:1.45;margin:12px 0}header p,.formula{font-size:14px;color:var(--brand-body)}.controls{display:flex;flex-wrap:wrap;align-items:center;gap:16px;padding:18px 0}.controls label{font-size:14px;display:flex;align-items:center;flex-wrap:wrap;gap:8px;min-height:44px}select{background:var(--card-bg);color:var(--brand-ink);border:1px solid var(--brand-line);border-radius:6px;padding:9px;max-width:100%}input{accent-color:var(--brand-accent);width:20px;height:20px}svg{width:100%;max-width:700px;height:auto;display:block;margin:auto}.metrics{display:flex;flex-wrap:wrap;gap:20px}.metrics p{border:1px solid var(--brand-line);border-radius:10px;min-width:110px;padding:12px 18px;font-size:13px}.metrics strong{display:block;font:600 28px/1.5 monospace;color:var(--brand-accent)}.insight-box{padding:17px;border-left:3px solid var(--brand-accent);font-size:14px;margin:18px 0;background:var(--brand-paper)}nav{display:flex;flex-wrap:wrap;gap:10px 20px;font-size:13px}a{color:var(--brand-accent);text-decoration:underline;text-underline-offset:4px}.mappings{padding-left:22px;font:14px/2 monospace}@media(max-width:600px){.reviewed-lens{padding:18px}.controls{display:grid}.metrics{gap:8px}.metrics p{min-width:80px;padding:10px}.metrics strong{font-size:24px}}
</style>
