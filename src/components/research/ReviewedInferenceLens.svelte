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
	for (const h of history) {
		const prev = states.at(-1) ?? 0;
		states.push((prev + Number(h)) / 2);
	}
	return states;
});
// Counterfactual twin trace (reversing the sequence order, e.g. 1100 <-> 0011)
const mirrorHistory = $derived(history.split("").reverse().join(""));
const mirrorTrace = $derived.by(() => {
	const states = [0];
	for (const h of mirrorHistory) {
		const prev = states.at(-1) ?? 0;
		states.push((prev + Number(h)) / 2);
	}
	return states;
});
const current = $derived(reset ? 0 : (trace.at(-1) ?? 0));
const readout = $derived(current + Number(support));
const points = $derived(
	trace.map((x, i) => `${40 + i * 120},${150 - x * 120}`).join(" "),
);
const mirrorPoints = $derived(
	mirrorTrace.map((x, i) => `${40 + i * 120},${150 - x * 120}`).join(" "),
);

let anchors = $state("none");
let hoveredCandidate = $state<number | null>(null);
const candidates = $derived(
	(
		semanticChecks.semantic as Record<
			string,
			{ candidates: Record<string, string>[] }
		>
	)[anchors]?.candidates ?? [],
);

const symbolPositions: Record<string, number> = { a: 120, b: 280, c: 440 };
const objectPositions: Record<string, number> = {
	圆: 120,
	方: 280,
	三角: 440,
};

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
		<div class="preset-strip">
			<button type="button" class="preset-chip" class:active={history === "1100" && support && !reset} onclick={() => { history = "1100"; support = true; reset = false; }}>{text("先学后停 (1100, U=1)", "Early input (1100, U=1)")}</button>
			<button type="button" class="preset-chip" class:active={history === "0011" && support && !reset} onclick={() => { history = "0011"; support = true; reset = false; }}>{text("晚近输入 (0011, U=1)", "Recent input (0011, U=1)")}</button>
			<button type="button" class="preset-chip" class:active={!support && !reset} onclick={() => { support = false; reset = false; }}>{text("撤除外部工具 (U=0)", "Remove tool (U=0)")}</button>
			<button type="button" class="preset-chip" class:active={reset} onclick={() => { reset = true; }}>{text("切断内部状态记忆 (Reset)", "Reset internal state")}</button>
		</div>
		<div class="controls"><label>{text("四步训练输入 H", "Four training inputs H")}<select data-history bind:value={history}>{#each ["0000", "1111", "1100", "0011"] as item}<option value={item}>{item}</option>{/each}</select></label><label><input type="checkbox" data-support bind:checked={support} />{text("测试时提供支持 U=1", "Test support U=1")}</label><label><input type="checkbox" data-reset bind:checked={reset} />{text("测试前将状态重置为0", "Reset state to zero before testing")}</label></div>
		<svg viewBox="0 0 560 190" role="img" aria-label={text("训练状态轨迹，横轴为0至4步；重置仅作用于之后的测试", "Training state over steps 0–4; reset applies only to the subsequent test")}>
			<path d="M40 22V150H525" fill="none" stroke="currentColor" opacity=".35"/>
			{#if mirrorHistory !== history}
				<polyline points={mirrorPoints} fill="none" stroke="var(--brand-clay)" stroke-width="1.8" stroke-dasharray="5 4" opacity=".55" />
			{/if}
			<polyline {points} fill="none" stroke="var(--brand-accent)" stroke-width="3"/>
			{#each trace as x, i}<circle cx={40+i*120} cy={150-x*120} r="5" fill="var(--brand-accent)"/><text x={40+i*120} y="175" text-anchor="middle">{i}</text>{/each}
			<text x="20" y="35">1</text><text x="20" y="150">0</text>
		</svg>
		<p class="formula">x₀=0 · xₜ₊₁=(xₜ+Hₜ)/2 · Y=x₄+U</p><div class="metrics" aria-live="polite"><p>{text("测试状态", "Test state")}<strong data-state>{current}</strong></p><p>{text("测试读数 Y", "Test readout Y")}<strong data-readout>{readout}</strong></p></div>
		<div class="insight-box">{text("把1100改为0011：输入次数相同，状态却不同；重置后差异消失。这里历史完全由当前状态中介。读数没有人类准确率、时间或认知边界的含义，不能据此裁决延展心智。", "Change 1100 to 0011: equal input counts, different states. Reset removes that difference. History is fully mediated by current state here. The readout is not human accuracy, time, or a cognitive boundary, and does not settle extended cognition.")}</div>
	{:else if mode === "semantics"}
		<div class="preset-strip">
			{#each [
				["none", text("0 锚点 (6 种全排列歧义)", "0 Anchors (6 bijections)")],
				["one", text("1 锚点：固定 a→圆 (剩 2 种)", "1 Anchor: a→circle (2 left)")],
				["two", text("2 锚点：唯一锁定 (剩 1 种)", "2 Anchors: unique (1 left)")],
				["conflict", text("矛盾约束 (0 相容解)", "Conflict (0 compatible)")],
			] as [val, label]}
				<button type="button" class="preset-chip" class:active={anchors === val} onclick={() => { anchors = val; }}>{label}</button>
			{/each}
		</div>
		<div class="controls"><label>{text("加入的映射约束", "Mapping constraints")}<select data-anchors bind:value={anchors}><option value="none">{text("无锚点", "No anchors")}</option><option value="one">a → {text("圆", "circle")}</option><option value="two">a → {text("圆", "circle")} · b → {text("方", "square")}</option><option value="conflict">a → {text("圆且方（矛盾）", "circle AND square (conflict)")}</option></select></label></div>
		<svg viewBox="0 0 560 148" role="img" aria-label={text("形式符号 a, b, c 到指称对象 圆, 方, 三角 的双部映射图", "Bipartite mapping graph from symbols a, b, c to objects circle, square, triangle")}>
			{#each candidates as mapping, cIdx}
				{@const isHovered = hoveredCandidate === cIdx}
				{@const isFocus = hoveredCandidate !== null ? isHovered : true}
				{#each Object.entries(mapping) as [sym, obj]}
					<line
						x1={symbolPositions[sym] ?? 120}
						y1="38"
						x2={objectPositions[obj] ?? 120}
						y2="106"
						stroke={isHovered ? "var(--brand-accent)" : candidates.length === 1 ? "var(--brand-accent)" : "var(--brand-clay)"}
						stroke-width={isHovered ? "3.2" : candidates.length === 1 ? "2.6" : "1.2"}
						stroke-dasharray={candidates.length === 1 || isHovered ? "none" : "3 2"}
						opacity={isFocus ? (isHovered || candidates.length === 1 ? "0.95" : "0.35") : "0.08"}
					/>
				{/each}
			{/each}
			{#each ["a", "b", "c"] as sym}
				<circle cx={symbolPositions[sym]} cy="30" r="15" fill="var(--brand-paper)" stroke="var(--brand-ink)" stroke-width="1.5" />
				<text x={symbolPositions[sym]} y="35" text-anchor="middle" font-family="monospace" font-size="13.5" font-weight="600" fill="currentColor">{sym}</text>
			{/each}
			{#each [
				["圆", text("圆", "circle")],
				["方", text("方", "square")],
				["三角", text("三角", "triangle")],
			] as [objKey, objLabel]}
				<rect x={(objectPositions[objKey] ?? 120) - 34} y="106" width="68" height="28" rx="6" fill="var(--brand-paper)" stroke="var(--brand-accent)" stroke-width="1.5" />
				<text x={objectPositions[objKey]} y="125" text-anchor="middle" font-size="13" fill="currentColor">{objLabel}</text>
			{/each}
		</svg>
		<p class="formula">{text("三符号与三对象构成双射；各符号独立等概率，序列abca在六种映射下的概率均为1/81。", "Three symbols map bijectively to three objects. Symbols are independent and equiprobable; sequence abca has probability 1/81 under all six mappings.")}</p>
		<div class="metrics" aria-live="polite"><p>{text("仍相容的映射", "Compatible mappings")}<strong data-candidates>{candidates.length}</strong></p></div>
		<ul class="mappings">
			{#each candidates as mapping, i}
				<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
				<li
					class:highlighted={hoveredCandidate === i}
					onpointerenter={() => { hoveredCandidate = i; }}
					onpointerleave={() => { if (hoveredCandidate === i) hoveredCandidate = null; }}
				>
					{Object.entries(mapping).map(([symbol, object]) => `${symbol} → ${english ? ({ 圆: "circle", 方: "square", 三角: "triangle" }[object]) : object}`).join(" · ")}
				</li>
			{/each}
		</ul>
		<div class="insight-box">{candidates.length === 0 ? text("零候选表示约束矛盾，不表示零理解。", "Zero candidates means inconsistent constraints, not zero understanding.") : text("候选从6到2再到1，只证明这个有限任务的映射被约束。", "The reduction from 6 to 2 to 1 shows only that this finite mapping task becomes constrained.")} {text("抽象概念可有内部角色；符号接地与操作者／整体是否理解仍须另行论证。", "Abstract concepts can have internal roles; grounding and whether an operator or a whole system understands require separate arguments.")}</div>
	{:else}
		<div class="preset-strip">
			<button type="button" class="preset-chip" class:active={stimulus && !reportRequired} onclick={() => { stimulus = true; reportRequired = false; }}>{text("无报告范式 (S=1, Q=0)：分离 N_A 与 N_B", "No-report paradigm (S=1, Q=0)")}</button>
			<button type="button" class="preset-chip" class:active={stimulus && reportRequired} onclick={() => { stimulus = true; reportRequired = true; }}>{text("标准报告范式 (S=1, Q=1)：两信号混淆", "Report paradigm (S=1, Q=1)")}</button>
			<button type="button" class="preset-chip" class:active={!stimulus && reportRequired} onclick={() => { stimulus = false; reportRequired = true; }}>{text("无刺激基线 (S=0, Q=1)", "No-stimulus baseline (S=0, Q=1)")}</button>
		</div>
		<div class="controls"><label><input type="checkbox" data-stimulus bind:checked={stimulus} />{text("刺激条件 S=1", "Stimulus condition S=1")}</label><label><input type="checkbox" data-report-required bind:checked={reportRequired} />{text("要求报告 Q=1", "Report required Q=1")}</label></div>
		<svg viewBox="0 0 600 140" role="img" aria-label={text("模型A的信号只依赖S，模型B的信号和两者报告依赖S与Q", "Signal A depends on S; signal B and both reports depend on S and Q")}>
			<g fill="none" stroke="currentColor" opacity=".45">
				<path d="M125 50H265" stroke={signalA ? "var(--brand-accent)" : "currentColor"} stroke-width={signalA ? "2.5" : "1.2"} />
				<path d="M125 50L265 105" stroke={signalB ? "var(--brand-clay)" : "currentColor"} stroke-width={signalB ? "2.2" : "1.2"} />
				<path d="M125 110L265 105" stroke={signalB ? "var(--brand-clay)" : "currentColor"} stroke-width={signalB ? "2.2" : "1.2"} />
				<rect x="265" y="18" width="300" height="47" rx="8" stroke={signalA ? "var(--brand-accent)" : "currentColor"} stroke-width={signalA ? "2" : "1"} />
				<rect x="265" y="82" width="300" height="47" rx="8" stroke={signalB ? "var(--brand-clay)" : "currentColor"} stroke-width={signalB ? "2" : "1"} />
			</g>
			<g fill="currentColor" font-size="19">
				<text x="25" y="55">S = {Number(stimulus)}</text>
				<text x="25" y="115">Q = {Number(reportRequired)}</text>
				<text x="285" y="49">N_A = S = {signalA}</text>
				<text x="285" y="113">N_B = R = S ∧ Q = {signalB}</text>
			</g>
		</svg>
		<div class="metrics" aria-live="polite"><p>{text("两模型报告 R", "Report R in both models")}<strong data-report>{report}</strong></p><p>N_A<strong data-signal-a>{signalA}</strong></p><p>N_B<strong data-signal-b>{signalB}</strong></p></div>
		<div class="insight-box">{text("保留刺激并取消报告要求：两模型都不报告，信号却不同。S是刺激条件、N是候选信号，均不是体验真值。这个依赖关系反例没有测量意识，也没有计算IIT的Φ；GNW与IIT的具体预测需要分别检验。", "Keep the stimulus and remove the report requirement: neither model reports, but their signals differ. S is a stimulus condition and N a candidate signal, neither a ground truth for experience. This counterexample neither measures consciousness nor computes IIT’s Φ; GNW and IIT predictions require separate tests.")}</div>
	{/if}
	<nav><a href={`/research/intelligence/${article}/#${mode === "history" ? "b3" : mode === "semantics" ? "m2" : "m3"}`}>{text("结论、反方与原文定位", "Conclusion, objections and sources")} →</a><a href={`/research/intelligence/downloads/${article}-protocol.md`}>{text("模型协议", "Model protocol")}</a><a href={`/research/intelligence/downloads/${article}-results.json`}>{text("精确核验", "Exact checks")}</a></nav>
</section>
<style>
	.reviewed-lens{border:1px solid var(--brand-line);border-radius:16px;padding:24px;margin:22px 0;background:var(--brand-surface);color:var(--brand-ink);line-height:1.8}.kicker{font:11px monospace;letter-spacing:.1em;color:var(--brand-muted)}h3{font-size:22px;line-height:1.45;margin:12px 0}header p,.formula{font-size:14px;color:var(--brand-body)}.preset-strip{display:flex;flex-wrap:wrap;gap:8px;padding-top:14px}.preset-chip{font:inherit;font-size:12.5px;padding:5px 12px;border:1px solid var(--brand-line);border-radius:999px;background:var(--brand-paper);color:var(--brand-ink);cursor:pointer;transition:background .15s,border-color .15s}.preset-chip.active{background:var(--brand-sage);border-color:var(--brand-accent);font-weight:500}.controls{display:flex;flex-wrap:wrap;align-items:center;gap:16px;padding:14px 0}.controls label{font-size:14px;display:flex;align-items:center;flex-wrap:wrap;gap:8px;min-height:44px}select{background:var(--card-bg);color:var(--brand-ink);border:1px solid var(--brand-line);border-radius:6px;padding:9px;max-width:100%}input{accent-color:var(--brand-accent);width:20px;height:20px}svg{width:100%;max-width:700px;height:auto;display:block;margin:auto}.metrics{display:flex;flex-wrap:wrap;gap:20px}.metrics p{border:1px solid var(--brand-line);border-radius:10px;min-width:110px;padding:12px 18px;font-size:13px}.metrics strong{display:block;font:600 28px/1.5 monospace;color:var(--brand-accent)}.insight-box{padding:17px;border-left:3px solid var(--brand-accent);font-size:14px;margin:18px 0;background:var(--brand-paper)}nav{display:flex;flex-wrap:wrap;gap:10px 20px;font-size:13px}a{color:var(--brand-accent);text-decoration:underline;text-underline-offset:4px}.mappings{padding-left:22px;font:14px/2 monospace}.mappings li{cursor:pointer;padding:2px 6px;border-radius:4px;transition:background .15s}.mappings li:hover,.mappings li.highlighted{background:var(--brand-sage);color:var(--brand-ink);font-weight:600}@media(max-width:600px){.reviewed-lens{padding:18px}.controls{display:grid}.metrics{gap:8px}.metrics p{min-width:80px;padding:10px}.metrics strong{font-size:24px}}
</style>
