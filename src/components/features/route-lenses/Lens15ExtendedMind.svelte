<script lang="ts">
/**
 * Lens 15: 延展认知与外部脚手架路线 (Clark & Chalmers: The Extended Mind)
 * 揭示：心智并不止步于颅骨与皮肤(Skull and Skin)。当外部笔记本、算盘或代码沙盒与生物大脑形成可靠双向耦合时，它们就是认知系统不可分割的组成部分。
 */
let english = $state(false);

$effect(() => {
	const syncLang = () => {
		english = document.documentElement.lang.toLowerCase().startsWith("en");
	};
	syncLang();
	const obs = new MutationObserver(syncLang);
	obs.observe(document.documentElement, {
		attributes: true,
		attributeFilter: ["lang"],
	});
	return () => obs.disconnect();
});

let taskDigits = $state(4); // Multiplication complexity: 2-digit vs 5-digit mental math
let useScaffolding = $state(true); // External scratchpad / symbolic tool enabled

const perf = $derived.by(() => {
	// Biological working memory bottleneck (~4-7 chunks)
	const wmRequired = taskDigits * 2;
	if (!useScaffolding) {
		const overload = Math.max(0, wmRequired - 5);
		const accuracy = Math.max(5, 98 - overload * 22);
		const latency = (1.5 + taskDigits ** 2.2 * 0.8).toFixed(1);
		return {
			wmSlotsUsed: `${Math.min(7, wmRequired)} / 7 (生物脑满载溢出)`,
			accuracy: `${accuracy}%`,
			latency: `${latency}s`,
			overloaded: overload > 0,
		};
	}
	const accuracy = 99;
	const latency = (1.2 + taskDigits * 0.9).toFixed(1);
	return {
		wmSlotsUsed: "2 / 7 (中间状态已卸载至外部纸笔)",
		accuracy: `${accuracy}%`,
		latency: `${latency}s`,
		overloaded: false,
	};
});
</script>

<div class="lens-container">
	<div class="lens-header">
		<span class="lens-tag">COGNITIVE LENS 15</span>
		<h3 class="lens-title">{english ? "Clark & Chalmers Extended Mind Loop" : "克拉克与查尔默斯：延展心智与奥托笔记本沙盒"}</h3>
		<p class="lens-subtitle">
			{english
				? "Andy Clark & David Chalmers (Parity Principle): If a part of the world functions as a process which, were it done in the head, we would have no hesitation in recognizing as part of the cognitive process, then that part of the world IS part of the cognitive process."
				: "安迪·克拉克与大卫·查尔默斯 (对等原则 Parity Principle)：如果外部媒质（如奥托的笔记本、数学草稿纸、外部检索工具）在功能上扮演了与大脑内部记忆/推演同等的回路角色，那么认知系统的边界就已经跨越颅骨延展到了外部世界。"}
		</p>
	</div>

	<div class="controls-bar">
		<div class="slider-col">
			<div class="label-row">
				<span>{english ? "Task Symbolic Complexity (N-Digit Multiplication)" : "认知任务符号复杂度 (N 位数乘法推演)"}</span>
				<strong class="val-pill">{taskDigits} 位数 × {taskDigits} 位数</strong>
			</div>
			<input type="range" min="2" max="6" step="1" bind:value={taskDigits} class="slider" />
		</div>

		<div class="toggle-col">
			<button
				type="button"
				class="scaffold-btn"
				class:active={useScaffolding}
				onclick={() => useScaffolding = !useScaffolding}
			>
				{useScaffolding ? "📓 已接入外部认知脚手架 (纸笔/外部存储)" : "🧠 仅限颅骨内部裸脑心算 (无外部工具)"}
			</button>
		</div>
	</div>

	<div class="coupling-diagram">
		<div class="entity-box bio" class:overloaded={perf.overloaded}>
			<div class="e-title">🧠 生物脑工作记忆 (In-Head WM)</div>
			<div class="e-metric">占用槽位：{perf.wmSlotsUsed}</div>
			<div class="e-sub">擅长：模式直觉、单步注意聚焦</div>
		</div>

		<div class="coupling-arrows">
			{#if useScaffolding}
				<span class="arrow active">⇄ 双向紧密耦合回路 (Epistemic Action)</span>
			{:else}
				<span class="arrow severed">✕ 外部回路切断 (孤立颅内计算)</span>
			{/if}
		</div>

		<div class="entity-box ext" class:disabled={!useScaffolding}>
			<div class="e-title">📓 外部符号媒质 (Otto's Notebook / Scratchpad)</div>
			<div class="e-metric">{useScaffolding ? "持久化存储进位与中间符号状态" : "未挂载"}</div>
			<div class="e-sub">擅长：零衰减精确记忆、空间对齐排列</div>
		</div>
	</div>

	<div class="metrics-row">
		<div class="m-card">
			<span class="lbl">推演准确率 (Accuracy)</span>
			<strong class="val" class:bad={perf.overloaded}>{perf.accuracy}</strong>
		</div>
		<div class="m-card">
			<span class="lbl">完成耗时 (Cognitive Latency)</span>
			<strong class="val">{perf.latency}</strong>
		</div>
		<div class="m-card">
			<span class="lbl">认知系统边界判定</span>
			<strong class="val">{useScaffolding ? "大脑 + 外部笔记本合成体" : "仅生物神经回路"}</strong>
		</div>
	</div>

	<div class="insight-box">
		<div class="insight-icon">📎</div>
		<div class="insight-text">
			<strong>{english ? "Implications for LLM Tool-Use & RAG:" : "对现代 Agent 工具调用与外挂记忆的启示："}</strong>
			{english
				? "Asking an LLM to do 10-digit arithmetic purely in forward-pass weights is as absurd as forbidding a mathematician from using chalk. True intelligence lies in seamlessly coupling neural intuition with external symbolic scaffolding."
				: "强迫大模型仅靠前向权重心算 10 位数乘法，就像剥夺数学家手中的粉笔与草稿纸一样荒谬。人类与高级 Agent 的真正力量，在于将神经直觉与外部精确符号脚手架（代码解释器、笔记本、知识库）融为单一延展认知系统。"}
		</div>
	</div>
</div>

<style>
.lens-container {
	border: 1px solid var(--border-color, rgba(120, 140, 130, 0.25));
	background: var(--brand-paper, #faf8f5);
	color: var(--brand-ink, #2d3748);
	border-radius: 12px;
	padding: 1.5rem;
	margin: 2rem 0;
	box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
}
:global(.dark) .lens-container { background: #141816; border-color: rgba(120, 150, 130, 0.2); color: #e2e8f0; }
.lens-tag { font-size: 0.75rem; font-weight: 700; letter-spacing: 0.1em; color: var(--brand-clay, #8a7356); text-transform: uppercase; }
.lens-title { font-size: 1.35rem; margin: 0.25rem 0 0.5rem; font-weight: 700; }
.lens-subtitle { font-size: 0.88rem; line-height: 1.5; color: var(--text-muted, #718096); margin-bottom: 1.25rem; }
.controls-bar {
	display: grid;
	grid-template-columns: 1.2fr 1fr;
	gap: 1.25rem;
	align-items: center;
	padding: 1rem;
	background: rgba(0, 0, 0, 0.03);
	border-radius: 8px;
	margin-bottom: 1.25rem;
}
:global(.dark) .controls-bar { background: rgba(255, 255, 255, 0.03); }
@media (max-width: 640px) { .controls-bar { grid-template-columns: 1fr; } }
.label-row { display: flex; justify-content: space-between; font-size: 0.82rem; font-weight: 600; margin-bottom: 0.35rem; }
.val-pill { font-family: monospace; color: var(--brand-clay, #8a7356); }
.slider { width: 100%; accent-color: var(--brand-clay, #8a7356); }
.scaffold-btn {
	width: 100%;
	padding: 0.65rem 0.85rem;
	border-radius: 8px;
	border: 1px solid rgba(0, 0, 0, 0.15);
	background: rgba(0, 0, 0, 0.04);
	color: inherit;
	font-size: 0.82rem;
	font-weight: 600;
	cursor: pointer;
	transition: all 0.2s;
}
.scaffold-btn.active {
	background: #0284c7;
	color: #fff;
	border-color: #0284c7;
}
.coupling-diagram {
	display: grid;
	grid-template-columns: 1fr auto 1fr;
	gap: 1rem;
	align-items: center;
	margin-bottom: 1.25rem;
}
@media (max-width: 680px) { .coupling-diagram { grid-template-columns: 1fr; text-align: center; } }
.entity-box {
	padding: 1rem;
	border-radius: 8px;
	background: rgba(255, 255, 255, 0.65);
	border: 2px solid rgba(2, 132, 199, 0.35);
}
:global(.dark) .entity-box { background: rgba(0, 0, 0, 0.25); }
.entity-box.overloaded { border-color: #ef4444; background: rgba(239, 68, 68, 0.08); }
.entity-box.disabled { opacity: 0.38; border-style: dashed; border-color: #94a3b8; }
.e-title { font-size: 0.85rem; font-weight: 700; margin-bottom: 0.35rem; }
.e-metric { font-size: 0.8rem; font-family: monospace; color: var(--brand-clay, #8a7356); margin-bottom: 0.25rem; }
.e-sub { font-size: 0.72rem; color: var(--text-muted, #718096); }
.arrow { font-size: 0.75rem; font-weight: 700; padding: 0.3rem 0.6rem; border-radius: 99px; }
.arrow.active { background: rgba(16, 185, 129, 0.15); color: #10b981; }
.arrow.severed { background: rgba(239, 68, 68, 0.15); color: #ef4444; }
.metrics-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.85rem; margin-bottom: 1.25rem; }
@media (max-width: 600px) { .metrics-row { grid-template-columns: 1fr; } }
.m-card { padding: 0.75rem; background: rgba(0, 0, 0, 0.02); border-radius: 6px; border: 1px solid rgba(0, 0, 0, 0.05); }
:global(.dark) .m-card { background: rgba(255, 255, 255, 0.02); border-color: rgba(255, 255, 255, 0.05); }
.m-card .lbl { font-size: 0.7rem; color: var(--text-muted, #718096); display: block; }
.m-card .val { font-size: 1rem; font-family: monospace; }
.m-card .val.bad { color: #ef4444; }
.insight-box {
	display: flex;
	gap: 0.75rem;
	padding: 0.85rem 1rem;
	background: rgba(138, 115, 86, 0.08);
	border-left: 3px solid var(--brand-clay, #8a7356);
	border-radius: 4px;
	font-size: 0.84rem;
	line-height: 1.5;
}
.insight-icon { font-size: 1.1rem; }
</style>
