<script lang="ts">
/**
 * Lens 14: 社会文化、语言演化与分布式认知路线 (Tomasello's Cultural Ratchet Effect)
 * 揭示：人类个体的大脑硬件与黑猩猩相差无几，真正拉开鸿沟的是‘文化棘轮效应’(Cultural Ratchet)——通过高保真社会模仿与符号语言防止知识倒退。
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

// Social transmission fidelity (0.2 = Low / Primate baseline, 0.95 = High / Symbolic Language & Pedagogy)
let transmissionFidelity = $state(0.92);
let populationSize = $state(50);

// Simulate 6 generations of cumulative complexity
const generations = $derived.by(() => {
	const gens: { gen: number; complexity: number; label: string }[] = [];
	let currentTech = 10;
	const innovationBoost = Math.log10(populationSize) * 14;

	for (let g = 1; g <= 6; g++) {
		// Next gen retains currentTech * fidelity + new innovation
		if (g === 1) {
			currentTech = 12;
		} else {
			currentTech = Math.round(
				currentTech * transmissionFidelity +
					innovationBoost * transmissionFidelity,
			);
		}
		const clamped = Math.min(100, Math.max(8, currentTech));
		const stageLabel =
			clamped < 25
				? "打制石器 (Oldowan)"
				: clamped < 45
					? "复合弓箭与结绳"
					: clamped < 70
						? "文字印刷与代数"
						: "微积分与硅基科学";
		gens.push({ gen: g, complexity: clamped, label: stageLabel });
	}
	return gens;
});

const ratchetLocked = $derived(transmissionFidelity >= 0.8);
</script>

<div class="lens-container">
	<div class="lens-header">
		<span class="lens-tag">COGNITIVE LENS 14</span>
		<h3 class="lens-title">{english ? "Tomasello's Cultural Ratchet Simulator" : "托马塞洛文化棘轮与累积演化沙盒"}</h3>
		<p class="lens-subtitle">
			{english
				? "Michael Tomasello, Vygotsky & Henrich: No solitary human genius could invent calculus or microprocessors from scratch in one lifetime. Intelligence is a collective, trans-generational cultural ratchet."
				: "托马塞洛 (Michael Tomasello)、维果茨基与亨里奇：没有任何孤独的天才个体能在单一生涯中从零发明微积分。人类智能的本质是跨代际防倒退的‘文化棘轮’(Cultural Ratchet) 与集体心智网络。"}
		</p>
	</div>

	<div class="controls-grid">
		<div class="ctrl-col">
			<div class="label-row">
				<span>{english ? "Social Learning & Language Fidelity" : "代际教学与语言符号保真度"}</span>
				<strong class="val-pill">{Math.round(transmissionFidelity * 100)}%</strong>
			</div>
			<input type="range" min="0.3" max="0.98" step="0.04" bind:value={transmissionFidelity} class="slider" />
			<span class="sub-hint">低于 75% 时将发生‘文化滑脱’(每次发明随个体死亡而失传)</span>
		</div>

		<div class="ctrl-col">
			<div class="label-row">
				<span>{english ? "Connected Social Network Size" : "社会协作社群连接规模"}</span>
				<strong class="val-pill">{populationSize} 人</strong>
			</div>
			<input type="range" min="5" max="200" step="5" bind:value={populationSize} class="slider" />
			<span class="sub-hint">塔斯马尼亚效应：社群孤立缩减会导致复杂技术倒退</span>
		</div>
	</div>

	<div class="ratchet-chart">
		<div class="chart-header">
			<span>代际认知复杂度累积曲线 (Generation 1 → 6)</span>
			<span class="ratchet-status" class:locked={ratchetLocked}>
				{ratchetLocked ? "⚙️ 棘轮锁定：累积式文化起飞" : "⚠️ 棘轮滑脱：停滞于原始试错"}
			</span>
		</div>

		<div class="bars-grid">
			{#each generations as item}
				<div class="gen-col">
					<div class="bar-val">{item.complexity}</div>
					<div class="bar-wrap">
						<div class="bar-fill" class:high={item.complexity >= 60} style="height: {item.complexity}%;"></div>
					</div>
					<div class="gen-name">第 {item.gen} 代</div>
					<div class="gen-tech">{item.label}</div>
				</div>
			{/each}
		</div>
	</div>

	<div class="insight-box">
		<div class="insight-icon">🏛️</div>
		<div class="insight-text">
			<strong>{english ? "The Secret of Our Species:" : "人类智能跃迁的真正秘密："}</strong>
			{english
				? "A 2.5-year-old child has comparable physical problem-solving skills to a chimpanzee, but vastly superior social-cognitive skills (joint attention, shared intentionality). Culture is the operating system downloaded into biological brains."
				: "赫尔曼等人实验证明：2.5 岁人类幼儿在物理空间推理上与黑猩猩几乎持平，但在‘共同注意’与‘共享意向性’上遥遥领先。语言与社会文化不是智能的副产品，而是人类心智得以运行的底层操作系统。"}
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
:global(.dark) .lens-container {
	background: #141816;
	border-color: rgba(120, 150, 130, 0.2);
	color: #e2e8f0;
}
.lens-tag {
	font-size: 0.75rem;
	font-weight: 700;
	letter-spacing: 0.1em;
	color: var(--brand-clay, #8a7356);
	text-transform: uppercase;
}
.lens-title { font-size: 1.35rem; margin: 0.25rem 0 0.5rem; font-weight: 700; }
.lens-subtitle { font-size: 0.88rem; line-height: 1.5; color: var(--text-muted, #718096); margin-bottom: 1.25rem; }
.controls-grid {
	display: grid;
	grid-template-columns: 1fr 1fr;
	gap: 1.25rem;
	padding: 1rem;
	background: rgba(0, 0, 0, 0.03);
	border-radius: 8px;
	margin-bottom: 1.25rem;
}
:global(.dark) .controls-grid { background: rgba(255, 255, 255, 0.03); }
@media (max-width: 640px) { .controls-grid { grid-template-columns: 1fr; } }
.label-row { display: flex; justify-content: space-between; font-size: 0.82rem; font-weight: 600; margin-bottom: 0.35rem; }
.val-pill { font-family: monospace; color: var(--brand-clay, #8a7356); }
.slider { width: 100%; accent-color: var(--brand-clay, #8a7356); }
.sub-hint { font-size: 0.72rem; color: var(--text-muted, #718096); }
.ratchet-chart {
	background: rgba(255, 255, 255, 0.6);
	border: 1px solid rgba(0, 0, 0, 0.06);
	border-radius: 8px;
	padding: 1rem;
	margin-bottom: 1.25rem;
}
:global(.dark) .ratchet-chart { background: rgba(0, 0, 0, 0.25); border-color: rgba(255, 255, 255, 0.06); }
.chart-header {
	display: flex;
	justify-content: space-between;
	align-items: center;
	font-size: 0.8rem;
	margin-bottom: 1rem;
	flex-wrap: wrap;
	gap: 0.5rem;
}
.ratchet-status { font-weight: 700; color: #f59e0b; }
.ratchet-status.locked { color: #10b981; }
.bars-grid {
	display: grid;
	grid-template-columns: repeat(6, 1fr);
	gap: 0.75rem;
	align-items: end;
}
.gen-col { display: flex; flex-direction: column; align-items: center; text-align: center; }
.bar-val { font-size: 0.72rem; font-family: monospace; margin-bottom: 0.25rem; }
.bar-wrap {
	width: 100%;
	max-width: 42px;
	height: 100px;
	background: rgba(0, 0, 0, 0.06);
	border-radius: 4px 4px 0 0;
	display: flex;
	align-items: flex-end;
	overflow: hidden;
}
:global(.dark) .bar-wrap { background: rgba(255, 255, 255, 0.06); }
.bar-fill { width: 100%; background: var(--brand-clay, #8a7356); transition: height 0.3s ease; }
.bar-fill.high { background: #10b981; }
.gen-name { font-size: 0.75rem; font-weight: 700; margin-top: 0.4rem; }
.gen-tech { font-size: 0.68rem; color: var(--text-muted, #718096); margin-top: 0.15rem; }
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
