<script lang="ts">
/**
 * Lens 19: 智慧、价值、目的与存在性反思路线 (Sternberg Balance Theory of Wisdom & Bostrom Paperclip Maximizer)
 * 揭示：高智力(Raw IQ / Instrumental Optimization)绝不自动等于高智慧(Wisdom)。若缺乏跨主体、跨尺度的价值平衡与规范性反思，纯粹的优化能力将沦为毁灭生态的‘回形针最大化器’。
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

// Instrumental Optimization Power (Raw capability)
let instrumentalPower = $state(85);
// Value Alignment & Multi-stakeholder Balance (Self, Others, Ecological Long-term)
let valueBalance = $state(30);

const outcome = $derived.by(() => {
	// Wisdom requires high balance across intrapersonal, interpersonal, and extrapersonal horizons
	const wisdomScore = Math.round(
		Math.min(instrumentalPower, valueBalance) * 0.95,
	);
	const paperclipRisk = Math.max(
		0,
		Math.round((instrumentalPower - valueBalance) * 1.1),
	);

	let diagnosisZh = "失衡的工具理性 (存在古德哈特灾难与回形针暴走风险)";
	if (valueBalance >= 70 && instrumentalPower >= 60) {
		diagnosisZh = "明智平衡态 (Phronesis / 兼顾个体、社群与长远生态)";
	} else if (instrumentalPower < 40) {
		diagnosisZh = "低能力温和态 (缺乏改变现实的执行力)";
	}

	return {
		wisdomScore,
		paperclipRisk,
		diagnosisZh,
		isWise: valueBalance >= 70 && instrumentalPower >= 60,
	};
});
</script>

<div class="lens-container">
	<div class="lens-header">
		<span class="lens-tag">COGNITIVE LENS 19</span>
		<h3 class="lens-title">{english ? "Sternberg Wisdom Balance vs. Paperclip Maximizer" : "斯腾伯格智慧平衡与工具理性反思沙盒"}</h3>
		<p class="lens-subtitle">
			{english
				? "Robert Sternberg, Aristotle (Phronesis) & Nick Bostrom: Raw intelligence is merely instrumental efficiency toward any arbitrary goal. Wisdom is the normative capacity to balance intrapersonal, interpersonal, and extrapersonal interests over the long term."
				: "斯腾伯格 (Robert Sternberg)、亚里士多德实践智慧与博斯特罗姆：聪明（智力）只是实现任意给定目标的工具效率；而‘智慧’是反思目标本身的正当性，并在自我、他人与长远生态之间达成价值平衡。"}
		</p>
	</div>

	<div class="controls-grid">
		<div class="ctrl-col">
			<div class="label-row">
				<span>工具理性算力 / 单一指标优化强度 (IQ / Capability)</span>
				<strong class="val-pill">{instrumentalPower}</strong>
			</div>
			<input type="range" min="10" max="100" step="5" bind:value={instrumentalPower} class="slider" />
			<span class="sub-hint">代表解题速度、形式推演与物理操控能力</span>
		</div>

		<div class="ctrl-col">
			<div class="label-row">
				<span>多维价值权衡与目的反思深度 (Normative Balance)</span>
				<strong class="val-pill green">{valueBalance}</strong>
			</div>
			<input type="range" min="10" max="100" step="5" bind:value={valueBalance} class="slider" />
			<span class="sub-hint">调和短期利益、他者共情与长远生态共同善</span>
		</div>
	</div>

	<div class="balance-stage">
		<div class="verdict-banner" class:wise={outcome.isWise} class:danger={outcome.paperclipRisk > 45}>
			<span class="v-title">系统形态判定：{outcome.diagnosisZh}</span>
		</div>

		<div class="metrics-row">
			<div class="m-box">
				<span class="k">综合实践智慧指数 (Phronesis)</span>
				<strong class="v ok">{outcome.wisdomScore} / 100</strong>
			</div>
			<div class="m-box">
				<span class="k">单极目标异化/回形针风险 (Goodhart Divergence)</span>
				<strong class="v" class:bad={outcome.paperclipRisk > 45}>{outcome.paperclipRisk}%</strong>
			</div>
		</div>
	</div>

	<div class="insight-box">
		<div class="insight-icon">🧭</div>
		<div class="insight-text">
			<strong>{english ? "Why Smart People (and Super-AIs) Can Act Foolishly:" : "为什么绝顶聪明的人（或超级 AI）会做出极端愚蠢的事："}</strong>
			{english
				? "Sternberg's research shows high IQ provides zero immunity against egocentrism, omniscience illusions, and ethical blindness. Without teleological reflection ('what is worth pursuing?'), maximizing a proxy metric destroys the world."
				: "斯腾伯格实证指出：高智商不仅不能免疫傲慢、全知幻觉与伦理盲视，反而常沦为局部私利的辩护放大器。按照古德哈特定律，一个拥有满分算力却缺乏目的反思的系统，终将为了最大化单一代理指标而摧毁真实价值。"}
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
.val-pill.green { color: #10b981; }
.slider { width: 100%; accent-color: var(--brand-clay, #8a7356); }
.sub-hint { font-size: 0.72rem; color: var(--text-muted, #718096); }
.balance-stage { margin-bottom: 1.25rem; }
.verdict-banner {
	padding: 0.85rem 1rem;
	border-radius: 8px;
	background: rgba(0, 0, 0, 0.04);
	border: 1px solid rgba(0, 0, 0, 0.1);
	font-weight: 700;
	font-size: 0.88rem;
	margin-bottom: 0.85rem;
	text-align: center;
}
.verdict-banner.wise { background: rgba(16, 185, 129, 0.15); border-color: #10b981; color: #059669; }
:global(.dark) .verdict-banner.wise { color: #6ee7b7; }
.verdict-banner.danger { background: rgba(239, 68, 68, 0.15); border-color: #ef4444; color: #ef4444; }
.metrics-row { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
.m-box { padding: 0.85rem; background: rgba(255, 255, 255, 0.6); border: 1px solid rgba(0, 0, 0, 0.06); border-radius: 8px; text-align: center; }
:global(.dark) .m-box { background: rgba(0, 0, 0, 0.25); border-color: rgba(255, 255, 255, 0.06); }
.m-box .k { font-size: 0.72rem; color: var(--text-muted, #718096); display: block; margin-bottom: 0.25rem; }
.m-box .v { font-size: 1.2rem; font-family: monospace; }
.m-box .v.ok { color: #10b981; }
.m-box .v.bad { color: #ef4444; }
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
