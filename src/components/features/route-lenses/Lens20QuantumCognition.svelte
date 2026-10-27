<script lang="ts">
/**
 * Lens 20: 边缘、异常与形而上学假说路线 (Quantum Decision & Non-Commutative Order Effects)
 * 揭示：经典布尔概率论要求 P(A ∩ B) = P(B ∩ A)；然而人类真实判断常表现出典型的‘非对易次序效应’(Non-commutative Projection)——先提问 A 会使心智叠加态坍缩至子空间 A，从而改变对问题 B 的回答概率。
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

let questionOrder = $state<"A_then_B" | "B_then_A">("A_then_B");

// Busemeyer & Bruza Quantum Cognition experiment (e.g., Clinton-Gore honesty poll / Contextuality)
const experimentResult = $derived.by(() => {
	if (questionOrder === "A_then_B") {
		return {
			seqTitle: "测量次序 1：先问 [A: 宽容直率？] → 再问 [B: 值得信任？]",
			projAngle: 28,
			probYesBoth: "68%",
			formula: "|| P_B · P_A |ψ⟩ ||² = 0.68",
			desc: "先激活‘直率宽容’语义基底 |A⟩，心理状态向量 |ψ⟩ 坍缩至高好感子空间，随后向 |B⟩ 投影时产生相长干涉。",
		};
	}
	return {
		seqTitle: "测量次序 2：先问 [B: 值得信任？] → 再问 [A: 宽容直率？]",
		projAngle: 64,
		probYesBoth: "49%",
		formula: "|| P_A · P_B |ψ⟩ ||² = 0.49",
		desc: "先用严苛审视标准触发‘信任度审查’基底 |B⟩，心智叠加态坍缩至怀疑子空间，导致同一批受试者赞同率骤降 19%！",
	};
});
</script>

<div class="lens-container">
	<div class="lens-header">
		<span class="lens-tag">COGNITIVE LENS 20</span>
		<h3 class="lens-title">{english ? "Quantum Cognition & Non-Commutative Projection" : "量子认知模型与非对易次序效应沙盒"}</h3>
		<p class="lens-subtitle">
			{english
				? "Jerome Busemeyer & Peter Bruza: Note—this is NOT saying the brain is a physical quantum computer, but that human concepts exist in contextual superposition and obey non-commutative Hilbert-space projection rather than classical Boolean set theory."
				: "布세迈尔 (Busemeyer) 量子认知数学模型（注：严禁与‘微管物理量子灵魂’混淆）：人类未决概念并非静态布尔集合，而是处于语境叠加态；提问本身如同一次正交投影测量，改变提问顺序将导致 P(A∧B) ≠ P(B∧A)。"}
		</p>
	</div>

	<div class="controls-bar">
		<button
			type="button"
			class="order-btn"
			class:active={questionOrder === "A_then_B"}
			onclick={() => questionOrder = "A_then_B"}
		>
			算符次序 P_B · P_A |ψ⟩ (先问 A 后问 B)
		</button>
		<button
			type="button"
			class="order-btn"
			class:active={questionOrder === "B_then_A"}
			onclick={() => questionOrder = "B_then_A"}
		>
			算符次序 P_A · P_B |ψ⟩ (先问 B 后问 A)
		</button>
	</div>

	<div class="hilbert-stage">
		<div class="svg-col">
			<svg viewBox="0 0 240 180" class="hilbert-svg">
				<!-- Orthogonal Axes -->
				<line x1="30" y1="150" x2="210" y2="150" stroke="currentColor" stroke-opacity="0.3" />
				<line x1="30" y1="20" x2="30" y2="150" stroke="currentColor" stroke-opacity="0.3" />
				<text x="195" y="166" font-size="10" fill="#0284c7">|A⟩ 子空间</text>
				<text x="40" y="32" font-size="10" fill="#ea580c">|B⟩ 子空间</text>

				<!-- Initial Mental State |psi> -->
				<line x1="30" y1="150" x2="155" y2="55" stroke="#10b981" stroke-width="2.5" stroke-dasharray="4 2" />
				<text x="160" y="52" font-size="10" font-weight="700" fill="#10b981">初始心智态 |ψ⟩</text>

				<!-- Projected Collapse Vector -->
				<line
					x1="30"
					y1="150"
					x2={30 + Math.cos((experimentResult.projAngle * Math.PI) / 180) * 135}
					y2={150 - Math.sin((experimentResult.projAngle * Math.PI) / 180) * 135}
					stroke="var(--brand-clay, #8a7356)"
					stroke-width="3"
				/>
			</svg>
		</div>

		<div class="result-col">
			<div class="seq-badge">{experimentResult.seqTitle}</div>
			<div class="prob-readout">
				<span>联合赞同率：</span>
				<strong>{experimentResult.probYesBoth}</strong>
				<code>{experimentResult.formula}</code>
			</div>
			<p class="exp-desc">{experimentResult.desc}</p>
		</div>
	</div>

	<div class="insight-box">
		<div class="insight-icon">🔮</div>
		<div class="insight-text">
			<strong>{english ? "Methodological Caution on Route 20:" : "第 20 路线的科学方法论边界声明："}</strong>
			{english
				? "We strictly separate mathematically rigorous 'Quantum-like Probabilistic Models of Cognition' (which model contextuality and interference in psychology) from speculative metaphysical claims (Penrose-Hameroff Orch-OR, panpsychism). Openness to anomalies requires falsifiable rigor."
				: "在本路线中，我们严格区分两部分：一是用希尔伯特空间数学工具精确刻画人类语境效应的‘类量子概率认知模型’（已获大量心理学实证）；二是彭罗斯微管引力坍缩、泛心论等形而上学猜想。对边缘假说保持开放，前提是坚守可证伪性与奥卡姆剃刀。"}
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
.controls-bar { display: flex; gap: 0.75rem; margin-bottom: 1.25rem; flex-wrap: wrap; }
.order-btn {
	flex: 1;
	padding: 0.55rem 0.85rem;
	border-radius: 8px;
	background: rgba(0, 0, 0, 0.04);
	border: 1px solid rgba(0, 0, 0, 0.1);
	color: inherit;
	font-size: 0.82rem;
	font-weight: 600;
	cursor: pointer;
}
.order-btn.active { background: var(--brand-ink, #2d3748); color: #fff; }
:global(.dark) .order-btn.active { background: #3b5043; }
.hilbert-stage {
	display: grid;
	grid-template-columns: 240px 1fr;
	gap: 1.25rem;
	align-items: center;
	background: rgba(255, 255, 255, 0.6);
	border: 1px solid rgba(0, 0, 0, 0.06);
	border-radius: 8px;
	padding: 1rem;
	margin-bottom: 1.25rem;
}
:global(.dark) .hilbert-stage { background: rgba(0, 0, 0, 0.25); border-color: rgba(255, 255, 255, 0.06); }
@media (max-width: 640px) { .hilbert-stage { grid-template-columns: 1fr; } }
.hilbert-svg { width: 100%; max-width: 240px; margin: 0 auto; display: block; }
.seq-badge { font-size: 0.8rem; font-weight: 700; color: var(--brand-clay, #8a7356); margin-bottom: 0.5rem; }
.prob-readout { display: flex; align-items: baseline; gap: 0.65rem; flex-wrap: wrap; margin-bottom: 0.5rem; }
.prob-readout strong { font-size: 1.45rem; font-family: monospace; color: #0284c7; }
.prob-readout code { font-size: 0.76rem; background: rgba(0, 0, 0, 0.05); padding: 0.15rem 0.4rem; border-radius: 4px; }
:global(.dark) .prob-readout code { background: rgba(255, 255, 255, 0.08); }
.exp-desc { font-size: 0.82rem; line-height: 1.5; margin: 0; }
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
