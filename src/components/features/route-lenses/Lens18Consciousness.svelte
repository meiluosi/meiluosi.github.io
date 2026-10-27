<script lang="ts">
/**
 * Lens 18: 意识、主观体验与全局工作空间路线 (Baars / Dehaene Global Neuronal Workspace & IIT)
 * 揭示：绝大多数专门化模块在无意识阈下并行运转；唯有当局部信号跨越非线性点火阈值(Ignition Threshold)进入‘全局工作空间’，信息才会向全脑广播并形成主观意识访问。
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

// Stimulus salience / attention amplification (0 to 100)
let stimulusSalience = $state(42);
const IGNITION_THRESHOLD = 65;

const isIgnited = $derived(stimulusSalience >= IGNITION_THRESHOLD);
const phiValue = $derived(
	isIgnited
		? (0.78 + (stimulusSalience - 65) * 0.006).toFixed(2)
		: (stimulusSalience * 0.003).toFixed(2),
);

const MODULES = [
	{ id: "vis", name: "初级视皮层 V1", role: "局部边缘检测" },
	{ id: "sem", name: "颞叶词汇库", role: "语义激活" },
	{ id: "emo", name: "杏仁核效价", role: "情绪显著性评估" },
	{ id: "mot", name: "运动准备区", role: "动作规划" },
	{ id: "mem", name: "海马体情景存贮", role: "长时记忆写入" },
];
</script>

<div class="lens-container">
	<div class="lens-header">
		<span class="lens-tag">COGNITIVE LENS 18</span>
		<h3 class="lens-title">{english ? "Global Neuronal Workspace Ignition" : "全局工作空间非线性点火与广播沙盒"}</h3>
		<p class="lens-subtitle">
			{english
				? "Bernard Baars, Stanislas Dehaene & Giulio Tononi: Unconscious processing is modular and localized. Conscious access occurs via a sudden non-linear phase transition ('Global Ignition') that broadcasts information across all cortical modules."
				: "巴尔斯 (Baars)、迪昂 (Dehaene) 与托诺尼 (IIT)：无意识加工是局部且模块化的；当刺激显著性跨越临界相变点，将触发全脑长程神经元的‘非线性全局点火’(Global Ignition)，向所有模块同步广播并涌现整合信息 Φ。"}
		</p>
	</div>

	<div class="controls-bar">
		<div class="label-row">
			<span>{english ? "Stimulus Duration & Attention Gain" : "输入刺激显著度与自上而下注意增益"}</span>
			<strong class="val-pill" class:ignited={isIgnited}>
				强度 {stimulusSalience}% {isIgnited ? "🔥 [已跨越 65% 点火阈值：全局广播激活]" : "🌑 [阈下无意识局部闪烁]"}
			</strong>
		</div>
		<input type="range" min="10" max="100" step="1" bind:value={stimulusSalience} class="slider" />
	</div>

	<div class="workspace-arena" class:ignited={isIgnited}>
		<div class="central-theater">
			<div class="spotlight-label">
				{isIgnited ? "🌟 全局工作空间中央剧场 (Conscious Broadcast)" : "💤 全局工作空间静默 (Subliminal Processing)"}
			</div>
			<div class="phi-readout">整合信息度 Φ = {phiValue}</div>
		</div>

		<div class="modules-ring">
			{#each MODULES as mod, i}
				{@const active = isIgnited || i === 0}
				<div class="mod-pill" class:broadcast={isIgnited} class:local-only={!isIgnited && i === 0}>
					<span class="m-name">{mod.name}</span>
					<span class="m-role">{isIgnited ? "⚡ 已接收全局广播同步" : i === 0 ? "局部阈下激活" : "未耦合隔离"}</span>
				</div>
			{/each}
		</div>
	</div>

	<div class="insight-box">
		<div class="insight-icon">🕯️</div>
		<div class="insight-text">
			<strong>{english ? "Access Consciousness vs. Phenomenal Qualia:" : "功能性广播与查尔默斯‘意识难问题’："}</strong>
			{english
				? "Global Workspace Theory explains the functional utility of consciousness (cross-module coordination and working memory reportability), while still leaving open Chalmers' Hard Problem: why does global broadcasting feel like anything at all?"
				: "全局工作空间理论完美解释了意识的‘功能性价值’——打破模块壁垒，实现跨脑区协同推理与语言报告；但查尔默斯的‘难问题’依然悬而未决：为什么信息广播会伴随着第一人称的主观感受质 (Qualia)？"}
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
.controls-bar { padding: 1rem; background: rgba(0, 0, 0, 0.03); border-radius: 8px; margin-bottom: 1.25rem; }
:global(.dark) .controls-bar { background: rgba(255, 255, 255, 0.03); }
.label-row { display: flex; justify-content: space-between; font-size: 0.82rem; font-weight: 600; margin-bottom: 0.35rem; flex-wrap: wrap; gap: 0.35rem; }
.val-pill { font-family: monospace; color: var(--text-muted, #64748b); }
.val-pill.ignited { color: #f59e0b; }
.slider { width: 100%; accent-color: #f59e0b; }
.workspace-arena {
	padding: 1.25rem;
	border-radius: 10px;
	background: rgba(0, 0, 0, 0.03);
	border: 1px solid rgba(0, 0, 0, 0.08);
	margin-bottom: 1.25rem;
	transition: all 0.35s ease;
}
:global(.dark) .workspace-arena { background: rgba(0, 0, 0, 0.3); border-color: rgba(255, 255, 255, 0.08); }
.workspace-arena.ignited {
	background: rgba(245, 158, 11, 0.08);
	border-color: #f59e0b;
	box-shadow: 0 0 24px rgba(245, 158, 11, 0.15);
}
.central-theater {
	text-align: center;
	padding: 0.85rem;
	border-radius: 8px;
	background: rgba(255, 255, 255, 0.6);
	margin-bottom: 1rem;
}
:global(.dark) .central-theater { background: rgba(255, 255, 255, 0.05); }
.spotlight-label { font-size: 0.9rem; font-weight: 700; }
.phi-readout { font-size: 0.8rem; font-family: monospace; color: var(--brand-clay, #8a7356); margin-top: 0.25rem; }
.modules-ring {
	display: grid;
	grid-template-columns: repeat(5, 1fr);
	gap: 0.65rem;
}
@media (max-width: 720px) { .modules-ring { grid-template-columns: 1fr 1fr; } }
.mod-pill {
	padding: 0.65rem;
	border-radius: 6px;
	background: rgba(0, 0, 0, 0.03);
	border: 1px solid rgba(0, 0, 0, 0.06);
	display: flex;
	flex-direction: column;
	text-align: center;
	opacity: 0.45;
	transition: all 0.3s;
}
:global(.dark) .mod-pill { background: rgba(255, 255, 255, 0.03); border-color: rgba(255, 255, 255, 0.06); }
.mod-pill.local-only { opacity: 0.9; border-color: #0284c7; }
.mod-pill.broadcast {
	opacity: 1;
	border-color: #f59e0b;
	background: rgba(245, 158, 11, 0.12);
	transform: translateY(-2px);
}
.m-name { font-size: 0.78rem; font-weight: 700; }
.m-role { font-size: 0.68rem; color: var(--text-muted, #718096); margin-top: 0.2rem; }
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
