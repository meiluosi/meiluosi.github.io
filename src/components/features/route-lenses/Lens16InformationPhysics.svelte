<script lang="ts">
/**
 * Lens 16: 信息物理、统计力学与热力学路线 (Tishby Information Bottleneck & Landauer Limit)
 * 揭示：智能是受热力学约束的信息压缩器。最优内部表征 T 必须在‘遗忘无关传感器噪声 I(X;T)’与‘保留目标预测相关互信息 I(T;Y)’之间达到帕累托边界。
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

// Lagrange multiplier beta: tradeoff between compression and prediction relevance (0.1 to 5.0)
let beta = $state(1.8);

const ibState = $derived.by(() => {
	// Rate of information retained from input X: I(X; T) in bits
	const infoCompressed = Number(Math.min(10, beta ** 1.15 * 1.6).toFixed(2));
	// Predictive mutual information about target Y: I(T; Y) saturates near 4.0 bits
	const infoRelevant = Number((4.0 * (1 - Math.exp(-0.9 * beta))).toFixed(2));
	// Landauer thermodynamic dissipation kT ln(2) per erased/processed bit
	const landauerEnergy = Number((infoCompressed * 2.87).toFixed(1));

	let regimeZh = "黄金抽象瓶颈 (Minimal Sufficient Statistic)";
	if (beta < 0.8) regimeZh = "过度压缩欠拟合 (Under-compressed / Lost Signal)";
	if (beta > 3.2) regimeZh = "过拟合死记噪声 (Memorizing Sensory Noise)";

	return {
		infoCompressed,
		infoRelevant,
		landauerEnergy,
		regimeZh,
		cx: 35 + (infoCompressed / 10) * 230,
		cy: 165 - (infoRelevant / 4.2) * 135,
	};
});
</script>

<div class="lens-container">
	<div class="lens-header">
		<span class="lens-tag">COGNITIVE LENS 16</span>
		<h3 class="lens-title">{english ? "Information Bottleneck & Thermodynamic Frontier" : "信息瓶颈 (Information Bottleneck) 与兰道尔能效沙盒"}</h3>
		<p class="lens-subtitle">
			{english
				? "Naftali Tishby, Karl Friston & Rolf Landauer: Intelligence is the art of selective forgetting. An optimal brain squeezes raw sensory data X through a narrow bottleneck T, discarding nuisance noise while preserving predictive power for future states Y."
				: "蒂什比 (Naftali Tishby)、弗里斯顿与兰道尔定律：理解的本质是‘有选择地遗忘’。最优智能系统通过狭窄的信息瓶颈压缩原始感官流 X，滤除无关细节噪声，仅保留对未来状态 Y 具预测价值的最小充分统计量。"}
		</p>
	</div>

	<div class="controls-bar">
		<div class="label-row">
			<span>{english ? "Lagrange Multiplier β (Compression vs. Relevance Trade-off)" : "信息瓶颈拉格朗日乘子 β (压缩率 vs 预测保真度权衡)"}</span>
			<strong class="val-pill">β = {beta.toFixed(1)} — {ibState.regimeZh}</strong>
		</div>
		<input type="range" min="0.2" max="4.5" step="0.1" bind:value={beta} class="slider" />
		<div class="scale-hints">
			<span>← 极致抽象遗忘 (低能耗/丢失细节)</span>
			<span>★ 最小充分统计量 (β ≈ 1.8)</span>
			<span>保留全量像素噪声 (过拟合/高能耗) →</span>
		</div>
	</div>

	<div class="ib-workspace">
		<div class="curve-box">
			<svg viewBox="0 0 300 190" class="ib-svg">
				<!-- Axes -->
				<line x1="35" y1="165" x2="280" y2="165" stroke="currentColor" stroke-opacity="0.3" />
				<line x1="35" y1="20" x2="35" y2="165" stroke="currentColor" stroke-opacity="0.3" />
				<text x="160" y="184" text-anchor="middle" font-size="9" fill="var(--text-muted, #64748b)">
					复杂度/保留输入比特 I(X; T) [Bits] →
				</text>
				<text x="12" y="95" text-anchor="middle" font-size="9" transform="rotate(-90, 12, 95)" fill="var(--text-muted, #64748b)">
					预测有效互信息 I(T; Y) →
				</text>

				<!-- Unachievable Forbidden Region -->
				<path d="M 35 165 Q 80 30 270 30 L 35 30 Z" fill="rgba(239, 68, 68, 0.06)" />
				<text x="95" y="48" font-size="8" fill="#ef4444">物理不可能区 (违反数据处理不等式)</text>

				<!-- Pareto Information Bottleneck Curve -->
				<path
					d="M 35 165 Q 85 38 270 35"
					fill="none"
					stroke="#0284c7"
					stroke-width="2.5"
				/>

				<!-- Current Operating Point -->
				<circle cx={ibState.cx} cy={ibState.cy} r="7" fill="#10b981" stroke="#fff" stroke-width="2" />
			</svg>
		</div>

		<div class="stats-col">
			<div class="stat-box">
				<span class="k">编码表征代价 I(X; T)</span>
				<strong class="v">{ibState.infoCompressed} bits</strong>
			</div>
			<div class="stat-box">
				<span class="k">目标预测价值 I(T; Y)</span>
				<strong class="v green">{ibState.infoRelevant} / 4.00 bits</strong>
			</div>
			<div class="stat-box">
				<span class="k">热力学擦除耗散 (Landauer Cost)</span>
				<strong class="v">{ibState.landauerEnergy} kT·ln2</strong>
			</div>
		</div>
	</div>

	<div class="insight-box">
		<div class="insight-icon">🔥</div>
		<div class="insight-text">
			<strong>{english ? "Compression IS Comprehension:" : "物理学家眼中的‘压缩即理解’："}</strong>
			{english
				? " Memorizing every falling apple's pixel trajectory requires gigabytes of entropy. Compressing all trajectories into F = G(m1*m2)/r² requires only a few bits yet achieves near-perfect prediction I(T;Y). Abstraction is thermodynamic optimization."
				: "死记硬背每一颗苹果落地的像素视频需要数 GB 的信息熵；而将其压缩为牛顿万有引力公式仅需几十比特，却保留了几乎 100% 的未来预测力 I(T;Y)。概念抽象本质上是智能体对抗热力学能耗的极致压缩。"}
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
	padding: 1rem;
	background: rgba(0, 0, 0, 0.03);
	border-radius: 8px;
	margin-bottom: 1.25rem;
}
:global(.dark) .controls-bar { background: rgba(255, 255, 255, 0.03); }
.label-row { display: flex; justify-content: space-between; font-size: 0.82rem; font-weight: 600; margin-bottom: 0.35rem; flex-wrap: wrap; gap: 0.4rem; }
.val-pill { font-family: monospace; color: var(--brand-clay, #8a7356); }
.slider { width: 100%; accent-color: var(--brand-clay, #8a7356); }
.scale-hints { display: flex; justify-content: space-between; font-size: 0.68rem; color: var(--text-muted, #718096); margin-top: 0.25rem; }
.ib-workspace {
	display: grid;
	grid-template-columns: 1.4fr 1fr;
	gap: 1.25rem;
	align-items: center;
	margin-bottom: 1.25rem;
}
@media (max-width: 640px) { .ib-workspace { grid-template-columns: 1fr; } }
.curve-box {
	background: rgba(255, 255, 255, 0.65);
	border: 1px solid rgba(0, 0, 0, 0.06);
	border-radius: 8px;
	padding: 0.75rem;
}
:global(.dark) .curve-box { background: rgba(0, 0, 0, 0.25); border-color: rgba(255, 255, 255, 0.06); }
.ib-svg { width: 100%; height: auto; }
.stats-col { display: flex; flex-direction: column; gap: 0.75rem; }
.stat-box { padding: 0.75rem; background: rgba(0, 0, 0, 0.02); border: 1px solid rgba(0, 0, 0, 0.06); border-radius: 6px; }
:global(.dark) .stat-box { background: rgba(255, 255, 255, 0.02); border-color: rgba(255, 255, 255, 0.06); }
.stat-box .k { font-size: 0.72rem; color: var(--text-muted, #718096); display: block; }
.stat-box .v { font-size: 1.05rem; font-family: monospace; }
.stat-box .v.green { color: #10b981; }
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
