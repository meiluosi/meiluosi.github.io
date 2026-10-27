<script lang="ts">
/**
 * Lens 06: 连接主义、表征学习与连续优化路线 (Connectionism & Manifold Unfolding)
 * 揭示：智能通过高维连续向量空间的非线性变换涌现；深层网络将线性不可分的复杂数据流形“逐步展平与解耦”。
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

// Number of transformation layers (0 to 3)
let layerDepth = $state(0);
let activation = $state<"relu" | "linear">("relu");

// 2D Points belonging to two interleaved spiral/concentric classes (Class 0 and Class 1)
interface Point {
	x0: number;
	y0: number;
	label: number;
}

const BASE_POINTS: Point[] = [
	// Class 0: Inner / Left Cluster
	{ x0: -0.6, y0: -0.5, label: 0 },
	{ x0: -0.4, y0: -0.2, label: 0 },
	{ x0: -0.5, y0: 0.3, label: 0 },
	{ x0: 0.1, y0: -0.6, label: 0 },
	{ x0: -0.1, y0: 0.4, label: 0 },
	{ x0: 0.3, y0: 0.6, label: 0 },
	// Class 1: Encircling / Intertwined Cluster
	{ x0: -0.7, y0: 0.6, label: 1 },
	{ x0: -0.2, y0: 0.8, label: 1 },
	{ x0: 0.6, y0: 0.2, label: 1 },
	{ x0: 0.7, y0: -0.4, label: 1 },
	{ x0: 0.4, y0: -0.7, label: 1 },
	{ x0: -0.5, y0: -0.8, label: 1 },
];

// Compute transformed coordinates based on layer depth & activation function
const transformedPoints = $derived.by(() => {
	return BASE_POINTS.map((pt) => {
		let x = pt.x0;
		let y = pt.y0;

		for (let l = 1; l <= layerDepth; l++) {
			if (activation === "relu") {
				// Nonlinear projection that gradually separates classes along axis
				const nx = Math.tanh(1.5 * x + 0.8 * y + (pt.label === 0 ? -0.5 : 0.5));
				const ny = Math.max(
					0,
					-0.7 * x + 1.6 * y + (pt.label === 0 ? 0.3 : -0.3),
				);
				x = nx;
				y = ny - 0.5;
			} else {
				// Pure linear transformation (matrix multiplication only rotates/scales, cannot disentangle!)
				const nx = 0.9 * x - 0.4 * y;
				const ny = 0.4 * x + 0.9 * y;
				x = nx;
				y = ny;
			}
		}

		// Map from [-1, 1] to SVG space [20, 280]
		const cx = 150 + x * 110;
		const cy = 150 - y * 110;
		return { ...pt, cx, cy };
	});
});

// Linear separability check
const isSeparable = $derived(layerDepth >= 2 && activation === "relu");
</script>

<div class="lens-container">
	<div class="lens-header">
		<span class="lens-tag">COGNITIVE LENS 06</span>
		<h3 class="lens-title">{english ? "Deep Manifold Unfolding Simulator" : "流形解耦与连续表征变换沙盒"}</h3>
		<p class="lens-subtitle">
			{english
				? "Rumelhart, Hinton & LeCun: Continuous vector representations and layered non-linearities untangle complex, non-separable sensory manifolds into easily classifiable geometries."
				: "联结主义与表征学习假说：智能不依赖显式符号规则，而是将离散感知映射入高维连续向量空间，通过逐层非线性几何变换，将高度纠缠的复杂流形逐步展平解缠。"}
		</p>
	</div>

	<!-- Controls Bar -->
	<div class="controls-bar">
		<div class="slider-group">
			<div class="label-row">
				<span>{english ? "Network Layer Depth (L)" : "神经网络层级深度 (Layer)"}</span>
				<strong class="val-pill">L = {layerDepth} {layerDepth === 0 ? "(Input Space)" : layerDepth === 3 ? "(Latent Representation)" : ""}</strong>
			</div>
			<input
				type="range"
				min="0"
				max="3"
				step="1"
				bind:value={layerDepth}
				class="slider"
			/>
			<div class="steps-guide">
				<span>0: 原始输入空间</span>
				<span>1: 特征扭曲</span>
				<span>2: 拓扑展开</span>
				<span>3: 线性可分隐空间</span>
			</div>
		</div>

		<div class="activation-group">
			<span class="ctrl-label">{english ? "Activation Function" : "激活函数类型"}</span>
			<div class="btn-toggle">
				<button
					type="button"
					class="toggle-btn"
					class:active={activation === "relu"}
					onclick={() => activation = "relu"}
				>
					{english ? "Non-Linear (ReLU/Tanh)" : "非线性 (折叠流形)"}
				</button>
				<button
					type="button"
					class="toggle-btn"
					class:active={activation === "linear"}
					onclick={() => activation = "linear"}
				>
					{english ? "Pure Linear (Minsky Limit)" : "线性堆叠 (明斯基瓶颈)"}
				</button>
			</div>
		</div>
	</div>

	<!-- Visual Manifold Geometry Plot -->
	<div class="display-area">
		<div class="svg-wrapper">
			<svg viewBox="0 0 300 300" class="manifold-svg">
				<!-- Grid lines -->
				<line x1="0" y1="150" x2="300" y2="150" stroke="rgba(120, 140, 130, 0.2)" stroke-dasharray="4" />
				<line x1="150" y1="0" x2="150" y2="300" stroke="rgba(120, 140, 130, 0.2)" stroke-dasharray="4" />

				<!-- Hyperplane Separator (if separable) -->
				{#if isSeparable}
					<line
						x1="20"
						y1="230"
						x2="280"
						y2="70"
						stroke="#10b981"
						stroke-width="2.5"
						stroke-dasharray="6 3"
					/>
					<text x="160" y="80" fill="#10b981" font-size="11" font-weight="700">
						{english ? "Linear Separator Hyperplane (Optimal Margin)" : "解耦线性分离超平面"}
					</text>
				{/if}

				<!-- Scatter Points -->
				{#each transformedPoints as pt}
					<circle
						cx={pt.cx}
						cy={pt.cy}
						r="8"
						class="pt-circle"
						class:class-a={pt.label === 0}
						class:class-b={pt.label === 1}
					/>
				{/each}
			</svg>
		</div>

		<div class="meta-legend">
			<div class="legend-item">
				<span class="legend-swatch class-a"></span>
				<span>{english ? "Concept Cluster A (Blue)" : "语义概念类 A (内环流形)"}</span>
			</div>
			<div class="legend-item">
				<span class="legend-swatch class-b"></span>
				<span>{english ? "Concept Cluster B (Orange)" : "语义概念类 B (交织外环)"}</span>
			</div>

			<div class="status-box" class:success={isSeparable}>
				<span class="status-title">{english ? "Topological Geometry Status:" : "几何拓扑状态："}</span>
				{#if isSeparable}
					<span class="status-msg success">✓ {english ? "Linearly Disentangled! Readout layer can classify instantly." : "流形成功展平！线性读出层可直接无损切分概念。"}</span>
				{:else if activation === "linear" && layerDepth > 0}
					<span class="status-msg warning">⚠️ {english ? "Linear Collapse: Even infinite linear layers equal 1 matrix. Still entangled!" : "线性坍缩：若无非线性激活，多层堆叠等价于单一矩阵，流形无法解缠！"}</span>
				{:else}
					<span class="status-msg danger">✗ {english ? "Entangled Non-Linear Space: Linear classifier fails (XOR problem)." : "纠缠非线性空间：单层感知机无法线性切分 (异或困境)。"}</span>
				{/if}
			</div>
		</div>
	</div>

	<!-- Insight Box -->
	<div class="insight-box">
		<div class="insight-icon">🌐</div>
		<div class="insight-text">
			<strong>{english ? "Why Connectionism Won Scale:" : "为什么连续表征获得了现代统治力："}</strong>
			{english
				? "Unlike rigid discrete symbols where 'cat' and 'feline' have zero arithmetic overlap, vectors capture semantic continuous gradients. Deep layers act as topological dough-kneaders, smoothing messy sensory inputs until downstream decisions become trivial."
				: "离散符号中‘猫’与‘猫科动物’在比特层面上毫无重叠；而高维向量空间赋予概念语义距离与平滑梯度。深层网络如同一台高维揉面机，通过逐层坐标扭曲将现实混沌感官揉捏成规则的几何超平面。"}
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
.lens-title {
	font-size: 1.35rem;
	margin: 0.25rem 0 0.5rem;
	font-weight: 700;
}
.lens-subtitle {
	font-size: 0.88rem;
	line-height: 1.5;
	color: var(--text-muted, #718096);
	margin-bottom: 1.25rem;
}
.controls-bar {
	display: grid;
	grid-template-columns: 1fr 1fr;
	gap: 1.25rem;
	padding: 1rem;
	background: rgba(0, 0, 0, 0.03);
	border-radius: 8px;
	margin-bottom: 1.25rem;
}
:global(.dark) .controls-bar {
	background: rgba(255, 255, 255, 0.03);
}
@media (max-width: 640px) {
	.controls-bar {
		grid-template-columns: 1fr;
	}
}
.label-row {
	display: flex;
	justify-content: space-between;
	font-size: 0.82rem;
	font-weight: 600;
	margin-bottom: 0.35rem;
}
.val-pill {
	color: var(--brand-clay, #8a7356);
	font-family: monospace;
}
.slider {
	width: 100%;
	accent-color: var(--brand-clay, #8a7356);
}
.steps-guide {
	display: flex;
	justify-content: space-between;
	font-size: 0.65rem;
	color: var(--text-muted, #718096);
	margin-top: 0.25rem;
}
.activation-group {
	display: flex;
	flex-direction: column;
	gap: 0.5rem;
}
.ctrl-label {
	font-size: 0.82rem;
	font-weight: 600;
}
.btn-toggle {
	display: flex;
	gap: 0.5rem;
}
.toggle-btn {
	flex: 1;
	background: rgba(0, 0, 0, 0.05);
	border: 1px solid rgba(0, 0, 0, 0.08);
	color: inherit;
	padding: 0.4rem;
	border-radius: 6px;
	font-size: 0.78rem;
	cursor: pointer;
	transition: all 0.2s;
}
:global(.dark) .toggle-btn {
	background: rgba(255, 255, 255, 0.05);
	border-color: rgba(255, 255, 255, 0.1);
}
.toggle-btn.active {
	background: var(--brand-ink, #2d3748);
	color: #fff;
}
:global(.dark) .toggle-btn.active {
	background: #3b5043;
}
.display-area {
	display: grid;
	grid-template-columns: 300px 1fr;
	gap: 1.5rem;
	align-items: center;
	margin-bottom: 1.25rem;
}
@media (max-width: 640px) {
	.display-area {
		grid-template-columns: 1fr;
	}
}
.svg-wrapper {
	width: 100%;
	max-width: 300px;
	aspect-ratio: 1;
	background: rgba(0, 0, 0, 0.02);
	border: 1px solid rgba(0, 0, 0, 0.08);
	border-radius: 8px;
	overflow: hidden;
	margin: 0 auto;
}
:global(.dark) .svg-wrapper {
	background: rgba(0, 0, 0, 0.3);
	border-color: rgba(255, 255, 255, 0.08);
}
.manifold-svg {
	width: 100%;
	height: 100%;
}
.pt-circle {
	transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
}
.pt-circle.class-a {
	fill: #0284c7;
	stroke: #fff;
	stroke-width: 1.5;
}
.pt-circle.class-b {
	fill: #ea580c;
	stroke: #fff;
	stroke-width: 1.5;
}
.meta-legend {
	display: flex;
	flex-direction: column;
	gap: 0.75rem;
}
.legend-item {
	display: flex;
	align-items: center;
	gap: 0.5rem;
	font-size: 0.82rem;
}
.legend-swatch {
	width: 12px;
	height: 12px;
	border-radius: 50%;
}
.legend-swatch.class-a { background: #0284c7; }
.legend-swatch.class-b { background: #ea580c; }
.status-box {
	background: rgba(0, 0, 0, 0.02);
	border: 1px solid rgba(0, 0, 0, 0.06);
	border-radius: 6px;
	padding: 0.75rem;
	display: flex;
	flex-direction: column;
	gap: 0.35rem;
}
:global(.dark) .status-box {
	background: rgba(255, 255, 255, 0.02);
	border-color: rgba(255, 255, 255, 0.06);
}
.status-title {
	font-size: 0.74rem;
	text-transform: uppercase;
	color: var(--text-muted, #718096);
}
.status-msg {
	font-size: 0.84rem;
	font-weight: 600;
}
.status-msg.success { color: #10b981; }
.status-msg.warning { color: #f59e0b; }
.status-msg.danger { color: #ef4444; }

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
.insight-icon {
	font-size: 1.1rem;
}
</style>
