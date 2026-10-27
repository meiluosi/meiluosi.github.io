<script lang="ts">
/**
 * Lens 02: 心理测量与能力结构透镜 (g-Factor & Process Overlap Spectrum)
 * 揭示：观察到一个统计上的共同因素 (g)，并不等于找到了单一的生物物理发动机。
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
let processOverlap = $state(0.65); // 0.1 ~ 0.95

// 4 项认知任务
const tasks = [
	{ nameZh: "工作记忆广度", nameEn: "Working Memory", base: 0.8 },
	{ nameZh: "空间三维旋转", nameEn: "Spatial Rotation", base: 0.6 },
	{ nameZh: "类比归纳推理", nameEn: "Analogical Reasoning", base: 0.75 },
	{ nameZh: "感知处理速度", nameEn: "Processing Speed", base: 0.5 },
];

// 计算派生的相关度矩阵与拟合出的 g 贡献率
const correlations = $derived(
	tasks.map((t1, i) =>
		tasks.map((t2, j) => {
			if (i === j) return 1.0;
			const r =
				(t1.base * t2.base * processOverlap + 0.1 * (1 - processOverlap)) * 0.9;
			return Math.min(0.95, Math.max(0.1, r));
		}),
	),
);

// 统计估计的 g-factor 方差解释率 (PCA 第一主成分模拟)
const gVarianceExplained = $derived(
	Math.round((0.35 + processOverlap * 0.45) * 100),
);
</script>

<div class="lens-container">
	<div class="lens-header">
		<span class="lens-tag">COGNITIVE LENS 02</span>
		<h3 class="lens-title">{english ? "The g-Factor & Overlap Simulator" : "过程重叠与 g 因素涌现模拟器"}</h3>
		<p class="lens-subtitle">
			{english
				? "Observe how a shared statistical 'general intelligence' factor emerges from overlapping basic cognitive processes without assuming a single biological engine."
				: "拖动调节底层认知过程的重叠度，观察统计上的‘一般智力因素 (g)’如何自发涌现，而无需预设一个单独的‘智力发动机’。"}
		</p>
	</div>

	<!-- 交互控制器 -->
	<div class="control-panel">
		<div class="control-row">
			<label for="overlap-slider" class="control-label">
				<span>{english ? "Shared Process Overlap (Executive Control / Attention):" : "基础认知过程重叠度 (执行控制 / 注意力共享)："}</span>
				<strong>{(processOverlap * 100).toFixed(0)}%</strong>
			</label>
			<input
				id="overlap-slider"
				type="range"
				min="0.1"
				max="0.95"
				step="0.05"
				bind:value={processOverlap}
				class="overlap-slider"
			/>
		</div>
	</div>

	<!-- 认知相关矩阵与主成分可视化 -->
	<div class="matrix-display">
		<div class="matrix-table-wrap">
			<span class="matrix-caption">{english ? "Inter-Task Correlation Matrix" : "认知任务间成绩相关矩阵"}</span>
			<table class="corr-table">
				<thead>
					<tr>
						<th></th>
						{#each tasks as t}
							<th>{english ? t.nameEn : t.nameZh}</th>
						{/each}
					</tr>
				</thead>
				<tbody>
					{#each tasks as t1, i}
						<tr>
							<td class="row-header">{english ? t1.nameEn : t1.nameZh}</td>
							{#each tasks as _, j}
								{@const val = correlations[i][j]}
								<td
									class="corr-cell"
									style={`background-color: rgba(65, 99, 79, ${val * 0.45}); font-weight: ${val > 0.6 ? '600' : '400'};`}
								>
									{val.toFixed(2)}
								</td>
							{/each}
						</tr>
					{/each}
				</tbody>
			</table>
		</div>

		<!-- 统计涌现读数 -->
		<div class="g-readout">
			<span class="g-label">{english ? "Statistical g-Factor Variance:" : "统计 g 因素方差解释率"}</span>
			<div class="g-gauge">
				<div class="gauge-bar" style={`width: ${gVarianceExplained}%;`}></div>
			</div>
			<div class="g-num">{gVarianceExplained}%</div>
			<p class="g-desc">
				{english
					? `At ${Math.round(processOverlap * 100)}% process overlap, distinct tasks show high correlation, creating an illusion of a unitary biological intelligence faculty.`
					: `当任务调用底层注意力/执行控制的重叠度达到 ${Math.round(processOverlap * 100)}% 时，不同任务自发产生高度正相关（Positive Manifold），在统计上被测量为一个强大的 g 因素。`}
			</p>
		</div>
	</div>
</div>

<style>
	.lens-container {
		background: rgb(250 251 248 / 95%);
		border: 1px solid var(--brand-line);
		border-radius: 10px;
		padding: 24px 28px;
		margin-bottom: 32px;
	}

	.lens-tag {
		font-family: var(--font-mono, monospace);
		font-size: 11px;
		letter-spacing: 0.14em;
		color: var(--brand-clay);
		font-weight: 600;
	}

	.lens-title {
		margin: 4px 0 6px;
		font-family: var(--font-serif, "Songti SC", Georgia, serif);
		font-size: 22px;
		color: var(--brand-ink);
	}

	.lens-subtitle {
		margin: 0 0 18px;
		font-size: 13.5px;
		color: var(--brand-muted);
		line-height: 1.6;
	}

	.control-panel {
		padding: 14px 18px;
		background: var(--brand-surface);
		border-radius: 8px;
		margin-bottom: 18px;
	}

	.control-label {
		display: flex;
		justify-content: space-between;
		font-size: 13px;
		color: var(--brand-body);
		margin-bottom: 8px;
	}

	.overlap-slider {
		width: 100%;
		accent-color: var(--brand-clay);
		cursor: pointer;
	}

	.matrix-display {
		display: grid;
		grid-template-columns: 1.4fr 1fr;
		gap: 20px;
		align-items: center;
	}

	.matrix-caption {
		font-size: 12px;
		font-family: var(--font-mono, monospace);
		color: var(--brand-muted);
		display: block;
		margin-bottom: 8px;
	}

	.corr-table {
		width: 100%;
		border-collapse: collapse;
		font-size: 12px;
		text-align: center;
	}

	.corr-table th, .corr-table td {
		padding: 6px 4px;
		border: 1px solid var(--brand-line);
	}

	.corr-table th {
		font-size: 11px;
		color: var(--brand-muted);
		font-weight: 500;
	}

	.row-header {
		text-align: left;
		font-size: 11px;
		color: var(--brand-ink);
		font-weight: 500;
		white-space: nowrap;
	}

	.corr-cell {
		font-family: var(--font-mono, monospace);
		color: var(--brand-ink);
	}

	.g-readout {
		padding: 18px 20px;
		border: 1px solid var(--brand-line);
		border-radius: 8px;
		background: var(--brand-paper);
		text-align: center;
	}

	.g-label {
		font-size: 12px;
		color: var(--brand-muted);
		display: block;
		margin-bottom: 6px;
	}

	.g-gauge {
		width: 100%;
		height: 10px;
		border-radius: 999px;
		background: var(--brand-surface);
		overflow: hidden;
		margin-bottom: 8px;
	}

	.gauge-bar {
		height: 100%;
		background: var(--brand-clay);
		transition: width 0.2s ease;
	}

	.g-num {
		font-family: var(--font-serif, serif);
		font-size: 32px;
		font-weight: 600;
		color: var(--brand-clay);
		margin-bottom: 6px;
	}

	.g-desc {
		margin: 0;
		font-size: 12px;
		line-height: 1.6;
		color: var(--brand-body);
		text-align: left;
	}

	@media (max-width: 680px) {
		.matrix-display {
			grid-template-columns: 1fr;
		}
	}
</style>
