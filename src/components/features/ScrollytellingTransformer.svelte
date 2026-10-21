<script lang="ts">
import { onMount } from "svelte";
import {
	playLatchClick,
	playMechanicalTick,
	playSlideTick,
} from "@/utils/haptics";

interface Act {
	id: string;
	titleZh: string;
	titleEn: string;
	subtitleZh: string;
	subtitleEn: string;
	formula: string;
	proseZh: string[];
	proseEn: string[];
}

const acts: Act[] = [
	{
		id: "act-1",
		titleZh: "第一幕：词元投影与连续向量流形",
		titleEn: "Act I: Discrete Tokens to Vector Manifolds",
		subtitleZh: "离散符号的高维连续化表征",
		subtitleEn: "Projecting discrete symbols into dense vector space",
		formula: "X_0 = W_e \\cdot \\text{one\\_hot}(w_t)",
		proseZh: [
			"自然语言是由离散符号构成的链条。计算机无法直接理解‘注意力’这三个汉字，因此必须通过嵌入矩阵（Embedding Matrix）将其映射为连续的高维向量。",
			"在这个 16 维的隐层几何空间中，语义相近的词元（如‘神经网络’与‘反向传播’）在欧几里得距离与余弦相似度上自然聚拢，构成算法可微优化的基础表征。",
		],
		proseEn: [
			"Natural language is a sequence of discrete symbols. Before transformers can reason, tokens must be mapped into continuous high-dimensional vector representations via an embedding matrix.",
			"In this 16-dimensional geometric space, semantically affine words naturally cluster together in cosine distance, laying the foundation for differentiable matrix operations.",
		],
	},
	{
		id: "act-2",
		titleZh: "第二幕：旋转位置编码（RoPE）的时空星盘",
		titleEn: "Act II: Rotary Position Embedding (RoPE) Phase Astrolabe",
		subtitleZh: "二维复数旋转赋能相对距离感知",
		subtitleEn: "Relative positional awareness via 2D complex plane rotations",
		formula:
			"\\mathbf{R}_{\\Theta, m}^d \\vec{x} = \\begin{pmatrix} \\cos m\\theta & -\\sin m\\theta \\\\ \\sin m\\theta & \\cos m\\theta \\end{pmatrix} \\vec{x}",
		proseZh: [
			"自注意力机制本身是置换对称的（Permutation Invariant）——若无位置编码，‘我吃苹果’与‘苹果吃我’在计算上毫无分别。",
			"现代前沿大模型普遍采用 RoPE。它将向量切分为两两成对的特征通道，并随着时序位置 m 进行角度为 mθ 的逆时针旋转。点积时，两词元之间的几何内积仅取决于相对距离 (m - n)，完美契合长文本外推需求。",
		],
		proseEn: [
			"Self-attention by default is strictly permutation invariant: without positional hints, words shuffle with zero structural awareness.",
			"Modern flagship architectures adopt RoPE: pairs of dimensions are rotated by angle mθ on 2D complex sub-planes. The resulting inner product depends purely on the relative distance (m - n), enabling natural length extrapolation.",
		],
	},
	{
		id: "act-3",
		titleZh: "第三幕：因果下三角注意力点积矩阵与掩码切片",
		titleEn: "Act III: Causal Lower-Triangular Attention Dot-Product",
		subtitleZh: "严格禁止窥探未来的因果掩码屏障",
		subtitleEn: "Lower-triangular causal mask preventing future token peeking",
		formula:
			"A = \\text{softmax}\\left( \\frac{Q K^T}{\\sqrt{d_k}} + M_{\\text{causal}} \\right)",
		proseZh: [
			"Query 向量代表‘当前词元在寻找什么’，Key 向量代表‘其他词元具有什么特征’。二者的点积除以标度因子 \\sqrt{d_k} 后，经过 Softmax 归一化为概率权重。",
			"在自回归（Decoder-only）语言模型中，未来位置被施加 -\\infty 强度的因果掩码遮蔽（Causal Mask），使得注意力热力图呈现出严格的下三角阶梯形态，严禁向后偷看。",
		],
		proseEn: [
			"Query vectors represent 'what this token seeks', while Key vectors represent 'what this token offers'. Their dot product scaled by sqrt(d_k) normalizes via Softmax into attention probabilities.",
			"In autoregressive generation, future positions are masked by -inf, producing a strictly lower-triangular causal matrix that preserves the temporal arrow of time.",
		],
	},
	{
		id: "act-4",
		titleZh: "第四幕：多头归纳回路与神经分工消融",
		titleEn: "Act IV: Multi-Head Induction Circuits & Head Ablation",
		subtitleZh: "局部语法归纳与宏观语义连贯的回路分工",
		subtitleEn: "Specialized syntax induction and semantic cohesion circuits",
		formula:
			"\\text{MultiHead}(Q, K, V) = \\text{Concat}(\\text{head}_0, \\text{head}_1) W_O",
		proseZh: [
			"单个注意力头难以兼顾多样化的语言模式，多头注意力（MHA）通过并行子空间实现劳动分工。",
			"在这里：Head 0 专司局部相邻词元的语法过渡与复制回路（[A][B]...[A] ➔ [B]），而 Head 1 负责跨越长距离追踪主题连贯性。一旦通过消融实验归零 Head 0，模型的局部句法立刻出现断崖式崩溃。",
		],
		proseEn: [
			"A single head cannot master diverse language patterns simultaneously. Multi-Head Attention allows heads to specialize across distinct orthogonal sub-spaces.",
			"Here, Head 0 specializes in local syntax transitions and induction copy loops, while Head 1 tracks macro-semantic coherence across paragraphs. Zeroing out Head 0 causes immediate syntactic collapse.",
		],
	},
	{
		id: "act-5",
		titleZh: "第五幕：SwiGLU 前馈门控、温度缩放与最终采样",
		titleEn: "Act V: SwiGLU FFN Gating, Temperature & Next-Token Sampling",
		subtitleZh: "非线性记忆提取与香农熵塌缩",
		subtitleEn: "Non-linear memory gating and probabilistic token collapse",
		formula: "P(w_{t+1}) = \\text{softmax}(\\vec{z} / T)",
		proseZh: [
			"注意力的本质是‘信息聚合’，而前馈网络（FFN）才是存储世界事实的‘非线性知识库’。SwiGLU 通过门控向量与激活函数的乘积，过滤并增强特征表达。",
			"最终，解嵌入头（LM Head）将 16 维表征投影至词表 Logits，并通过采样温度 T 控制分布的尖锐程度。高温让分布更平坦（高熵发散），低温让概率急剧收敛在最高置信度的候选项上。",
		],
		proseEn: [
			"Attention aggregates context, but Feed-Forward Networks store factual knowledge. SwiGLU filters and amplifies representations via SiLU gated non-linear projections.",
			"Finally, the unembedding head projects vectors back to vocabulary logits. Sampling temperature T sharpens or flattens the distribution, collapsing probabilistic quantum uncertainty into the next output token.",
		],
	},
];

let activeActIdx = $state(0);
let english = $state(false);
let canvasEl: HTMLCanvasElement;
let animId: number;

onMount(() => {
	english = document.documentElement.lang.startsWith("en");
	const handleLocale = () => {
		english = document.documentElement.lang.startsWith("en");
	};
	document.addEventListener("site-locale-change", handleLocale);

	// IntersectionObserver for scrollytelling acts
	const observer = new IntersectionObserver(
		(entries) => {
			for (const entry of entries) {
				if (entry.isIntersecting) {
					const idx = Number(entry.target.getAttribute("data-act-idx"));
					if (!Number.isNaN(idx) && idx !== activeActIdx) {
						activeActIdx = idx;
						playSlideTick(1.0 + idx * 0.1);
					}
				}
			}
		},
		{ threshold: 0.55 },
	);

	const sections = document.querySelectorAll(".scrolly-act-step");
	sections.forEach((sec) => {
		observer.observe(sec);
	});

	// Canvas visual loop
	let frame = 0;
	const renderCanvas = () => {
		frame++;
		if (canvasEl) {
			const ctx = canvasEl.getContext("2d");
			if (ctx) {
				const w = canvasEl.width;
				const h = canvasEl.height;
				ctx.clearRect(0, 0, w, h);

				// ACT 0: High-dimensional Vector Manifold
				if (activeActIdx === 0) {
					ctx.save();
					ctx.translate(w / 2, h / 2);
					const numPoints = 16;
					for (let i = 0; i < numPoints; i++) {
						const angle = (i / numPoints) * Math.PI * 2 + frame * 0.015;
						const r = 90 + Math.sin(frame * 0.03 + i) * 25;
						const x = Math.cos(angle) * r;
						const y = Math.sin(angle) * r;
						ctx.beginPath();
						ctx.arc(x, y, 6, 0, Math.PI * 2);
						ctx.fillStyle =
							i % 2 === 0 ? "var(--brand-accent)" : "rgba(224, 122, 95, 0.85)";
						ctx.fill();

						// Vector radiating lines
						ctx.beginPath();
						ctx.moveTo(0, 0);
						ctx.lineTo(x, y);
						ctx.strokeStyle = "rgba(92, 122, 107, 0.25)";
						ctx.stroke();
					}
					// Center origin
					ctx.beginPath();
					ctx.arc(0, 0, 7, 0, Math.PI * 2);
					ctx.fillStyle = "var(--brand-ink)";
					ctx.fill();
					ctx.restore();
				}

				// ACT 1: RoPE Rotary Phase Astrolabe
				else if (activeActIdx === 1) {
					ctx.save();
					ctx.translate(w / 2, h / 2);
					const radius = 100;
					ctx.beginPath();
					ctx.arc(0, 0, radius, 0, Math.PI * 2);
					ctx.strokeStyle = "rgba(92, 122, 107, 0.35)";
					ctx.lineWidth = 2;
					ctx.stroke();

					// Rotating phase hands for token positions m = 0, 1, 2, 3
					for (let m = 0; m < 4; m++) {
						const phase = m * 0.8 + frame * 0.02;
						const x = Math.cos(phase) * radius;
						const y = Math.sin(phase) * radius;
						ctx.beginPath();
						ctx.moveTo(0, 0);
						ctx.lineTo(x, y);
						ctx.strokeStyle =
							m === 0 ? "var(--brand-clay, #e07a5f)" : "var(--brand-accent)";
						ctx.lineWidth = 3;
						ctx.stroke();

						ctx.beginPath();
						ctx.arc(x, y, 6, 0, Math.PI * 2);
						ctx.fillStyle = "var(--brand-accent)";
						ctx.fill();

						ctx.font = "11px monospace";
						ctx.fillStyle = "var(--brand-ink)";
						ctx.fillText(`pos[${m}]`, x + 10, y + 4);
					}
					ctx.restore();
				}

				// ACT 2: Causal Mask Matrix & Arcs
				else if (activeActIdx === 2) {
					ctx.save();
					const cellSize = 38;
					const startX = (w - 4 * cellSize) / 2;
					const startY = (h - 4 * cellSize) / 2;
					for (let r = 0; r < 4; r++) {
						for (let c = 0; c < 4; c++) {
							const x = startX + c * cellSize;
							const y = startY + r * cellSize;
							ctx.beginPath();
							ctx.rect(x, y, cellSize - 3, cellSize - 3);
							if (c > r) {
								ctx.fillStyle = "rgba(200, 200, 200, 0.15)";
								ctx.fill();
							} else {
								const weight = 0.3 + 0.6 * Math.sin(frame * 0.04 + r * c);
								ctx.fillStyle = `rgba(92, 122, 107, ${Math.max(0.2, weight)})`;
								ctx.fill();
							}
							ctx.strokeStyle = "rgba(92, 122, 107, 0.3)";
							ctx.stroke();
						}
					}
					ctx.restore();
				}

				// ACT 3: Multi-Head Induction Circuit
				else if (activeActIdx === 3) {
					ctx.save();
					const cy = h / 2;
					// Head 0 box
					ctx.beginPath();
					ctx.roundRect(w / 4 - 45, cy - 50, 90, 100, [8]);
					ctx.fillStyle = "rgba(92, 122, 107, 0.15)";
					ctx.fill();
					ctx.strokeStyle = "var(--brand-accent)";
					ctx.lineWidth = 2;
					ctx.stroke();

					ctx.font = "bold 12px sans-serif";
					ctx.fillStyle = "var(--brand-accent)";
					ctx.textAlign = "center";
					ctx.fillText("Head 0 (Syntax)", w / 4, cy - 25);

					// Head 1 box
					ctx.beginPath();
					ctx.roundRect((3 * w) / 4 - 45, cy - 50, 90, 100, [8]);
					ctx.fillStyle = "rgba(224, 122, 95, 0.15)";
					ctx.fill();
					ctx.strokeStyle = "var(--brand-clay, #e07a5f)";
					ctx.lineWidth = 2;
					ctx.stroke();

					ctx.fillStyle = "var(--brand-clay, #e07a5f)";
					ctx.fillText("Head 1 (Topic)", (3 * w) / 4, cy - 25);

					// Stream pulses
					const pulse = (Math.sin(frame * 0.08) + 1) * 0.5;
					ctx.beginPath();
					ctx.arc(w / 4, cy + 15, 12 * pulse + 4, 0, Math.PI * 2);
					ctx.fillStyle = "var(--brand-accent)";
					ctx.fill();

					ctx.beginPath();
					ctx.arc((3 * w) / 4, cy + 15, 12 * (1 - pulse) + 4, 0, Math.PI * 2);
					ctx.fillStyle = "var(--brand-clay, #e07a5f)";
					ctx.fill();
					ctx.restore();
				}

				// ACT 4: Probabilities & Temperature
				else if (activeActIdx === 4) {
					ctx.save();
					const candidates = [
						{ name: "模型", prob: 0.62 },
						{ name: "架构", prob: 0.22 },
						{ name: "推断", prob: 0.11 },
						{ name: "注意力", prob: 0.05 },
					];
					const barMaxW = 160;
					const startY = h / 2 - 50;
					candidates.forEach((cand, idx) => {
						const y = startY + idx * 30;
						ctx.font = "12px monospace";
						ctx.fillStyle = "var(--brand-ink)";
						ctx.textAlign = "right";
						ctx.fillText(cand.name, w / 2 - 45, y + 14);

						const barW = cand.prob * barMaxW;
						ctx.beginPath();
						ctx.roundRect(w / 2 - 35, y, barW, 16, [4]);
						ctx.fillStyle =
							idx === 0 ? "var(--brand-accent)" : "rgba(92, 122, 107, 0.35)";
						ctx.fill();

						ctx.textAlign = "left";
						ctx.fillStyle = "var(--brand-muted)";
						ctx.fillText(
							`${(cand.prob * 100).toFixed(0)}%`,
							w / 2 - 30 + barW + 8,
							y + 13,
						);
					});
					ctx.restore();
				}
			}
		}
		animId = requestAnimationFrame(renderCanvas);
	};

	renderCanvas();

	return () => {
		observer.disconnect();
		document.removeEventListener("site-locale-change", handleLocale);
		cancelAnimationFrame(animId);
	};
});
</script>

<div class="scrolly-theatre-container">
	<!-- Left Narrative Column -->
	<div class="scrolly-narrative-col">
		<header class="scrolly-header">
			<span class="theatre-pill">🎭 {english ? "Interactive Scrollytelling Essay" : "交互式长卷双栏叙事"}</span>
			<h2 class="theatre-title">
				{english ? "Transformer In-Depth: A Five-Act Computational Journey" : "Transformer 架构剖析：五幕交互计算巡礼"}
			</h2>
			<p class="theatre-lead">
				{english
					? "Scroll down to experience the mathematical gears of the Transformer architecture unfolding in synchronization with narrative prose."
					: "向下滚动页面，右侧计算舞台将随叙事篇章无缝演变，带你亲眼目睹词元如何一步步汇聚注意力并最终塌缩为下一个预测。"}
			</p>
		</header>

		{#each acts as act, idx}
			<section
				class="scrolly-act-step"
				class:active={activeActIdx === idx}
				data-act-idx={idx}
			>
				<div class="step-card">
					<div class="step-meta">
						<span class="step-badge">ACT 0{idx + 1}</span>
						<span class="step-sub">{english ? act.subtitleEn : act.subtitleZh}</span>
					</div>
					<h3 class="step-heading">{english ? act.titleEn : act.titleZh}</h3>

					<div class="step-formula-box">
						<code>{act.formula}</code>
					</div>

					<div class="step-prose">
						{#each (english ? act.proseEn : act.proseZh) as p}
							<p>{p}</p>
						{/each}
					</div>
				</div>
			</section>
		{/each}

		<footer class="theatre-footer">
			<a href="/lab/transformer-forward/" class="launch-lab-btn" onclick={() => playMechanicalTick(1.2)}>
				⚡ {english ? "Launch Full Micro-Transformer Forward Pass Lab" : "前往完整的 Micro-Transformer 前向推演实验室"} ➔
			</a>
		</footer>
	</div>

	<!-- Right Sticky Interactive Stage -->
	<div class="scrolly-stage-col">
		<div class="sticky-stage-card">
			<div class="stage-header">
				<div class="stage-status">
					<span class="live-dot"></span>
					<strong>{english ? "Live Calculation Stage" : "实时计算舞台"}</strong>
				</div>
				<span class="active-act-tag">
					{acts[activeActIdx] ? (english ? acts[activeActIdx].titleEn.split(":")[0] : acts[activeActIdx].titleZh.split("：")[0]) : ""}
				</span>
			</div>

			<div class="stage-canvas-wrap">
				<canvas
					bind:this={canvasEl}
					width="380"
					height="320"
					class="theatre-canvas"
				></canvas>
			</div>

			<div class="stage-footer">
				<span class="stage-tip">
					{english
						? "Scroll narrative to advance stage dynamically"
						: "滚动左侧篇章，右侧几何模型与激活热力图实时联动变换"}
				</span>
			</div>
		</div>
	</div>
</div>

<style>
	.scrolly-theatre-container {
		display: grid;
		grid-template-columns: 1fr 420px;
		gap: 2rem;
		align-items: start;
		margin: 2.5rem 0;
	}

	@media (max-width: 960px) {
		.scrolly-theatre-container {
			grid-template-columns: 1fr;
		}

		.scrolly-stage-col {
			position: sticky;
			top: 70px;
			z-index: 20;
		}
	}

	.scrolly-header {
		margin-bottom: 2rem;
	}

	.theatre-pill {
		display: inline-block;
		font-size: 11.5px;
		font-weight: 600;
		padding: 3px 10px;
		border-radius: 999px;
		background: rgba(92, 122, 107, 0.12);
		color: var(--brand-accent);
		border: 1px solid var(--brand-line);
		margin-bottom: 0.75rem;
	}

	.theatre-title {
		margin: 0 0 0.75rem;
		font-size: 1.6rem;
		font-weight: 700;
		line-height: 1.3;
		color: var(--brand-ink);
	}

	.theatre-lead {
		font-size: 14px;
		line-height: 1.6;
		color: var(--brand-body);
		margin: 0;
	}

	.scrolly-act-step {
		min-height: 70vh;
		display: flex;
		align-items: center;
		padding: 1rem 0;
	}

	.step-card {
		padding: 1.5rem;
		border-radius: 12px;
		background: var(--brand-surface);
		border: 1px solid var(--brand-line);
		transition: all 0.25s ease;
		width: 100%;
	}

	.scrolly-act-step.active .step-card {
		border-color: var(--brand-accent);
		box-shadow: 0 8px 30px rgba(92, 122, 107, 0.12);
	}

	.step-meta {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		margin-bottom: 0.5rem;
	}

	.step-badge {
		font-family: var(--font-mono, monospace);
		font-size: 11px;
		font-weight: 700;
		color: var(--brand-accent);
		padding: 2px 6px;
		background: rgba(92, 122, 107, 0.15);
		border-radius: 4px;
	}

	.step-sub {
		font-size: 12px;
		color: var(--brand-muted);
	}

	.step-heading {
		margin: 0 0 1rem;
		font-size: 1.25rem;
		color: var(--brand-ink);
		line-height: 1.35;
	}

	.step-formula-box {
		padding: 0.6rem 0.9rem;
		border-radius: 6px;
		background: var(--brand-paper);
		border: 1px dashed var(--brand-line);
		margin-bottom: 1rem;
		font-family: var(--font-mono, monospace);
		font-size: 13px;
		color: var(--brand-accent);
		overflow-x: auto;
	}

	.step-prose p {
		margin: 0 0 0.75rem;
		font-size: 13.5px;
		line-height: 1.65;
		color: var(--brand-body);
	}

	.step-prose p:last-child {
		margin-bottom: 0;
	}

	.theatre-footer {
		margin-top: 2rem;
		padding-bottom: 2rem;
	}

	.launch-lab-btn {
		display: inline-flex;
		align-items: center;
		padding: 0.8rem 1.4rem;
		border-radius: 8px;
		background: var(--brand-accent);
		color: #ffffff;
		font-weight: 600;
		font-size: 13.5px;
		text-decoration: none;
		transition: opacity 0.15s ease;
	}

	.launch-lab-btn:hover {
		opacity: 0.9;
	}

	/* Right Sticky Stage */
	.scrolly-stage-col {
		position: sticky;
		top: 15vh;
	}

	.sticky-stage-card {
		padding: 1.25rem;
		border-radius: 14px;
		background: var(--brand-paper);
		border: 1px solid var(--brand-line);
		box-shadow: 0 6px 24px rgba(0, 0, 0, 0.05);
	}

	.stage-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		margin-bottom: 1rem;
		padding-bottom: 0.65rem;
		border-bottom: 1px solid var(--brand-line);
	}

	.stage-status {
		display: flex;
		align-items: center;
		gap: 6px;
		font-size: 12.5px;
		color: var(--brand-ink);
	}

	.live-dot {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		background: #10b981;
		box-shadow: 0 0 8px #10b981;
	}

	.active-act-tag {
		font-size: 11px;
		font-family: var(--font-mono, monospace);
		color: var(--brand-accent);
		font-weight: 600;
	}

	.stage-canvas-wrap {
		width: 100%;
		display: flex;
		justify-content: center;
		background: var(--brand-surface);
		border-radius: 8px;
		padding: 10px;
		border: 1px solid var(--brand-line);
	}

	.theatre-canvas {
		width: 100%;
		height: auto;
		display: block;
		max-width: 380px;
	}

	.stage-footer {
		margin-top: 0.75rem;
		text-align: center;
	}

	.stage-tip {
		font-size: 11px;
		color: var(--brand-muted);
	}
</style>
