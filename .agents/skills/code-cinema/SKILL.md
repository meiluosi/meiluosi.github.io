---
name: code-cinema
description: >-
  Design, script, and implement multi-minute, code-driven interactive motion explainers (3Blue1Brown,
  Paperclip, and Chai Zhidao style) in Svelte 5 for route monographs, visual essays, and technical concepts.
  Use this skill whenever the user asks to create an animated cinema, long-form programmatic explainer,
  or multi-act scripted interactive theatre.
---

# Code Cinema & Scripted Explainer Theatre Skill

This skill codifies the architecture, modular scene composition, closed-loop mathematical data layer, and Svelte 5 engineering standards for authoring **code-driven interactive motion explainers (Code Cinema)** across the site (`/essays/*`, `/lab/*`, and `/books/*`).

---

## 1. Core Philosophy (Beyond Passive Video)

1. **Scene as a Pure Function of Time (`Scene = f(t, params)`)**
   - Every scene component is a pure SVG vector projection of `(actTime, params)`.
   - All geometric positions, path definitions, color interpolations, and opacities derive deterministically from clamped timeline primitives:
     ```ts
     const progress = seg(actTime, startSec, endSec, ease);
     ```
   - **Zero imperative side effects**: no internal `requestAnimationFrame`, `setTimeout`, or component-level mutable accumulation. The user can scrub, reverse, jump, or play at $2\times$ speed with instantaneous, pixel-perfect fidelity.

2. **Lessons Learned from Remotion & Manim**
   - **Clamped Interpolation**: Primitives must strictly clamp outside active temporal windows (equivalent to Remotion's `interpolate(t, [a, b], [0, 1], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' })`). Unclamped linear functions cause out-of-bounds geometry.
   - **Rate Functions**: Standardize on composable easing curves (`linear`, `easeOut`, `easeInOut`, `easeOutBack`, `easeOutQuart`) mirroring 3Blue1Brown's Manim `rate_func` primitives.
   - **Deterministic Pseudo-Randomness**: Never use bare `Math.random()`. Stochastic demonstrations (e.g., empirical Monte Carlo variance tests, particle scatter, sampling roulettes) must use a seedable PRNG (such as `mulberry32`) with fixed seeds so frames are 100% reproducible across machines and scrub directions.

3. **Separated Mathematical Data Layer (Closed-Loop Ground Truth)**
   - Never hardcode derived calculations, matrix outputs, attention percentages, or token probabilities into scene visuals or subtitle scripts.
   - Keep a dedicated `data/modelData.ts` backed by pure linear algebra functions (`data/tensor.ts`):
     - **Raw Parameters Only**: Word embedding matrix $E$, weight matrices $W_Q, W_K, W_V, W_O, W_1, W_2, W_U$, and positional frequencies.
     - **Real-Time Pure Computations**: Projections $Q, K, V$, scaled inner products $q \cdot k / \sqrt{d_k}$, causal masking ($-\infty$), Softmax distributions, context synthesis $Z = \sum \alpha_i V_i$, LayerNorm, FFN activations, and KV Cache state.
     - **Test-Driven Verification**: Enforce mathematical consistency with an automated test suite (`npx tsx <sceneData.test.ts>`), verifying that attention weights sum to 1, causal masks zero future tokens, and variances stay bounded.

4. **Collision-Free Panoramic Widescreen Layout**
   - Universal stage coordinate system: $1100 \times 600$ vector canvas (`viewBox="-550 -300 1100 600"`).
   - Divide the canvas into **Dedicated Vertical Spatial Lanes** to guarantee zero visual occlusions:
     - **Top Runway Lane ($y \in [-250, -150]$)**: Sequence token pills, context tags, global prompt indicators.
     - **Center Dynamic Amphitheater ($y \in [-130, 130]$)**: Core algorithmic operations—QKV projection gates, attention radar arcs, value vector pulling, neural activations, probability bars.
     - **Bottom Telemetry & Status Lane ($y \in [150, 240]$)**: Live mathematical equations, scalar readouts, LayerNorm statistics, KV cache memory footprint.
   - Respect horizontal bounding budgets ($|x| \le 500$, card gap $\ge 20\text{px}$, explicit label clipping/ellipsis).

5. **Documentary Subtitle Stream & Humble Pedagogical Tone**
   - Subtitles render in an accessible documentary bar outside the vector canvas.
   - Tone must remain evidence-based, humble, and mathematically rigorous. Replace exaggerated dramatizations (e.g. "电击/金刀/斩下") with precise academic mechanisms and literature citations (Vaswani 2017, Sennrich 2016, Clark 2019, Geva 2021, Holtzman 2020).
   - Full bilingual parity: Every entry requires matched `textZh` and `textEn`.

6. **Interactive Checkpoints (The Active Stop)**
   - When playback hits an act checkpoint (`pauseAtSec`), the director pauses smoothly.
   - An interactive HUD surfaces, letting the reader tune critical physical/mathematical parameters (e.g. Temperature $T$, Nucleus Top-p, Learning Rate) to observe phase bifurcation in real time before resuming.

---

## 2. Directory & Module Structure

```
src/components/features/cinema/
├── types.ts                     # CinemaScript, CinemaAct, CinemaSubtitle, CinemaCheckpoint
├── timeline.ts                  # seg, easeOut, easeOutBack, fmt, lerp, PALETTE
├── CinemaDirector.svelte        # Master player (scrubber, act chips, subtitles, checkpoint HUD)
├── data/
│   ├── tensor.ts                # Pure functional vector & matrix linear algebra
│   ├── modelData.ts             # Closed-loop mathematical ground truth (source of all numbers)
│   └── modelData.test.ts        # Node:test assertions verifying mathematical closure
├── scenes/
│   ├── Scene1VocabEmbedding.svelte
│   ├── Scene2QKVProjection.svelte
│   ├── Scene3AttentionMask.svelte
│   ├── Scene4SoftmaxTemperature.svelte
│   └── Scene5AutoregressiveClosure.svelte
└── scripts/
    └── transformerScript.ts     # Declarative multi-act script with synchronized timings
```

---

## 3. Data-Scene Closed-Loop Contract

### Linear Algebra Primitives (`data/tensor.ts`)
```ts
export type Vec = number[];
export type Mat = number[][];

export const dot = (a: Vec, b: Vec): number =>
	a.reduce((sum, val, i) => sum + val * b[i], 0);

export const softmax = (logits: Vec, temperature = 1.0): Vec => {
	const safeT = Math.max(1e-4, temperature);
	const maxL = Math.max(...logits);
	const exps = logits.map((l) => Math.exp((l - maxL) / safeT));
	const sumExp = exps.reduce((a, b) => a + b, 0);
	return exps.map((e) => e / sumExp);
};
```

### Scene Temporal Progression (`Scene = f(actTime)`)
```svelte
<script lang="ts">
import { HEAD1, HEAD2 } from "../data/modelData";
import { easeInOut, easeOut, fmt, PALETTE, seg } from "../timeline";

interface Props {
	actTime: number; // 0 <= actTime <= actDurationSec
	english?: boolean;
}

const { actTime, english = false }: Props = $props();

// Clamped progress primitives (0..1)
const enterAlpha = $derived(seg(actTime, 0.0, 1.5, easeOut));
const radarSweep = $derived(seg(actTime, 1.5, 6.0, easeInOut));
const weightCardAlpha = $derived(seg(actTime, 5.0, 8.0, easeOut));
</script>

<g opacity={enterAlpha}>
	<!-- Dynamic SVG scene strictly bounded within designated lanes -->
</g>
```

---

## 4. Engineering & Verification Rules

1. **Pure Vector Sharpness**: All animations render in 100% SVG vectors; no heavy canvas or MP4 video clips.
2. **Strict Accessibility**:
   - Keyboard navigation: Spacebar toggles Play/Pause; Arrow keys seek $\pm 5$s.
   - Respect `prefers-reduced-motion`.
   - Provide an accessible text transcript and step-by-step breakdown beneath the player for screen readers and text-first readers.
3. **KaTeX & Audit Compliance**:
   - All in-text mathematical variables and symbols must use valid KaTeX (`$T$`, `$\mathbf{Q}$`, `$W_Q$`, `$\sqrt{d_k}$`, `$\mathcal{O}(1)$`).
   - Chinese annotations inside formulas must be wrapped in `\text{...}`.
4. **Verification Pipeline**:
   - Mathematical tests pass: `npx tsx <sceneData.test.ts>`
   - Biome formatting & linter pass: `pnpm format && pnpm lint`
   - Astro & TypeScript typechecks pass: `pnpm check && pnpm type-check`
   - Production build succeeds: `pnpm build`
