<script lang="ts">
import { onMount } from "svelte";

/**
 * Canonical Interactive Explainer Blueprint (Svelte 5)
 * Copy and adapt this template when creating new physics/math/cognitive explainers.
 */
let english = $state(false);
let controlParam = $state(0.5);
let phase = $state(0);
let isPlaying = $state(true);

const simulation = $derived.by(() => {
	const isResonant = Math.abs(controlParam - 0.62) < 0.15;
	const amplitude = isResonant ? 1.0 : 0.35;
	const x = Math.cos(phase) * 55 * amplitude;
	const y = Math.sin(phase * 2) * 35 * amplitude;
	return {
		isResonant,
		amplitude,
		x,
		y,
		statusZh: isResonant ? "黄金共振吸引子 (Resonant Attractor)" : "耗散非共振区 (Damped Regime)",
		statusEn: isResonant ? "Resonant Attractor" : "Damped Regime",
	};
});

onMount(() => {
	const syncLang = () => {
		english = document.documentElement.lang.toLowerCase().startsWith("en");
	};
	syncLang();
	const obs = new MutationObserver(syncLang);
	obs.observe(document.documentElement, {
		attributes: true,
		attributeFilter: ["lang"],
	});

	if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
		isPlaying = false;
	}

	let rafId = 0;
	const tick = () => {
		if (isPlaying) {
			phase = (phase + 0.04) % (Math.PI * 2);
		}
		rafId = requestAnimationFrame(tick);
	};
	rafId = requestAnimationFrame(tick);

	return () => {
		obs.disconnect();
		cancelAnimationFrame(rafId);
	};
});
</script>

<div class="lens-container">
	<div class="lens-header">
		<div class="header-top-row">
			<span class="lens-tag">INTERACTIVE EXPLAINER</span>
			<button type="button" class="anim-toggle" onclick={() => (isPlaying = !isPlaying)}>
				{isPlaying ? (english ? "⏸ Pause" : "⏸ 暂停律动") : (english ? "▶ Play" : "▶ 开启律动")}
			</button>
		</div>
		<h3 class="lens-title">{english ? "Phase-Space Attractor Sandbox" : "相空间吸引子与分岔沙盒"}</h3>
	</div>

	<div class="controls-bar">
		<label class="label-row">
			<span>{english ? "Coupling Parameter (κ)" : "耦合控制参数 (κ)"}</span>
			<strong>{controlParam.toFixed(2)}</strong>
			<input type="range" min="0.1" max="1.0" step="0.02" bind:value={controlParam} />
		</label>
	</div>

	<svg viewBox="-100 -60 200 120" class="stage-svg">
		<circle cx={simulation.x} cy={simulation.y} r="8" fill="#10b981" />
	</svg>
</div>
