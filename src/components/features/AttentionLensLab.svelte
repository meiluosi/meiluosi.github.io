<script lang="ts">
import { onMount } from "svelte";
import {
	type AttentionHeadType,
	CANONICAL_HEADS,
	computeAttentionMatrix,
	SAMPLE_PROMPTS,
	tokenizePrompt,
} from "@/utils/attention-lens";
import { playLatchClick, playMechanicalTick } from "@/utils/haptics";

let english = $state(false);
let rawPrompt = $state("Alice visited Paris . Later Alice visited Rome");
let selectedHead = $state<AttentionHeadType>("induction");
let hoveredCell = $state<{ q: number; k: number; weight: number } | null>(null);
let activeQueryToken = $state<number | null>(null);

const tokens = $derived(tokenizePrompt(rawPrompt));
const lens = $derived(computeAttentionMatrix(tokens, selectedHead));

function selectPreset(text: string) {
	rawPrompt = text;
	playLatchClick();
}

function handleHeadChange(head: AttentionHeadType) {
	selectedHead = head;
	playLatchClick();
}

function handleCellEnter(q: number, k: number, weight: number) {
	hoveredCell = { q, k, weight };
	activeQueryToken = q;
	playMechanicalTick(0.8 + weight * 0.8);
}

function handleCellLeave() {
	hoveredCell = null;
}

onMount(() => {
	const sync = () => {
		english = document.documentElement.lang.startsWith("en");
	};
	sync();
	document.addEventListener("site-locale-change", sync);
	return () => {
		document.removeEventListener("site-locale-change", sync);
	};
});
</script>

<section
	class="attention-lens"
	aria-label={english
		? "Transformer Attention Lens & Circuit Explorer"
		: "Transformer 机制可解释性：注意力头探针"}
>
	<div class="controls-bar">
		<div class="preset-strip">
			<span class="label">{english ? "Presets:" : "典型语料范例:"}</span>
			{#each SAMPLE_PROMPTS as sample}
				<button
					type="button"
					class:active={rawPrompt === sample.text}
					onclick={() => selectPreset(sample.text)}
				>
					{sample.label}
				</button>
			{/each}
		</div>

		<label for="prompt-input" class="prompt-field">
			<span>{english ? "Input Prompt Sequence:" : "输入词元序列 (空格分隔):"}</span>
			<input
				id="prompt-input"
				type="text"
				bind:value={rawPrompt}
				placeholder="Type tokens..."
			/>
		</label>

		<div class="heads-group" role="group" aria-label="Attention head selection">
			{#each CANONICAL_HEADS as head}
				<button
					type="button"
					class="head-pill"
					class:active={selectedHead === head.id}
					onclick={() => handleHeadChange(head.id)}
				>
					<strong>{head.id.toUpperCase()}</strong>
					<span>{english ? head.nameEn : head.nameZh}</span>
				</button>
			{/each}
		</div>
	</div>

	<div class="circuit-banner">
		<div class="banner-text">
			<h4>{english ? lens.headMeta.nameEn : lens.headMeta.nameZh}</h4>
			<p>{english ? lens.headMeta.roleEn : lens.headMeta.roleZh}</p>
		</div>
		<div class="hover-readout">
			{#if hoveredCell}
				<span>Query: <code>{tokens[hoveredCell.q]}</code> (pos {hoveredCell.q})</span>
				<span>Key: <code>{tokens[hoveredCell.k]}</code> (pos {hoveredCell.k})</span>
				<strong>A(q, k) = {(hoveredCell.weight * 100).toFixed(1)}%</strong>
			{:else}
				<span class="hint">
					{english
						? "Hover matrix cell to inspect attention weight & token pair"
						: "悬停矩阵单元格，查看 Query 与 Key 的具体注意力权重"}
				</span>
			{/if}
		</div>
	</div>

	<div class="matrix-wrapper" tabindex="0" role="region" aria-label="Attention matrix">
		<table class="attention-table">
			<thead>
				<tr>
					<th class="corner-header">Q \ K</th>
					{#each tokens as tok, k}
						<th class:focused={hoveredCell?.k === k}>
							<span class="tok-label">{tok}</span>
							<small class="tok-idx">{k}</small>
						</th>
					{/each}
				</tr>
			</thead>
			<tbody>
				{#each lens.matrix as row, q}
					<tr>
						<th class="row-header" class:focused={hoveredCell?.q === q}>
							<span class="tok-label">{tokens[q]}</span>
							<small class="tok-idx">{q}</small>
						</th>
						{#each row as weight, k}
							{#if k <= q}
								<td
									class="matrix-cell"
									class:highlighted={hoveredCell?.q === q && hoveredCell?.k === k}
									style={`background-color: color-mix(in srgb, var(--brand-accent) ${(weight * 100).toFixed(0)}%, var(--brand-paper)); color: ${weight > 0.4 ? "var(--brand-paper)" : "var(--brand-ink)"};`}
									onmouseenter={() => handleCellEnter(q, k, weight)}
									onmouseleave={handleCellLeave}
								>
									{weight > 0.05 ? (weight * 100).toFixed(0) : ""}
								</td>
							{:else}
								<td class="masked-cell" aria-hidden="true"></td>
							{/if}
						{/each}
					</tr>
				{/each}
			</tbody>
		</table>
	</div>

	<div class="flows-display">
		<h4>
			{english
				? "Attention Sinks & Flow Paths (Selected Query Position)"
				: "注意力流向流线 (当前 Query 词元聚焦路径)"}
		</h4>
		<div class="token-flow-strip">
			{#each tokens as tok, idx}
				{@const qIdx = activeQueryToken ?? tokens.length - 1}
				{@const weight = lens.matrix[qIdx]?.[idx] ?? 0}
				<div
					class="flow-token"
					class:is-query={idx === qIdx}
					class:is-attended={weight > 0.15}
				>
					<span class="f-tok">{tok}</span>
					<div class="f-bar">
						<i style={`width:${(weight * 100).toFixed(1)}%`}></i>
					</div>
					<small class="f-pct">{(weight * 100).toFixed(0)}%</small>
				</div>
			{/each}
		</div>
	</div>
</section>

<style>
	.attention-lens {
		border: 1px solid var(--brand-line);
		border-radius: 18px;
		background: var(--brand-surface);
		padding: 24px;
		color: var(--brand-ink);
	}
	.controls-bar {
		display: grid;
		gap: 16px;
		padding-bottom: 20px;
		border-bottom: 1px solid var(--brand-line);
	}
	.preset-strip {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 8px;
	}
	.preset-strip .label {
		font: 11px var(--font-mono);
		color: var(--brand-muted);
	}
	button {
		font: 11px var(--font-mono);
		border: 1px solid var(--brand-line);
		border-radius: 999px;
		padding: 6px 12px;
		background: var(--brand-paper);
		color: var(--brand-body);
		cursor: pointer;
	}
	button.active {
		background: var(--brand-accent);
		color: var(--brand-paper);
		border-color: var(--brand-accent);
	}
	.prompt-field {
		display: grid;
		gap: 6px;
		font-size: 12px;
	}
	.prompt-field input {
		padding: 9px 12px;
		border-radius: 8px;
		border: 1px solid var(--brand-line);
		background: var(--brand-paper);
		color: var(--brand-ink);
		font: 13px var(--font-mono);
	}
	.heads-group {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
	}
	.head-pill {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		border-radius: 8px;
		padding: 6px 12px;
	}
	.head-pill strong {
		color: var(--brand-accent);
	}
	.head-pill.active strong {
		color: var(--brand-paper);
	}
	.circuit-banner {
		display: flex;
		justify-content: space-between;
		align-items: center;
		flex-wrap: wrap;
		gap: 14px;
		margin: 18px 0;
		padding: 14px 18px;
		background: var(--brand-paper);
		border: 1px solid var(--brand-line);
		border-radius: 12px;
	}
	.banner-text h4 {
		margin: 0 0 4px;
		font-size: 14px;
		font-weight: 600;
	}
	.banner-text p {
		margin: 0;
		font-size: 11.5px;
		color: var(--brand-body);
	}
	.hover-readout {
		display: flex;
		align-items: center;
		gap: 12px;
		font: 11px var(--font-mono);
	}
	.hover-readout strong {
		color: var(--brand-accent);
		font-size: 13px;
	}
	.hover-readout .hint {
		color: var(--brand-muted);
	}
	.matrix-wrapper {
		overflow-x: auto;
		background: var(--brand-paper);
		border: 1px solid var(--brand-line);
		border-radius: 12px;
		padding: 16px;
	}
	.attention-table {
		border-collapse: separate;
		border-spacing: 2px;
		font: 10px var(--font-mono);
		margin: 0 auto;
	}
	.corner-header {
		color: var(--brand-muted);
		padding: 4px;
	}
	th {
		padding: 4px 6px;
		font-weight: 500;
		color: var(--brand-body);
		text-align: center;
		vertical-align: bottom;
	}
	th.focused {
		color: var(--brand-accent);
		font-weight: 700;
	}
	.row-header {
		text-align: right;
		padding-right: 8px;
		white-space: nowrap;
	}
	.tok-label {
		display: block;
		max-width: 56px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.tok-idx {
		display: block;
		color: var(--brand-muted);
		font-size: 8.5px;
	}
	.matrix-cell {
		width: 32px;
		height: 32px;
		text-align: center;
		border-radius: 4px;
		cursor: crosshair;
		user-select: none;
		transition: transform 0.08s ease;
	}
	.matrix-cell:hover,
	.matrix-cell.highlighted {
		transform: scale(1.15);
		box-shadow: 0 0 0 2px var(--brand-clay);
		z-index: 2;
	}
	.masked-cell {
		background: color-mix(in srgb, var(--brand-surface) 40%, transparent);
		border-radius: 4px;
	}
	.flows-display {
		margin-top: 18px;
		padding: 16px;
		background: var(--brand-paper);
		border: 1px solid var(--brand-line);
		border-radius: 12px;
	}
	.flows-display h4 {
		margin: 0 0 10px;
		font-size: 13px;
	}
	.token-flow-strip {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(64px, 1fr));
		gap: 8px;
	}
	.flow-token {
		padding: 8px;
		border-radius: 8px;
		background: var(--brand-surface);
		border: 1px solid var(--brand-line);
		display: grid;
		gap: 4px;
		text-align: center;
	}
	.flow-token.is-query {
		border-color: var(--brand-accent);
		background: color-mix(in srgb, var(--brand-sage) 30%, var(--brand-paper));
	}
	.flow-token.is-attended {
		border-color: var(--brand-clay);
	}
	.f-tok {
		font: 11px var(--font-mono);
		font-weight: 500;
	}
	.f-bar {
		height: 5px;
		border-radius: 999px;
		background: var(--brand-paper);
		overflow: hidden;
	}
	.f-bar i {
		display: block;
		height: 100%;
		background: var(--brand-accent);
	}
	.f-pct {
		font: 9.5px var(--font-mono);
		color: var(--brand-muted);
	}
</style>
