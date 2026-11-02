<script lang="ts">
import { onMount } from "svelte";
import { playLatchClick, playMechanicalTick } from "@/utils/haptics";

interface RelatedEvidence {
	label: string;
	url: string;
}

interface AxisItem {
	id: string;
	title: string;
	judgment: string;
	objection: string;
	test: string;
	conclusions: string[];
	questionIds: string[];
	relatedEvidence: RelatedEvidence[];
}

interface ComparisonItem {
	type: string;
	title: string;
	text: string;
	axes: string[];
}

let {
	axes = [],
	comparisons = [],
}: {
	axes: AxisItem[];
	comparisons: ComparisonItem[];
} = $props();

let selectedAxisId = $state<string>("all");
let searchQuery = $state("");
let english = $state(false);

const axisTitleEn: Record<string, string> = {
	performance: "Performance, Capability & Generalization",
	mechanism: "Correlation Structure & Generative Mechanism",
	development: "Learning, Development & Open Environments",
	boundary: "Body, Environment & System Boundaries",
	collective: "Collective Coordination & Culture",
	semantics: "Representation, Reference & Meaning",
	experience: "Access, Consciousness & Explanation",
	norms: "Resources, Goals & Values",
	physical: "Formalism, Realization & Physical Hypotheses",
};

const activeAxis = $derived(axes.find((a) => a.id === selectedAxisId) ?? null);

const filteredAxes = $derived(
	axes.filter((axis) => {
		if (!searchQuery.trim()) return true;
		const q = searchQuery.trim().toLowerCase();
		return (
			axis.title.toLowerCase().includes(q) ||
			(axisTitleEn[axis.id]?.toLowerCase().includes(q) ?? false) ||
			axis.judgment.toLowerCase().includes(q) ||
			axis.objection.toLowerCase().includes(q) ||
			axis.test.toLowerCase().includes(q)
		);
	}),
);

const matchingComparisons = $derived(
	selectedAxisId === "all"
		? comparisons
		: comparisons.filter((c) => c.axes.includes(selectedAxisId)),
);

function selectDomain(axisId: string) {
	selectedAxisId = selectedAxisId === axisId ? "all" : axisId;
	playLatchClick();
}

function jumpToDomainSection(axisId: string) {
	playMechanicalTick();
	const target = document.getElementById(axisId);
	target?.scrollIntoView({ behavior: "smooth", block: "center" });
}

onMount(() => {
	const syncLocale = () => {
		english = document.documentElement.lang.toLowerCase().startsWith("en");
	};
	syncLocale();
	document.addEventListener("site-locale-change", syncLocale);
	return () => {
		document.removeEventListener("site-locale-change", syncLocale);
	};
});
</script>

<section class="interactive-qmap" aria-label="九大问题域导览与分歧关系导引器">
	<div class="qmap-header">
		<div>
			<span class="qmap-kicker">QUESTION DOMAIN NAVIGATOR · 9 AXES · 15 CARDS</span>
			<h3>
				{english
					? "Nine Question Domains Navigator & Disagreement Guide"
					: "九大问题域导览与分歧关系导引器"}
			</h3>
		</div>
		<div class="qmap-search-box">
			<input
				type="search"
				placeholder={english
					? "Filter domains, objections or tests…"
					: "搜索判断、强反方或区分检验…"}
				bind:value={searchQuery}
			/>
			{#if selectedAxisId !== "all" || searchQuery}
				<button
					type="button"
					class="reset-pill"
					onclick={() => {
						selectedAxisId = "all";
						searchQuery = "";
						playLatchClick();
					}}
				>
					{english ? "Reset Filter" : "显示全部 (9)"}
				</button>
			{/if}
		</div>
	</div>

	<div class="domain-pills-grid">
		{#each filteredAxes as axis, idx}
			<button
				type="button"
				class="domain-card-btn"
				class:active={selectedAxisId === axis.id}
				onclick={() => selectDomain(axis.id)}
			>
				<span class="domain-no">0{idx + 1}</span>
				<strong>{english ? (axisTitleEn[axis.id] ?? axis.title) : axis.title}</strong>
				<span class="domain-meta">
					{axis.conclusions.length} {english ? "conclusions" : "条结论"} · {axis.questionIds.length} {english ? "cards" : "张问题卡"}
				</span>
			</button>
		{/each}
	</div>

	{#if activeAxis}
		<div class="active-domain-spotlight">
			<div class="spotlight-top">
				<div>
					<span class="qmap-kicker">ACTIVE DOMAIN SPOTLIGHT</span>
					<h4>{activeAxis.title}</h4>
				</div>
				<button
					type="button"
					class="jump-btn"
					onclick={() => jumpToDomainSection(activeAxis.id)}
				>
					{english ? "Scroll to full section ↓" : "定位至下方完整条目 ↓"}
				</button>
			</div>
			<div class="spotlight-tri-grid">
				<div class="tri-cell">
					<span class="cell-tag">当前判断 · JUDGMENT</span>
					<p>{activeAxis.judgment}</p>
				</div>
				<div class="tri-cell warn">
					<span class="cell-tag">强反方 · STRONG OBJECTION</span>
					<p>{activeAxis.objection}</p>
				</div>
				<div class="tri-cell test">
					<span class="cell-tag">如何区分 · DISTINGUISHING TEST</span>
					<p>{activeAxis.test}</p>
				</div>
			</div>
			{#if matchingComparisons.length > 0}
				<div class="connected-bridges">
					<span class="bridge-label">关联的跨路线互补与分歧：</span>
					<div class="bridge-chips">
						{#each matchingComparisons as comp}
							<span class="bridge-chip">
								<b>[{comp.type}]</b> {comp.title}
							</span>
						{/each}
					</div>
				</div>
			{/if}
		</div>
	{/if}
</section>

<style>
	.interactive-qmap{border:1px solid var(--brand-line);border-radius:12px;padding:18px 20px;margin:20px 0 28px;background:color-mix(in srgb,var(--brand-sage) 12%,transparent)}
	.qmap-header{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:flex-end;gap:12px;margin-bottom:14px}
	.qmap-kicker{display:block;font:11px "JetBrains Mono",monospace;letter-spacing:.08em;color:var(--brand-clay)}
	.qmap-header h3{font:500 19px/1.4 "Noto Serif SC","Songti SC",Georgia,serif;margin:4px 0 0;color:var(--brand-ink)}
	.qmap-search-box{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
	.qmap-search-box input{font:inherit;font-size:13px;padding:6px 12px;border:1px solid var(--brand-line);border-radius:6px;background:var(--brand-paper);color:var(--brand-ink);min-width:210px}
	.reset-pill,.jump-btn{font:inherit;font-size:12px;padding:6px 12px;border:1px solid var(--brand-clay);border-radius:999px;background:var(--brand-peach);color:var(--brand-ink);cursor:pointer;font-weight:500}
	.domain-pills-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}
	.domain-card-btn{text-align:left;padding:10px 12px;border:1px solid var(--brand-line);border-radius:8px;background:color-mix(in srgb,var(--brand-paper) 84%,transparent);color:var(--brand-ink);cursor:pointer;display:flex;flex-direction:column;gap:3px;transition:border-color .15s,background .15s}
	.domain-card-btn:hover{border-color:var(--brand-accent)}
	.domain-card-btn.active{background:var(--brand-sage);border-color:var(--brand-accent);box-shadow:inset 3px 0 0 var(--brand-clay)}
	.domain-no{font:11px "JetBrains Mono",monospace;color:var(--brand-clay)}
	.domain-card-btn strong{font-size:13.5px;line-height:1.35}
	.domain-meta{font-size:11.5px;color:var(--brand-muted)}
	.active-domain-spotlight{margin-top:14px;padding:14px 16px;border:1px solid var(--brand-line);border-radius:8px;background:var(--brand-paper)}
	.spotlight-top{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:8px;margin-bottom:10px}
	.spotlight-top h4{font:500 17px/1.4 "Noto Serif SC","Songti SC",Georgia,serif;margin:2px 0 0}
	.spotlight-tri-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}
	.tri-cell{padding:10px 12px;border-radius:6px;border:1px solid var(--brand-line);background:color-mix(in srgb,var(--brand-sage) 10%,transparent);font-size:13px;line-height:1.65}
	.tri-cell.warn{background:color-mix(in srgb,var(--brand-peach) 18%,transparent);border-color:color-mix(in srgb,var(--brand-clay) 45%,transparent)}
	.tri-cell.test{background:color-mix(in srgb,var(--brand-paper) 90%,transparent);border-left:3px solid var(--brand-accent)}
	.cell-tag{display:block;font:600 10.5px "JetBrains Mono",monospace;color:var(--brand-muted);margin-bottom:4px}
	.tri-cell p{margin:0;color:var(--brand-body)}
	.connected-bridges{margin-top:10px;padding-top:10px;border-top:1px dashed var(--brand-line);display:flex;flex-wrap:wrap;align-items:center;gap:8px;font-size:12px}
	.bridge-label{color:var(--brand-muted)}
	.bridge-chips{display:flex;flex-wrap:wrap;gap:6px}
	.bridge-chip{padding:3px 9px;border-radius:999px;background:color-mix(in srgb,var(--brand-sage) 22%,transparent);color:var(--brand-ink)}
	@media(max-width:760px){
		.domain-pills-grid,.spotlight-tri-grid{grid-template-columns:1fr}
	}
</style>
