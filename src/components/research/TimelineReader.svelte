<script lang="ts">
import { tick } from "svelte";
import type { TimelineRoute } from "@/data/intelligenceTimelines";
let { route }: { route: TimelineRoute } = $props();
let english = $state(false);
$effect(() => {
	const sync = () => {
		english = document.documentElement.lang.toLowerCase().startsWith("en");
	};
	sync();
	const observer = new MutationObserver(sync);
	observer.observe(document.documentElement, {
		attributes: true,
		attributeFilter: ["lang"],
	});
	return () => observer.disconnect();
});
const tr = (zh: string, en: string) => (english ? en : zh);
const claimLabels: Record<string, string> = {
	"source-reported": "原文主张",
	derived: "本站推导",
	"editorial-inference": "本站分析",
};
const readingLabels: Record<string, string> = {
	"selected-sections": "指定片段",
	"full-text": "正文已读",
	"abstract-only": "仅摘要",
};
let period = $state("all");
let query = $state("");
const nodes = $derived(
	[...route.nodes].sort(
		(a, b) =>
			(a.year ?? Infinity) - (b.year ?? Infinity) || a.id.localeCompare(b.id),
	),
);
const visible = $derived(
	nodes.filter((n) => {
		const match =
			period === "all" ||
			(period === "unknown"
				? n.year === null
				: n.year !== null &&
					(period === "early"
						? n.year < 1950
						: period === "middle"
							? n.year >= 1950 && n.year < 2000
							: n.year >= 2000));
		return (
			match &&
			`${n.title} ${n.question} ${n.mechanism} ${n.branch} ${n.sources.map((s) => s.title).join(" ")}`
				.toLowerCase()
				.includes(query.trim().toLowerCase())
		);
	}),
);
$effect(() => {
	const reveal = (hash: string) => {
		const id = hash.slice(1);
		if (!route.nodes.some((node) => node.id === id)) return;
		period = "all";
		query = "";
		void tick().then(() =>
			document.getElementById(id)?.scrollIntoView({ block: "start" }),
		);
	};
	let handledHash = "";
	const onHash = () => {
		if (handledHash === location.hash) {
			handledHash = "";
			return;
		}
		handledHash = "";
		reveal(location.hash);
	};
	const onClick = (event: MouseEvent) => {
		if (!(event.target instanceof Element)) return;
		const link = event.target.closest("a[href]");
		if (!link) return;
		const target = new URL(link.getAttribute("href")!, location.href);
		if (
			target.origin === location.origin &&
			target.pathname === location.pathname &&
			target.hash &&
			route.nodes.some((node) => `#${node.id}` === target.hash)
		) {
			handledHash = target.hash;
			reveal(target.hash);
		}
	};
	window.addEventListener("hashchange", onHash);
	document.addEventListener("click", onClick);
	return () => {
		window.removeEventListener("hashchange", onHash);
		document.removeEventListener("click", onClick);
	};
});
</script>
<div class="timeline-reader">
	<div class="controls">
		<label>{tr("按年代查看", "Period")}<select bind:value={period} data-period><option value="all">{tr("全部年代", "All periods")}</option><option value="early">{tr("1950年以前", "Before 1950")}</option><option value="middle">1950–1999</option><option value="recent">{tr("2000年至今", "2000 onward")}</option><option value="unknown">{tr("年代未定", "Date uncertain")}</option></select></label>
		<label class="search">{tr("搜索问题、方法或来源", "Search questions, methods or sources")}<input type="search" bind:value={query} data-query maxlength="200" /></label>
		<output aria-live="polite" data-count>{visible.length} / {nodes.length}</output>
	</div>
	<p class="note">{tr("筛选作用于下方节点清单；上方图始终显示完整路线。点击图中的节点会恢复完整清单并跳转。", "Filters apply to the list below; the diagram remains complete. Selecting a diagram node resets the filters. Research notes are in Chinese.")}</p>
	{#if visible.length === 0}<p data-empty>{tr("没有匹配的节点，请调整筛选条件。", "No matching nodes. Adjust the filters.")}</p>{/if}
	{#each visible as node (node.id)}
		<article class="node" id={node.id} data-node={node.id}>
			<div class="meta"><span>{node.dateLabel}</span><span>{node.branch}</span><span>{node.batchLabel}</span></div>
			<h3>{node.title}</h3>
			<dl><dt>{tr("问题", "Question")}</dt><dd>{node.question}</dd><dt>{tr("方法与核心变化", "Approach")}</dt><dd>{node.mechanism}</dd><dt>{tr("限制", "Limits")}</dt><dd>{node.limits}</dd>{#if node.nextQuestion}<dt>{tr("接下来的问题", "Next question")}</dt><dd>{node.nextQuestion}</dd>{/if}</dl>
			<details><summary>{tr("前提、原文定位与阅读范围", "Assumptions, source locations and reading scope")}</summary>
				{#if node.assumptions}<p><strong>{tr("前提：", "Assumptions: ")}</strong>{node.assumptions}</p>{/if}
				<p class="note">{tr("以下是原批次记录的主张类型；“原文报告”与独立复现分别登记。", "Claim types follow the original research batch; source reports and independent replication are distinct.")}</p>
				{#each node.claims as claim (claim.id)}<section class="claim" id={claim.id}><span class="claim-id">{claim.id} · {tr(claimLabels[claim.basis] ?? claim.basis, claim.basis)}{claim.independentlyReplicated ? tr(" · 独立核验", " · independently checked") : ""}</span><p>{claim.text}</p><p class="note">{tr("定位：", "Location: ")}{claim.locator} · {claim.sourceId}</p></section>{/each}
				{#each node.sources as source (source.id)}<section class="source"><h4><a href={source.url} target="_blank" rel="noopener noreferrer">{source.title} ↗</a></h4>{#if source.authors.length}<p>{source.authors.join(" · ")}</p>{/if}{#each source.corrections as correction}<p class="note">{tr("当前书目更正：", "Bibliographic correction: ")}{correction.old} → {correction.new}。{correction.reason} <a href={correction.evidence} target="_blank" rel="noopener noreferrer">{tr("更正依据", "Evidence")} ↗</a></p>{/each}<p><strong>{tr("版本：", "Version: ")}</strong>{source.version}</p><p><strong>{tr("已读：", "Read: ")}</strong>{source.readScope} <span class="note">({tr(readingLabels[source.readingStatus] ?? source.readingStatus, source.readingStatus)})</span></p><p><strong>{tr("未读／未核验：", "Not read / verified: ")}</strong>{source.unreadScope}</p><p class="note">{source.verification}</p></section>{/each}
			</details>
		</article>
	{/each}
</div>
<style>
.timeline-reader{color:var(--brand-ink);line-height:1.8}.controls{display:flex;flex-wrap:wrap;align-items:end;gap:16px;margin:24px 0}.controls label{display:grid;gap:6px;font-size:13px}.controls select,.controls input{font:inherit;color:var(--brand-ink);background:var(--card-bg);border:1px solid var(--brand-line);border-radius:8px;padding:10px;min-height:44px;max-width:100%}.search{flex:1;min-width:200px}.controls output{padding-bottom:10px;font:13px monospace}.note{font-size:13px;color:var(--brand-muted)}.node{border-top:1px solid var(--brand-line);padding:27px 0;scroll-margin-top:110px}.node:target{border-top:3px solid var(--brand-accent)}.meta{display:flex;flex-wrap:wrap;gap:8px 16px;color:var(--brand-muted);font-size:12px}.node h3{font-size:24px;margin:8px 0 16px;line-height:1.45}.node dl{display:grid;grid-template-columns:120px 1fr;gap:6px 18px;margin:12px 0}.node dt{font-weight:650}.node dd{margin:0;color:var(--brand-body)}details{margin:18px 0}summary{cursor:pointer;color:var(--brand-accent);padding:10px 0;min-height:44px}summary:focus-visible,a:focus-visible{outline:3px solid var(--brand-accent);outline-offset:4px}.claim,.source{border-top:1px solid var(--brand-line);padding:14px 0}.claim-id{font:12px monospace;color:var(--brand-muted);overflow-wrap:anywhere}.claim p,.source p{margin:8px 0;font-size:14px}.source h4{font-size:16px;margin:8px 0}.source a{color:var(--brand-accent);text-decoration:underline;text-underline-offset:4px}@media(max-width:640px){.controls{display:grid}.node dl{grid-template-columns:1fr;gap:3px}.node dd{margin-bottom:12px}.node h3{font-size:21px}}
</style>
