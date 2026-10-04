<script lang="ts">
import { onMount, tick } from "svelte";
import { researchThreads } from "@/data/researchThreads";
import type { WorldController, WorldLocale, WorldQuality } from "./world/create-world";
import WorldPoster from "./world/WorldPoster.svelte";

interface Props { immersive?: boolean; }
let { immersive = false }: Props = $props();
let stage: HTMLDivElement;
let canvas: HTMLCanvasElement;
let panel = $state<HTMLElement>();
let controller: WorldController | null = null;
let disposed = false;
let locale = $state<WorldLocale>("zh-CN");
let noticeZh = $state("研究工作室 / 点选一个方向，或启动 3D 探索");
let noticeEn = $state("RESEARCH WORKSHOP / CHOOSE A STATION OR START 3D");
let webglAvailable = $state(true);
let sceneReady = $state(false);
let loading = $state(false);
let animationPaused = $state(false);
let reducedMotion = $state(false);
let followRover = $state(false);
let quality = $state<WorldQuality>("light");
let selected = $state<(typeof researchThreads)[number] | null>(null);
let panelCollapsed = $state(false);
let previewTrigger: HTMLElement | null = null;
const destinations = researchThreads;
const stationContent = {
	"llm-systems": {
		slug: "local-1b-model-workflow", zh: "小模型全链路训练", en: "End-to-end small-model training",
		experiment: "/lab/model-pipeline/", experimentZh: "从数据、分词到训练、评测与部署", experimentEn: "From data and tokenization to training, evaluation, and deployment",
		path: "/paths/models-to-systems/", pathZh: "从模型到智能系统", pathEn: "From models to intelligent systems",
	},
	"reinforcement-learning": {
		slug: "alphazero-gomoku", zh: "AlphaZero 五子棋", en: "AlphaZero Gomoku",
		experiment: "/lab/mcts/", experimentZh: "运行一次 MCTS 搜索", experimentEn: "Run a Monte Carlo tree search",
		path: "/paths/search-to-self-play/", pathZh: "从搜索到自我对弈", pathEn: "From search to self-play",
	},
	"causal-inference": {
		slug: "ylearn-causal-inference", zh: "YLearn 因果分析", en: "YLearn causal inference",
		experiment: "/lab/causal-playground/", experimentZh: "改变分配方式，观察效果", experimentEn: "Change assignment and observe the effect",
		path: "/paths/evaluating-improvement/", pathZh: "怎样判断改进有效", pathEn: "How to evaluate improvement",
	},
};
const panelId = $derived(immersive ? "immersive-station-panel" : "home-station-panel");
const instructionId = $derived(immersive ? "immersive-world-instructions" : "home-world-instructions");

function localized(item: { sceneLabel: string; sceneLabelEn: string }): string {
	return locale === "en" ? item.sceneLabelEn : item.sceneLabel;
}
async function openStation(destination: (typeof destinations)[number]): Promise<void> {
	if (!selected) previewTrigger = document.activeElement instanceof HTMLElement ? document.activeElement : null;
	selected = destination;
	panelCollapsed = false;
	controller?.selectStation(destination.id);
	await tick();
	panel?.focus({ preventScroll: true });
	if (immersive && panel) {
		const bounds = panel.getBoundingClientRect();
		if (bounds.top < 80 || bounds.bottom > window.innerHeight - 16) {
			panel.scrollIntoView({ block: "center", behavior: "instant" });
		}
	}
}
function closeStation(): void {
	selected = null;
	controller?.resetView();
	previewTrigger?.focus({ preventScroll: true });
}
function resetView(): void {
	selected = null;
	controller?.resetView();
}
function collapsePanel(): void {
	panelCollapsed = true;
	canvas.focus({ preventScroll: true });
}
function toggleAnimation(): void {
	animationPaused = !animationPaused;
	controller?.setAnimationPaused(animationPaused);
}
function toggleFollow(): void {
	followRover = !followRover;
	controller?.setFollowRover(followRover);
}
function changeQuality(): void {
	quality = quality === "light" ? "full" : "light";
	controller?.setQuality(quality);
}
function onWindowKeyDown(event: KeyboardEvent): void {
	if (event.key === "Escape" && selected) {
		event.preventDefault();
		closeStation();
	}
}

function showFallback(): void {
	loading = false;
	webglAvailable = false;
	noticeZh = "3D 暂时不可用 · 站点内容仍可直接打开";
	noticeEn = "3D IS UNAVAILABLE · ALL STATION CONTENT REMAINS ACCESSIBLE";
}
async function startScene(): Promise<void> {
	if (loading || sceneReady || disposed) return;
	loading = true;
	noticeZh = "正在准备研究岛 · 可先点选下方站点";
	noticeEn = "PREPARING THE ISLAND · STATION BUTTONS ARE READY";
	try {
		const { createWorld } = await import("./world/create-world");
		if (disposed) return;
		controller = createWorld({
			canvas, stage, stations: destinations, immersive, locale,
			onReady: () => { sceneReady = true; loading = false; },
			onSelect: (id) => { const station = destinations.find((item) => item.id === id); if (station) void openStation(station); },
			onOverview: () => { selected = null; },
			onNotice: (zh, en) => { noticeZh = zh; noticeEn = en; },
			onMotionPreference: (reduced) => { reducedMotion = reduced; },
			onFollowChange: (follow) => { followRover = follow; },
			onError: showFallback,
		});
		if (selected) controller.selectStation(selected.id);
	} catch { if (!disposed) showFallback(); }
}

onMount(() => {
	locale = document.documentElement.lang.toLowerCase().startsWith("en") ? "en" : "zh-CN";
	const onLocaleChange = (event: Event) => {
		locale = (event as CustomEvent<string>).detail === "en" ? "en" : "zh-CN";
		controller?.setLocale(locale);
	};
	document.addEventListener("site-locale-change", onLocaleChange);
	if (immersive) void startScene();
	return () => {
		disposed = true;
		controller?.dispose();
		controller = null;
		document.removeEventListener("site-locale-change", onLocaleChange);
	};
});
</script>

<svelte:window onkeydown={onWindowKeyDown} />

<div class="explore-world" class:immersive class:offline={!webglAvailable} class:ready={sceneReady} class:has-selection={selected !== null && !panelCollapsed} bind:this={stage}>
	<div class="world-poster" aria-hidden="true"><WorldPoster /></div>
	<canvas bind:this={canvas} tabindex={webglAvailable && sceneReady ? 0 : -1} aria-hidden={!webglAvailable || !sceneReady} aria-describedby={instructionId} aria-label={locale === "en" ? "Interactive research island. Select a station, drag to rotate, or use WASD and arrow keys to move. Press Enter near a station. R returns the rover to the start; Home restores the view." : "互动研究岛。点选站点、拖动旋转，或用 WASD 与方向键移动。靠近站点后按 Enter；R 返回起点，Home 恢复视角。"}></canvas>
	{#if !immersive && !sceneReady && webglAvailable}
		<button type="button" class="world-launch" onclick={startScene} disabled={loading}><span aria-hidden="true">{loading ? "◌" : "▷"}</span>{loading ? (locale === "en" ? "Preparing…" : "正在准备…") : (locale === "en" ? "Start 3D here" : "在此启动 3D")}</button>
	{/if}
	<div class="world-status" role="status"><span class:offline={!webglAvailable}></span><span>{locale === "en" ? noticeEn : noticeZh}</span></div>
	{#if sceneReady && webglAvailable}
		<div class="world-toolbar" role="group" aria-label={locale === "en" ? "World controls" : "研究岛控制"}>
			<button type="button" onclick={resetView} title={locale === "en" ? "Restore the overview (Home)" : "恢复总览视角（Home）"}><span aria-hidden="true">⌖</span> {locale === "en" ? "Overview" : "总览"}</button>
			<button type="button" onclick={() => controller?.resetRover()} title={locale === "en" ? "Return rover to the start (R)" : "小车回到起点（R）"}><span aria-hidden="true">↺</span> {locale === "en" ? "Reset rover" : "回起点"}</button>
			<button type="button" onclick={changeQuality} aria-pressed={quality === "full"} title={locale === "en" ? "Full quality adds shadows and finer resolution" : "完整画质增加投影和显示精度"}>{locale === "en" ? "Quality:" : "画质："} {quality === "full" ? (locale === "en" ? "Full" : "完整") : (locale === "en" ? "Light" : "轻量")}</button>
			<button type="button" onclick={toggleAnimation} disabled={reducedMotion} aria-pressed={animationPaused || reducedMotion} title={locale === "en" ? "Pause ambient animation; driving and camera controls stay available" : "暂停环境动效，仍可驾驶与调整镜头"}>{reducedMotion ? (locale === "en" ? "Reduced motion" : "已减少动态效果") : animationPaused ? (locale === "en" ? "Resume motion" : "继续动效") : (locale === "en" ? "Pause motion" : "暂停动效")}</button>
			{#if immersive}<button type="button" onclick={toggleFollow} aria-pressed={followRover}>{locale === "en" ? "Follow rover" : "跟随小车"}</button>{/if}
		</div>
	{/if}
	<p class="world-controls" id={instructionId}>{sceneReady && webglAvailable ? (locale === "en" ? "Select to travel · Drag to orbit · Focus the scene to drive with WASD" : "点选自动前往 · 拖动旋转 · 点画布后可用 WASD 驾驶") : (locale === "en" ? "Three questions. One connected research journey." : "三个研究问题，沿尝试、反馈与评估相连。")}</p>
	<nav class="world-shortcuts" aria-label={locale === "en" ? "Research stations" : "研究站点"}>
		{#each destinations as destination}
			<button type="button" class:active={selected?.id === destination.id} onclick={() => openStation(destination)} aria-expanded={selected?.id === destination.id} aria-controls={panelId}><span class="station-number">{destination.number}</span><span>{locale === "en" ? destination.labelEn : destination.label}</span><span class="station-arrow" aria-hidden="true">↗</span></button>
		{/each}
	</nav>
	{#if selected && panelCollapsed}
		<button type="button" class="reopen-preview" onclick={() => selected && openStation(selected)} aria-controls={panelId} aria-expanded="false">{locale === "en" ? "Open station details" : "展开站点内容"} ↗</button>
	{/if}
	{#if selected && !panelCollapsed}
		{@const content = stationContent[selected.id]}
		<section class="station-preview" id={panelId} bind:this={panel} tabindex="-1" aria-labelledby={`${panelId}-title`}>
			<button class="preview-close" type="button" onclick={closeStation} aria-label={locale === "en" ? "Close station details" : "关闭站点详情"}>×</button>
			<p class="preview-kicker">{localized(selected)}</p>
			<h2 id={`${panelId}-title`}>{locale === "en" ? selected.questionEn : selected.question}</h2>
			<p class="preview-body">{locale === "en" ? selected.workTextEn : selected.workText}</p>
			{#if sceneReady && webglAvailable}<button type="button" class="preview-explore" onclick={collapsePanel}>{locale === "en" ? "Hide panel & explore the installation" : "收起面板，查看装置与行驶过程"} <span aria-hidden="true">↙</span></button>{/if}
			<div class="preview-links">
				<a href={content.experiment}><span>{locale === "en" ? "01 / INTERACTIVE EXPERIMENT" : "01 / 互动实验"}</span><strong>{locale === "en" ? content.experimentEn : content.experimentZh}</strong><b aria-hidden="true">↗</b></a>
				<a href={`/work/${content.slug}/`}><span>{locale === "en" ? "02 / RESEARCH CASE" : "02 / 研究案例"}</span><strong>{locale === "en" ? content.en : content.zh}</strong><b aria-hidden="true">↗</b></a>
				<a href={content.path}><span>{locale === "en" ? "03 / READING PATH" : "03 / 阅读路线"}</span><strong>{locale === "en" ? content.pathEn : content.pathZh}</strong><b aria-hidden="true">↗</b></a>
			</div>
		</section>
	{/if}
	<noscript><nav class="world-no-script" aria-label="研究方向"><a href="/lab/model-pipeline/">小模型全链路训练</a><a href="/lab/mcts/">MCTS 搜索实验</a><a href="/lab/causal-playground/">因果实验</a></nav></noscript>
</div>

<style>
.explore-world { position: absolute; inset: 0 -5% 0 auto; width: 68%; z-index: 0; pointer-events: none; overflow: hidden; }
.explore-world canvas { position: absolute; inset: 0; width: 100%; height: 100%; outline: none; pointer-events: none; touch-action: none; cursor: grab; opacity: 0; transition: opacity 650ms ease; }
.explore-world.ready canvas { opacity: 1; pointer-events: auto; }
.explore-world.offline canvas { opacity: 0; pointer-events: none; }
.explore-world canvas:focus-visible { outline: 2px solid var(--brand-accent); outline-offset: -6px; border-radius: 16px; }
.world-poster { position: absolute; inset: 14% 1% 18%; display: grid; place-items: center; transition: opacity 400ms ease; }
.ready .world-poster { opacity: 0; }
.offline .world-poster { opacity: 1; }
.world-launch { position: absolute; right: 10%; bottom: 22%; display: flex; align-items: center; gap: 9px; padding: 11px 17px; min-height: 44px; background: var(--brand-accent); color: var(--brand-paper); border: 0; border-radius: 24px; font: 11px/1.6 "JetBrains Mono", monospace; cursor: pointer; pointer-events: auto; box-shadow: 0 5px 20px rgb(37 60 52 / 12%); }
.world-launch:hover { background: var(--brand-ink); }
.world-launch:disabled { opacity: .7; cursor: progress; }
.world-launch span { font-size: 17px; }
.world-status { position: absolute; right: 10%; top: 11%; display: flex; align-items: center; gap: 8px; max-width: 74%; color: var(--brand-body); font: 8px/1.7 "JetBrains Mono", monospace; letter-spacing: .035em; }
.world-status > span:first-child { flex: 0 0 5px; height: 5px; border-radius: 50%; background: var(--brand-accent); }
.world-status > span.offline:first-child { background: var(--brand-clay); }
.world-toolbar { position: absolute; right: 10%; top: 17%; display: flex; flex-wrap: wrap; gap: 5px; pointer-events: auto; }
.world-toolbar button { min-height: 44px; padding: 7px 9px; border: 1px solid var(--brand-line); border-radius: 20px; background: var(--brand-paper); color: var(--brand-accent); font: 9px/1.5 "JetBrains Mono", monospace; cursor: pointer; }
.world-toolbar button:hover, .world-toolbar button[aria-pressed="true"] { background: var(--brand-sage); }
.world-toolbar button:disabled { opacity: .65; cursor: default; }
.world-toolbar button > span { font-size: 13px; vertical-align: -1px; }
.world-controls { position: absolute; right: 10%; bottom: 16%; margin: 0; color: var(--brand-muted); font: 8px/1.8 "JetBrains Mono", monospace; letter-spacing: .02em; }
.world-shortcuts { position: absolute; right: 10%; bottom: 6%; display: flex; gap: 6px; max-width: 90%; pointer-events: auto; }
.world-shortcuts button { display: flex; align-items: center; gap: 7px; cursor: pointer; min-height: 44px; padding: 10px 11px; border: 1px solid var(--brand-line); border-radius: 10px; background: var(--brand-paper); color: var(--brand-accent); text-align: left; font: 10px/1.5 "JetBrains Mono", monospace; transition: background 180ms ease, border-color 180ms ease; }
.world-shortcuts button:hover, .world-shortcuts button.active { background: var(--brand-sage); border-color: var(--brand-accent); }
.station-number { color: var(--brand-muted); font-size: 8px; }
.station-arrow { font-size: 13px; }
.station-preview { position: absolute; right: 9%; bottom: 20%; width: min(350px, 82%); max-height: min(440px, 64%); box-sizing: border-box; overflow: auto; overscroll-behavior: contain; pointer-events: auto; padding: 25px; border: 1px solid var(--brand-line); border-radius: 17px; background: var(--brand-paper); color: var(--brand-ink); box-shadow: 0 14px 60px rgb(37 60 52 / 15%); scrollbar-width: thin; }
.station-preview:focus { outline: none; }
.station-preview:focus-visible { outline: 2px solid var(--brand-accent); outline-offset: -4px; }
.preview-close { position: absolute; right: 6px; top: 4px; border: 0; border-radius: 50%; background: none; color: var(--brand-accent); font-size: 26px; cursor: pointer; width: 44px; height: 44px; }
.preview-close:hover { background: var(--brand-sage); }
.preview-kicker { font: 8px/1.8 "JetBrains Mono", monospace; color: var(--brand-accent); margin: 0 18px 12px 0; }
.station-preview h2 { font: 400 22px/1.5 Georgia, "Songti SC", serif; margin: 0; }
.preview-body { font-size: 12px; line-height: 1.9; color: var(--brand-body); margin: 12px 0 18px; }
.preview-explore { display: flex; align-items: center; justify-content: space-between; width: 100%; min-height: 44px; margin: -3px 0 12px; padding: 6px 0; color: var(--brand-accent); background: none; border: 0; border-bottom: 1px solid var(--brand-line); text-align: left; font-size: 11px; line-height: 1.7; cursor: pointer; }
.preview-explore:hover { color: var(--brand-clay); }
.reopen-preview { position: absolute; right: 10%; bottom: 23%; max-width: 80%; pointer-events: auto; min-height: 44px; padding: 10px 15px; border: 1px solid var(--brand-line); border-radius: 10px; color: var(--brand-accent); background: var(--brand-paper); font-size: 11px; cursor: pointer; box-shadow: 0 5px 25px rgb(37 60 52 / 10%); }
.preview-links { display: grid; gap: 7px; }
.preview-links a { display: grid; grid-template-columns: 1fr auto; gap: 5px 9px; padding: 12px; border: 1px solid var(--brand-line); border-radius: 9px; color: var(--brand-ink); text-decoration: none; }
.preview-links a:first-child { background: var(--brand-sage); }
.preview-links a:hover { border-color: var(--brand-accent); }
.preview-links span { grid-column: 1; font: 8px/1.5 "JetBrains Mono", monospace; color: var(--brand-accent); }
.preview-links strong { grid-column: 1; font-size: 12px; font-weight: 500; line-height: 1.6; }
.preview-links b { grid-column: 2; grid-row: 1 / span 2; align-self: center; font-size: 16px; font-weight: 400; color: var(--brand-accent); }
.world-no-script { position: absolute; inset: auto 10% 30%; display: grid; gap: 12px; pointer-events: auto; font-size: 13px; }
.world-no-script a { color: var(--brand-accent); }
.immersive { position: relative; inset: auto; width: 100%; height: 100%; min-height: 500px; isolation: isolate; border-radius: inherit; background: radial-gradient(ellipse at 50% 45%, #e9efe1 0, var(--brand-paper) 73%); }
.immersive .world-status { left: 22px; right: auto; top: auto; bottom: 88px; font-size: 9px; max-width: calc(100% - 44px); }
.immersive .world-toolbar { left: 22px; right: auto; top: 20px; }
.immersive .world-toolbar button { font-size: 10px; padding: 8px 12px; }
.immersive .world-controls { left: 22px; right: 22px; top: auto; bottom: 114px; max-width: none; font-size: 9px; }
.immersive .world-shortcuts { inset: auto 22px 20px; justify-content: center; max-width: none; }
.immersive .world-shortcuts button { flex: 1; max-width: 250px; justify-content: space-between; min-height: 48px; font-size: 11px; }
.immersive .station-preview { top: 75px; right: 20px; bottom: auto; width: 320px; max-height: calc(100% - 170px); }
.immersive .reopen-preview { right: 22px; top: 80px; bottom: auto; }
@media (max-width: 1000px) {
	.world-shortcuts button { font-size: 8px; padding: 9px 7px; gap: 5px; }
	.station-number { display: none; }
	.world-controls { max-width: 77%; font-size: 7px; }
	.immersive .world-controls { display: none; }
}
@media (max-width: 700px) {
	.explore-world { inset: auto 0 52px; width: 100%; height: 360px; }
	.explore-world canvas { touch-action: pan-y; }
	.world-poster { inset: 3% 2% 19%; }
	.world-launch { right: 6%; bottom: 23%; padding: 9px 14px; font-size: 10px; }
	.world-status { right: 6%; left: 6%; top: 1%; max-width: none; font-size: 8px; }
	.world-toolbar { top: 9%; right: 6%; left: 6%; display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); }
	.world-toolbar button { padding: 5px 8px; font-size: 8px; }
	.world-controls { right: 6%; bottom: 17%; max-width: 88%; font-size: 7px; }
	.world-shortcuts { inset: auto 5% 0; display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); max-width: none; gap: 5px; }
	.world-shortcuts button { padding: 8px 6px; min-height: 48px; font-size: 9px; justify-content: center; text-align: center; }
	.station-arrow { display: none; }
	.station-preview { position: fixed; inset: auto 14px 14px; width: auto; max-height: min(520px, 72svh); z-index: 40; padding: 23px; }
	.station-preview h2 { font-size: 23px; }
	.immersive { position: relative; inset: auto; height: 100%; min-height: 500px; }
	.immersive .world-toolbar { top: 14px; left: 14px; right: 14px; grid-template-columns: repeat(3, minmax(0, 1fr)); }
	.immersive .world-toolbar button { padding: 7px 9px; font-size: 9px; }
	.immersive .reopen-preview { top: 117px; right: 14px; font-size: 10px; }
	.immersive .world-shortcuts { inset: auto 12px 12px; }
	.immersive .world-shortcuts button { font-size: 9px; min-height: 48px; }
	.immersive .world-status { left: 15px; right: 15px; bottom: 72px; top: auto; max-width: none; font-size: 8px; }
	.immersive .station-preview { position: absolute; inset: auto 12px 102px; width: auto; max-height: calc(100% - 236px); padding: 21px; }
	.immersive .station-preview h2 { font-size: 21px; }
	.immersive.has-selection .world-status { display: none; }
}
@media (prefers-reduced-motion: reduce) {
	.explore-world canvas, .world-poster, .world-shortcuts button { transition: none; }
}
</style>
