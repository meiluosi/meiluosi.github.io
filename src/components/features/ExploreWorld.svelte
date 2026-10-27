<script lang="ts">
import { onMount, tick } from "svelte";
import { researchThreads } from "@/data/researchThreads";
import { computeCelestialState, getCurrentSolarHour } from "@/utils/celestial";
import { playLatchClick } from "@/utils/haptics";
import type {
	WorldController,
	WorldLocale,
	WorldQuality,
} from "./world/create-world";
import WorldPoster from "./world/WorldPoster.svelte";

interface Props {
	immersive?: boolean;
	compact?: boolean;
}
let { immersive = false, compact = false }: Props = $props();
let stage: HTMLDivElement;
let canvas = $state<HTMLCanvasElement>();
let panel = $state<HTMLElement>();
let controller: WorldController | null = null;
let disposed = false;
let locale = $state<WorldLocale>("zh-CN");
let noticeZh = $state("桌上有几个还没想完的问题，随意漫步看看。");
let noticeEn = $state(
	"A few unfinished questions are in the garden. Wander freely around.",
);
let webglAvailable = $state(true);
let sceneReady = $state(false);
let loading = $state(false);
let animationPaused = $state(false);
let reducedMotion = $state(false);
let followRover = $state(true);
let quality = $state<WorldQuality>("light");
let selected = $state<(typeof destinations)[number] | null>(null);
let panelCollapsed = $state(false);
let previewTrigger: HTMLElement | null = null;
let solarHour = $state(getCurrentSolarHour());

// HUD States for Genshin-style mechanics
let oculiCount = $state(0);
let nearStationId = $state<string | null>(null);
let bannerMessage = $state<{ title: string; subtitle: string } | null>(null);
let bannerTimer: ReturnType<typeof setTimeout> | null = null;

function showBanner(title: string, subtitle: string): void {
	bannerMessage = { title, subtitle };
	if (bannerTimer) clearTimeout(bannerTimer);
	bannerTimer = setTimeout(() => {
		bannerMessage = null;
	}, 4500);
}

const celestial = $derived(computeCelestialState(solarHour));
const celestialEmoji = $derived(
	celestial.period === "dawn"
		? "🌅"
		: celestial.period === "noon"
			? "☀️"
			: celestial.period === "sunset"
				? "🌇"
				: "🌙",
);
const celestialLabel = $derived(
	locale === "en" ? celestial.labelEn : celestial.labelZh,
);
const stationCopy = {
	"llm-systems": {
		label: "文字桌",
		labelEn: "Word desk",
		sceneLabel: "文字桌",
		sceneLabelEn: "The word desk",
		question: "一句话怎样变成机器眼中的数字？",
		questionEn: "How does a sentence become numbers to a machine?",
		workText: "从纸上的一句话开始，拆成词元，看看预测、训练与评测怎样接上。",
		workTextEn:
			"Start with a sentence on the page, turn it into tokens, and follow the connections between prediction, training, and evaluation.",
	},
	"reinforcement-learning": {
		label: "棋盘",
		labelEn: "The board",
		sceneLabel: "棋盘边",
		sceneLabelEn: "At the board",
		question: "多想一步，选择会改变吗？",
		questionEn: "Would thinking one step further change the choice?",
		workText:
			"棋盘上的一步，还可以往前想多远？展开一棵搜索树，看看尝试怎样留下线索。",
		workTextEn:
			"How far ahead can we look before placing a stone? Open a search tree and see what each attempt leaves behind.",
	},
	"causal-inference": {
		label: "观察台",
		labelEn: "Observation",
		sceneLabel: "观察台",
		sceneLabelEn: "The observation desk",
		question: "怎样辨认一次改变的原因？",
		questionEn: "How can we find the cause of a change?",
		workText:
			"两组人看起来不同，差异来自哪里？换一种安排，在模拟数据里试着分辨。",
		workTextEn:
			"Two groups look different, but why? Change how things are assigned and explore the distinction with simulated data.",
	},
};
const destinations = researchThreads.map((thread) => ({
	...thread,
	...stationCopy[thread.id],
}));
const stationContent = {
	"llm-systems": {
		slug: "local-1b-model-workflow",
		zh: "小模型全链路训练",
		en: "End-to-end small-model training",
		experiment: "/lab/model-pipeline/",
		experimentZh: "拆开一句话，试着预测下一个词",
		experimentEn: "Take a sentence apart and predict the next token",
		path: "/work/local-1b-model-workflow/",
		pathZh: "从模型到智能系统",
		pathEn: "From models to intelligent systems",
	},
	"reinforcement-learning": {
		slug: "alphazero-gomoku",
		zh: "AlphaZero 五子棋",
		en: "AlphaZero Gomoku",
		experiment: "/lab/mcts/",
		experimentZh: "沿着搜索树，多想一步",
		experimentEn: "Think one step further with a search tree",
		path: "/work/alphazero-gomoku/",
		pathZh: "从搜索到自我对弈",
		pathEn: "From search to self-play",
	},
	"causal-inference": {
		slug: "ylearn-causal-inference",
		zh: "YLearn 因果分析",
		en: "YLearn causal inference",
		experiment: "/lab/causal-playground/",
		experimentZh: "换一种安排，观察差异",
		experimentEn: "Change the assignment and observe the difference",
		path: "/work/ylearn-causal-inference/",
		pathZh: "怎样判断改进有效",
		pathEn: "How to evaluate improvement",
	},
};
const panelId = $derived(
	immersive ? "immersive-station-panel" : "home-station-panel",
);
const instructionId = $derived(
	immersive ? "immersive-world-instructions" : "home-world-instructions",
);

function localized(item: { sceneLabel: string; sceneLabelEn: string }): string {
	return locale === "en" ? item.sceneLabelEn : item.sceneLabel;
}
async function openStation(
	destination: (typeof destinations)[number],
): Promise<void> {
	if (!selected)
		previewTrigger =
			document.activeElement instanceof HTMLElement
				? document.activeElement
				: null;
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
	canvas?.focus({ preventScroll: true });
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
function cycleDaylight(): void {
	const hours = [6.5, 12.0, 18.0, 23.5];
	const nextIdx = hours.findIndex((h) => h > solarHour + 0.5);
	const nextHour = nextIdx === -1 ? hours[0] : hours[nextIdx];
	solarHour = nextHour;
	controller?.setSolarHour(solarHour);
	playLatchClick();
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
	if (loading || sceneReady || disposed || compact || !canvas) return;
	loading = true;
	noticeZh = "正在准备探索花园 · 可先点选下方站点";
	noticeEn = "PREPARING THE GARDEN · STATION BUTTONS ARE READY";
	try {
		const { createWorld } = await import("./world/create-world");
		if (disposed || !canvas) return;
		controller = createWorld({
			canvas,
			stage,
			stations: destinations,
			immersive,
			locale,
			onReady: () => {
				sceneReady = true;
				loading = false;
				controller?.setSolarHour(solarHour);
			},
			onSelect: (id) => {
				const station = destinations.find((item) => item.id === id);
				if (station) void openStation(station);
			},
			onOverview: () => {
				selected = null;
			},
			onNotice: (zh, en) => {
				noticeZh = zh;
				noticeEn = en;
			},
			onMotionPreference: (reduced) => {
				reducedMotion = reduced;
			},
			onFollowChange: (follow) => {
				followRover = follow;
			},
			onError: showFallback,
			onCollectOculus: (_idx, titleZh, titleEn) => {
				oculiCount += 1;
				playLatchClick();
				showBanner(
					locale === "en"
						? "✨ Inspiration Spark Collected!"
						: "✨ 获得灵感神瞳！",
					locale === "en" ? titleEn : titleZh,
				);
			},
			onStepPlate: (active) => {
				if (active) {
					playLatchClick();
					showBanner(
						locale === "en"
							? "⚙️ Ancient Mechanism Triggered"
							: "⚙️ 触发遗迹机关",
						locale === "en"
							? "Floating stones have aligned as a walkway!"
							: "悬空浮石正在升起拼合为桥梁！",
					);
				}
			},
			onProximityStation: (stId) => {
				nearStationId = stId;
			},
		});
		if (selected) controller.selectStation(selected.id);
	} catch (error) {
		if (import.meta.env.DEV)
			console.warn("[ExploreWorld] Scene initialization failed", error);
		if (!disposed) showFallback();
	}
}

onMount(() => {
	locale = document.documentElement.lang.toLowerCase().startsWith("en")
		? "en"
		: "zh-CN";
	const onLocaleChange = (event: Event) => {
		locale = (event as CustomEvent<string>).detail === "en" ? "en" : "zh-CN";
		controller?.setLocale(locale);
	};
	document.addEventListener("site-locale-change", onLocaleChange);
	if (immersive && !compact) void startScene();
	return () => {
		disposed = true;
		if (bannerTimer) clearTimeout(bannerTimer);
		controller?.dispose();
		controller = null;
		document.removeEventListener("site-locale-change", onLocaleChange);
	};
});
</script>

<svelte:window onkeydown={onWindowKeyDown} />

<div
	class="explore-world"
	class:immersive
	class:compact
	class:offline={!webglAvailable}
	class:ready={sceneReady}
	class:has-selection={selected !== null && !panelCollapsed}
	bind:this={stage}
>
	<div class="world-poster" aria-hidden="true"><WorldPoster {locale} /></div>
	{#if compact}
		<div class="compact-caption">
			<p>{locale === "en" ? "A desk, a board, and a place to look closer." : "一张桌、一盘棋，还有几个想继续追问的问题。"}</p>
			<a href="/world/">{locale === "en" ? "Have a look around" : "进来随意看看"}<span aria-hidden="true">↗</span></a>
		</div>
	{:else}
		<canvas
			bind:this={canvas}
			tabindex={webglAvailable && sceneReady ? 0 : -1}
			aria-hidden={!webglAvailable || !sceneReady}
			aria-describedby={instructionId}
			aria-label={locale === "en" ? "Interactive 3D digital garden. Use WASD to walk, Space to jump, E to inspect notes, or click destinations." : "三维数字花园。用 WASD 漫步，空格跳跃，E 翻看手稿，或点选站点自动前往。"}
		></canvas>

		{#if !immersive && !sceneReady && webglAvailable}
			<button type="button" class="world-launch" onclick={startScene} disabled={loading}>
				<span aria-hidden="true">{loading ? "◌" : "▷"}</span>
				{loading ? (locale === "en" ? "Preparing…" : "正在准备…") : (locale === "en" ? "Start 3D here" : "在此启动 3D")}
			</button>
		{/if}

		<!-- Genshin-style Pickup / Achievement Banner -->
		{#if bannerMessage}
			<div class="genshin-banner">
				<h4>{bannerMessage.title}</h4>
				<p>{bannerMessage.subtitle}</p>
			</div>
		{/if}

		<!-- Near Station Interaction Prompt -->
		{#if nearStationId && (!selected || panelCollapsed)}
			{@const stObj = destinations.find((d) => d.id === nearStationId)}
			{#if stObj}
				<div class="interaction-prompt">
					<button type="button" class="interact-action-btn" onclick={() => openStation(stObj)}>
						<span class="key-pill">E</span>
						<div class="prompt-text">
							<strong>{locale === "en" ? stObj.sceneLabelEn : stObj.sceneLabel}</strong>
							<small>{locale === "en" ? "Press E or click to open notes ↗" : "按 E 或点击翻开手稿 ↗"}</small>
						</div>
					</button>
				</div>
			{/if}
		{/if}

		<div class="world-status" role="status">
			<span class:offline={!webglAvailable}></span>
			<span>{locale === "en" ? noticeEn : noticeZh}</span>
		</div>

		{#if sceneReady && webglAvailable}
			<div class="world-toolbar" role="group" aria-label={locale === "en" ? "World controls" : "研究岛控制"}>
				<!-- Inspiration Oculi Counter -->
				<div class="oculi-counter" title={locale === "en" ? "Inspiration Oculi gathered" : "已收集的灵感神瞳"}>
					<span class="star-icon">✦</span>
					<span class="count-num">{oculiCount} / 3</span>
				</div>
				<button type="button" onclick={resetView} title={locale === "en" ? "Restore overview (Home)" : "恢复总览视角（Home）"}>
					<span aria-hidden="true">⌖</span> {locale === "en" ? "Overview" : "总览"}
				</button>
				<button type="button" onclick={() => controller?.resetRover()} title={locale === "en" ? "Return traveler to start (R)" : "漫步者回起点（R）"}>
					<span aria-hidden="true">↺</span> {locale === "en" ? "Reset" : "回起点"}
				</button>
				<button type="button" onclick={changeQuality} aria-pressed={quality === "full"} title={locale === "en" ? "Full quality adds shadows and finer resolution" : "完整画质增加投影和显示精度"}>
					{locale === "en" ? "Quality:" : "画质："} {quality === "full" ? (locale === "en" ? "Full" : "完整") : (locale === "en" ? "Light" : "轻量")}
				</button>
				<button type="button" class="celestial-btn" onclick={cycleDaylight} title={locale === "en" ? "Cycle celestial daylight (Dawn / Noon / Sunset / Night)" : "切换天象光影周期（晨曦 / 正午 / 晚霞 / 静夜）"}>
					<span aria-hidden="true">{celestialEmoji}</span> {celestialLabel}
				</button>
				<button type="button" onclick={toggleAnimation} disabled={reducedMotion} aria-pressed={animationPaused || reducedMotion} title={locale === "en" ? "Pause ambient animation" : "暂停环境动效"}>
					{reducedMotion ? (locale === "en" ? "Reduced motion" : "已减动效") : animationPaused ? (locale === "en" ? "Resume" : "继续动效") : (locale === "en" ? "Pause" : "暂停动效")}
				</button>
				{#if immersive}
					<button type="button" onclick={toggleFollow} aria-pressed={followRover}>
						{locale === "en" ? "Follow" : "跟随镜头"}
					</button>
				{/if}
			</div>
		{/if}

		<p class="world-controls" id={instructionId}>
			{sceneReady && webglAvailable
				? (locale === "en"
					? "WASD to walk · Space to jump · E to inspect · Drag to orbit · Click to navigate"
					: "WASD 漫步 · 空格跳跃 · E 翻看手稿 · 拖动旋转 · 点选花径导引")
				: (locale === "en"
					? "Three questions. One connected research journey."
					: "三个研究问题，沿尝试、反馈与评估相连。")}
		</p>

		<nav class="world-shortcuts" aria-label={locale === "en" ? "Research stations" : "研究站点"}>
			{#each destinations as destination}
				<button
					type="button"
					class:active={selected?.id === destination.id}
					onclick={() => openStation(destination)}
					aria-expanded={selected?.id === destination.id}
					aria-controls={panelId}
				>
					<span class="station-number">{destination.number}</span>
					<span>{locale === "en" ? destination.labelEn : destination.label}</span>
					<span class="station-arrow" aria-hidden="true">↗</span>
				</button>
			{/each}
		</nav>

		{#if selected && panelCollapsed}
			<button
				type="button"
				class="reopen-preview"
				onclick={() => selected && openStation(selected)}
				aria-controls={panelId}
				aria-expanded="false"
			>
				{locale === "en" ? "Open station details" : "展开手稿记录"} ↗
			</button>
		{/if}

		{#if selected && !panelCollapsed}
			{@const content = stationContent[selected.id]}
			<section
				class="station-preview"
				id={panelId}
				bind:this={panel}
				tabindex="-1"
				aria-labelledby={`${panelId}-title`}
			>
				<button class="preview-close" type="button" onclick={closeStation} aria-label={locale === "en" ? "Close station details" : "关闭手稿记录"}>×</button>
				<p class="preview-kicker">{localized(selected)}</p>
				<h2 id={`${panelId}-title`}>{locale === "en" ? selected.questionEn : selected.question}</h2>
				<a class="preview-main-link" href={content.experiment}>
					<span>{locale === "en" ? "Try it" : "动手试试"}</span>
					<strong>{locale === "en" ? content.experimentEn : content.experimentZh}</strong>
					<b aria-hidden="true">↗</b>
				</a>
				<p class="preview-body">{locale === "en" ? selected.workTextEn : selected.workText}</p>
				<div class="preview-links">
					<a href={`/work/${content.slug}/`}>
						<span>{locale === "en" ? "Read notes" : "翻翻记录"}</span>
						<strong>{locale === "en" ? content.en : content.zh}</strong>
						<b aria-hidden="true">↗</b>
					</a>
					<a href={content.path}>
						<span>{locale === "en" ? "Research case" : "研究案例"}</span>
						<strong>{locale === "en" ? content.pathEn : content.pathZh}</strong>
						<b aria-hidden="true">↗</b>
					</a>
				</div>
				{#if sceneReady && webglAvailable}
					<button type="button" class="preview-explore" onclick={collapsePanel}>
						{locale === "en" ? "Put note aside and continue walking" : "先放下纸条，在花园里逛逛"} <span aria-hidden="true">↙</span>
					</button>
				{/if}
			</section>
		{/if}

		<noscript>
			<nav class="world-no-script" aria-label="研究方向">
				<a href="/lab/model-pipeline/">小模型全链路训练</a>
				<a href="/lab/mcts/">MCTS 搜索实验</a>
				<a href="/lab/causal-playground/">因果实验</a>
			</nav>
		</noscript>
	{/if}
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
.world-launch { position: absolute; right: 10%; bottom: 22%; display: flex; align-items: center; gap: 9px; padding: 11px 17px; min-height: 44px; background: var(--brand-accent); color: var(--brand-paper); border: 0; border-radius: 24px; font-size: 12px; line-height: 1.6; cursor: pointer; pointer-events: auto; box-shadow: 0 5px 20px rgb(37 60 52 / 12%); }
.world-launch:hover { background: var(--brand-ink); }
.world-launch:disabled { opacity: .7; cursor: progress; }
.world-launch span { font-size: 17px; }

/* Genshin Style Pickup Banner */
.genshin-banner {
	position: absolute;
	top: 10%;
	left: 50%;
	transform: translateX(-50%);
	background: linear-gradient(135deg, rgba(37, 56, 43, 0.94), rgba(48, 73, 56, 0.94));
	border: 1px solid #c9b183;
	border-radius: 8px;
	color: #f7f3e8;
	padding: 10px 24px;
	box-shadow: 0 8px 24px rgba(0, 0, 0, 0.22), 0 0 16px rgba(201, 177, 131, 0.25);
	animation: bannerIn 0.3s cubic-bezier(0.18, 0.89, 0.32, 1.25);
	z-index: 25;
	text-align: center;
	pointer-events: none;
}
.genshin-banner h4 { margin: 0; font-size: 13px; color: #ffd88a; letter-spacing: 0.04em; }
.genshin-banner p { margin: 3px 0 0; font-size: 11px; color: #e5ede3; }

@keyframes bannerIn {
	from { opacity: 0; transform: translate(-50%, -15px) scale(0.94); }
	to { opacity: 1; transform: translate(-50%, 0) scale(1); }
}

/* Near Station Interaction Prompt */
.interaction-prompt {
	position: absolute;
	bottom: 22%;
	left: 45%;
	transform: translateX(-50%);
	z-index: 15;
	pointer-events: auto;
	animation: floatPrompt 2s ease-in-out infinite;
}
.interact-action-btn {
	display: flex;
	align-items: center;
	gap: 10px;
	background: rgba(30, 48, 36, 0.92);
	backdrop-filter: blur(8px);
	border: 1.5px solid #d4be8d;
	border-radius: 28px;
	padding: 6px 16px 6px 10px;
	color: #fff;
	cursor: pointer;
	box-shadow: 0 6px 20px rgba(0, 0, 0, 0.22);
	transition: transform 0.15s;
}
.interact-action-btn:hover { transform: scale(1.05); }
.key-pill {
	display: inline-flex;
	align-items: center;
	justify-content: center;
	width: 26px;
	height: 26px;
	background: #eed8a1;
	color: #2b3a2e;
	font-weight: 700;
	font-size: 12px;
	border-radius: 50%;
}
.prompt-text { display: flex; flex-direction: column; text-align: left; }
.prompt-text strong { font-size: 12px; color: #fcebd0; }
.prompt-text small { font-size: 10px; color: #a9c7b0; }

@keyframes floatPrompt {
	0%, 100% { transform: translate(-50%, 0); }
	50% { transform: translate(-50%, -4px); }
}

.world-status { position: absolute; right: 10%; top: 7%; display: flex; align-items: center; gap: 8px; max-width: 74%; color: var(--brand-body); font-size: 12px; line-height: 1.7; }
.world-status > span:first-child { flex: 0 0 5px; height: 5px; border-radius: 50%; background: var(--brand-accent); }
.world-status > span.offline:first-child { background: var(--brand-clay); }

.world-toolbar { position: absolute; right: 10%; top: 12%; display: flex; flex-wrap: wrap; gap: 6px; pointer-events: auto; align-items: center; }
.oculi-counter {
	display: inline-flex;
	align-items: center;
	gap: 5px;
	padding: 6px 11px;
	background: var(--brand-paper);
	border: 1px solid var(--brand-line);
	border-radius: 20px;
	font-size: 12px;
	color: var(--brand-accent);
}
.star-icon { color: #d98e3b; font-size: 13px; }
.count-num { font-family: monospace; font-size: 12px; font-weight: 600; }

.world-toolbar button { min-height: 38px; padding: 6px 11px; border: 1px solid var(--brand-line); border-radius: 20px; background: var(--brand-paper); color: var(--brand-accent); font-size: 12px; line-height: 1.5; cursor: pointer; transition: background 150ms; }
.world-toolbar button:hover, .world-toolbar button[aria-pressed="true"] { background: var(--brand-sage); }
.world-toolbar button:disabled { opacity: .65; cursor: default; }
.world-toolbar button > span { font-size: 13px; vertical-align: -1px; }

.world-controls { position: absolute; right: 10%; bottom: 15%; margin: 0; color: var(--brand-muted); font-size: 12px; line-height: 1.8; }
.world-shortcuts { position: absolute; right: 10%; bottom: 6%; display: flex; gap: 6px; max-width: 90%; pointer-events: auto; }
.world-shortcuts button { display: flex; align-items: center; gap: 7px; cursor: pointer; min-height: 44px; padding: 10px 11px; border: 1px solid var(--brand-line); border-radius: 8px; background: var(--brand-paper); color: var(--brand-accent); text-align: left; font-size: 14px; line-height: 1.5; transition: background 180ms ease, border-color 180ms ease; }
.world-shortcuts button:hover, .world-shortcuts button.active { background: var(--brand-sage); border-color: var(--brand-accent); }
.station-number { color: var(--brand-muted); font-size: 12px; }
.station-arrow { font-size: 13px; }

.station-preview { position: absolute; right: 9%; bottom: 18%; width: min(350px, 82%); max-height: min(450px, 64%); box-sizing: border-box; overflow: auto; overscroll-behavior: contain; pointer-events: auto; padding: 25px; border: 1px solid var(--brand-line); border-radius: 17px; background: var(--brand-paper); color: var(--brand-ink); box-shadow: 0 14px 60px rgb(37 60 52 / 15%); scrollbar-width: thin; }
.station-preview:focus { outline: none; }
.station-preview:focus-visible { outline: 2px solid var(--brand-accent); outline-offset: -4px; }
.preview-close { position: absolute; right: 6px; top: 4px; border: 0; border-radius: 50%; background: none; color: var(--brand-accent); font-size: 26px; cursor: pointer; width: 44px; height: 44px; }
.preview-close:hover { background: var(--brand-sage); }
.preview-kicker { font-size: 12px; color: var(--brand-clay); margin: 0 0 8px; }
.station-preview h2 { font: 400 24px/1.5 Georgia, "Songti SC", serif; margin: 0 0 16px; }
.preview-main-link { display: flex; flex-direction: column; gap: 4px; padding: 14px; margin-bottom: 16px; border: 1px solid var(--brand-line); border-radius: 10px; text-decoration: none; background: rgba(157, 189, 166, 0.12); position: relative; }
.preview-main-link span { font-size: 11px; color: var(--brand-clay); }
.preview-main-link strong { font-size: 14px; color: var(--brand-ink); font-weight: 500; }
.preview-main-link b { position: absolute; right: 12px; top: 12px; font-size: 14px; color: var(--brand-accent); }
.preview-body { font-size: 13px; line-height: 1.7; color: var(--brand-body); margin: 0 0 16px; }
.preview-links { display: grid; gap: 8px; margin-bottom: 16px; }
.preview-links a { display: flex; justify-content: space-between; align-items: center; padding: 10px 12px; border: 1px solid var(--brand-line); border-radius: 8px; text-decoration: none; color: var(--brand-ink); font-size: 12px; }
.preview-links a span { color: var(--brand-muted); }
.preview-links a strong { font-weight: 500; }
.preview-links a b { color: var(--brand-accent); }
.preview-explore { width: 100%; padding: 9px; background: transparent; border: 1px dashed var(--brand-line); border-radius: 8px; color: var(--brand-accent); font-size: 12px; cursor: pointer; transition: background 150ms; }
.preview-explore:hover { background: var(--brand-sage); }
.reopen-preview { position: absolute; right: 10%; bottom: 20%; padding: 8px 14px; background: var(--brand-paper); border: 1px solid var(--brand-accent); border-radius: 20px; color: var(--brand-accent); font-size: 12px; cursor: pointer; pointer-events: auto; }

.world-no-script { display: none; }
</style>
