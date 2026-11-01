<script lang="ts">
import { onMount } from "svelte";
import { TOKENS } from "./data/modelData";
import Scene1VocabEmbedding from "./scenes/Scene1VocabEmbedding.svelte";
import Scene2QKVProjection from "./scenes/Scene2QKVProjection.svelte";
import Scene3AttentionMask from "./scenes/Scene3AttentionMask.svelte";
import Scene4SoftmaxTemperature from "./scenes/Scene4SoftmaxTemperature.svelte";
import Scene5AutoregressiveClosure from "./scenes/Scene5AutoregressiveClosure.svelte";
import { TRANSFORMER_CINEMA_SCRIPT } from "./scripts/transformerScript";
import { PALETTE } from "./timeline";
import type { CinemaAct, CinemaScript, CinemaSubtitle } from "./types";

interface Props {
	script?: CinemaScript;
}

const { script = TRANSFORMER_CINEMA_SCRIPT }: Props = $props();

let english = $state(false);
let heroIdx = $state(3);
let currentTime = $state(0);
let isPlaying = $state(false);
let playbackSpeed = $state(1.0);
let isFullscreen = $state(false);
let containerEl: HTMLElement | null = $state(null);

// 交互驻留点参数字典 (例如 temperature, topP)
let checkpointParams = $state<Record<string, number>>({
	temperature: 0.7,
	topP: 0.95,
});

// 已触发过的驻留点防重复自锁
const visitedCheckpoints = new Set<string>();
let isCheckpointPaused = $state(false);

// 当前激活的幕索引与幕对象
const activeActIndex = $derived.by<number>(() => {
	for (let i = 0; i < script.acts.length; i++) {
		const act = script.acts[i];
		if (
			currentTime >= act.startSec &&
			currentTime < act.startSec + act.durationSec
		) {
			return i;
		}
	}
	return script.acts.length - 1;
});

const activeAct = $derived.by<CinemaAct>(() => script.acts[activeActIndex]);

// 当前幕内已经过去的时间 (秒)
const actTime = $derived.by<number>(() =>
	Math.max(
		0,
		Math.min(activeAct.durationSec, currentTime - activeAct.startSec),
	),
);

// 当前激活的字幕 (Active Subtitle)
const currentSubtitle = $derived.by<CinemaSubtitle | null>(() => {
	if (!activeAct.subtitles) return null;
	for (const sub of activeAct.subtitles) {
		if (currentTime >= sub.timeRange[0] && currentTime < sub.timeRange[1]) {
			return sub;
		}
	}
	return null;
});

function formatTime(seconds: number): string {
	const mins = Math.floor(seconds / 60);
	const secs = Math.floor(seconds % 60);
	return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

function seekTo(sec: number) {
	currentTime = Math.max(0, Math.min(script.totalDurationSec, sec));
	isCheckpointPaused = false;
}

function skip(deltaSec: number) {
	seekTo(currentTime + deltaSec);
}

function togglePlay() {
	if (isCheckpointPaused) {
		isCheckpointPaused = false;
	}
	isPlaying = !isPlaying;
}

function toggleFullscreen() {
	if (!containerEl) return;
	if (!document.fullscreenElement) {
		containerEl
			.requestFullscreen()
			.then(() => (isFullscreen = true))
			.catch(() => {});
	} else {
		document
			.exitFullscreen()
			.then(() => (isFullscreen = false))
			.catch(() => {});
	}
}

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

	let lastRafTime = performance.now();
	let rafId = 0;

	const loop = (now: number) => {
		const dt = (now - lastRafTime) / 1000;
		lastRafTime = now;

		if (isPlaying && !isCheckpointPaused) {
			const nextTime = currentTime + dt * playbackSpeed;

			// 检查是否恰好撞击到新一幕的交互驻留点
			for (const act of script.acts) {
				if (act.checkpoint && !visitedCheckpoints.has(act.id)) {
					const pauseThreshold = act.startSec + act.checkpoint.pauseAtSec;
					if (currentTime < pauseThreshold && nextTime >= pauseThreshold) {
						visitedCheckpoints.add(act.id);
						currentTime = pauseThreshold;
						isPlaying = false;
						isCheckpointPaused = true;
						break;
					}
				}
			}

			if (!isCheckpointPaused) {
				if (nextTime >= script.totalDurationSec) {
					currentTime = script.totalDurationSec;
					isPlaying = false;
				} else {
					currentTime = nextTime;
				}
			}
		}

		rafId = requestAnimationFrame(loop);
	};

	rafId = requestAnimationFrame(loop);

	const handleKeyDown = (e: KeyboardEvent) => {
		if (
			e.target instanceof HTMLInputElement ||
			e.target instanceof HTMLTextAreaElement
		)
			return;
		if (e.code === "Space") {
			e.preventDefault();
			togglePlay();
		} else if (e.code === "ArrowLeft") {
			e.preventDefault();
			skip(-5);
		} else if (e.code === "ArrowRight") {
			e.preventDefault();
			skip(5);
		}
	};
	window.addEventListener("keydown", handleKeyDown);

	return () => {
		obs.disconnect();
		cancelAnimationFrame(rafId);
		window.removeEventListener("keydown", handleKeyDown);
	};
});
</script>

<div
	class="code-cinema-wrapper"
	bind:this={containerEl}
	class:is-fullscreen={isFullscreen}
	aria-label="Code Cinema Interactive Theater"
>
	<!-- ════════ 顶部导演控制甲板 (Director Deck Topbar) ════════ -->
	<header class="cinema-topbar">
		<div class="meta-col">
			<span class="act-kicker-pill">{activeAct.actNumber}</span>
			<h2 class="act-title">
				{english ? activeAct.titleEn : activeAct.titleZh}
			</h2>
			<div
				class="hero-token-switcher"
				role="group"
				aria-label={english ? "Select Hero Query token" : "选择主角 Query 词元"}
			>
				<span class="switcher-label">
					{english ? "Hero Query:" : "主角 Query:"}
				</span>
				{#each TOKENS as t, idx}
					<button
						type="button"
						class="token-chip-btn"
						class:active={heroIdx === idx}
						onclick={() => (heroIdx = idx)}
						title={english
							? `Set "${t.text}" as active Query token`
							: `设定【${t.text}】为激活 Query 词元`}
					>
						<span class="tok-text">{t.text}</span>
						<span class="sub-pos">p{t.pos}</span>
					</button>
				{/each}
			</div>
		</div>

		<div class="actions-col">
			<div class="view-mode-pill" title={english ? "Pure Vector Panoramic View" : "纯矢量全景画幅"}>
				📐 <span>{english ? "Panoramic Stage (1100×600)" : "全景恒定画幅 (1100×600)"}</span>
			</div>

			<div class="speed-selector" role="group" aria-label={english ? "Playback speed" : "播放倍速"}>
				{#each [1.0, 1.5, 2.0] as speed}
					<button
						type="button"
						class="speed-btn"
						class:active={playbackSpeed === speed}
						onclick={() => (playbackSpeed = speed)}
					>
						{speed}x
					</button>
				{/each}
			</div>

			<button
				type="button"
				class="cinema-btn icon-btn"
				onclick={toggleFullscreen}
				title={english ? "Toggle fullscreen" : "全屏沉浸视口"}
				aria-label={english ? "Toggle fullscreen" : "全屏切换"}
			>
				{isFullscreen ? "🗗" : "⛶"}
			</button>
		</div>
	</header>

	<!-- ════════ 核心矢量演化舞台 (SVG Virtual Stage) ════════ -->
	<div class="stage-viewport">
		<svg
			viewBox="-550 -300 1100 600"
			class="cinema-svg"
			preserveAspectRatio="xMidYMid meet"
		>
			<defs>
				<!-- 网格微底纹 -->
				<pattern
					id="cinema-grid"
					width="60"
					height="60"
					patternUnits="userSpaceOnUse"
				>
					<path
						d="M 60 0 L 0 0 0 60"
						fill="none"
						stroke="rgba(120, 140, 130, 0.08)"
						stroke-width="1"
					/>
				</pattern>

				<!-- 危险电击斜纹 (因果掩码屏障) -->
				<pattern
					id="hazard-stripes"
					width="20"
					height="20"
					patternTransform="rotate(45 0 0)"
					patternUnits="userSpaceOnUse"
				>
					<line
						x1="0"
						y1="0"
						x2="0"
						y2="20"
						stroke="#ef4444"
						stroke-width="8"
					/>
					<line
						x1="10"
						y1="0"
						x2="10"
						y2="20"
						stroke="#1e293b"
						stroke-width="12"
					/>
				</pattern>

				<!-- 发光滤镜 -->
				<filter id="hero-glow" x="-50%" y="-50%" width="200%" height="200%">
					<feGaussianBlur in="SourceGraphic" stdDeviation="6" result="blur" />
					<feMerge>
						<feMergeNode in="blur" />
						<feMergeNode in="SourceGraphic" />
					</feMerge>
				</filter>
			</defs>

			<!-- 全局空间背景网格 -->
			<rect
				x="-600"
				y="-350"
				width="1200"
				height="700"
				fill="url(#cinema-grid)"
			/>

			<!-- 模块化五幕渲染 (每幕为 actTime 的纯函数，零状态干扰) -->
			{#if activeActIndex === 0}
				<Scene1VocabEmbedding
					{actTime}
					{english}
					{heroIdx}
					onSelectHero={(idx) => (heroIdx = idx)}
				/>
			{:else if activeActIndex === 1}
				<Scene2QKVProjection
					{actTime}
					{english}
					{heroIdx}
					onSelectHero={(idx) => (heroIdx = idx)}
				/>
			{:else if activeActIndex === 2}
				<Scene3AttentionMask
					{actTime}
					{english}
					{heroIdx}
					onSelectHero={(idx) => (heroIdx = idx)}
				/>
			{:else if activeActIndex === 3}
				<Scene4SoftmaxTemperature
					{actTime}
					temperature={checkpointParams.temperature}
					topP={checkpointParams.topP}
					{english}
					{heroIdx}
				/>
			{:else if activeActIndex === 4}
				<Scene5AutoregressiveClosure {actTime} {english} {heroIdx} />
			{/if}
		</svg>

		<!-- 交互驻留点 HUD 弹出层 (Interactive Checkpoint Overlay) -->
		{#if isCheckpointPaused && activeAct.checkpoint}
			<div
				class="checkpoint-hud-overlay"
				role="dialog"
				aria-label={english ? "Interactive Checkpoint Console" : "交互驻留控制台"}
			>
				<div class="hud-card">
					<div class="hud-badge">INTERACTIVE CHECKPOINT · ACT IV</div>
					<h3 class="hud-prompt">
						{english
							? activeAct.checkpoint.promptEn
							: activeAct.checkpoint.promptZh}
					</h3>
					<p class="hud-guidance">
						{english
							? activeAct.checkpoint.guidanceEn
							: activeAct.checkpoint.guidanceZh}
					</p>

					<div class="hud-controls">
						{#each activeAct.checkpoint.params as p}
							<div class="hud-param-row">
								<div class="p-label-box">
									<span class="p-label"
										>{english
											? p.labelEn
											: p.labelZh}</span
									>
									<strong class="p-val"
										>{checkpointParams[p.id]?.toFixed(
											2,
										)}</strong
									>
								</div>
								<input
									type="range"
									min={p.min}
									max={p.max}
									step={p.step}
									bind:value={checkpointParams[p.id]}
									class="hud-slider"
								/>
							</div>
						{/each}
					</div>

					<div class="hud-footer">
						<button
							type="button"
							class="resume-cinema-btn"
							onclick={togglePlay}
						>
							▶
							{english
								? "Resume Cinema Rollout ➔"
								: "完成调节，继续长片推演 ➔"}
						</button>
					</div>
				</div>
			</div>
		{/if}
	</div>

	<!-- ════════ 电影级双语动态字幕条 (Cinematic Subtitle Bar) ════════ -->
	<section class="cinema-subtitle-bar" aria-live="polite">
		{#if currentSubtitle}
			<div class="subtitle-content">
				<div class="speaker-tag-row">
					{#if currentSubtitle.speakerZh}
						<span class="speaker-pill">
							🎙️ {english ? (currentSubtitle.speakerEn || currentSubtitle.speakerZh) : currentSubtitle.speakerZh}
						</span>
					{/if}
					{#if currentSubtitle.tagZh}
						<span class="scene-badge">
							{english ? (currentSubtitle.tagEn || currentSubtitle.tagZh) : currentSubtitle.tagZh}
						</span>
					{/if}
				</div>
				{#if english}
					<p class="sub-line-zh">{currentSubtitle.textEn}</p>
				{:else}
					<p class="sub-line-zh">{currentSubtitle.textZh}</p>
					<p class="sub-line-en">{currentSubtitle.textEn}</p>
				{/if}
			</div>
		{:else}
			<div class="subtitle-placeholder">
				<span class="placeholder-dot">●</span>
				<span>{english ? "Cinema Rollout in Progress..." : "代码电影推演中..."}</span>
			</div>
		{/if}
	</section>

	<!-- ════════ 底部进度条与分幕指示器 (Scrubber & Act Nav) ════════ -->
	<footer class="cinema-footer">
		<!-- 分幕时间轴轨道 -->
		<div
			class="timeline-track-wrap"
			onclick={(e) => {
				const rect = e.currentTarget.getBoundingClientRect();
				const ratio = (e.clientX - rect.left) / rect.width;
				seekTo(ratio * script.totalDurationSec);
			}}
			role="slider"
			tabindex="0"
			aria-valuemin="0"
			aria-valuemax={script.totalDurationSec}
			aria-valuenow={currentTime}
			aria-label={english ? "Cinema timeline progress" : "电影时间轴进度"}
			onkeydown={(e) => {
				if (e.key === "ArrowLeft") skip(-5);
				if (e.key === "ArrowRight") skip(5);
			}}
		>
			<div
				class="timeline-fill"
				style={`width: ${(currentTime / script.totalDurationSec) * 100}%`}
			></div>

			<!-- 各幕分界点标记 -->
			{#each script.acts as act}
				{@const actPosRatio =
					(act.startSec / script.totalDurationSec) * 100}
				<div
					class="act-marker"
					style={`left: ${actPosRatio}%`}
					title={`${act.actNumber}: ${english ? act.titleEn : act.titleZh}`}
				>
					{#if act.checkpoint}
						<span class="checkpoint-gem" title={english ? "Contains interactive checkpoint" : "含交互驻留点"}
							>⚡</span
						>
					{/if}
				</div>
			{/each}
		</div>

		<!-- 播放器底栏控制按钮组 -->
		<div class="footer-controls-row">
			<div class="left-btn-group">
				<button
					type="button"
					class="cinema-btn main-play-btn"
					onclick={togglePlay}
					title={isPlaying ? (english ? "Pause" : "暂停") : (english ? "Play" : "播放")}
					aria-label={isPlaying ? (english ? "Pause" : "暂停") : (english ? "Play" : "播放")}
				>
					{isPlaying ? "⏸" : "▶"}
				</button>
				<button
					type="button"
					class="cinema-btn step-btn"
					onclick={() => skip(-5)}
					title={english ? "Rewind 5 seconds" : "快退 5 秒"}
					aria-label={english ? "Rewind 5 seconds" : "快退 5 秒"}
				>
					-5s
				</button>
				<button
					type="button"
					class="cinema-btn step-btn"
					onclick={() => skip(5)}
					title={english ? "Forward 5 seconds" : "快进 5 秒"}
					aria-label={english ? "Forward 5 seconds" : "快进 5 秒"}
				>
					+5s
				</button>
				<div class="time-readout">
					<strong>{formatTime(currentTime)}</strong> / {formatTime(
						script.totalDurationSec,
					)}
				</div>
			</div>

			<div class="act-chips-row">
				{#each script.acts as act, idx}
					<button
						type="button"
						class="act-chip"
						class:active={activeActIndex === idx}
						onclick={() => seekTo(act.startSec)}
					>
						<span class="chip-num">{act.actNumber}</span>
						<span class="chip-name"
							>{english ? act.titleEn : act.titleZh}</span
						>
					</button>
				{/each}
			</div>
		</div>
	</footer>
</div>

<style>
	.code-cinema-wrapper {
		position: relative;
		width: 100%;
		background: #0b1118;
		border: 1px solid rgba(138, 160, 180, 0.2);
		border-radius: 12px;
		overflow: hidden;
		display: flex;
		flex-direction: column;
		box-shadow: 0 12px 40px rgba(0, 0, 0, 0.35);
		margin: 2rem 0;
		color: #e8eef5;
	}

	.code-cinema-wrapper.is-fullscreen {
		position: fixed;
		inset: 0;
		z-index: 99999;
		margin: 0;
		border-radius: 0;
		height: 100vh;
		width: 100vw;
	}

	.cinema-topbar {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 0.85rem 1.25rem;
		border-bottom: 1px solid rgba(255, 255, 255, 0.08);
		background: #111a24;
	}

	.meta-col {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		flex-wrap: wrap;
	}
	.act-kicker-pill {
		font-size: 0.72rem;
		font-weight: 800;
		letter-spacing: 0.08em;
		padding: 0.2rem 0.55rem;
		border-radius: 99px;
		background: #4ade80;
		color: #0b1118;
	}
	.hero-token-switcher {
		display: inline-flex;
		align-items: center;
		gap: 0.35rem;
		background: rgba(255, 255, 255, 0.05);
		padding: 2px 6px;
		border-radius: 99px;
		border: 1px solid rgba(251, 191, 36, 0.3);
	}
	.switcher-label {
		font-size: 0.7rem;
		font-weight: 700;
		color: #8aa0b4;
		margin-right: 2px;
	}
	.token-chip-btn {
		display: inline-flex;
		align-items: center;
		gap: 3px;
		border: 1px solid transparent;
		background: transparent;
		color: #e8eef5;
		font-size: 0.72rem;
		font-weight: 600;
		padding: 0.15rem 0.45rem;
		border-radius: 99px;
		cursor: pointer;
		transition: all 0.18s cubic-bezier(0.16, 1, 0.3, 1);
	}
	.token-chip-btn .tok-text {
		font-weight: 700;
	}
	.token-chip-btn .sub-pos {
		font-size: 0.62rem;
		font-family: monospace;
		opacity: 0.75;
	}
	.token-chip-btn:hover {
		background: rgba(255, 255, 255, 0.1);
		color: #ffffff;
	}
	.token-chip-btn.active {
		background: rgba(251, 191, 36, 0.22);
		color: #fbbf24;
		border-color: #fbbf24;
		box-shadow: 0 0 10px rgba(251, 191, 36, 0.25);
	}
	.act-title {
		font-size: 1.05rem;
		font-weight: 700;
		margin: 0;
		color: #f8fafc;
	}

	.actions-col {
		display: flex;
		align-items: center;
		gap: 0.6rem;
	}
	.view-mode-pill {
		font-size: 0.74rem;
		font-weight: 600;
		padding: 0.25rem 0.55rem;
		border-radius: 4px;
		background: rgba(255, 255, 255, 0.06);
		color: #8aa0b4;
	}

	.speed-selector {
		display: flex;
		background: rgba(255, 255, 255, 0.06);
		border-radius: 6px;
		padding: 2px;
	}
	.speed-btn {
		border: none;
		background: transparent;
		font-size: 0.72rem;
		font-weight: 600;
		padding: 0.2rem 0.45rem;
		border-radius: 4px;
		cursor: pointer;
		color: #8aa0b4;
	}
	.speed-btn.active {
		background: #4ade80;
		color: #0b1118;
	}

	.cinema-btn {
		border: 1px solid rgba(255, 255, 255, 0.15);
		background: transparent;
		color: inherit;
		border-radius: 6px;
		cursor: pointer;
		display: inline-flex;
		align-items: center;
		justify-content: center;
		transition: all 0.15s ease;
	}
	.cinema-btn:hover {
		background: rgba(255, 255, 255, 0.08);
	}
	.icon-btn {
		width: 28px;
		height: 28px;
		font-size: 0.95rem;
	}

	.stage-viewport {
		position: relative;
		width: 100%;
		height: 540px;
		background: #0b1118;
		overflow: hidden;
		display: flex;
		align-items: center;
		justify-content: center;
	}
	.is-fullscreen .stage-viewport {
		flex: 1;
		height: auto;
	}

	.cinema-svg {
		width: 100%;
		height: 100%;
		display: block;
		user-select: none;
	}

	/* 🎬 电影字幕栏 */
	.cinema-subtitle-bar {
		background: #090d12;
		color: #ffffff;
		padding: 1rem 1.5rem;
		min-height: 86px;
		display: flex;
		flex-direction: column;
		justify-content: center;
		border-top: 1px solid rgba(255, 255, 255, 0.08);
		border-bottom: 1px solid rgba(255, 255, 255, 0.06);
	}

	.speaker-tag-row {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		margin-bottom: 0.35rem;
	}
	.speaker-pill {
		font-size: 0.72rem;
		font-weight: 700;
		color: #fbbf24;
		background: rgba(251, 191, 36, 0.15);
		padding: 0.15rem 0.45rem;
		border-radius: 4px;
	}
	.scene-badge {
		font-size: 0.7rem;
		color: #8aa0b4;
		font-family: monospace;
	}
	.sub-line-zh {
		font-size: 1.05rem;
		font-weight: 600;
		line-height: 1.45;
		margin: 0;
		color: #f8fafc;
	}
	.sub-line-en {
		font-size: 0.82rem;
		line-height: 1.4;
		margin: 0.2rem 0 0;
		color: #8aa0b4;
	}
	.subtitle-placeholder {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		color: #8aa0b4;
		font-size: 0.85rem;
	}
	.placeholder-dot {
		color: #4ade80;
		animation: pulse 1.5s infinite;
	}
	@keyframes pulse {
		0%, 100% { opacity: 0.3; }
		50% { opacity: 1; }
	}

	/* 交互驻留 HUD 遮罩 */
	.checkpoint-hud-overlay {
		position: absolute;
		inset: 0;
		background: rgba(0, 0, 0, 0.75);
		backdrop-filter: blur(6px);
		display: flex;
		align-items: center;
		justify-content: center;
		padding: 1.5rem;
		z-index: 10;
		animation: fadeIn 0.2s ease-out;
	}
	@keyframes fadeIn {
		from { opacity: 0; }
		to { opacity: 1; }
	}

	.hud-card {
		background: #111a24;
		color: #e8eef5;
		max-width: 480px;
		width: 100%;
		border-radius: 12px;
		padding: 1.5rem;
		border: 2px solid #4ade80;
		box-shadow: 0 16px 48px rgba(0, 0, 0, 0.5);
	}
	.hud-badge {
		font-size: 0.72rem;
		font-weight: 800;
		letter-spacing: 0.1em;
		color: #4ade80;
		margin-bottom: 0.35rem;
	}
	.hud-prompt {
		font-size: 1.15rem;
		margin: 0 0 0.5rem;
		font-weight: 700;
		color: #f8fafc;
	}
	.hud-guidance {
		font-size: 0.85rem;
		line-height: 1.5;
		color: #8aa0b4;
		margin-bottom: 1.25rem;
	}
	.hud-controls {
		display: flex;
		flex-direction: column;
		gap: 1rem;
		background: rgba(0, 0, 0, 0.25);
		padding: 1rem;
		border-radius: 8px;
		margin-bottom: 1.25rem;
	}
	.p-label-box {
		display: flex;
		justify-content: space-between;
		font-size: 0.82rem;
		font-weight: 600;
		margin-bottom: 0.35rem;
	}
	.p-val {
		font-family: monospace;
		color: #4ade80;
	}
	.hud-slider {
		width: 100%;
		accent-color: #4ade80;
	}
	.resume-cinema-btn {
		width: 100%;
		padding: 0.75rem;
		border: none;
		border-radius: 8px;
		background: #4ade80;
		color: #0b1118;
		font-weight: 800;
		font-size: 0.95rem;
		cursor: pointer;
		transition: background 0.15s ease;
	}
	.resume-cinema-btn:hover {
		background: #22c55e;
	}

	.cinema-footer {
		background: #0d141d;
		border-top: 1px solid rgba(255, 255, 255, 0.08);
		padding: 0.75rem 1.25rem 1rem;
	}

	.timeline-track-wrap {
		position: relative;
		width: 100%;
		height: 8px;
		background: rgba(255, 255, 255, 0.1);
		border-radius: 99px;
		cursor: pointer;
		margin-bottom: 0.85rem;
	}
	.timeline-fill {
		height: 100%;
		background: #4ade80;
		border-radius: 99px;
		pointer-events: none;
	}
	.act-marker {
		position: absolute;
		top: -2px;
		width: 2px;
		height: 12px;
		background: rgba(255, 255, 255, 0.35);
		pointer-events: none;
	}
	.checkpoint-gem {
		position: absolute;
		top: -12px;
		left: -6px;
		font-size: 0.7rem;
	}

	.footer-controls-row {
		display: flex;
		justify-content: space-between;
		align-items: center;
		flex-wrap: wrap;
		gap: 0.75rem;
	}
	.left-btn-group {
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}
	.main-play-btn {
		width: 34px;
		height: 34px;
		background: #4ade80;
		color: #0b1118;
		border-color: #4ade80;
		font-size: 1rem;
		font-weight: 800;
	}
	.step-btn {
		padding: 0.25rem 0.5rem;
		font-size: 0.72rem;
		font-weight: 600;
	}
	.time-readout {
		font-size: 0.8rem;
		font-family: monospace;
		margin-left: 0.4rem;
		color: #8aa0b4;
	}
	.time-readout strong {
		color: #f8fafc;
	}

	.act-chips-row {
		display: flex;
		gap: 0.35rem;
		overflow-x: auto;
		max-width: 100%;
	}
	.act-chip {
		display: inline-flex;
		align-items: center;
		gap: 0.35rem;
		border: 1px solid rgba(255, 255, 255, 0.12);
		background: transparent;
		color: #8aa0b4;
		border-radius: 99px;
		padding: 0.2rem 0.55rem;
		font-size: 0.72rem;
		cursor: pointer;
		white-space: nowrap;
		transition: all 0.15s ease;
	}
	.act-chip.active {
		border-color: #4ade80;
		background: rgba(74, 222, 128, 0.15);
		color: #4ade80;
		font-weight: 700;
	}
	.chip-num {
		font-weight: 800;
		opacity: 0.8;
	}
</style>
