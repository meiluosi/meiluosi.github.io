<script lang="ts">
import { onMount } from "svelte";

let progress = $state(0);
let visible = $state(false);
let english = $state(false);

onMount(() => {
	const article =
		document.querySelector<HTMLElement>("#post-container") ?? document.body;
	let frame = 0;
	let disposed = false;
	// 文章几何只在测量时读取布局一次；scroll 回调只用缓存值
	let articleTop = 0;
	let articleHeight = 1;
	let viewportHeight = window.innerHeight;

	const measure = () => {
		if (disposed) return;
		const scrollTop = window.scrollY;
		const rect = article.getBoundingClientRect();
		articleTop = rect.top + scrollTop;
		articleHeight = rect.height;
		viewportHeight = window.innerHeight;
	};

	const render = () => {
		frame = 0;
		const scrollTop = window.scrollY;
		const range = Math.max(1, articleHeight - viewportHeight);
		const current = scrollTop - articleTop;
		progress = Math.min(Math.max((current / range) * 100, 0), 100);
		visible = scrollTop > articleTop - 100;
	};

	const schedule = () => {
		if (disposed || frame) return;
		frame = requestAnimationFrame(render);
	};

	const syncLocale = () => {
		english = document.documentElement.lang.startsWith("en");
	};

	// 尺寸/字体变化后先重算缓存，下一帧再刷新进度
	const remeasure = () => {
		if (disposed) return;
		measure();
		schedule();
	};

	syncLocale();
	document.addEventListener("site-locale-change", syncLocale);
	window.addEventListener("scroll", schedule, { passive: true });
	window.addEventListener("resize", remeasure);
	const resizeObserver = new ResizeObserver(remeasure);
	resizeObserver.observe(article);
	// 自定义字体加载完成后行高会变化，需要重新测量文章高度
	document.fonts?.ready.then(remeasure).catch(() => {});
	measure();
	render();

	return () => {
		disposed = true;
		if (frame) {
			cancelAnimationFrame(frame);
			frame = 0;
		}
		window.removeEventListener("scroll", schedule);
		window.removeEventListener("resize", remeasure);
		document.removeEventListener("site-locale-change", syncLocale);
		resizeObserver.disconnect();
	};
});
</script>

{#if visible}
	<div class="reading-progress-container">
		<div
			class="reading-progress-bar"
			style="width: {progress}%"
			role="progressbar"
			aria-valuenow={Math.round(progress)}
			aria-valuemin="0"
			aria-valuemax="100"
			aria-label={english ? "Reading progress" : "阅读进度"}
		></div>
	</div>
{/if}

<style>
	.reading-progress-container {
		position: fixed;
		top: 0;
		left: 0;
		width: 100%;
		height: 3px;
		z-index: 10000;
		background: transparent;
		pointer-events: none;
	}

	.reading-progress-bar {
		height: 100%;
		background: linear-gradient(
			90deg,
			var(--primary, hsl(165, 70%, 50%)),
			var(--primary-light, hsl(165, 80%, 60%))
		);
		border-radius: 0 2px 2px 0;
		box-shadow: 0 0 8px var(--primary, hsl(165, 70%, 50%));
	}
</style>
