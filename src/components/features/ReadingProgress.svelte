<script lang="ts">
	import { onMount } from "svelte";

	let progress = $state(0);
	let visible = $state(false);
	let english = $state(false);

	onMount(() => {
		const article = document.querySelector<HTMLElement>("#post-container") ?? document.body;
		const syncLocale = () => { english = document.documentElement.lang.startsWith("en"); };
		syncLocale(); document.addEventListener("site-locale-change", syncLocale);
		const updateProgress = () => {
			const articleTop = article.getBoundingClientRect().top + window.scrollY;
			const articleHeight = article.offsetHeight;
			const scrollTop = window.scrollY;
			const viewportHeight = window.innerHeight;

			const start = articleTop;
			const range = Math.max(1, articleHeight - viewportHeight);
			const current = scrollTop - start;

			progress = range > 0 ? Math.min(Math.max((current / range) * 100, 0), 100) : 0;
			visible = scrollTop > articleTop - 100;
		};

		window.addEventListener("scroll", updateProgress, { passive: true });
		window.addEventListener("resize", updateProgress);
		const resize = new ResizeObserver(updateProgress);
		resize.observe(article);
		updateProgress();

		return () => {
			window.removeEventListener("scroll", updateProgress);
			window.removeEventListener("resize", updateProgress);
			document.removeEventListener("site-locale-change", syncLocale);
			resize.disconnect();
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
		transition: width 0.15s linear;
		box-shadow: 0 0 8px var(--primary, hsl(165, 70%, 50%));
	}
</style>
