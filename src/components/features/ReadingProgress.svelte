<script lang="ts">
	import { onMount } from "svelte";

	let progress = $state(0);
	let visible = $state(false);

	onMount(() => {
		const article = document.querySelector("article");
		if (!article) {
			// fallback: use document body
			const updateProgress = () => {
				const scrollTop = window.scrollY;
				const docHeight = document.documentElement.scrollHeight - window.innerHeight;
				progress = docHeight > 0 ? Math.min((scrollTop / docHeight) * 100, 100) : 0;
				visible = scrollTop > 100;
			};
			window.addEventListener("scroll", updateProgress, { passive: true });
			updateProgress();
			return () => window.removeEventListener("scroll", updateProgress);
		}

		const updateProgress = () => {
			const articleTop = article.offsetTop;
			const articleHeight = article.offsetHeight;
			const scrollTop = window.scrollY;
			const viewportHeight = window.innerHeight;

			// progress starts when article top reaches viewport bottom
			const start = articleTop - viewportHeight;
			const range = articleHeight;
			const current = scrollTop - start;

			progress = range > 0 ? Math.min(Math.max((current / range) * 100, 0), 100) : 0;
			visible = scrollTop > articleTop - 100;
		};

		window.addEventListener("scroll", updateProgress, { passive: true });
		updateProgress();

		return () => window.removeEventListener("scroll", updateProgress);
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
			aria-label="阅读进度"
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