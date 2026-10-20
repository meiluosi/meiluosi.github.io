<script lang="ts">
import { onMount } from "svelte";
import {
	isHapticsEnabled,
	playLatchClick,
	setHapticsEnabled,
} from "@/utils/haptics";

let enabled = $state(true);
let english = $state(false);

function toggle() {
	const next = !enabled;
	enabled = next;
	setHapticsEnabled(next);
	if (next) {
		playLatchClick();
	}
}

onMount(() => {
	enabled = isHapticsEnabled();
	english = document.documentElement.lang.startsWith("en");
	const syncLang = () => {
		english = document.documentElement.lang.startsWith("en");
	};
	const syncAudio = (e: Event) => {
		const custom = e as CustomEvent<{ enabled: boolean }>;
		if (custom.detail) enabled = custom.detail.enabled;
	};
	document.addEventListener("site-locale-change", syncLang);
	window.addEventListener("blog-haptics-change", syncAudio);
	return () => {
		document.removeEventListener("site-locale-change", syncLang);
		window.removeEventListener("blog-haptics-change", syncAudio);
	};
});
</script>

<button
	type="button"
	class="haptic-toggle"
	class:active={enabled}
	onclick={toggle}
	title={enabled
		? english
			? "Mechanical sound effects: On (click to mute)"
			: "机械微音效：开启（点击静音）"
		: english
			? "Mechanical sound effects: Muted (click to enable)"
			: "机械微音效：静音（点击开启）"}
	aria-label={enabled
		? english
			? "Mute mechanical sounds"
			: "静音机械音效"
		: english
			? "Enable mechanical sounds"
			: "开启机械音效"}
>
	<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
		{#if enabled}
			<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
			<path d="M15.54 8.46a5 5 0 0 1 0 7.07" />
			<path d="M19.07 4.93a10 10 0 0 1 0 14.14" />
		{:else}
			<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
			<line x1="23" y1="9" x2="17" y2="15" />
			<line x1="17" y1="9" x2="23" y2="15" />
		{/if}
	</svg>
</button>

<style>
	.haptic-toggle {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		flex-shrink: 0;
		height: 36px;
		width: 36px;
		padding: 0;
		border: 1px solid var(--brand-line);
		border-radius: 999px;
		background: transparent;
		color: var(--brand-muted);
		cursor: pointer;
		transition:
			border-color 0.15s ease,
			color 0.15s ease,
			background-color 0.15s ease;
	}
	.haptic-toggle:hover {
		border-color: var(--brand-accent);
		color: var(--brand-ink);
	}
	.haptic-toggle.active {
		color: var(--brand-accent);
	}
</style>
