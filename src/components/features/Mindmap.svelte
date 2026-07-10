<script lang="ts">
	interface Props {
		headings: Array<{ depth: number; text: string; slug: string }>;
	}

	let { headings }: Props = $props();

	const mindmapCode = $derived.by(() => {
		if (headings.length === 0) return "";

		// Find the minimum depth to root the mindmap
		const minDepth = Math.min(...headings.map((h) => h.depth));

		// Build Mermaid mindmap syntax
		const lines: string[] = ["mindmap"];
		const stack: number[] = [];

		for (const h of headings) {
			const indent = "  ".repeat(Math.max(0, h.depth - minDepth));
			// Clean the text for Mermaid (remove special chars)
			const cleanText = h.text
				.replace(/[\[\]{}()<>]/g, "")
				.replace(/"/g, "'")
				.replace(/:/g, "：")
				.trim();
			lines.push(`${indent}${cleanText}`);
		}

		return lines.join("\n");
	});

	const show = $derived(mindmapCode.length > 0);
</script>

{#if show}
	<div class="mindmap-section">
		<h3 class="mindmap-title">🧠 文章思维导图</h3>
		<div class="mermaid-wrapper">
			<pre class="mermaid">{mindmapCode}</pre>
		</div>
	</div>
{/if}

<style>
	.mindmap-section {
		margin-top: 2.5rem;
		padding: 1.5rem;
		border-radius: var(--radius-large, 12px);
		background: var(--card-bg, #fff);
		border: 1px solid var(--border-color, rgba(0, 0, 0, 0.08));
	}

	.mindmap-title {
		font-size: 1.1rem;
		font-weight: 600;
		margin-bottom: 1rem;
		color: var(--text-color, #1f2937);
	}

	.mermaid-wrapper {
		overflow-x: auto;
	}

	:global(.dark) .mindmap-section {
		background: var(--card-bg, #1f2937);
		border-color: rgba(255, 255, 255, 0.08);
	}
</style>