<script lang="ts">
import { onMount } from "svelte";
import { playMechanicalTick, playShutterTick } from "@/utils/haptics";

let {
	code = "",
	title = "Python 算例验证",
	titleEn = "Interactive Python Sandbox",
	language = "python",
} = $props<{
	code?: string;
	title?: string;
	titleEn?: string;
	language?: string;
}>();

let english = $state(false);
let editableCode = $state(code);
let status = $state<"idle" | "loading" | "running" | "success" | "error">(
	"idle",
);
let stdout = $state("");
let stderr = $state("");
let executionTimeMs = $state(0);

// Global Pyodide singleton on window
declare global {
	interface Window {
		loadPyodide?: (config: { indexURL: string }) => Promise<any>;
		__pyodideInstance?: any;
		__pyodideLoadingPromise?: Promise<any>;
	}
}

async function getPyodide(): Promise<any> {
	if (typeof window === "undefined") return null;
	if (window.__pyodideInstance) return window.__pyodideInstance;

	if (window.__pyodideLoadingPromise) {
		return await window.__pyodideLoadingPromise;
	}

	window.__pyodideLoadingPromise = (async () => {
		if (!window.loadPyodide) {
			await new Promise<void>((resolve, reject) => {
				const script = document.createElement("script");
				script.src = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js";
				script.async = true;
				script.onload = () => resolve();
				script.onerror = () =>
					reject(new Error("Failed to load Pyodide from CDN"));
				document.head.appendChild(script);
			});
		}

		if (!window.loadPyodide) {
			throw new Error("Pyodide script loaded but loadPyodide is not defined");
		}

		const pyodide = await window.loadPyodide({
			indexURL: "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/",
		});
		window.__pyodideInstance = pyodide;
		return pyodide;
	})();

	return await window.__pyodideLoadingPromise;
}

async function runCode(): Promise<void> {
	playShutterTick();
	status = "loading";
	stdout = "";
	stderr = "";

	const startTime = performance.now();

	try {
		const pyodide = await getPyodide();
		status = "running";

		// Python harness capturing stdout and stderr
		const harness = `
import sys, io
__stdout_buf = io.StringIO()
__stderr_buf = io.StringIO()
sys.stdout = __stdout_buf
sys.stderr = __stderr_buf

try:
    exec(${JSON.stringify(editableCode)}, globals())
except Exception as e:
    import traceback
    traceback.print_exc(file=sys.stderr)
finally:
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

__captured_out = __stdout_buf.getvalue()
__captured_err = __stderr_buf.getvalue()
`;
		await pyodide.runPythonAsync(harness);
		const out = pyodide.globals.get("__captured_out") || "";
		const err = pyodide.globals.get("__captured_err") || "";

		executionTimeMs = Math.round(performance.now() - startTime);
		stdout = out;
		stderr = err;

		if (err) {
			status = "error";
		} else {
			status = "success";
			playMechanicalTick(1.2);
		}
	} catch (e: any) {
		executionTimeMs = Math.round(performance.now() - startTime);
		status = "error";
		stderr = e.message || String(e);
	}
}

function resetCode(): void {
	editableCode = code;
	stdout = "";
	stderr = "";
	status = "idle";
}

onMount(() => {
	const sync = () => {
		english = document.documentElement.lang.startsWith("en");
	};
	sync();
	document.addEventListener("site-locale-change", sync);
	return () => {
		document.removeEventListener("site-locale-change", sync);
	};
});
</script>

<div class="code-sandbox-card">
	<header class="sandbox-header">
		<div class="header-left">
			<span class="python-badge">
				<span class="badge-dot"></span>
				{language.toUpperCase()} 3.12 (PYODIDE WASM)
			</span>
			<span class="sandbox-title">{english ? titleEn : title}</span>
		</div>
		<div class="header-actions">
			<button
				type="button"
				class="action-btn run-btn"
				disabled={status === "loading" || status === "running"}
				onclick={runCode}
			>
				{#if status === "loading"}
					<span class="spinner"></span> {english ? "Loading..." : "加载中..."}
				{:else if status === "running"}
					<span class="spinner"></span> {english ? "Executing..." : "运行中..."}
				{:else}
					▶ {english ? "Run" : "运行算例"}
				{/if}
			</button>
			{#if editableCode !== code || stdout || stderr}
				<button type="button" class="action-btn reset-btn" onclick={resetCode}>
					{english ? "Reset" : "重置"}
				</button>
			{/if}
		</div>
	</header>

	<div class="editor-wrap">
		<textarea
			class="code-textarea"
			bind:value={editableCode}
			spellcheck="false"
			rows={Math.max(4, Math.min(16, editableCode.split("\n").length))}
			aria-label={english ? "Python code" : "Python 算例代码"}
		></textarea>
	</div>

	{#if status !== "idle" || stdout || stderr}
		<div class="terminal-output" class:has-error={status === "error"}>
			<div class="terminal-bar">
				<span class="term-label">
					{english ? "STDOUT & VERIFICATION" : "标准输出与算例核验"}
				</span>
				{#if status === "success"}
					<span class="term-status success">
						✓ {english ? "Executed in" : "执行成功"} {executionTimeMs}ms
					</span>
				{:else if status === "error"}
					<span class="term-status error">
						✕ {english ? "Execution failed" : "执行异常"} ({executionTimeMs}ms)
					</span>
				{:else}
					<span class="term-status loading">
						{english ? "Evaluating in WebAssembly sandbox..." : "在浏览器沙盒执行中..."}
					</span>
				{/if}
			</div>
			<pre class="terminal-content"><code>{#if stdout}{stdout}{/if}{#if stderr}<span class="stderr-text">{stderr}</span>{/if}{#if !stdout && !stderr && status === "success"}<i>{english ? "(Code executed with empty stdout)" : "(代码执行完毕，无终端输出)"}</i>{/if}</code></pre>
		</div>
	{/if}
</div>

<style>
	.code-sandbox-card {
		background: var(--brand-paper);
		border: 1px solid var(--brand-line);
		border-radius: 12px;
		margin: 22px 0;
		overflow: hidden;
		box-shadow: 0 4px 18px rgba(37, 60, 52, 0.05);
	}
	.sandbox-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 10px 14px;
		background: var(--brand-surface);
		border-bottom: 1px solid var(--brand-line);
		flex-wrap: wrap;
		gap: 8px;
	}
	.header-left {
		display: flex;
		align-items: center;
		gap: 10px;
		flex-wrap: wrap;
	}
	.python-badge {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		font: 10px var(--font-mono);
		letter-spacing: 0.06em;
		color: var(--brand-accent);
		background: var(--brand-paper);
		border: 1px solid var(--brand-line);
		padding: 2px 7px;
		border-radius: 999px;
	}
	.badge-dot {
		width: 6px;
		height: 6px;
		border-radius: 50%;
		background: var(--brand-accent);
	}
	.sandbox-title {
		font-size: 13px;
		font-weight: 500;
		color: var(--brand-ink);
	}
	.header-actions {
		display: flex;
		align-items: center;
		gap: 8px;
	}
	.action-btn {
		font: 500 12px var(--font-mono);
		padding: 4px 12px;
		border-radius: 6px;
		border: 1px solid var(--brand-line);
		cursor: pointer;
		transition: all 0.15s ease;
		display: inline-flex;
		align-items: center;
		gap: 6px;
	}
	.run-btn {
		background: var(--brand-peach);
		color: var(--brand-ink);
		border-color: var(--brand-clay);
	}
	.run-btn:hover:not(:disabled) {
		background: color-mix(in srgb, var(--brand-peach) 85%, var(--brand-clay));
		transform: translateY(-1px);
	}
	.run-btn:disabled {
		opacity: 0.6;
		cursor: wait;
	}
	.reset-btn {
		background: transparent;
		color: var(--brand-muted);
	}
	.reset-btn:hover {
		color: var(--brand-ink);
		border-color: var(--brand-accent);
	}
	.editor-wrap {
		position: relative;
	}
	.code-textarea {
		width: 100%;
		background: var(--brand-paper);
		color: var(--brand-ink);
		font: 13px/1.55 var(--font-mono);
		padding: 14px 16px;
		border: none;
		outline: none;
		resize: vertical;
		display: block;
		box-sizing: border-box;
		tab-size: 4;
	}
	.code-textarea:focus {
		background: var(--brand-surface);
	}
	.terminal-output {
		background: #181f1b;
		color: #e5ece7;
		border-top: 1px solid var(--brand-line);
		font-family: var(--font-mono);
	}
	.terminal-bar {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 6px 14px;
		background: #131916;
		border-bottom: 1px solid #25332c;
		font-size: 11px;
	}
	.term-label {
		color: #8fa598;
		letter-spacing: 0.05em;
	}
	.term-status.success {
		color: #6ee7b7;
	}
	.term-status.error {
		color: #fca5a5;
	}
	.term-status.loading {
		color: #fcd34d;
	}
	.terminal-content {
		margin: 0;
		padding: 12px 14px;
		font-size: 12px;
		line-height: 1.6;
		white-space: pre-wrap;
		word-break: break-all;
		max-height: 240px;
		overflow-y: auto;
	}
	.stderr-text {
		color: #fca5a5;
	}
	.spinner {
		display: inline-block;
		width: 10px;
		height: 10px;
		border: 2px solid currentColor;
		border-right-color: transparent;
		border-radius: 50%;
		animation: spin 0.75s linear infinite;
	}
	@keyframes spin {
		to {
			transform: rotate(360deg);
		}
	}
</style>
