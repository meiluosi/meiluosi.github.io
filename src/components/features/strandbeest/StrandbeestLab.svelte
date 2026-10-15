<script lang="ts">
import { onMount } from "svelte";
import {
	DEFAULT_CRANK,
	polygonPath,
	rodPath,
	solveStrandbeest,
	STRANDBEEST_LINKS,
	traceStrandbeest,
} from "@/utils/strandbeest-geometry";

let { mode = "glance" }: { mode?: "glance" | "study" } = $props();
const uid = $props.id();
let root: HTMLDivElement;
let angle = $state(205);
let crank = $state(DEFAULT_CRANK);
let step = $state(0);
let playing = $state(false);
let ready = $state(false);
let english = $state(false);
let reduced = $state(false);
let visible = false;
let frame = 0;
let lastTime = 0;
const baseline = traceStrandbeest();
const pose = $derived(solveStrandbeest(angle, crank));
const trace = $derived(traceStrandbeest(crank));
const full = $derived(mode === "glance" || step === 2);
const say = (zh: string, en: string) => (english ? en : zh);
const stepText = $derived(
	[
		say(
			"先看桃色的短杆。它绕固定点 O 转动，另一端 C 只能沿着圆走。拖动角度，看它带动整条腿。",
			"Start with the terracotta crank. It turns around fixed point O, so its other end C must follow a circle. Drag the angle and watch the leg respond.",
		),
		say(
			"K 同时连着 G 和 C，两根杆的长度不变。以 G、C 为圆心画圆，K 只能在交点上。这里固定选择同一侧的装配分支。",
			"K is connected to both G and C by rods of fixed length. Draw a circle around each: K must lie at their intersection. This model keeps the same assembly branch.",
		),
		say(
			"重复这个约束，就得到脚端 F。虚线是一整圈的计算轨迹。试着把曲柄从 15 拉长到 30，再慢慢转一圈：有些角度会无法闭合。",
			"Repeat the constraint to locate foot F. The dashed line is its calculated path over one turn. Lengthen the crank from 15 to 30, then turn it slowly: some angles become unreachable.",
		),
	][step],
);

function stopFrame() {
	cancelAnimationFrame(frame);
	frame = 0;
	lastTime = 0;
}
function tick(time: number) {
	frame = 0;
	if (!playing || !visible || document.hidden) {
		lastTime = 0;
		return;
	}
	if (!lastTime) lastTime = time;
	const elapsed = time - lastTime;
	if (elapsed >= 1000 / 30) {
		angle = (angle + Math.min(elapsed, 65) * 0.028) % 360;
		lastTime = time;
	}
	frame = requestAnimationFrame(tick);
}
function syncFrame() {
	if (!ready) return;
	if (playing && visible && !document.hidden) {
		if (!frame) frame = requestAnimationFrame(tick);
	} else stopFrame();
}
function toggle() {
	playing = !playing;
	syncFrame();
}
function manual() {
	playing = false;
	stopFrame();
}
function reset() {
	manual();
	angle = 205;
	crank = DEFAULT_CRANK;
}

onMount(() => {
	ready = true;
	const media = window.matchMedia("(prefers-reduced-motion: reduce)");
	reduced = media.matches;
	const locale = () => {
		english = document.documentElement.lang.startsWith("en");
	};
	const motion = () => {
		reduced = media.matches;
		if (reduced) manual();
	};
	locale();
	playing = mode === "glance" && !reduced;
	const observer = new IntersectionObserver(
		([entry]) => {
			visible = entry.isIntersecting;
			syncFrame();
		},
		{ threshold: 0.08 },
	);
	observer.observe(root);
	media.addEventListener("change", motion);
	document.addEventListener("visibilitychange", syncFrame);
	document.addEventListener("site-locale-change", locale);
	return () => {
		stopFrame();
		observer.disconnect();
		media.removeEventListener("change", motion);
		document.removeEventListener("visibilitychange", syncFrame);
		document.removeEventListener("site-locale-change", locale);
	};
});
</script>

<div class:study={mode === "study"} class="linkage" bind:this={root}>
	{#if mode === "study"}
		<div class="steps" aria-label={say("拆开看这条腿", "Look inside the leg")}>
			{#each [say("1 · 一个圆", "1 · One circle"), say("2 · 两个交点", "2 · Two intersections"), say("3 · 一条腿", "3 · A leg")] as label, index}
				<button type="button" class:active={step === index} aria-pressed={step === index} disabled={!ready} onclick={() => { step = index; }}>{label}</button>
			{/each}
		</div>
	{/if}
	<div class="drawing">
		<svg viewBox={mode === "study" && step === 1 ? "-94 -76 211 183" : "-96 -62 184 168"} role="img" aria-label={say("Jansen 式平面连杆与脚端轨迹", "Jansen-style planar linkage and foot path")}>
			<title>{say("从圆周运动到脚端轨迹", "From rotation to a foot path")}</title>
			<desc>{say("两处固定铰，桃色曲柄带动十根连杆。脚端路径由圆交点逐步计算。", "Two fixed pivots and a terracotta crank drive ten rods. The foot path is calculated through successive circle intersections.")}</desc>
			{#if full}<path d={baseline.path} class="baseline" />{/if}
			{#if full}<path d={trace.path} class="trace" />{/if}
			<g opacity={full ? 1 : .2}>
				<path d={polygonPath(pose, ["G", "K", "L"])} class="triangle upper" />
				<path d={polygonPath(pose, ["M", "N", "F"])} class="triangle lower" />
				<path d={rodPath(pose)} class="rods" />
			</g>
			{#if pose.points.O}
				<circle cx={pose.points.O.x} cy={-pose.points.O.y} r={crank} class="construction" />
			{/if}
			{#if mode === "study" && step === 1 && pose.points.C}
				<circle cx="0" cy="0" r="41.5" class="construction circle-g" />
				<circle cx={pose.points.C.x} cy={-pose.points.C.y} r="50" class="construction circle-c" />
				<path d={rodPath(pose, STRANDBEEST_LINKS.slice(0, 2))} class="rods" />
			{/if}
			{#if pose.points.O && pose.points.C}<line x1={pose.points.O.x} y1={-pose.points.O.y} x2={pose.points.C.x} y2={-pose.points.C.y} class="crank" />{/if}
			{#each Object.entries(pose.points) as [name, point]}
				{#if point}
					<circle cx={point.x} cy={-point.y} r={name === "F" ? 1.7 : 1.1} class:foot={name === "F"} class="joint" opacity={full || ["G", "O", "C", ...(step === 1 ? ["K"] : [])].includes(name) ? 1 : .2} />
					{#if ["G", "O"].includes(name)}<circle cx={point.x} cy={-point.y} r="2.8" class="fixed" />{/if}
					{#if mode === "study" && (["G", "O", "C"].includes(name) || (step === 1 && name === "K") || (step === 2 && name === "F"))}
						<text x={point.x + 3.8} y={-point.y - 3.8}>{name}</text>
					{/if}
				{/if}
			{/each}
		</svg>
		<div class="legend"><span class="legend-crank"></span>{say("曲柄", "Crank")}<span class="legend-trace"></span>{say("脚端轨迹", "Foot path")}</div>
	</div>
	{#if mode === "study"}<p class="explanation" aria-live="polite">{stepText}</p>{/if}
	<div class="controls">
		<button type="button" class="play" disabled={!ready} onclick={toggle} aria-label={playing ? say("暂停连杆运动", "Pause the linkage") : say("播放连杆运动", "Play the linkage")}><span aria-hidden="true">{playing ? "Ⅱ" : "▷"}</span> {playing ? say("暂停", "Pause") : say("播放", "Play")}</button>
		<label for={`${uid}-angle`}><span>{say("手动转动", "Turn by hand")} <output>{Math.round(angle)}°</output></span><input id={`${uid}-angle`} type="range" min="0" max="360" step="1" bind:value={angle} oninput={manual} disabled={!ready} /></label>
		{#if mode === "study"}<button type="button" class="reset" disabled={!ready} onclick={reset}>{say("恢复初始", "Reset")}</button>{/if}
	</div>
	{#if mode === "study" && step === 2}
		<div class="length-control"><label for={`${uid}-length`}><span>{say("只改变曲柄长度", "Change only the crank")} <output>{crank.toFixed(1)}</output></span><input id={`${uid}-length`} type="range" min="5" max="40" step=".5" bind:value={crank} oninput={manual} disabled={!ready} /></label><p>{say("默认 15 · 其余杆长不变 · 无量纲", "Default 15 · All other rods stay fixed · Dimensionless")}</p></div>
		<p class="cycle-state">{say("每隔 1° 采样：", "Sampled every 1°: ")}{trace.validSamples} / 360 {say("个相位可闭合。这是数值采样，不是连续可达性的证明。", "phases close. This is a numerical sample, not a proof of continuous reachability.")}</p>
	{/if}
	<p class:invalid={!pose.valid} class="geometry-status" aria-live="polite">{#if !pose.valid}{say(`当前角度无法闭合：关节 ${pose.failedJoint} 的两个圆没有可用交点。图中只保留已求出的杆件；请转动曲柄或恢复初始长度。`, `This angle cannot close: joint ${pose.failedJoint} has no usable circle intersection. Only solved rods remain. Turn the crank or restore the initial length.`)}{/if}</p>
	{#if reduced}<p class="motion-note">{say("已遵循系统减少动态效果设置，默认静止；可手动转动或主动播放。", "Reduced motion is on. The diagram starts still; turn it by hand or choose Play.")}</p>{/if}
</div>

<style>
	.linkage{color:var(--brand-ink);min-width:0}.drawing{position:relative;max-width:720px;margin:auto}.drawing svg{display:block;width:100%;height:clamp(285px,45vw,465px);overflow:hidden}.drawing text{font:4.5px "JetBrains Mono",monospace;fill:var(--brand-muted)}.rods{fill:none;stroke:var(--brand-ink);stroke-width:1.15;stroke-linecap:round;stroke-linejoin:round}.triangle{fill-opacity:.55}.upper{fill:var(--brand-sage)}.lower{fill:var(--brand-peach)}.crank{stroke:var(--brand-clay);stroke-width:2;stroke-linecap:round}.joint{fill:var(--brand-paper);stroke:var(--brand-ink);stroke-width:.7}.joint.foot{fill:var(--brand-clay)}.fixed{fill:none;stroke:var(--brand-ink);stroke-width:.5}.construction{fill:none;stroke:var(--brand-muted);stroke-width:.55;stroke-dasharray:1 1.6;opacity:.65}.circle-g{stroke:var(--brand-accent)}.circle-c{stroke:var(--brand-clay)}.trace{fill:none;stroke:var(--brand-clay);stroke-width:.8;stroke-dasharray:1.2 1.8}.baseline{fill:none;stroke:var(--brand-muted);stroke-width:.55;opacity:.26}.legend{display:flex;justify-content:center;align-items:center;gap:8px;font-size:13px;color:var(--brand-muted);margin:8px 0 20px}.legend-crank{width:18px;border-top:3px solid var(--brand-clay)}.legend-trace{width:18px;border-top:1px dashed var(--brand-clay);margin-left:18px}.controls{display:flex;gap:24px;align-items:center;max-width:600px;margin:auto;padding:12px 0}.controls label{flex:1;min-width:0}label{display:block;font-size:14px;color:var(--brand-body)}label>span{display:flex;justify-content:space-between;gap:15px}output{font-variant-numeric:tabular-nums;font-family:"JetBrains Mono",monospace}input[type=range]{width:100%;height:32px;margin:4px 0 0;accent-color:var(--brand-clay);cursor:pointer}button{font:inherit;font-size:14px;border:1px solid var(--brand-line);border-radius:4px;color:var(--brand-ink);background:transparent;padding:10px 16px;min-height:44px;cursor:pointer;white-space:nowrap}button:disabled{cursor:default;opacity:.5}button:hover:not(:disabled){background:var(--brand-sage)}button:focus-visible,input:focus-visible{outline:2px solid var(--brand-accent);outline-offset:4px}.play{min-width:88px}.play span{font-size:15px;margin-right:5px}.steps{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:14px}.steps button{flex:1;padding:10px 12px}.steps .active{background:var(--brand-sage);border-color:var(--brand-accent)}.explanation{font-size:15px;line-height:1.9;min-height:5.7em;max-width:620px;margin:14px auto 10px;color:var(--brand-body)}.length-control{max-width:600px;margin:14px auto 0;padding-top:18px;border-top:1px solid var(--brand-line)}.length-control p,.cycle-state,.motion-note{font-size:13px;line-height:1.8;color:var(--brand-muted);max-width:600px;margin:4px auto}.geometry-status{font-size:13px;line-height:1.8;max-width:600px;margin:0 auto}.geometry-status.invalid{border-left:2px solid var(--brand-clay);color:var(--brand-clay);padding:8px 12px;margin-top:12px}.geometry-status:empty{display:none}@media(max-width:500px){.controls{gap:12px;flex-wrap:wrap}.controls .reset{width:100%;border:none;text-decoration:underline;text-underline-offset:4px;padding:4px}.steps{gap:5px}.steps button{font-size:13px;padding:8px 5px}.explanation{font-size:14px;min-height:7.6em}.drawing svg{height:320px}}
</style>
