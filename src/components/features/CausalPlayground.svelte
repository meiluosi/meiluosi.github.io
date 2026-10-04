<script lang="ts">
	import { onMount } from "svelte";
	let english = $state(false);
	let bias = $state(60);
	let effect = $state(20);
	const share = $derived(50 + bias / 2);
	const treated = $derived(40 + 80 * share / 100 + effect);
	const control = $derived(40 + 80 * (1 - share / 100));
	const observed = $derived(treated - control);
	const signed = (value: number) => `${value > 0 ? "+" : ""}${value.toFixed(0)}`;
	onMount(() => {
		const sync = () => { english = document.documentElement.lang.startsWith("en"); };
		sync(); document.addEventListener("site-locale-change", sync);
		return () => document.removeEventListener("site-locale-change", sync);
	});
</script>

<section class="experiment" aria-label={english ? "Coupon selection experiment" : "优惠券选择偏差实验"}>
	<div class="controls">
		<p class="eyebrow">{english ? "01 / CHANGE THE ASSIGNMENT" : "01 / 改变分配方式"}</p>
		<h2>{english ? "Who gets the coupon?" : "优惠券，更偏向发给谁？"}</h2>
		<p>{english ? "Two equally sized groups spend 120 and 40 units without a coupon. Turn the sliders to change who receives it and what it actually does." : "两类顾客人数相同，不用券时的消费分别是 120 与 40。拖动滑块，改变发券偏好和优惠券真正带来的增量。"}</p>
		<label for="selection-bias">{english ? "Preference for high-spending customers" : "偏向高消费人群的程度"}<output for="selection-bias">{signed(bias)} {english ? "pp" : "个百分点"}</output></label>
		<input id="selection-bias" type="range" min="-80" max="80" step="10" bind:value={bias} aria-valuetext={english ? `${signed(bias)} percentage points difference in coupon probability` : `两类人群获券概率相差 ${signed(bias)} 个百分点`} />
		<div class="range-labels"><span>{english ? "Lower spend" : "偏向低消费"}</span><span>{english ? "Random" : "随机分配"}</span><span>{english ? "Higher spend" : "偏向高消费"}</span></div>
		<label for="coupon-effect">{english ? "Actual effect of the coupon" : "设定优惠券的真实效果"}<output for="coupon-effect">{signed(effect)}</output></label>
		<input id="coupon-effect" type="range" min="-20" max="40" step="5" bind:value={effect} />
		<div class="range-labels"><span>−20</span><span>0</span><span>+40</span></div>
		<div class="mobile-result" aria-hidden="true"><span>{english ? "Raw difference" : "直接比较"}<strong>{signed(observed)}</strong></span><span>{english ? "Within-type effect" : "组内比较"}<strong>{signed(effect)}</strong></span></div>
		<div class="actions"><button type="button" onclick={() => { bias = 0; }}>{english ? "Try random assignment" : "试试随机分配"}</button><button class="reset" type="button" onclick={() => { bias = 60; effect = 20; }}>{english ? "Reset" : "重置"}</button></div>
	</div>
	<div class="results">
		<p class="eyebrow">{english ? "02 / WATCH THE COMPOSITION" : "02 / 看看人群构成"}</p>
		<div class="legend"><span class="high-dot"></span>{english ? "High baseline: 120" : "高消费：120"}<span class="low-dot"></span>{english ? "Low baseline: 40" : "低消费：40"}</div>
		<div class="composition"><div><span>{english ? "With coupon" : "拿到券"}</span><b>{share}% / {100 - share}%</b></div><div class="bar" role="img" aria-label={english ? `${share}% high-spending, ${100 - share}% low-spending` : `高消费占 ${share}%，低消费占 ${100 - share}%`}><span style:width={`${share}%`}></span></div></div>
		<div class="composition"><div><span>{english ? "Without coupon" : "没拿到券"}</span><b>{100 - share}% / {share}%</b></div><div class="bar" role="img" aria-label={english ? `${100 - share}% high-spending, ${share}% low-spending` : `高消费占 ${100 - share}%，低消费占 ${share}%`}><span style:width={`${100 - share}%`}></span></div></div>
		<div class="score-grid" aria-live="polite" aria-atomic="true">
			<div class="score naive"><span>{english ? "Compare everyone" : "直接比较两组"}</span><strong>{signed(observed)}</strong><small>{treated.toFixed(0)} − {control.toFixed(0)}</small></div>
			<div class="score adjusted"><span>{english ? "Compare within each type" : "在同类人群内比较"}</span><strong>{signed(effect)}</strong><small>{english ? "Matches the effect you set" : "等于你设定的真实效果"}</small></div>
		</div>
		<p class="insight">{bias === 0 ? (english ? "With equal assignment, both groups have the same composition. The raw difference now matches the causal effect." : "随机分配后，两组人群构成相同。直接比较得到的差异，此时才与因果效果一致。") : (english ? `The raw difference includes ${signed(observed - effect)} units from who received the coupon. A different mix of customers can exaggerate, hide, or even reverse the apparent effect.` : `直接比较中，有 ${signed(observed - effect)} 的差异来自“券发给了谁”。人群构成不同，就可能夸大、掩盖，甚至颠倒你看到的效果。`)}</p>
	</div>
</section>
<details class="assumptions"><summary>{english ? "What does this little model assume?" : "这个小模型做了哪些假设？"}</summary><div><p>{english ? "This is a deterministic teaching model showing expected values, not collected customer data. Both customer types are equally common. The coupon adds the same effect for everyone; there is no random noise or hidden confounder. Assignment probability is 50% + bias/2 for high spenders and 50% − bias/2 for low spenders." : "这是按设定计算期望值的教学模型，不是顾客实测数据。两类人群各占一半，优惠券对每个人的增量相同，不含随机噪声或未观测混杂因素。高消费人群获券概率为 50% + 偏好/2，低消费人群为 50% − 偏好/2。"}</p><p>{english ? "Within each type, compare (baseline + effect) with baseline, then average the two differences. In real observational data, adjustment only works under additional assumptions, including measuring the relevant confounders and having overlap. A slider cannot establish causality in a real study." : "组内比较先分别计算（原有消费 + 效果）− 原有消费，再平均两类人的差值。真实观察数据中的校正还需要满足混杂因素被恰当观测、存在可比较样本等条件，不能仅凭分组比较就认定因果。"}</p></div></details>

<style>
	.mobile-result{display:none}@media(max-width:760px){.mobile-result{display:flex;gap:25px;margin-top:22px;padding:14px 0;border-block:1px solid var(--brand-line);font-size:11px;color:var(--brand-body)}.mobile-result>span{display:flex;gap:12px;align-items:center}.mobile-result strong{font:400 26px Georgia,serif;color:var(--brand-clay)}.mobile-result>span:last-child strong{color:var(--brand-accent)}}
	.experiment{display:grid;grid-template-columns:1fr 1fr;border:1px solid var(--brand-line);border-radius:22px;overflow:hidden;background:var(--brand-paper)}.controls,.results{padding:clamp(22px,4vw,44px)}.results{background:var(--brand-surface);border-left:1px solid var(--brand-line)}.eyebrow{font:10px/1.8 "JetBrains Mono",monospace;letter-spacing:.08em;color:var(--brand-accent);margin:0 0 22px}h2{font-size:24px;line-height:1.5;font-weight:450;margin:0 0 15px}.controls>p:not(.eyebrow),.insight,.assumptions p{font-size:13px;line-height:2;color:var(--brand-body)}label{display:flex;justify-content:space-between;gap:20px;font-size:12px;line-height:1.8;margin:30px 0 10px}output{font-variant-numeric:tabular-nums;color:var(--brand-accent);white-space:nowrap}input{display:block;accent-color:var(--brand-accent);width:100%;height:28px;cursor:pointer}.range-labels{display:flex;justify-content:space-between;font-size:10px;color:var(--brand-muted);margin-top:6px}.actions{display:flex;gap:12px;margin-top:30px;flex-wrap:wrap}button{cursor:pointer;font-size:12px;padding:12px 18px;border:1px solid var(--brand-line);border-radius:25px;background:var(--brand-peach);color:var(--brand-ink)}button.reset{background:transparent}.legend{display:flex;gap:7px;align-items:center;flex-wrap:wrap;font-size:10px;color:var(--brand-body);margin-bottom:26px}.high-dot,.low-dot{width:9px;height:9px;border-radius:50%;background:var(--brand-accent)}.low-dot{background:var(--brand-peach);margin-left:12px}.composition{margin:22px 0}.composition>div:first-child{display:flex;justify-content:space-between;font-size:12px;margin-bottom:10px}.composition b{font:11px "JetBrains Mono",monospace;color:var(--brand-muted)}.bar{height:20px;background:var(--brand-peach);border-radius:6px;overflow:hidden}.bar span{display:block;height:100%;background:var(--brand-accent);transition:width .2s}.score-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:30px}.score{padding:18px 14px;border:1px solid var(--brand-line);border-radius:12px;background:var(--brand-paper)}.score>span{display:block;font-size:11px;color:var(--brand-body)}.score strong{display:block;font:400 clamp(36px,5vw,58px)/1.2 Georgia,serif;letter-spacing:-.05em;margin:14px 0;color:var(--brand-clay);font-variant-numeric:tabular-nums}.adjusted strong{color:var(--brand-accent)}.score small{font-size:10px;line-height:1.6;color:var(--brand-muted)}.insight{margin:24px 0 0}.assumptions{margin-top:24px;border-bottom:1px solid var(--brand-line);padding:0 0 22px}summary{cursor:pointer;font-size:13px;color:var(--brand-accent)}.assumptions>div{max-width:760px}@media(max-width:760px){.experiment{grid-template-columns:1fr}.results{border-left:0;border-top:1px solid var(--brand-line)}.score strong{font-size:46px}}@media(prefers-reduced-motion:reduce){.bar span{transition:none}}
</style>
