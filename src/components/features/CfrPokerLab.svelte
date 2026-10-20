<script lang="ts">
import { onMount } from "svelte";
import { type KuhnCard, runKuhnCfr, terminalUtility } from "@/utils/cfr-poker";

let english = $state(false);
let iterations = $state(250);
let showMode = $state<"average" | "current">("average");
let running = $state(false);
let timer: ReturnType<typeof setInterval> | undefined;

const snapshot = $derived(runKuhnCfr(iterations));

const p1Open = $derived(snapshot.infoSets.filter((s) => s.history === ""));
const p2AfterPass = $derived(
	snapshot.infoSets.filter((s) => s.history === "p"),
);
const p2AfterBet = $derived(snapshot.infoSets.filter((s) => s.history === "b"));
const p1AfterPassBet = $derived(
	snapshot.infoSets.filter((s) => s.history === "pb"),
);

const jOpenBet = $derived(
	snapshot.infoSets.find((s) => s.key === "J")?.averageStrategy[1] ?? 0,
);
const kOpenBet = $derived(
	snapshot.infoSets.find((s) => s.key === "K")?.averageStrategy[1] ?? 0,
);
const qbCall = $derived(
	snapshot.infoSets.find((s) => s.key === "Qb")?.averageStrategy[1] ?? 0,
);
const bluffRatio = $derived(
	jOpenBet > 0.005 ? (kOpenBet / jOpenBet).toFixed(2) : "—",
);

// Mini-game state: Human (Player 1) vs CFR Bot (Player 2)
const deck: KuhnCard[] = ["J", "Q", "K"];
let humanCard = $state<KuhnCard>("Q");
let botCard = $state<KuhnCard>("K");
let handHistory = $state("");
let botNote = $state("");
let handScore = $state(0);
let handsPlayed = $state(0);

function cardIndex(card: KuhnCard): number {
	return card === "J" ? 0 : card === "Q" ? 1 : 2;
}

function dealHand() {
	const c1 = Math.floor(Math.random() * 3);
	let c2 = Math.floor(Math.random() * 2);
	if (c2 >= c1) c2 += 1;
	humanCard = deck[c1];
	botCard = deck[c2];
	handHistory = "";
	botNote = "";
}

function playHumanAction(action: "p" | "b") {
	if (isTerminal(handHistory)) return;
	const next = `${handHistory}${action}`;
	handHistory = next;

	if (isTerminal(next)) {
		settleHand(next);
		return;
	}

	// Bot (Player 2) responds using current averageStrategy
	const key = `${botCard}${next}`;
	const info = snapshot.infoSets.find((s) => s.key === key);
	const probBet = info ? info.averageStrategy[1] : 0.5;
	const botAction = Math.random() < probBet ? "b" : "p";
	const afterBot = `${next}${botAction}`;
	botNote = english
		? `CFR Bot at [${key}] chose ${botAction === "b" ? "Bet/Call" : "Pass/Fold"} (P(Bet)=${(probBet * 100).toFixed(1)}%)`
		: `CFR 在信息集 [${key}] 选择了 ${botAction === "b" ? "下注/跟注 (Bet)" : "过牌/弃牌 (Pass)"}（策略概率 ${(probBet * 100).toFixed(1)}%）`;
	handHistory = afterBot;

	if (isTerminal(afterBot)) {
		settleHand(afterBot);
	}
}

function isTerminal(h: string): boolean {
	return h === "pp" || h === "bp" || h === "bb" || h === "pbp" || h === "pbb";
}

function settleHand(h: string) {
	const uActive = terminalUtility(h, cardIndex(humanCard), cardIndex(botCard));
	const activePlayer = h.length % 2 === 0 ? 1 : 2;
	const humanDelta = activePlayer === 1 ? uActive : -uActive;
	handScore += humanDelta;
	handsPlayed += 1;
}

const handFinished = $derived(isTerminal(handHistory));
const lastPayoff = $derived.by(() => {
	if (!handFinished) return 0;
	const uActive = terminalUtility(
		handHistory,
		cardIndex(humanCard),
		cardIndex(botCard),
	);
	const activePlayer = handHistory.length % 2 === 0 ? 1 : 2;
	return activePlayer === 1 ? uActive : -uActive;
});

function pause() {
	if (timer) clearInterval(timer);
	timer = undefined;
	running = false;
}

function togglePlay() {
	if (running) {
		pause();
		return;
	}
	if (iterations >= 2000) iterations = 0;
	running = true;
	timer = setInterval(() => {
		const step = iterations < 50 ? 2 : iterations < 300 ? 15 : 50;
		iterations = Math.min(2000, iterations + step);
		if (iterations >= 2000) pause();
	}, 90);
}

onMount(() => {
	const sync = () => {
		english = document.documentElement.lang.startsWith("en");
	};
	sync();
	document.addEventListener("site-locale-change", sync);
	dealHand();
	return () => {
		pause();
		document.removeEventListener("site-locale-change", sync);
	};
});
</script>

<section
	class="cfr-lab"
	aria-label={english
		? "Kuhn Poker Counterfactual Regret Minimization Workbench"
		: "Kuhn Poker 反事实遗憾最小化（CFR）实验台"}
>
	<div class="controls-bar">
		<label for="cfr-iterations" class="slider-label">
			<span>
				{english ? "CFR Self-Play Iterations (T)" : "CFR 自我博弈迭代轮数 (T)"}
				<strong>{iterations}</strong>
			</span>
			<input
				id="cfr-iterations"
				type="range"
				min="0"
				max="2000"
				step="1"
				value={iterations}
				oninput={(e) => {
					pause();
					iterations = Number(e.currentTarget.value);
				}}
			/>
		</label>

		<div class="preset-group" role="group" aria-label="Iteration presets">
			{#each [0, 1, 10, 50, 250, 1000, 2000] as preset}
				<button
					type="button"
					class="chip"
					class:active={iterations === preset}
					onclick={() => {
						pause();
						iterations = preset;
					}}
				>
					T={preset}
				</button>
			{/each}
			<button type="button" class="play-btn" onclick={togglePlay}>
				{running
					? english
						? "Pause"
						: "暂停"
					: english
						? "Animate T → 2000"
						: "动态演化"}
			</button>
		</div>

		<div class="mode-switch" role="group" aria-label="Strategy view mode">
			<button
				type="button"
				class:active={showMode === "average"}
				onclick={() => (showMode = "average")}
			>
				{english ? "Average Strategy (Nash)" : "平均策略 (收敛至纳什均衡)"}
			</button>
			<button
				type="button"
				class:active={showMode === "current"}
				onclick={() => (showMode = "current")}
			>
				{english ? "Current Regret-Matching" : "单步遗憾匹配策略 (瞬时震荡)"}
			</button>
		</div>
	</div>

	<div class="metrics-strip">
		<div class="metric-card">
			<small>{english ? "P1 EXPECTED VALUE" : "先手期望收益 EV₁"}</small>
			<strong>{snapshot.expectedValueP1.toFixed(4)}</strong>
			<span>
				{english
					? `Theoretical Nash: -1/18 ≈ ${snapshot.nashValueP1.toFixed(4)}`
					: `理论纳什值: -1/18 ≈ ${snapshot.nashValueP1.toFixed(4)}`}
			</span>
		</div>
		<div class="metric-card">
			<small>{english ? "VALUE : BLUFF RATIO (K : J)" : "先手价值下注 : 诈唬比 (K : J)"}</small>
			<strong>{bluffRatio} : 1</strong>
			<span>
				{english
					? "Nash optimal ratio: 3α : α = 3.00 : 1"
					: "理论最优比例: 3α : α = 3.00 : 1"}
			</span>
		</div>
		<div class="metric-card">
			<small>{english ? "P2 QUEEN CALL FREQUENCY" : "后手 Q 抓诈唬频率 P(Call | Qb)"}</small>
			<strong>{(qbCall * 100).toFixed(1)}%</strong>
			<span>
				{english
					? "Indifference threshold: 1/3 ≈ 33.3%"
					: "令先手诈唬无差异阈值: 1/3 ≈ 33.3%"}
			</span>
		</div>
	</div>

	<div class="stages-grid">
		{#each [
			{
				title: english ? "01 / P1 Opening (Root)" : "01 / 先手开局决策 (Root)",
				subtitle: english
					? "J bluffs with α; Q checks; K value-bets with 3α"
					: "J 以 α 概率诈唬；Q 稳健过牌；K 以 3α 价值下注",
				items: p1Open,
			},
			{
				title: english ? "02 / P2 Facing Check (History: p)" : "02 / 后手面对过牌 (历史: p)",
				subtitle: english
					? "J bluffs 1/3 of the time; Q checks behind; K always bets"
					: "J 以 1/3 概率偷池；Q 过牌摊牌；K 100% 价值下注",
				items: p2AfterPass,
			},
			{
				title: english ? "03 / P2 Facing Bet (History: b)" : "03 / 后手面对下注 (历史: b)",
				subtitle: english
					? "J folds; Q calls 1/3 to prevent bluffing; K always calls"
					: "J 弃牌；Q 以 1/3 概率跟注抓诈唬；K 100% 跟注",
				items: p2AfterBet,
			},
			{
				title: english
					? "04 / P1 Facing Check-Bet (History: pb)"
					: "04 / 先手过牌后面对下注 (历史: pb)",
				subtitle: english
					? "J folds; Q calls α + 1/3; K traps and always calls"
					: "J 弃牌；Q 以 α + 1/3 概率跟注；K 慢打后 100% 跟注",
				items: p1AfterPassBet,
			},
		] as group}
			<div class="stage-panel">
				<header>
					<h3>{group.title}</h3>
					<p>{group.subtitle}</p>
				</header>
				<div class="infoset-rows">
					{#each group.items as item}
						{@const strat =
							showMode === "average"
								? item.averageStrategy
								: item.currentStrategy}
						<div class="infoset-row">
							<div class="infoset-meta">
								<span class="card-badge" data-card={item.card}>{item.card}</span>
								<code>{item.key}</code>
								<span class="prob-readout">
									Pass {(strat[0] * 100).toFixed(0)}% ·
									<b>Bet {(strat[1] * 100).toFixed(0)}%</b>
								</span>
							</div>
							<div class="bar-track" aria-hidden="true">
								<span
									class="bar-pass"
									style={`width:${(strat[0] * 100).toFixed(1)}%`}
								></span>
								<span
									class="bar-bet"
									style={`width:${(strat[1] * 100).toFixed(1)}%`}
								></span>
							</div>
						</div>
					{/each}
				</div>
			</div>
		{/each}
	</div>

	<div class="play-box">
		<div class="play-header">
			<div>
				<span class="kicker">INTERACTIVE SPARRING</span>
				<h3>
					{english
						? "Play a Hand vs. Current CFR Policy"
						: "与当前迭代的 CFR 策略实战一手"}
				</h3>
			</div>
			<div class="score-pill">
				{english ? "Session Chips" : "累计筹码"}:
				<strong class:pos={handScore > 0} class:neg={handScore < 0}>
					{handScore > 0 ? `+${handScore}` : handScore}
				</strong>
				<span>({handsPlayed} {english ? "hands" : "局"})</span>
			</div>
		</div>

		<div class="table-arena">
			<div class="seat">
				<small>{english ? "YOU (PLAYER 1)" : "你 (先手 Player 1)"}</small>
				<div class="playing-card" data-card={humanCard}>{humanCard}</div>
			</div>

			<div class="pot-center">
				<p class="history-line">
					{english ? "Action History:" : "当前行动序列:"}
					<code>{handHistory || (english ? "(Opening)" : "(等待开局动作)")}</code>
				</p>
				{#if botNote}
					<p class="bot-note">{botNote}</p>
				{/if}
				{#if !handFinished}
					<div class="action-buttons">
						<button type="button" onclick={() => playHumanAction("p")}>
							{handHistory === "pb"
								? english
									? "Pass (Fold)"
									: "Pass (弃牌)"
								: english
									? "Pass (Check)"
									: "Pass (过牌)"}
						</button>
						<button
							type="button"
							class="bet-btn"
							onclick={() => playHumanAction("b")}
						>
							{handHistory === "pb"
								? english
									? "Bet (Call +1)"
									: "Bet (跟注 +1)"
								: english
									? "Bet (Bet +1)"
									: "Bet (下注 +1)"}
						</button>
					</div>
				{:else}
					<div class="showdown-banner">
						<strong>
							{lastPayoff > 0
								? english
									? `You win +${lastPayoff} chip${lastPayoff > 1 ? "s" : ""}!`
									: `本局获胜 +${lastPayoff} 筹码！`
								: english
									? `You lose ${lastPayoff} chip${lastPayoff < -1 ? "s" : ""}.`
									: `本局失利 ${lastPayoff} 筹码。`}
						</strong>
						<button type="button" onclick={dealHand}>
							{english ? "Deal Next Hand →" : "发下一手牌 →"}
						</button>
					</div>
				{/if}
			</div>

			<div class="seat">
				<small>{english ? "CFR BOT (PLAYER 2)" : "CFR 对手 (后手 Player 2)"}</small>
				<div
					class="playing-card"
					class:hidden-card={!handFinished}
					data-card={handFinished ? botCard : "?"}
				>
					{handFinished ? botCard : "?"}
				</div>
			</div>
		</div>
	</div>
</section>

<style>
	.cfr-lab {
		border: 1px solid var(--brand-line);
		border-radius: 18px;
		background: var(--brand-surface);
		padding: 26px;
		color: var(--brand-ink);
	}
	.controls-bar {
		display: grid;
		gap: 16px;
		padding-bottom: 20px;
		border-bottom: 1px solid var(--brand-line);
	}
	.slider-label {
		display: grid;
		gap: 8px;
		font-size: 13px;
	}
	.slider-label span {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
	}
	.slider-label strong {
		font: 600 16px var(--font-mono);
		color: var(--brand-accent);
	}
	input[type="range"] {
		width: 100%;
		accent-color: var(--brand-accent);
	}
	.preset-group,
	.mode-switch {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
	}
	button {
		font: 12px var(--font-mono);
		border: 1px solid var(--brand-line);
		border-radius: 999px;
		padding: 8px 14px;
		background: var(--brand-paper);
		color: var(--brand-body);
		cursor: pointer;
		transition:
			background-color 0.15s ease,
			color 0.15s ease,
			border-color 0.15s ease;
	}
	button:hover {
		border-color: var(--brand-accent);
		color: var(--brand-ink);
	}
	button.active {
		background: var(--brand-accent);
		color: var(--brand-paper);
		border-color: var(--brand-accent);
	}
	.play-btn {
		background: var(--brand-peach);
		color: var(--brand-ink);
		margin-left: auto;
	}
	.metrics-strip {
		display: grid;
		grid-template-columns: repeat(3, minmax(0, 1fr));
		gap: 14px;
		margin: 20px 0;
	}
	.metric-card {
		background: var(--brand-paper);
		border: 1px solid var(--brand-line);
		border-radius: 12px;
		padding: 14px 16px;
		display: grid;
		gap: 4px;
	}
	.metric-card small {
		font: 10px var(--font-mono);
		letter-spacing: 0.08em;
		color: var(--brand-muted);
	}
	.metric-card strong {
		font: 400 26px/1.2 var(--font-serif);
		color: var(--brand-accent);
	}
	.metric-card span {
		font-size: 11px;
		color: var(--brand-body);
	}
	.stages-grid {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 16px;
	}
	.stage-panel {
		background: var(--brand-paper);
		border: 1px solid var(--brand-line);
		border-radius: 12px;
		padding: 16px;
	}
	.stage-panel h3 {
		font: 600 14px var(--font-sans);
		margin: 0 0 4px;
		color: var(--brand-ink);
	}
	.stage-panel p {
		font-size: 11px;
		color: var(--brand-muted);
		margin: 0 0 14px;
		line-height: 1.5;
	}
	.infoset-rows {
		display: grid;
		gap: 10px;
	}
	.infoset-row {
		display: grid;
		gap: 5px;
	}
	.infoset-meta {
		display: flex;
		align-items: center;
		gap: 8px;
		font: 11px var(--font-mono);
	}
	.card-badge {
		display: inline-grid;
		place-items: center;
		width: 20px;
		height: 20px;
		border-radius: 5px;
		background: var(--brand-surface);
		border: 1px solid var(--brand-line);
		font-weight: 600;
		color: var(--brand-accent);
	}
	.card-badge[data-card="K"] {
		background: var(--brand-peach);
		color: var(--brand-clay);
	}
	.prob-readout {
		margin-left: auto;
		color: var(--brand-muted);
	}
	.prob-readout b {
		color: var(--brand-accent);
	}
	.bar-track {
		display: flex;
		height: 10px;
		border-radius: 999px;
		overflow: hidden;
		background: var(--brand-surface);
	}
	.bar-pass {
		background: var(--brand-sage);
		transition: width 0.15s ease;
	}
	.bar-bet {
		background: var(--brand-accent);
		transition: width 0.15s ease;
	}
	.play-box {
		margin-top: 22px;
		padding: 18px 20px;
		border-radius: 14px;
		border: 1px solid var(--brand-line);
		background: var(--brand-paper);
	}
	.play-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		flex-wrap: wrap;
		gap: 12px;
		margin-bottom: 14px;
	}
	.kicker {
		font: 10px var(--font-mono);
		letter-spacing: 0.1em;
		color: var(--brand-accent);
	}
	.play-header h3 {
		margin: 2px 0 0;
		font: 400 20px var(--font-serif);
	}
	.score-pill {
		font: 12px var(--font-mono);
		padding: 6px 12px;
		border-radius: 999px;
		background: var(--brand-surface);
		border: 1px solid var(--brand-line);
	}
	.score-pill strong.pos {
		color: var(--brand-accent);
	}
	.score-pill strong.neg {
		color: var(--brand-clay);
	}
	.table-arena {
		display: grid;
		grid-template-columns: 120px 1fr 120px;
		align-items: center;
		gap: 16px;
	}
	.seat {
		text-align: center;
		display: grid;
		place-items: center;
		gap: 8px;
	}
	.seat small {
		font: 10px var(--font-mono);
		color: var(--brand-muted);
	}
	.playing-card {
		width: 64px;
		height: 86px;
		border-radius: 10px;
		border: 1.5px solid var(--brand-accent);
		background: var(--brand-surface);
		display: grid;
		place-items: center;
		font: 400 32px var(--font-serif);
		color: var(--brand-ink);
	}
	.playing-card.hidden-card {
		border-style: dashed;
		color: var(--brand-muted);
	}
	.pot-center {
		text-align: center;
		display: grid;
		gap: 10px;
	}
	.history-line {
		margin: 0;
		font-size: 13px;
		color: var(--brand-body);
	}
	.bot-note {
		margin: 0;
		font: 11px var(--font-mono);
		color: var(--brand-accent);
	}
	.action-buttons,
	.showdown-banner {
		display: flex;
		justify-content: center;
		align-items: center;
		gap: 12px;
		flex-wrap: wrap;
	}
	.bet-btn {
		background: var(--brand-peach);
		color: var(--brand-ink);
		border-color: var(--brand-clay);
	}
	@media (max-width: 720px) {
		.metrics-strip,
		.stages-grid,
		.table-arena {
			grid-template-columns: 1fr;
		}
	}
</style>
