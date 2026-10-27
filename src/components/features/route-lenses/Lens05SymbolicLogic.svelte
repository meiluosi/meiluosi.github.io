<script lang="ts">
/**
 * Lens 05: 符号、逻辑、搜索与组合路线 (Physical Symbol Systems & Search)
 * 揭示：智能产生于物理符号系统的模式匹配、结构推演与启发式搜索，核心在于如何剪除组合爆炸的庞大状态空间。
 */
let english = $state(false);

$effect(() => {
	const syncLang = () => {
		english = document.documentElement.lang.toLowerCase().startsWith("en");
	};
	syncLang();
	const obs = new MutationObserver(syncLang);
	obs.observe(document.documentElement, {
		attributes: true,
		attributeFilter: ["lang"],
	});
	return () => obs.disconnect();
});

type SearchStrategy = "bfs" | "dfs" | "astar";

interface TreeNode {
	id: string;
	label: string;
	depth: number;
	heuristic: number; // h(n) estimated distance to goal
	cost: number; // g(n) actual cost from root
	isGoal?: boolean;
	children?: string[];
}

// Tree representation of a navigation/reasoning task
const NODES: Record<string, TreeNode> = {
	root: {
		id: "root",
		label: "起始状态 S₀",
		depth: 0,
		heuristic: 4,
		cost: 0,
		children: ["A", "B"],
	},
	A: {
		id: "A",
		label: "规则推演 A (企图展开)",
		depth: 1,
		heuristic: 3,
		cost: 1,
		children: ["A1", "A2"],
	},
	B: {
		id: "B",
		label: "规则推演 B (次优分支)",
		depth: 1,
		heuristic: 5,
		cost: 1,
		children: ["B1", "B2"],
	},
	A1: {
		id: "A1",
		label: "定理导出 A1 (靠近目标)",
		depth: 2,
		heuristic: 1,
		cost: 2,
		children: ["GOAL", "A1_dead"],
	},
	A2: {
		id: "A2",
		label: "矛盾无效 A2 (死胡同)",
		depth: 2,
		heuristic: 6,
		cost: 2,
		children: [],
	},
	B1: {
		id: "B1",
		label: "组合冗余 B1",
		depth: 2,
		heuristic: 7,
		cost: 2,
		children: [],
	},
	B2: {
		id: "B2",
		label: "组合冗余 B2",
		depth: 2,
		heuristic: 8,
		cost: 2,
		children: [],
	},
	GOAL: {
		id: "GOAL",
		label: "目标定理 Q.E.D",
		depth: 3,
		heuristic: 0,
		cost: 3,
		isGoal: true,
	},
	A1_dead: {
		id: "A1_dead",
		label: "冗余旁支",
		depth: 3,
		heuristic: 4,
		cost: 3,
	},
};

let currentStrategy = $state<SearchStrategy>("astar");
let visitedNodes = $state<string[]>([]);
let isSearching = $state(false);
let goalFound = $state(false);

function runSearch(strategy: SearchStrategy) {
	currentStrategy = strategy;
	visitedNodes = [];
	goalFound = false;
	isSearching = true;

	const order: string[] = [];

	if (strategy === "bfs") {
		// Breadth First: root, A, B, A1, A2, B1, B2, GOAL
		order.push("root", "A", "B", "A1", "A2", "B1", "B2", "GOAL");
	} else if (strategy === "dfs") {
		// Depth First (explore blind deep branch first): root, B, B1, B2, A, A2, A1, GOAL
		order.push("root", "B", "B1", "B2", "A", "A2", "A1", "GOAL");
	} else {
		// A* Heuristic: f(n) = g(n) + h(n). Directly prioritizes S0 -> A (1+3=4) -> A1 (2+1=3) -> GOAL (3+0=3)
		order.push("root", "A", "A1", "GOAL");
	}

	// Animate step-by-step
	let step = 0;
	const timer = setInterval(() => {
		if (step < order.length) {
			visitedNodes = [...visitedNodes, order[step]];
			if (order[step] === "GOAL") {
				goalFound = true;
				isSearching = false;
				clearInterval(timer);
			}
			step++;
		} else {
			isSearching = false;
			clearInterval(timer);
		}
	}, 350);
}

function reset() {
	visitedNodes = [];
	goalFound = false;
	isSearching = false;
}
</script>

<div class="lens-container">
	<div class="lens-header">
		<span class="lens-tag">COGNITIVE LENS 05</span>
		<h3 class="lens-title">{english ? "Symbolic Tree & Heuristic Pruning" : "物理符号系统与启发式搜索剪枝沙盒"}</h3>
		<p class="lens-subtitle">
			{english
				? "Newell & Simon: General intelligence operates by manipulating discrete symbols, applying formal deduction rules, and using heuristics to prune exponential search trees."
				: "纽厄尔 (Allen Newell) 与西蒙：物理符号系统假说认为，思考即符号的组合、规则匹配与状态空间遍历。面对指数爆炸，启发式评估函数 (Heuristic) 是智能战胜蛮力搜索的核心武器。"}
		</p>
	</div>

	<!-- Search Strategy Selector -->
	<div class="strategy-bar">
		<div class="btn-group">
			<button
				type="button"
				class="strat-btn"
				class:active={currentStrategy === "bfs"}
				onclick={() => runSearch("bfs")}
			>
				{english ? "Breadth-First (Brute Force)" : "广度优先 BFS (蛮力层序遍历)"}
			</button>
			<button
				type="button"
				class="strat-btn"
				class:active={currentStrategy === "dfs"}
				onclick={() => runSearch("dfs")}
			>
				{english ? "Depth-First (Blind Dive)" : "深度优先 DFS (盲目深探)"}
			</button>
			<button
				type="button"
				class="strat-btn highlight"
				class:active={currentStrategy === "astar"}
				onclick={() => runSearch("astar")}
			>
				★ {english ? "A* Heuristic Search (f = g + h)" : "A* 启发式搜索 (剪枝直达)"}
			</button>
		</div>

		<button type="button" class="reset-btn" onclick={reset}>
			{english ? "Reset Tree" : "重置状态树"}
		</button>
	</div>

	<!-- Interactive Search Tree Grid -->
	<div class="tree-canvas">
		<div class="tree-level">
			<div class="level-tag">Depth 0 (Root)</div>
			<div class="nodes-row">
				<div class="node-badge" class:visited={visitedNodes.includes("root")}>
					<span class="node-name">S₀</span>
					<span class="node-meta">h=4</span>
				</div>
			</div>
		</div>

		<div class="connector-line"></div>

		<div class="tree-level">
			<div class="level-tag">Depth 1 (Premises)</div>
			<div class="nodes-row">
				<div class="node-badge" class:visited={visitedNodes.includes("A")}>
					<span class="node-name">Rule A</span>
					<span class="node-meta">f = 1+3 = 4</span>
				</div>
				<div class="node-badge" class:visited={visitedNodes.includes("B")}>
					<span class="node-name">Rule B</span>
					<span class="node-meta">f = 1+5 = 6</span>
				</div>
			</div>
		</div>

		<div class="connector-line"></div>

		<div class="tree-level">
			<div class="level-tag">Depth 2 (Intermediate Lemmata)</div>
			<div class="nodes-row">
				<div class="node-badge" class:visited={visitedNodes.includes("A1")}>
					<span class="node-name">Lemma A1</span>
					<span class="node-meta">f = 2+1 = 3</span>
				</div>
				<div class="node-badge" class:visited={visitedNodes.includes("A2")}>
					<span class="node-name">Branch A2</span>
					<span class="node-meta">Dead End</span>
				</div>
				<div class="node-badge" class:visited={visitedNodes.includes("B1")}>
					<span class="node-name">Redundant B1</span>
					<span class="node-meta">f = 2+7 = 9</span>
				</div>
				<div class="node-badge" class:visited={visitedNodes.includes("B2")}>
					<span class="node-name">Redundant B2</span>
					<span class="node-meta">f = 2+8 = 10</span>
				</div>
			</div>
		</div>

		<div class="connector-line"></div>

		<div class="tree-level">
			<div class="level-tag">Depth 3 (Goal & Proof)</div>
			<div class="nodes-row">
				<div class="node-badge goal" class:visited={visitedNodes.includes("GOAL")}>
					<span class="node-name">🎯 Q.E.D (Goal State)</span>
					<span class="node-meta">h = 0</span>
				</div>
			</div>
		</div>
	</div>

	<!-- Statistics & Comparison -->
	<div class="search-stats">
		<div class="stat-pill">
			<span>{english ? "Nodes Expanded: " : "已展开探索节点："}</span>
			<strong>{visitedNodes.length} / 8</strong>
		</div>
		<div class="stat-pill">
			<span>{english ? "Pruned Subtrees: " : "剪枝规避冗余分支："}</span>
			<strong>{8 - visitedNodes.length}</strong>
		</div>
		<div class="stat-pill" class:success={goalFound}>
			<span>{english ? "Target Status: " : "目标状态："}</span>
			<strong>{goalFound ? "Solved (证明完成)" : isSearching ? "Searching..." : "Pending"}</strong>
		</div>
	</div>

	<!-- Insight Box -->
	<div class="insight-box">
		<div class="insight-icon">🔍</div>
		<div class="insight-text">
			<strong>{english ? "Core Dilemma of Symbolism:" : "符号路线的根本分水岭："}</strong>
			{english
				? "Logic guarantees rigor, but combinatorial explosion guarantees failure without heuristics. Real intelligence lies in designing domain representations and evaluation functions that guide reasoning directly toward truth without brute-forcing every branch."
				: "形式逻辑保证了严格性，但未加引导的演绎必然陷入维数灾难与组合爆炸。符号智能的精髓不在于‘全量死算’，而在于将世界结构化表征为知识库，并通过启发式估值引导搜索直接切中命题要害。"}
		</div>
	</div>
</div>

<style>
.lens-container {
	border: 1px solid var(--border-color, rgba(120, 140, 130, 0.25));
	background: var(--brand-paper, #faf8f5);
	color: var(--brand-ink, #2d3748);
	border-radius: 12px;
	padding: 1.5rem;
	margin: 2rem 0;
	box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
}
:global(.dark) .lens-container {
	background: #141816;
	border-color: rgba(120, 150, 130, 0.2);
	color: #e2e8f0;
}
.lens-tag {
	font-size: 0.75rem;
	font-weight: 700;
	letter-spacing: 0.1em;
	color: var(--brand-clay, #8a7356);
	text-transform: uppercase;
}
.lens-title {
	font-size: 1.35rem;
	margin: 0.25rem 0 0.5rem;
	font-weight: 700;
}
.lens-subtitle {
	font-size: 0.88rem;
	line-height: 1.5;
	color: var(--text-muted, #718096);
	margin-bottom: 1.25rem;
}
.strategy-bar {
	display: flex;
	justify-content: space-between;
	align-items: center;
	flex-wrap: wrap;
	gap: 0.75rem;
	margin-bottom: 1.25rem;
	padding-bottom: 1rem;
	border-bottom: 1px dashed var(--border-color, rgba(120, 140, 130, 0.2));
}
.btn-group {
	display: flex;
	gap: 0.5rem;
	flex-wrap: wrap;
}
.strat-btn {
	background: rgba(0, 0, 0, 0.04);
	border: 1px solid rgba(0, 0, 0, 0.08);
	color: inherit;
	padding: 0.4rem 0.8rem;
	border-radius: 6px;
	font-size: 0.82rem;
	cursor: pointer;
	transition: all 0.2s;
}
:global(.dark) .strat-btn {
	background: rgba(255, 255, 255, 0.04);
	border-color: rgba(255, 255, 255, 0.1);
}
.strat-btn.active {
	background: var(--brand-ink, #2d3748);
	color: #fff;
	border-color: var(--brand-ink, #2d3748);
}
:global(.dark) .strat-btn.active {
	background: #3b5043;
}
.strat-btn.highlight.active {
	background: #0284c7;
	border-color: #0284c7;
}
.reset-btn {
	background: transparent;
	border: 1px solid var(--border-color, rgba(120, 140, 130, 0.3));
	color: var(--text-muted, #718096);
	padding: 0.35rem 0.7rem;
	border-radius: 6px;
	font-size: 0.78rem;
	cursor: pointer;
}
.tree-canvas {
	background: rgba(0, 0, 0, 0.02);
	border: 1px solid rgba(0, 0, 0, 0.06);
	border-radius: 8px;
	padding: 1.25rem;
	display: flex;
	flex-direction: column;
	gap: 0.5rem;
	align-items: center;
	margin-bottom: 1.25rem;
}
:global(.dark) .tree-canvas {
	background: rgba(0, 0, 0, 0.25);
	border-color: rgba(255, 255, 255, 0.06);
}
.tree-level {
	width: 100%;
	display: flex;
	flex-direction: column;
	align-items: center;
	gap: 0.35rem;
}
.level-tag {
	font-size: 0.65rem;
	font-weight: 700;
	color: var(--text-muted, #94a3b8);
	text-transform: uppercase;
}
.nodes-row {
	display: flex;
	gap: 0.75rem;
	justify-content: center;
	flex-wrap: wrap;
}
.node-badge {
	display: flex;
	flex-direction: column;
	align-items: center;
	padding: 0.4rem 0.8rem;
	border-radius: 6px;
	background: rgba(255, 255, 255, 0.8);
	border: 1px solid rgba(0, 0, 0, 0.1);
	min-width: 90px;
	opacity: 0.4;
	transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
:global(.dark) .node-badge {
	background: rgba(255, 255, 255, 0.04);
	border-color: rgba(255, 255, 255, 0.1);
}
.node-badge.visited {
	opacity: 1;
	transform: scale(1.05);
	background: #e0f2fe;
	border-color: #38bdf8;
	color: #0369a1;
	box-shadow: 0 4px 12px rgba(56, 189, 248, 0.2);
}
:global(.dark) .node-badge.visited {
	background: #082f49;
	border-color: #0284c7;
	color: #bae6fd;
}
.node-badge.goal.visited {
	background: #dcfce7;
	border-color: #22c55e;
	color: #15803d;
	box-shadow: 0 4px 14px rgba(34, 197, 94, 0.3);
}
:global(.dark) .node-badge.goal.visited {
	background: #052e16;
	border-color: #16a34a;
	color: #bbf7d0;
}
.node-name {
	font-size: 0.82rem;
	font-weight: 600;
}
.node-meta {
	font-size: 0.68rem;
	opacity: 0.8;
	font-family: monospace;
}
.connector-line {
	width: 2px;
	height: 12px;
	background: var(--border-color, rgba(120, 140, 130, 0.25));
}
.search-stats {
	display: flex;
	gap: 1rem;
	justify-content: space-around;
	margin-bottom: 1.25rem;
	flex-wrap: wrap;
}
.stat-pill {
	font-size: 0.82rem;
	padding: 0.4rem 0.8rem;
	background: rgba(0, 0, 0, 0.03);
	border-radius: 6px;
}
:global(.dark) .stat-pill {
	background: rgba(255, 255, 255, 0.03);
}
.stat-pill.success strong {
	color: #10b981;
}
.insight-box {
	display: flex;
	gap: 0.75rem;
	padding: 0.85rem 1rem;
	background: rgba(138, 115, 86, 0.08);
	border-left: 3px solid var(--brand-clay, #8a7356);
	border-radius: 4px;
	font-size: 0.84rem;
	line-height: 1.5;
}
.insight-icon {
	font-size: 1.1rem;
}
</style>
