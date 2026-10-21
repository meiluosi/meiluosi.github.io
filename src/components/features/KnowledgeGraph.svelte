<script lang="ts">
import { onMount } from "svelte";
import {
	type GraphEdge,
	type GraphNode,
	MATURITY_DEFINITIONS,
	type MaturityLevel,
} from "@/utils/digital-garden";
import {
	playLatchClick,
	playMechanicalTick,
	playShutterTick,
} from "@/utils/haptics";

interface Props {
	initialNodes: GraphNode[];
	initialEdges: GraphEdge[];
}

let { initialNodes, initialEdges }: Props = $props();

let english = $state(false);
let canvasEl = $state<HTMLCanvasElement | null>(null);
let activeFilter = $state<MaturityLevel | "all">("all");
let searchQuery = $state("");
let selectedNode = $state<GraphNode | null>(null);
let hoveredNode = $state<GraphNode | null>(null);

// 3D Celestial Cosmos & 2D Layout State
let viewMode = $state<"2d" | "3d">("3d");
let rotX = $state(0.24);
let rotY = $state(0.0);
let isDragging3D = false;
let drag3DStart = { x: 0, y: 0 };

// Motion and visibility policy. The cosmos must not spin on its own when the
// reader asked for reduced motion, and it must not burn frames while it is
// scrolled out of view or the tab is in the background.
let reduceMotion = $state(false);
let isVisible = $state(true);
let settleFrames = 0;
let animRunning = false;

interface SimNodeExt extends GraphNode {
	x3d?: number;
	y3d?: number;
	z3d?: number;
	orbitRadius?: number;
	orbitSpeed?: number;
	orbitAngle?: number;
	projX?: number;
	projY?: number;
	projZ?: number;
	projScale?: number;
}

// Background cosmic stars
const stars = Array.from({ length: 80 }, (_, i) => ({
	x: Math.sin(i * 19.3) * 0.5 + 0.5,
	y: Math.cos(i * 37.7) * 0.5 + 0.5,
	r: (i % 3) * 0.5 + 0.7,
	alpha: Math.sin(i * 7.1) * 0.25 + 0.45,
}));

// Transform / Pan / Zoom state
let transform = $state({ x: 0, y: 0, k: 1 });
let isDraggingCanvas = false;
let dragStart = { x: 0, y: 0 };
let draggedNode: SimNodeExt | null = null;

// Dynamic simulation nodes
let simNodes = $state<SimNodeExt[]>([]);
let simEdges = $state<GraphEdge[]>([]);

const activeMaturityCount = $derived.by(() => {
	const counts: Record<string, number> = { all: simNodes.length };
	for (const n of simNodes) {
		counts[n.maturity] = (counts[n.maturity] || 0) + 1;
	}
	return counts;
});

// The canvas is a spatial view; this list is the same graph in plain text.
// It is what keyboard and screen-reader users navigate, and it is the static
// representation that stays usable without JavaScript.
const listedNodes = $derived.by(() => {
	const q = searchQuery.trim().toLowerCase();
	return simNodes.filter((n) => {
		const matchesFilter = activeFilter === "all" || n.maturity === activeFilter;
		if (!matchesFilter) return false;
		if (!q) return true;
		return (
			n.label.toLowerCase().includes(q) ||
			(n.labelEn ?? "").toLowerCase().includes(q) ||
			(n.category ?? "").toLowerCase().includes(q) ||
			Boolean(n.tags?.some((t) => t.toLowerCase().includes(q)))
		);
	});
});

// Node-kind labels live in both languages so the plain-text index can be
// localized by the same data-i18n mechanism the rest of the site uses. That
// keeps the English page from flashing Chinese labels before hydration.
const kindLabels: Record<GraphNode["kind"], { zh: string; en: string }> = {
	hub: { zh: "研究方向", en: "Research direction" },
	lab: { zh: "交互实验", en: "Interactive lab" },
	post: { zh: "文章", en: "Article" },
};

// Connected neighbors for the selected node
const connectedNeighbors = $derived.by(() => {
	if (!selectedNode) return [];
	const id = selectedNode.id;
	const neighborIds = new Set<string>();
	for (const e of simEdges) {
		if (e.source === id) neighborIds.add(e.target);
		if (e.target === id) neighborIds.add(e.source);
	}
	return simNodes.filter((n) => neighborIds.has(n.id));
});

// Build the node list without touching the canvas, so the same data can be
// rendered on the server. The plain-text node index below must exist in the
// static HTML: it is the keyboard, screen-reader and no-JS path into the
// graph, and the canvas is only its spatial view.
function buildNodes(): SimNodeExt[] {
	return initialNodes.map((n, i) => {
		const angle = (i / initialNodes.length) * 2 * Math.PI;
		const radius = 120 + (i % 4) * 55;

		// 3D celestial orbits
		let r3d = 200;
		let inclination = ((i % 5) - 2) * 0.18;
		let speed = 0.0006 + (i % 3) * 0.0003;
		if (n.kind === "hub") {
			r3d = 30 + i * 8;
			inclination = 0.05;
			speed = 0.0002;
		} else if (n.kind === "lab") {
			r3d = 135;
			inclination = 0.15 * (i % 2 === 0 ? 1 : -1);
			speed = 0.0012;
		} else if (n.maturity === "evergreen") {
			r3d = 210;
			inclination = 0.25 * (i % 2 === 0 ? 1 : -1);
			speed = 0.0008;
		} else {
			r3d = 280 + (i % 3) * 35;
			inclination = Math.sin(i * 1.5) * 0.45;
			speed = 0.0005;
		}

		// Deterministic jitter: the server render and the client hydration must
		// agree, so nothing here may depend on Math.random().
		const jitter = (seed: number) =>
			Math.sin(seed * 12.9898) * 43758.5453 - Math.floor(Math.sin(seed * 12.9898) * 43758.5453) - 0.5;

		return {
			...n,
			x: Math.cos(angle) * radius,
			y: Math.sin(angle) * radius,
			vx: jitter(i + 1),
			vy: jitter(i + 101),
			orbitRadius: r3d,
			orbitSpeed: speed,
			orbitAngle: angle,
			x3d: Math.cos(angle) * r3d,
			y3d: Math.sin(inclination) * r3d,
			z3d: Math.sin(angle) * r3d,
			projX: 0,
			projY: 0,
			projZ: 0,
			projScale: 1,
		};
	});
}

// The node list exists before the canvas does, so the server-rendered page
// already carries the full graph as plain links.
simNodes = buildNodes();
simEdges = [...initialEdges];

function initSimulation() {
	if (!canvasEl) return;
	const rect = canvasEl.getBoundingClientRect();
	const width = rect.width || 800;
	const height = rect.height || 600;

	transform.x = width / 2;
	transform.y = height / 2;
}

let animFrameId: number;

// Start or resume the render loop. With reduced motion the loop only runs for
// a short settle window after an interaction, then parks until something
// asks for a new frame.
function ensureAnimating(frames = 240) {
	if (reduceMotion) settleFrames = Math.max(settleFrames, frames);
	if (animRunning) return;
	animRunning = true;
	animFrameId = requestAnimationFrame(tickSimulation);
}

function parkAnimation() {
	if (!animRunning) return;
	cancelAnimationFrame(animFrameId);
	animRunning = false;
	settleFrames = 0;
}

function scheduleNextFrame() {
	if (!isVisible) {
		animRunning = false;
		return;
	}
	if (reduceMotion) {
		settleFrames -= 1;
		if (settleFrames <= 0) {
			animRunning = false;
			return;
		}
	}
	animFrameId = requestAnimationFrame(tickSimulation);
}

function tickSimulation() {
	if (!canvasEl) return;
	const ctx = canvasEl.getContext("2d");
	if (!ctx) return;

	const rect = canvasEl.getBoundingClientRect();
	const width = canvasEl.width;
	const height = canvasEl.height;

	// 3D CELESTIAL COSMOS MODE
	if (viewMode === "3d") {
		// Update cosmic rotation
		if (!isDragging3D && !reduceMotion) {
			rotY += 0.002;
		}

		// Calculate 3D positions and project
		const cosY = Math.cos(rotY);
		const sinY = Math.sin(rotY);
		const cosX = Math.cos(rotX);
		const sinX = Math.sin(rotX);
		const focal = 500;

		for (const n of simNodes) {
			if (!reduceMotion) {
				n.orbitAngle = (n.orbitAngle || 0) + (n.orbitSpeed || 0.0005);
			}
			const r = n.orbitRadius || 200;
			const rawX = Math.cos(n.orbitAngle) * r;
			const rawZ = Math.sin(n.orbitAngle) * r;
			const rawY = n.y3d || 0;

			// Rotate around Y
			const x1 = rawX * cosY - rawZ * sinY;
			const z1 = rawX * sinY + rawZ * cosY;

			// Rotate around X
			const y2 = rawY * cosX - z1 * sinX;
			const z2 = rawY * sinX + z1 * cosX;
			const x2 = x1;

			const scale = Math.max(0.2, focal / (focal + z2 + 250));
			n.projScale = scale;
			n.projZ = z2;
			n.projX = x2 * scale;
			n.projY = y2 * scale;
		}

		// Render 3D Scene
		ctx.clearRect(0, 0, width, height);

		// Draw twinkling stars
		ctx.save();
		for (const star of stars) {
			const sx = star.x * width;
			const sy = star.y * height;
			ctx.beginPath();
			ctx.arc(sx, sy, star.r, 0, Math.PI * 2);
			ctx.fillStyle = `rgba(180, 205, 195, ${star.alpha})`;
			ctx.fill();
		}
		ctx.restore();

		ctx.save();
		ctx.translate(transform.x, transform.y);
		ctx.scale(transform.k, transform.k);

		// Draw faint 3D orbital rings
		const orbitalRadii = [135, 210, 290];
		ctx.save();
		ctx.lineWidth = 1 / transform.k;
		for (const or of orbitalRadii) {
			ctx.beginPath();
			ctx.strokeStyle = "rgba(92, 122, 107, 0.15)";
			for (let step = 0; step <= 48; step++) {
				const theta = (step / 48) * Math.PI * 2;
				const rx = Math.cos(theta) * or;
				const rz = Math.sin(theta) * or;
				const rx1 = rx * cosY - rz * sinY;
				const rz1 = rx * sinY + rz * cosY;
				const ry2 = -rz1 * sinX;
				const rz2 = rz1 * cosX;
				const rscale = Math.max(0.2, focal / (focal + rz2 + 250));
				const px = rx1 * rscale;
				const py = ry2 * rscale;
				if (step === 0) ctx.moveTo(px, py);
				else ctx.lineTo(px, py);
			}
			ctx.stroke();
		}
		ctx.restore();

		// Draw edges in 3D
		const nodeMap = new Map(simNodes.map((n) => [n.id, n]));
		for (const e of simEdges) {
			const s = nodeMap.get(e.source);
			const t = nodeMap.get(e.target);
			if (!s || !t) continue;

			const isConnectedToHover =
				hoveredNode && (s.id === hoveredNode.id || t.id === hoveredNode.id);
			const isConnectedToSelected =
				selectedNode && (s.id === selectedNode.id || t.id === selectedNode.id);

			ctx.beginPath();
			ctx.moveTo(s.projX || 0, s.projY || 0);
			ctx.lineTo(t.projX || 0, t.projY || 0);

			if (isConnectedToHover || isConnectedToSelected) {
				ctx.strokeStyle = "rgba(92, 122, 107, 0.85)";
				ctx.lineWidth = 2.5 / transform.k;
			} else {
				const edgeAlpha = Math.max(
					0.08,
					Math.min(0.35, ((s.projScale || 1) + (t.projScale || 1)) * 0.16),
				);
				ctx.strokeStyle = `rgba(130, 160, 145, ${edgeAlpha})`;
				ctx.lineWidth = 1 / transform.k;
			}
			ctx.stroke();
		}

		// Sort nodes by projZ ascending (draw back to front)
		const sortedNodes = [...simNodes].sort(
			(a, b) => (a.projZ || 0) - (b.projZ || 0),
		);

		// Draw 3D nodes
		const qLower = searchQuery.trim().toLowerCase();
		for (const n of sortedNodes) {
			const isHovered = hoveredNode?.id === n.id;
			const isSelected = selectedNode?.id === n.id;
			const matchesSearch =
				qLower.length === 0 ||
				n.label.toLowerCase().includes(qLower) ||
				Boolean(n.tags?.some((t) => t.toLowerCase().includes(qLower)));
			const matchesFilter =
				activeFilter === "all" || n.maturity === activeFilter;

			const baseAlpha = matchesFilter && matchesSearch ? 1 : 0.15;
			const depthAlpha = Math.max(0.3, Math.min(1.0, (n.projScale || 1) * 1.1));
			const alpha = baseAlpha * depthAlpha;

			const meta =
				MATURITY_DEFINITIONS[n.maturity] || MATURITY_DEFINITIONS.budding;
			const baseRadius = n.kind === "hub" ? 15 : n.kind === "lab" ? 12 : 8;
			const radius = baseRadius * (n.projScale || 1) * (isHovered ? 1.35 : 1);

			ctx.save();
			ctx.globalAlpha = alpha;

			// Glow aura
			if (isHovered || isSelected || n.kind === "hub") {
				ctx.beginPath();
				ctx.arc(
					n.projX || 0,
					n.projY || 0,
					radius + 6 * (n.projScale || 1),
					0,
					Math.PI * 2,
				);
				ctx.fillStyle =
					n.kind === "hub"
						? "rgba(92, 122, 107, 0.35)"
						: "rgba(92, 122, 107, 0.22)";
				ctx.fill();
			}

			// Sphere node
			ctx.beginPath();
			ctx.arc(n.projX || 0, n.projY || 0, radius, 0, Math.PI * 2);
			ctx.fillStyle = meta.color;
			ctx.fill();
			ctx.strokeStyle = "#ffffff";
			ctx.lineWidth = (1.5 * (n.projScale || 1)) / transform.k;
			ctx.stroke();

			// Label
			if (
				(n.projScale || 1) > 0.55 ||
				isHovered ||
				isSelected ||
				n.kind === "hub"
			) {
				ctx.font = `${n.kind === "hub" ? "bold 13px" : "11px"} -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`;
				ctx.fillStyle =
					isHovered || isSelected ? "#1b2a26" : "rgba(40, 50, 45, 0.9)";
				ctx.textAlign = "center";
				const textY = (n.projY || 0) + radius + 13 * (n.projScale || 1);
				const displayLabel = english && n.labelEn ? n.labelEn : n.label;
				ctx.fillText(displayLabel, n.projX || 0, textY);
			}

			ctx.restore();
		}

		ctx.restore();
		scheduleNextFrame();
		return;
	}

	// 2D PLANAR SPRING FORCE ATLAS MODE
	const damping = 0.88;
	const repulsion = 1800;
	const springLen = 70;
	const springK = 0.04;

	// 1. Repulsion between all pairs
	for (let i = 0; i < simNodes.length; i++) {
		const n1 = simNodes[i];
		for (let j = i + 1; j < simNodes.length; j++) {
			const n2 = simNodes[j];
			const dx = (n2.x || 0) - (n1.x || 0);
			const dy = (n2.y || 0) - (n1.y || 0);
			const distSq = dx * dx + dy * dy + 100;
			const dist = Math.sqrt(distSq);
			const force = repulsion / distSq;
			const fx = (dx / dist) * force;
			const fy = (dy / dist) * force;

			n1.vx = (n1.vx || 0) - fx;
			n1.vy = (n1.vy || 0) - fy;
			n2.vx = (n2.vx || 0) + fx;
			n2.vy = (n2.vy || 0) + fy;
		}

		// Centering gravity
		const cDist = Math.hypot(n1.x || 0, n1.y || 0);
		n1.vx = (n1.vx || 0) - (n1.x || 0) * 0.0008;
		n1.vy = (n1.vy || 0) - (n1.y || 0) * 0.0008;
	}

	// 2. Spring attraction along edges
	const nodeMap = new Map(simNodes.map((n) => [n.id, n]));
	for (const edge of simEdges) {
		const s = nodeMap.get(edge.source);
		const t = nodeMap.get(edge.target);
		if (!s || !t) continue;
		const dx = (t.x || 0) - (s.x || 0);
		const dy = (t.y || 0) - (s.y || 0);
		const dist = Math.hypot(dx, dy) || 1;
		const delta = dist - springLen;
		const force = delta * springK * edge.weight;
		const fx = (dx / dist) * force;
		const fy = (dy / dist) * force;

		s.vx = (s.vx || 0) + fx;
		s.vy = (s.vy || 0) + fy;
		t.vx = (t.vx || 0) - fx;
		t.vy = (t.vy || 0) - fy;
	}

	// 3. Update positions
	for (const n of simNodes) {
		if (n === draggedNode) continue;
		n.vx = (n.vx || 0) * damping;
		n.vy = (n.vy || 0) * damping;
		n.x = (n.x || 0) + n.vx;
		n.y = (n.y || 0) + n.vy;
	}

	// 4. Render Canvas
	ctx.clearRect(0, 0, width, height);
	ctx.save();
	ctx.translate(transform.x, transform.y);
	ctx.scale(transform.k, transform.k);

	// Draw edges
	const qLower = searchQuery.trim().toLowerCase();
	for (const e of simEdges) {
		const s = nodeMap.get(e.source);
		const t = nodeMap.get(e.target);
		if (!s || !t) continue;

		const isConnectedToHover =
			hoveredNode && (s.id === hoveredNode.id || t.id === hoveredNode.id);
		const isConnectedToSelected =
			selectedNode && (s.id === selectedNode.id || t.id === selectedNode.id);

		ctx.beginPath();
		ctx.moveTo(s.x || 0, s.y || 0);
		ctx.lineTo(t.x || 0, t.y || 0);

		if (isConnectedToHover || isConnectedToSelected) {
			ctx.strokeStyle = "rgba(92, 122, 107, 0.8)";
			ctx.lineWidth = 2 / transform.k;
		} else {
			ctx.strokeStyle = "rgba(160, 160, 160, 0.2)";
			ctx.lineWidth = 1 / transform.k;
		}
		ctx.stroke();
	}

	// Draw nodes
	for (const n of simNodes) {
		const isHovered = hoveredNode?.id === n.id;
		const isSelected = selectedNode?.id === n.id;
		const matchesSearch =
			qLower.length === 0 ||
			n.label.toLowerCase().includes(qLower) ||
			Boolean(n.tags?.some((t) => t.toLowerCase().includes(qLower)));
		const matchesFilter = activeFilter === "all" || n.maturity === activeFilter;

		const alpha = matchesFilter && matchesSearch ? 1 : 0.18;
		const meta =
			MATURITY_DEFINITIONS[n.maturity] || MATURITY_DEFINITIONS.budding;
		const radius =
			(n.kind === "hub" ? 14 : n.kind === "lab" ? 11 : 8) *
			(isHovered ? 1.3 : 1);

		ctx.save();
		ctx.globalAlpha = alpha;

		// Glow or ring
		if (isHovered || isSelected) {
			ctx.beginPath();
			ctx.arc(n.x || 0, n.y || 0, radius + 5, 0, Math.PI * 2);
			ctx.fillStyle = "rgba(92, 122, 107, 0.25)";
			ctx.fill();
		}

		// Node circle
		ctx.beginPath();
		ctx.arc(n.x || 0, n.y || 0, radius, 0, Math.PI * 2);
		ctx.fillStyle = meta.color;
		ctx.fill();
		ctx.strokeStyle = "#ffffff";
		ctx.lineWidth = 1.5 / transform.k;
		ctx.stroke();

		// Label text
		if (transform.k > 0.6 || isHovered || isSelected || n.kind === "hub") {
			ctx.font = `${n.kind === "hub" ? "bold 13px" : "11px"} -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`;
			ctx.fillStyle =
				isHovered || isSelected ? "#1b2a26" : "rgba(40, 50, 45, 0.85)";
			ctx.textAlign = "center";
			const textY = (n.y || 0) + radius + 14;
			const displayLabel = english && n.labelEn ? n.labelEn : n.label;
			ctx.fillText(displayLabel, n.x || 0, textY);
		}

		ctx.restore();
	}

	ctx.restore();
	scheduleNextFrame();
}

function getNodeAt(x: number, y: number): SimNodeExt | null {
	if (viewMode === "3d") {
		// Screen space coords relative to transform center and scale
		const sx = (x - transform.x) / transform.k;
		const sy = (y - transform.y) / transform.k;

		// Check closest from front to back (descending projZ)
		const sorted = [...simNodes].sort(
			(a, b) => (b.projZ || 0) - (a.projZ || 0),
		);
		for (const n of sorted) {
			const r = (n.kind === "hub" ? 18 : 13) * (n.projScale || 1);
			const dist = Math.hypot((n.projX || 0) - sx, (n.projY || 0) - sy);
			if (dist <= r) return n;
		}
		return null;
	}

	// 2D mode
	const wx = (x - transform.x) / transform.k;
	const wy = (y - transform.y) / transform.k;

	for (let i = simNodes.length - 1; i >= 0; i--) {
		const n = simNodes[i];
		const r = n.kind === "hub" ? 16 : 12;
		const dist = Math.hypot((n.x || 0) - wx, (n.y || 0) - wy);
		if (dist <= r) return n;
	}
	return null;
}

function handleMouseDown(e: MouseEvent) {
	if (!canvasEl) return;
	const rect = canvasEl.getBoundingClientRect();
	const mx = e.clientX - rect.left;
	const my = e.clientY - rect.top;

	const hit = getNodeAt(mx, my);
	if (hit) {
		draggedNode = hit;
		selectedNode = hit;
		playLatchClick();
	} else if (viewMode === "3d") {
		isDragging3D = true;
		drag3DStart = { x: e.clientX, y: e.clientY };
	} else {
		isDraggingCanvas = true;
		dragStart = { x: e.clientX - transform.x, y: e.clientY - transform.y };
	}
}

function handleMouseMove(e: MouseEvent) {
	if (!canvasEl) return;
	const rect = canvasEl.getBoundingClientRect();
	const mx = e.clientX - rect.left;
	const my = e.clientY - rect.top;

	if (viewMode === "3d" && isDragging3D) {
		rotY += (e.clientX - drag3DStart.x) * 0.006;
		rotX = Math.max(
			-1.1,
			Math.min(1.1, rotX + (e.clientY - drag3DStart.y) * 0.006),
		);
		drag3DStart = { x: e.clientX, y: e.clientY };
		ensureAnimating(120);
		return;
	}

	if (draggedNode && viewMode === "2d") {
		draggedNode.x = (mx - transform.x) / transform.k;
		draggedNode.y = (my - transform.y) / transform.k;
		draggedNode.vx = 0;
		draggedNode.vy = 0;
		return;
	}

	if (isDraggingCanvas) {
		transform.x = e.clientX - dragStart.x;
		transform.y = e.clientY - dragStart.y;
		return;
	}

	// Check hover
	const hit = getNodeAt(mx, my);
	if (hit !== hoveredNode) {
		hoveredNode = hit;
		if (hit) {
			playMechanicalTick(0.85);
		}
	}
}

function handleMouseUp() {
	draggedNode = null;
	isDraggingCanvas = false;
	isDragging3D = false;
}

function handleWheel(e: WheelEvent) {
	e.preventDefault();
	if (!canvasEl) return;
	const rect = canvasEl.getBoundingClientRect();
	const mx = e.clientX - rect.left;
	const my = e.clientY - rect.top;

	const zoomFactor = e.deltaY < 0 ? 1.08 : 0.92;
	const newK = Math.max(0.4, Math.min(3.5, transform.k * zoomFactor));

	transform.x = mx - ((mx - transform.x) / transform.k) * newK;
	transform.y = my - ((my - transform.y) / transform.k) * newK;
	transform.k = newK;
	ensureAnimating(120);
}

function resetView() {
	if (!canvasEl) return;
	const rect = canvasEl.getBoundingClientRect();
	transform.x = rect.width / 2;
	transform.y = rect.height / 2;
	transform.k = 1.0;
	playShutterTick();
	ensureAnimating(120);
}

function setFilter(filter: MaturityLevel | "all") {
	activeFilter = filter;
	playLatchClick();
	ensureAnimating(180);
}

function focusNode(node: GraphNode) {
	selectedNode = node;
	transform.x = (canvasEl?.width || 800) / 2 - (node.x || 0) * transform.k;
	transform.y = (canvasEl?.height || 600) / 2 - (node.y || 0) * transform.k;
	playLatchClick();
	ensureAnimating(120);
}

function closeDrawer() {
	selectedNode = null;
	playLatchClick();
}

onMount(() => {
	const sync = () => {
		english = document.documentElement.lang.startsWith("en");
	};
	sync();
	document.addEventListener("site-locale-change", sync);

	// Reduced motion: the reader asked for a still cosmos. Start on the 2D
	// atlas, stop the automatic rotation and orbit drift, and only re-render
	// for a short settle window after each interaction.
	const motionQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
	const applyMotionPreference = () => {
		reduceMotion = motionQuery.matches;
		if (reduceMotion) {
			viewMode = "2d";
			ensureAnimating(180);
		} else {
			ensureAnimating();
		}
	};
	applyMotionPreference();
	motionQuery.addEventListener("change", applyMotionPreference);

	const handleResize = () => {
		if (canvasEl) {
			const rect = canvasEl.getBoundingClientRect();
			canvasEl.width = rect.width * window.devicePixelRatio;
			canvasEl.height = rect.height * window.devicePixelRatio;
			const ctx = canvasEl.getContext("2d");
			if (ctx) ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
		}
		ensureAnimating(120);
	};

	handleResize();
	initSimulation();
	ensureAnimating();
	window.addEventListener("resize", handleResize);

	// Off-screen and background tabs must not keep the loop running.
	const container = canvasEl?.closest(".canvas-container") ?? null;
	let visibilityObserver: IntersectionObserver | undefined;
	if (container && "IntersectionObserver" in window) {
		visibilityObserver = new IntersectionObserver(
			(entries) => {
				const visible = entries.some((entry) => entry.isIntersecting);
				isVisible = visible;
				if (visible) ensureAnimating(120);
				else parkAnimation();
			},
			{ threshold: 0 },
		);
		visibilityObserver.observe(container);
	}
	const handleVisibility = () => {
		if (document.hidden) {
			parkAnimation();
		} else {
			isVisible = true;
			ensureAnimating(120);
		}
	};
	document.addEventListener("visibilitychange", handleVisibility);

	return () => {
		parkAnimation();
		cancelAnimationFrame(animFrameId);
		visibilityObserver?.disconnect();
		motionQuery.removeEventListener("change", applyMotionPreference);
		window.removeEventListener("resize", handleResize);
		document.removeEventListener("visibilitychange", handleVisibility);
		document.removeEventListener("site-locale-change", sync);
	};
});
</script>

<div class="knowledge-cosmos-wrapper">
	<!-- Top Controls Toolbar -->
	<div class="cosmos-toolbar">
		<div class="filter-strip">
			<button
				type="button"
				class="filter-pill"
				class:active={activeFilter === "all"}
				onclick={() => setFilter("all")}
			>
				{english ? "All Nodes" : "全部节点"}
				<span class="count">{simNodes.length}</span>
			</button>
			{#each Object.entries(MATURITY_DEFINITIONS) as [lvl, meta]}
				<button
					type="button"
					class="filter-pill"
					class:active={activeFilter === lvl}
					style="--pill-color: {meta.color}"
					onclick={() => setFilter(lvl as MaturityLevel)}
				>
					<span class="icon">{meta.icon}</span>
					{english ? meta.labelEn : meta.labelZh}
				</button>
			{/each}
		</div>

		<div class="search-and-reset">
			<!-- 2D / 3D Mode Switcher -->
			<div class="view-mode-toggle">
				<button
					type="button"
					class="view-pill"
					class:active={viewMode === "3d"}
					onclick={() => {
						viewMode = "3d";
						playShutterTick();
						ensureAnimating(180);
					}}
				>
					🪐 {english ? "3D Cosmos" : "3D 天球"}
				</button>
				<button
					type="button"
					class="view-pill"
					class:active={viewMode === "2d"}
					onclick={() => {
						viewMode = "2d";
						playShutterTick();
						ensureAnimating(180);
					}}
				>
					🗺️ {english ? "2D Atlas" : "2D 拓扑"}
				</button>
			</div>

			<input
				type="search"
				class="cosmos-search-input"
				placeholder={english ? "Search concept or note…" : "搜索知识概念或笔记…"}
				bind:value={searchQuery}
				oninput={() => ensureAnimating(120)}
			/>
			<button type="button" class="btn-icon" onclick={resetView} title={english ? "Reset View" : "重置视图"}>
				⟲
			</button>
		</div>
	</div>

	<!-- Canvas Graph Arena -->
	<div class="canvas-container">
		<canvas
			bind:this={canvasEl}
			aria-hidden="true"
			onmousedown={handleMouseDown}
			onmousemove={handleMouseMove}
			onmouseup={handleMouseUp}
			onwheel={handleWheel}
		></canvas>

		<!-- Watermark / Hint -->
		<div class="canvas-hint">
			<span>
				{#if viewMode === "3d"}
					{english
						? "🪐 Drag to orbit in 3D · Scroll to zoom · Click celestial note to open card"
						: "🪐 拖拽可 360° 旋转宇宙星系 · 滚轮缩放天球 · 点击星体展开卡片"}
				{:else}
					{english
						? "🗺️ Drag to pan · Scroll to zoom · Click node to inspect"
						: "🗺️ 按住拖拽平移 · 滚轮缩放 · 点击节点展开卡片"}
				{/if}
			</span>
		</div>

		<!-- Andy Matuschak Sliding Drawer -->
		{#if selectedNode}
			<aside class="sliding-pane-drawer">
				<div class="drawer-header">
					<div class="drawer-badge" style="background: {MATURITY_DEFINITIONS[selectedNode.maturity]?.badgeBg || 'transparent'}; color: {MATURITY_DEFINITIONS[selectedNode.maturity]?.color || 'inherit'}">
						<span>{MATURITY_DEFINITIONS[selectedNode.maturity]?.icon}</span>
						<strong>{english ? MATURITY_DEFINITIONS[selectedNode.maturity]?.labelEn : MATURITY_DEFINITIONS[selectedNode.maturity]?.labelZh}</strong>
					</div>
					<button type="button" class="btn-close" onclick={closeDrawer} aria-label="Close">✕</button>
				</div>

				<h3 class="drawer-title">{english && selectedNode.labelEn ? selectedNode.labelEn : selectedNode.label}</h3>

				<div class="drawer-meta">
					<span class="meta-tag">{selectedNode.category}</span>
					<span class="meta-tag">{selectedNode.kind.toUpperCase()}</span>
				</div>

				{#if selectedNode.excerpt}
					<p class="drawer-excerpt">{selectedNode.excerpt}</p>
				{/if}

				{#if selectedNode.tags && selectedNode.tags.length > 0}
					<div class="drawer-tags">
						{#each selectedNode.tags as t}
							<span class="tag-pill">{t}</span>
						{/each}
					</div>
				{/if}

				<!-- Connected Nodes -->
				<div class="neighbors-section">
					<h4>{english ? "Connected Thoughts & Experiments:" : "关联思想与实验:"}</h4>
					<div class="neighbor-list">
						{#each connectedNeighbors as nbr}
							<button type="button" class="neighbor-pill" onclick={() => focusNode(nbr)}>
								<span>{MATURITY_DEFINITIONS[nbr.maturity]?.icon}</span>
								<span>{english && nbr.labelEn ? nbr.labelEn : nbr.label}</span>
							</button>
						{/each}
					</div>
				</div>

				<div class="drawer-actions">
					<a href={selectedNode.href} class="btn-action-primary">
						{english ? "Open Note / Experiment ↗" : "打开完整笔记 / 实验 ↗"}
					</a>
				</div>
			</aside>
		{/if}
	</div>

	<!-- Plain-text node index: the keyboard and no-JS path into the same graph.
		It lives outside the canvas container on purpose: the container keeps a
		fixed height, so anything inside it would be clipped away. -->
	<details class="node-index">
		<summary>
			<span
				class="node-index-title"
				data-i18n="graph.node.index"
				data-i18n-zh="节点目录"
				data-i18n-en="Node index"
			>
				节点目录
			</span>
			<span class="node-index-count">{listedNodes.length}</span>
		</summary>
		<p
			class="node-index-note"
			data-i18n="graph.node.note"
			data-i18n-zh="把图谱里的每个节点以纯文本链接列出，顺序与拓扑图一致。画布只是同一份列表的空间化视图。"
			data-i18n-en="Every node as a plain link, in the order the atlas draws them. The canvas is a spatial view of the same list."
		>
			把图谱里的每个节点以纯文本链接列出，顺序与拓扑图一致。画布只是同一份列表的空间化视图。
		</p>
		<ul>
			{#each listedNodes as node (node.id)}
				<li>
					<a href={node.href}>
						<span
							class="node-index-badge"
							style="background: {MATURITY_DEFINITIONS[node.maturity]?.badgeBg || 'transparent'}; color: {MATURITY_DEFINITIONS[node.maturity]?.color || 'inherit'}"
						>
							{MATURITY_DEFINITIONS[node.maturity]?.icon}
						</span>
						<span
							class="node-index-label"
							data-i18n="graph.node.title"
							data-i18n-zh={node.label}
							data-i18n-en={node.labelEn ?? node.label}
						>
							{english && node.labelEn ? node.labelEn : node.label}
						</span>
						<span
							class="node-index-kind"
							data-i18n="graph.node.kind"
							data-i18n-zh={kindLabels[node.kind].zh}
							data-i18n-en={kindLabels[node.kind].en}
						>
							{english ? kindLabels[node.kind].en : kindLabels[node.kind].zh}
						</span>
					</a>
				</li>
			{/each}
		</ul>
		{#if listedNodes.length === 0}
			<p
				class="node-index-empty"
				data-i18n="graph.node.empty"
				data-i18n-zh="没有节点符合当前的筛选与搜索条件。"
				data-i18n-en="No node matches the current filter and search."
			>
				没有节点符合当前的筛选与搜索条件。
			</p>
		{/if}
	</details>
</div>

<style>
	.knowledge-cosmos-wrapper {
		display: flex;
		flex-direction: column;
		gap: 14px;
		background: var(--brand-paper);
		border: 1px solid var(--brand-line);
		border-radius: 14px;
		overflow: hidden;
		box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
	}

	.cosmos-toolbar {
		display: flex;
		justify-content: space-between;
		align-items: center;
		flex-wrap: wrap;
		gap: 12px;
		padding: 16px 20px;
		border-bottom: 1px solid var(--brand-line);
		background: var(--brand-surface);
	}

	.filter-strip {
		display: flex;
		align-items: center;
		flex-wrap: wrap;
		gap: 6px;
	}

	.filter-pill {
		display: inline-flex;
		align-items: center;
		gap: 5px;
		padding: 6px 12px;
		border-radius: 999px;
		border: 1px solid var(--brand-line);
		background: transparent;
		color: var(--brand-body);
		font-size: 12px;
		cursor: pointer;
		transition: all 0.15s ease;
	}

	.filter-pill:hover,
	.filter-pill.active {
		border-color: var(--pill-color, var(--brand-accent));
		color: var(--pill-color, var(--brand-accent));
		background: rgba(92, 122, 107, 0.08);
	}

	.filter-pill.active {
		font-weight: 600;
	}

	.count {
		font-size: 10px;
		opacity: 0.7;
	}

	.search-and-reset {
		display: flex;
		align-items: center;
		flex-wrap: wrap;
		gap: 8px;
	}

	.view-mode-toggle {
		display: inline-flex;
		align-items: center;
		padding: 2px;
		background: var(--brand-paper);
		border: 1px solid var(--brand-line);
		border-radius: 999px;
	}

	.view-pill {
		font-size: 11.5px;
		padding: 4px 10px;
		border-radius: 999px;
		border: none;
		background: transparent;
		color: var(--brand-muted);
		cursor: pointer;
		transition: all 0.15s ease;
		white-space: nowrap;
	}

	.view-pill.active {
		background: var(--brand-surface);
		color: var(--brand-accent);
		font-weight: 600;
		box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08);
	}

	.cosmos-search-input {
		padding: 6px 12px;
		border-radius: 8px;
		border: 1px solid var(--brand-line);
		background: var(--brand-paper);
		color: var(--brand-ink);
		font-size: 12px;
		outline: none;
		min-width: 180px;
	}

	.cosmos-search-input:focus {
		border-color: var(--brand-accent);
	}

	.btn-icon {
		padding: 6px 10px;
		border-radius: 8px;
		border: 1px solid var(--brand-line);
		background: transparent;
		color: var(--brand-body);
		cursor: pointer;
		font-size: 13px;
	}

	.btn-icon:hover {
		border-color: var(--brand-accent);
		color: var(--brand-accent);
	}

	.canvas-container {
		position: relative;
		width: 100%;
		height: 640px;
		background: radial-gradient(circle at center, rgba(92, 122, 107, 0.03) 0%, transparent 70%);
	}

	canvas {
		width: 100%;
		height: 100%;
		display: block;
		cursor: grab;
	}

	canvas:active {
		cursor: grabbing;
	}

	.canvas-hint {
		position: absolute;
		bottom: 12px;
		left: 16px;
		font-size: 11px;
		color: var(--brand-muted);
		pointer-events: none;
		background: rgba(255, 255, 255, 0.7);
		backdrop-filter: blur(4px);
		padding: 4px 8px;
		border-radius: 6px;
	}

	/* Plain-text node index: the accessible, no-JS path into the same graph */
	.node-index {
		margin: 0;
		border: 1px solid var(--brand-line);
		border-radius: 12px;
		background: var(--brand-surface);
	}

	.node-index > summary {
		display: flex;
		align-items: center;
		gap: 10px;
		padding: 12px 16px;
		cursor: pointer;
		list-style: none;
	}

	.node-index > summary::-webkit-details-marker {
		display: none;
	}

	.node-index-title {
		font: 500 13px/1.5 var(--font-sans, Inter, sans-serif);
		color: var(--brand-ink);
	}

	.node-index-count {
		margin-left: auto;
		font: 10px/1.8 "JetBrains Mono", monospace;
		color: var(--brand-muted);
	}

	.node-index-note {
		margin: 0 16px 12px;
		font-size: 12px;
		line-height: 1.8;
		color: var(--brand-muted);
	}

	.node-index ul {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 2px 18px;
		margin: 0;
		padding: 0 16px 16px;
		list-style: none;
	}

	.node-index li a {
		display: flex;
		align-items: baseline;
		gap: 8px;
		padding: 7px 0;
		border-bottom: 1px solid var(--brand-line);
		color: var(--brand-ink);
		text-decoration: none;
	}

	.node-index li a:hover .node-index-label {
		color: var(--brand-accent);
	}

	.node-index-badge {
		flex: 0 0 auto;
		font-size: 12px;
		line-height: 1.6;
	}

	.node-index-label {
		min-width: 0;
		font-size: 13px;
		line-height: 1.6;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.node-index-kind {
		margin-left: auto;
		flex: 0 0 auto;
		font: 9px/1.8 "JetBrains Mono", monospace;
		color: var(--brand-muted);
	}

	.node-index-empty {
		margin: 0;
		padding: 0 16px 18px;
		font-size: 12px;
		color: var(--brand-muted);
	}

	.node-index a:focus-visible,
	.node-index summary:focus-visible {
		outline: 2px solid var(--brand-accent);
		outline-offset: 3px;
	}

	@media (max-width: 700px) {
		.node-index ul {
			grid-template-columns: minmax(0, 1fr);
		}
	}

	/* Andy Matuschak Sliding Drawer */
	.sliding-pane-drawer {
		position: absolute;
		top: 14px;
		right: 14px;
		bottom: 14px;
		width: 340px;
		max-width: calc(100% - 28px);
		background: var(--brand-paper);
		border: 1px solid var(--brand-line);
		border-radius: 12px;
		box-shadow: -4px 0 24px rgba(0, 0, 0, 0.08);
		padding: 20px;
		display: flex;
		flex-direction: column;
		gap: 14px;
		overflow-y: auto;
		z-index: 10;
		animation: slideIn 0.2s cubic-bezier(0.16, 1, 0.3, 1);
	}

	@keyframes slideIn {
		from {
			transform: translateX(30px);
			opacity: 0;
		}
		to {
			transform: translateX(0);
			opacity: 1;
		}
	}

	.drawer-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
	}

	.drawer-badge {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		padding: 4px 10px;
		border-radius: 6px;
		font-size: 12px;
	}

	.btn-close {
		border: none;
		background: transparent;
		color: var(--brand-muted);
		cursor: pointer;
		font-size: 14px;
		padding: 4px;
	}

	.btn-close:hover {
		color: var(--brand-ink);
	}

	.drawer-title {
		margin: 0;
		font-size: 18px;
		font-family: var(--font-serif, serif);
		line-height: 1.4;
	}

	.drawer-meta {
		display: flex;
		gap: 6px;
	}

	.meta-tag {
		font-size: 11px;
		color: var(--brand-muted);
		background: var(--brand-surface);
		border: 1px solid var(--brand-line);
		padding: 2px 7px;
		border-radius: 4px;
	}

	.drawer-excerpt {
		font-size: 13px;
		line-height: 1.7;
		color: var(--brand-body);
		margin: 0;
	}

	.drawer-tags {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}

	.tag-pill {
		font-size: 11px;
		color: var(--brand-muted);
		border: 1px solid var(--brand-line);
		padding: 2px 8px;
		border-radius: 999px;
	}

	.neighbors-section h4 {
		margin: 0 0 8px;
		font-size: 12px;
		color: var(--brand-muted);
	}

	.neighbor-list {
		display: flex;
		flex-direction: column;
		gap: 6px;
	}

	.neighbor-pill {
		display: flex;
		align-items: center;
		gap: 8px;
		padding: 6px 10px;
		border: 1px solid var(--brand-line);
		border-radius: 6px;
		background: var(--brand-surface);
		color: var(--brand-body);
		font-size: 12px;
		cursor: pointer;
		text-align: left;
		transition: all 0.15s ease;
	}

	.neighbor-pill:hover {
		border-color: var(--brand-accent);
		color: var(--brand-accent);
	}

	.drawer-actions {
		margin-top: auto;
		padding-top: 14px;
		border-top: 1px solid var(--brand-line);
	}

	.btn-action-primary {
		display: block;
		text-align: center;
		padding: 10px 16px;
		background: var(--brand-accent);
		color: #fff;
		border-radius: 8px;
		font-size: 13px;
		font-weight: 600;
		text-decoration: none;
		transition: opacity 0.15s ease;
	}

	.btn-action-primary:hover {
		opacity: 0.9;
	}
</style>
