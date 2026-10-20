import * as THREE from "three";
import { planWorldRoute, WORLD_ROVER_CLEARANCE } from "./navigation";
import { type CelestialState, computeCelestialState } from "@/utils/celestial";

export type WorldQuality = "light" | "full";
export type WorldLocale = "zh-CN" | "en";
export interface WorldStation {
	id: string;
	x: number;
	z: number;
	color: number;
	sceneLabel: string;
	sceneLabelEn: string;
}
export interface WorldController {
	selectStation: (id: string) => void;
	resetView: () => void;
	resetRover: () => void;
	setQuality: (quality: WorldQuality) => void;
	setAnimationPaused: (paused: boolean) => void;
	setFollowRover: (follow: boolean) => void;
	setLocale: (locale: WorldLocale) => void;
	setSolarHour: (hour: number) => CelestialState;
	dispose: () => void;
}
interface WorldOptions {
	canvas: HTMLCanvasElement;
	stage: HTMLDivElement;
	stations: readonly WorldStation[];
	immersive: boolean;
	locale: WorldLocale;
	onReady: () => void;
	onSelect: (id: string) => void;
	onOverview: () => void;
	onNotice: (zh: string, en: string) => void;
	onMotionPreference: (reduced: boolean) => void;
	onFollowChange: (follow: boolean) => void;
	onError: () => void;
}

/** The scene stays independent of the accessible station links and content panel. */
export function createWorld(options: WorldOptions): WorldController {
	const { canvas, stage, stations, immersive } = options;
	let locale = options.locale;
	let quality: WorldQuality = "light";
	let usable = true;
	let active = true;
	let disposed = false;
	let ready = false;
	let frame = 0;
	let previousTime = 0;
	let elapsed = 0;
	let dragging = false;
	let activePointer: number | null = null;
	let dragMoved = false;
	let pointerStartX = 0;
	let pointerStartY = 0;
	let lastX = 0;
	let lastY = 0;
	let hovered: string | null = null;
	let selected: string | null = null;
	let nearby: string | null = null;
	let arrival: string | null = null;
	let animationPaused = false;
	let followRover = false;
	let yaw = -0.06;
	let pitch = 0.3;
	let radius = immersive ? 19 : 20;
	let desiredYaw = yaw;
	let desiredPitch = pitch;
	let desiredRadius = radius;
	let autoDestination: THREE.Vector3 | null = null;
	let autoRoute: { x: number; z: number }[] = [];
	const keys = new Set<string>();
	const motionPreference = window.matchMedia("(prefers-reduced-motion: reduce)");
	let reducedMotion = motionPreference.matches;
	options.onMotionPreference(reducedMotion);
	const renderer = new THREE.WebGLRenderer({
		canvas,
		alpha: true,
		antialias: true,
		powerPreference: "low-power",
	});
	renderer.setClearColor(0xf3f6f1, 0);
	renderer.outputColorSpace = THREE.SRGBColorSpace;
	renderer.toneMapping = THREE.ACESFilmicToneMapping;
	renderer.toneMappingExposure = 1.05;
	renderer.shadowMap.type = THREE.PCFShadowMap;
	const scene = new THREE.Scene();
	const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 90);
	const target = new THREE.Vector3(0, -0.55, 0);
	const desiredTarget = target.clone();
	const world = new THREE.Group();
	world.rotation.y = -0.12;
	scene.add(world);
	const hemi = new THREE.HemisphereLight(0xfff5e1, 0x849181, 1.4);
	scene.add(hemi);
	const sun = new THREE.DirectionalLight(0xffe4ba, 2.15);
	sun.position.set(-5, 12, 7);
	sun.castShadow = true;
	sun.shadow.mapSize.set(1024, 1024);
	sun.shadow.camera.left = -9;
	sun.shadow.camera.right = 9;
	sun.shadow.camera.top = 9;
	sun.shadow.camera.bottom = -9;
	sun.shadow.camera.near = 0.5;
	sun.shadow.camera.far = 35;
	sun.shadow.normalBias = 0.035;
	scene.add(sun);
	const fill = new THREE.DirectionalLight(0xc5dfd3, 1.0);
	fill.position.set(6, 4, -6);
	scene.add(fill);

	const materials = {
		forest: new THREE.MeshStandardMaterial({ color: 0x315844, roughness: 0.72 }),
		sage: new THREE.MeshStandardMaterial({ color: 0x9dbda6, roughness: 0.9 }),
		paper: new THREE.MeshStandardMaterial({ color: 0xf3ead6, roughness: 0.95 }),
		peach: new THREE.MeshStandardMaterial({ color: 0xeeb28c, roughness: 0.75 }),
		blue: new THREE.MeshStandardMaterial({ color: 0x587989, roughness: 0.7 }),
		ink: new THREE.MeshStandardMaterial({ color: 0x263c35, roughness: 0.88 }),
		wood: new THREE.MeshStandardMaterial({ color: 0xb58c61, roughness: 0.88 }),
		woodEdge: new THREE.MeshStandardMaterial({ color: 0x8c684c, roughness: 0.86 }),
		brass: new THREE.MeshStandardMaterial({ color: 0x9e8058, roughness: 0.42, metalness: 0.48 }),
		lamplight: new THREE.MeshBasicMaterial({ color: 0xffe6ad }),
	};
	function mesh(
		geometry: THREE.BufferGeometry,
		material: THREE.Material,
		parent: THREE.Object3D,
		x = 0,
		y = 0,
		z = 0,
	): THREE.Mesh {
		const object = new THREE.Mesh(geometry, material);
		object.position.set(x, y, z);
		object.castShadow = true;
		object.receiveShadow = true;
		parent.add(object);
		return object;
	}
	function box(parent: THREE.Object3D, size: [number, number, number], position: [number, number, number], material: THREE.Material): THREE.Mesh {
		return mesh(new THREE.BoxGeometry(...size), material, parent, ...position);
	}
	function connection(parent: THREE.Object3D, from: THREE.Vector3, to: THREE.Vector3, material: THREE.Material, thickness = 0.024): THREE.Mesh {
		const direction = to.clone().sub(from);
		const object = mesh(new THREE.CylinderGeometry(thickness, thickness, direction.length(), 6), material, parent);
		object.position.copy(from).add(to).multiplyScalar(0.5);
		object.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.normalize());
		return object;
	}
	function notebook(parent: THREE.Object3D, x: number, y: number, z: number, angle = 0): void {
		const book = new THREE.Group();
		book.position.set(x, y, z);
		book.rotation.y = angle;
		parent.add(book);
		box(book, [0.38, 0.035, 0.29], [0, 0, 0], materials.forest);
		box(book, [0.34, 0.035, 0.27], [0.005, 0.029, 0], materials.paper);
		for (let line = 0; line < 3; line++) box(book, [0.19 - line * 0.025, 0.005, 0.008], [0.01, 0.05, -0.065 + line * 0.045], materials.woodEdge);
		connection(book, new THREE.Vector3(-0.22, 0.05, -0.13), new THREE.Vector3(-0.18, 0.05, 0.15), materials.peach, 0.015);
	}
	function deskLamp(parent: THREE.Object3D, x: number, y: number, z: number): void {
		const lamp = new THREE.Group();
		lamp.position.set(x, y, z);
		parent.add(lamp);
		mesh(new THREE.CylinderGeometry(0.115, 0.13, 0.035, 12), materials.brass, lamp);
		connection(lamp, new THREE.Vector3(0, 0.02, 0), new THREE.Vector3(0, 0.53, 0), materials.brass, 0.018);
		connection(lamp, new THREE.Vector3(0, 0.53, 0), new THREE.Vector3(0.22, 0.62, 0.04), materials.brass, 0.021);
		mesh(new THREE.ConeGeometry(0.14, 0.15, 16, 1, true), materials.brass, lamp, 0.22, 0.58, 0.04);
		mesh(new THREE.SphereGeometry(0.048, 8, 6), materials.lamplight, lamp, 0.22, 0.505, 0.04).castShadow = false;
		// A soft tabletop pool suggests lamplight without an extra dynamic light.
		const glow = mesh(new THREE.CircleGeometry(0.26, 20), new THREE.MeshBasicMaterial({ color: 0xffd694, transparent: true, opacity: 0.14, depthWrite: false }), lamp, 0.22, 0.021, 0.04);
		glow.rotation.x = -Math.PI / 2;
		glow.castShadow = false;
	}

	const rock = mesh(new THREE.CylinderGeometry(6.4, 4.6, 1.7, 10), materials.forest, world, 0, -0.9, 0);
	rock.scale.z = 0.79;
	rock.rotation.y = 0.07;
	const shelf = mesh(new THREE.CylinderGeometry(6.65, 6.35, 0.28, 10), materials.sage, world, 0, 0.01, 0);
	shelf.scale.z = 0.79;
	shelf.rotation.y = 0.07;
	const ground = mesh(new THREE.CylinderGeometry(6.44, 6.44, 0.06, 64), new THREE.MeshStandardMaterial({ color: 0xc7d5b7, roughness: 1 }), world, 0, 0.18, 0);
	ground.scale.z = 0.775;
	ground.castShadow = false;
	const rim = mesh(new THREE.TorusGeometry(6.33, 0.026, 5, 80), materials.paper, world, 0, 0.235, 0);
	rim.rotation.x = -Math.PI / 2;
	rim.scale.y = 0.775;
	rim.castShadow = false;

	// A small radial texture supplies contact shadows even in the light quality mode.
	const shadowCanvas = document.createElement("canvas");
	shadowCanvas.width = 64;
	shadowCanvas.height = 64;
	const shadowContext = shadowCanvas.getContext("2d");
	if (shadowContext) {
		const gradient = shadowContext.createRadialGradient(32, 32, 2, 32, 32, 32);
		gradient.addColorStop(0, "rgba(28, 52, 39, .42)");
		gradient.addColorStop(0.45, "rgba(28, 52, 39, .20)");
		gradient.addColorStop(1, "rgba(28, 52, 39, 0)");
		shadowContext.fillStyle = gradient;
		shadowContext.fillRect(0, 0, 64, 64);
	}
	const shadowTexture = new THREE.CanvasTexture(shadowCanvas);
	const shadowMaterial = new THREE.MeshBasicMaterial({ map: shadowTexture, transparent: true, depthWrite: false });
	function contactShadow(parent: THREE.Object3D, width: number, depth: number, y: number): void {
		const shadow = mesh(new THREE.PlaneGeometry(width, depth), shadowMaterial, parent, 0, y, 0);
		shadow.rotation.x = -Math.PI / 2;
		shadow.castShadow = false;
		shadow.receiveShadow = false;
	}
	contactShadow(world, 17, 12, -1.82);
	const labelUpdates: (() => void)[] = [];
	function label(station: WorldStation, parent: THREE.Object3D): void {
		const surface = document.createElement("canvas");
		surface.width = 768;
		surface.height = 140;
		const context = surface.getContext("2d");
		if (!context) return;
		const texture = new THREE.CanvasTexture(surface);
		texture.colorSpace = THREE.SRGBColorSpace;
		const update = () => {
			context.clearRect(0, 0, 768, 140);
			context.fillStyle = "rgba(246, 239, 223, .97)";
			context.beginPath();
			context.roundRect(2, 2, 764, 136, 10);
			context.fill();
			context.strokeStyle = "#b8aa8f";
			context.lineWidth = 3;
			context.stroke();
			context.fillStyle = "#315844";
			context.textAlign = "center";
			context.textBaseline = "middle";
			const text = locale === "en" ? station.sceneLabelEn : station.sceneLabel;
			let fontSize = 60;
			context.font = `600 ${fontSize}px system-ui, sans-serif`;
			while (context.measureText(text).width > 708 && fontSize > 24) {
				fontSize--;
				context.font = `600 ${fontSize}px system-ui, sans-serif`;
			}
			context.fillText(text, 384, 72);
			texture.needsUpdate = true;
		};
		update();
		labelUpdates.push(update);
		const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, depthTest: false, transparent: true, toneMapped: false }));
		sprite.position.set(0, 2.6, 0);
		sprite.scale.set(2.6, 0.475, 1);
		sprite.userData.station = station.id;
		targets.push(sprite);
		parent.add(sprite);
	}

	const targets: THREE.Object3D[] = [];
	const stationHalos = new Map<string, THREE.Mesh>();
	const stationAnimations: ((time: number) => void)[] = [];
	const detailed = new THREE.Group();
	world.add(detailed);
	for (const station of stations) {
		const group = new THREE.Group();
		group.position.set(station.x, 0, station.z);
		world.add(group);
		contactShadow(group, 3.7, 3.1, 0.218);
		mesh(new THREE.CylinderGeometry(1.12, 1.23, 0.19, 12), materials.paper, group, 0, 0.32, 0);
		const haloMaterial = new THREE.MeshBasicMaterial({ color: station.color, transparent: true, opacity: 0.45 });
		const halo = mesh(new THREE.TorusGeometry(1.19, 0.028, 6, 64), haloMaterial, group, 0, 0.245, 0);
		halo.rotation.x = -Math.PI / 2;
		halo.castShadow = false;
		stationHalos.set(station.id, halo);
		const hotArea = mesh(new THREE.CylinderGeometry(1.35, 1.35, 2.1, 10), new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false }), group, 0, 1.18, 0);
		hotArea.castShadow = false;
		hotArea.userData.station = station.id;
		targets.push(hotArea);
		label(station, group);

		if (station.id === "llm-systems") {
			box(group, [1.96, 0.12, 1.12], [0, 0.52, 0], materials.wood);
			for (const x of [-0.76, 0.76]) for (const z of [-0.4, 0.4]) box(group, [0.09, 0.22, 0.09], [x, 0.39, z], materials.woodEdge);
			for (const z of [-0.36, 0.02, 0.39]) box(group, [1.82, 0.004, 0.007], [0, 0.582, z], materials.woodEdge);
			box(group, [0.56, 0.86, 0.7], [0, 0.96, 0], materials.sage);
			for (let layer = 0; layer < 4; layer++) {
				box(group, [0.65, 0.065, 0.76], [0, 0.66 + layer * 0.22, 0], materials.forest);
			}
			const tokens: THREE.Mesh[] = [];
			for (let i = 0; i < 4; i++) {
				tokens.push(box(group, [0.16, 0.13, 0.3], [-0.84 + i * 0.56, 0.64, 0], i < 2 ? materials.paper : materials.peach));
			}
			for (let i = 0; i < 3; i++) {
				box(group, [0.1, 0.18 + i * 0.13, 0.13], [0.7 + i * 0.14, 0.66 + i * 0.065, -0.21], materials.peach);
			}
			notebook(group, -0.53, 0.615, 0.36, -0.18);
			deskLamp(group, -0.79, 0.605, -0.31);
			stationAnimations.push((time) => {
				tokens.forEach((token, i) => { token.position.x = ((time * 0.28 + i * 0.46) % 1.84) - 0.92; });
			});
		} else if (station.id === "reinforcement-learning") {
			box(group, [1.43, 0.12, 1.18], [0, 0.5, 0.03], materials.wood);
			for (let i = -2; i <= 2; i++) {
				box(group, [0.016, 0.01, 0.99], [i * 0.24, 0.566, 0.03], materials.forest);
				box(group, [1.18, 0.01, 0.016], [0, 0.567, 0.03 + i * 0.2], materials.forest);
			}
			for (const [i, [x, z]] of [[-0.24, 0.03], [0, 0.03], [0.24, 0.23], [-0.24, -0.17]].entries()) {
				mesh(new THREE.SphereGeometry(0.09, 12, 8), i % 2 ? materials.paper : materials.ink, group, x, 0.62, z).scale.y = 0.5;
			}
			const points = [
				new THREE.Vector3(0, 1.91, -0.17),
				new THREE.Vector3(-0.49, 1.38, -0.17),
				new THREE.Vector3(0.49, 1.38, -0.17),
				new THREE.Vector3(-0.73, 0.94, -0.17),
				new THREE.Vector3(-0.24, 0.94, -0.17),
				new THREE.Vector3(0.24, 0.94, -0.17),
				new THREE.Vector3(0.73, 0.94, -0.17),
			];
			for (let i = 1; i < points.length; i++) connection(group, points[Math.floor((i - 1) / 2)], points[i], materials.brass);
			points.forEach((point, i) => mesh(new THREE.SphereGeometry(i === 0 ? 0.13 : 0.09, 12, 8), i === 0 || i === 2 || i === 6 ? materials.peach : materials.forest, group, point.x, point.y, point.z));
			const signal = mesh(new THREE.SphereGeometry(0.065, 10, 8), materials.paper, group);
			mesh(new THREE.CylinderGeometry(0.14, 0.12, 0.11, 14), materials.woodEdge, group, -0.84, 0.47, 0.48);
			mesh(new THREE.SphereGeometry(0.075, 10, 8), materials.paper, group, -0.84, 0.53, 0.48).scale.y = 0.45;
			notebook(group, 0.66, 0.435, 0.71, 0.28);
			stationAnimations.push((time) => {
				const phase = (time * 0.6) % 4;
				const route = [points[0], points[2], points[6], points[2], points[0]];
				signal.position.lerpVectors(route[Math.floor(phase)], route[Math.floor(phase) + 1], phase % 1);
			});
		} else {
			box(group, [1.55, 0.07, 0.94], [0, 0.455, 0.08], materials.wood);
			box(group, [1.36, 0.015, 0.74], [0, 0.505, 0.08], materials.paper);
			for (let line = 0; line < 4; line++) box(group, [0.31 + line * 0.07, 0.006, 0.007], [0.15, 0.519, 0.29 - line * 0.065], materials.sage);
			connection(group, new THREE.Vector3(-0.61, 0.53, -0.22), new THREE.Vector3(-0.49, 0.53, 0.4), materials.peach, 0.018);
			const points = [new THREE.Vector3(0, 1.99, -0.15), new THREE.Vector3(-0.67, 1.03, 0.1), new THREE.Vector3(0.67, 1.03, 0.1)];
			for (const [a, b] of [[0, 1], [0, 2], [1, 2]]) {
				connection(group, points[a], points[b], materials.blue, 0.035);
				const direction = points[b].clone().sub(points[a]).normalize();
				const arrow = mesh(new THREE.ConeGeometry(0.086, 0.2, 8), materials.blue, group);
				arrow.position.copy(points[b]).addScaledVector(direction, -0.29);
				arrow.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
			}
			points.forEach((point, i) => {
				mesh(new THREE.SphereGeometry(0.2, 20, 14), i === 1 ? materials.peach : materials.blue, group, point.x, point.y, point.z);
			});
			const intervention = mesh(new THREE.TorusGeometry(0.31, 0.036, 6, 40), materials.peach, group, -0.67, 1.03, 0.1);
			intervention.rotation.y = 0.3;
			stationAnimations.push((time) => { intervention.rotation.y = 0.3 + Math.sin(time * 0.8) * 0.24; });
		}
	}

	// Three curved tracks make the question → experiment → evaluation loop visible.
	for (let i = 0; i < stations.length; i++) {
		const from = stations[i];
		const to = stations[(i + 1) % stations.length];
		const curve = new THREE.CatmullRomCurve3([
			new THREE.Vector3(from.x * 0.72, 0.245, from.z * 0.7),
			new THREE.Vector3((from.x + to.x) * 0.31, 0.25, (from.z + to.z) * 0.38),
			new THREE.Vector3(to.x * 0.72, 0.245, to.z * 0.7),
		]);
		mesh(new THREE.TubeGeometry(curve, 36, 0.025, 5, false), materials.paper, world).castShadow = false;
		const signal = mesh(new THREE.SphereGeometry(0.065, 8, 6), materials.peach, world);
		stationAnimations.push((time) => { signal.position.copy(curve.getPoint((time * 0.065 + i * 0.33) % 1)); });
	}
	const core = mesh(new THREE.TorusGeometry(0.44, 0.11, 8, 32), materials.forest, world, 0, 0.9, -0.3);
	core.rotation.x = Math.PI / 3;
	mesh(new THREE.CylinderGeometry(0.62, 0.72, 0.16, 12), materials.paper, world, 0, 0.33, -0.3);
	for (let i = 0; i < 8; i++) {
		const angle = i * Math.PI / 4 + 0.3;
		mesh(new THREE.IcosahedronGeometry(0.13 + (i % 3) * 0.04), i % 2 ? materials.sage : materials.peach, detailed, Math.cos(angle) * 5.8, 0.33, Math.sin(angle) * 4.1);
	}

	const rover = new THREE.Group();
	rover.position.set(0, 0.25, 1.35);
	world.add(rover);
	contactShadow(rover, 1.45, 1.8, -0.03);
	const body = new THREE.Group();
	rover.add(body);
	box(body, [0.65, 0.22, 0.94], [0, 0.29, 0], materials.peach);
	box(body, [0.43, 0.3, 0.43], [0, 0.55, -0.12], materials.forest);
	box(body, [0.34, 0.16, 0.02], [0, 0.57, 0.106], materials.paper);
	box(body, [0.45, 0.05, 0.16], [0, 0.36, 0.44], materials.paper);
	mesh(new THREE.SphereGeometry(0.06, 10, 8), materials.peach, body, 0, 0.78, -0.12);
	const wheels: THREE.Group[] = [];
	for (const x of [-0.36, 0.36]) {
		for (const z of [-0.3, 0.3]) {
			const axle = new THREE.Group();
			axle.position.set(x, 0.15, z);
			const tyre = mesh(new THREE.CylinderGeometry(0.15, 0.15, 0.1, 14), materials.ink, axle);
			tyre.rotation.z = Math.PI / 2;
			const hub = mesh(new THREE.CylinderGeometry(0.065, 0.065, 0.112, 8), materials.paper, axle);
			hub.rotation.z = Math.PI / 2;
			box(axle, [0.12, 0.035, 0.22], [0, 0, 0], materials.sage);
			rover.add(axle);
			wheels.push(axle);
		}
	}
	const velocity = new THREE.Vector2();
	const movement = new THREE.Vector2();
	const pointer = new THREE.Vector2();
	const raycaster = new THREE.Raycaster();
	const routeGuide = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineDashedMaterial({
		color: 0x9b6849, dashSize: 0.14, gapSize: 0.1, transparent: true, opacity: 0.8,
	}));
	routeGuide.visible = false;
	world.add(routeGuide);
	function updateRouteGuide(route: { x: number; z: number }[]): void {
		routeGuide.geometry.dispose();
		routeGuide.geometry = new THREE.BufferGeometry().setFromPoints([rover.position, ...route].map((point) => new THREE.Vector3(point.x, 0.264, point.z)));
		routeGuide.computeLineDistances();
		routeGuide.visible = route.length > 0;
	}

	function noticeForStation(id: string): void {
		const station = stations.find((item) => item.id === id);
		if (station) options.onNotice(`${station.sceneLabel} · 点开看看`, `${station.sceneLabelEn} · Take a closer look`);
	}
	function defaultNotice(): void {
		options.onNotice("点一张桌子，或开着小车慢慢逛。", "Choose a desk, or take the rover for a little wander.");
	}
	function arrivalNotice(id: string): void {
		const station = stations.find((item) => item.id === id);
		if (station) options.onNotice(`到${station.sceneLabel}了，看看这里有什么。`, `You've arrived at ${station.sceneLabelEn.toLowerCase()}. Take a closer look.`);
	}
	function proximityNotice(id: string): void {
		const station = stations.find((item) => item.id === id);
		if (station) options.onNotice(`${station.sceneLabel} · 已靠近，按 Enter 打开`, `${station.sceneLabelEn} · Press Enter to explore`);
	}
	function setFollowRover(follow: boolean): void {
		followRover = follow;
		options.onFollowChange(follow);
		if (follow) { desiredRadius = immersive ? 14.8 : 17; desiredPitch = 0.42; }
		schedule();
	}
	function schedule(): void {
		if (!disposed && usable && active && !document.hidden && frame === 0) frame = requestAnimationFrame(render);
	}
	function resize(): void {
		if (!stage.clientWidth || !stage.clientHeight) return;
		const ratio = Math.min(window.devicePixelRatio || 1, quality === "full" ? (stage.clientWidth < 700 ? 1.5 : 1.85) : 1.15);
		if (renderer.getPixelRatio() !== ratio) renderer.setPixelRatio(ratio);
		renderer.setSize(stage.clientWidth, stage.clientHeight, false);
		camera.aspect = stage.clientWidth / stage.clientHeight;
		camera.fov = stage.clientWidth < 460 ? 52 : stage.clientWidth < 800 ? 42 : 35;
		camera.updateProjectionMatrix();
		schedule();
	}
	function setQuality(value: WorldQuality): void {
		quality = value;
		renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, value === "full" ? (stage.clientWidth < 700 ? 1.5 : 1.85) : 1.15));
		renderer.shadowMap.enabled = value === "full";
		detailed.visible = value === "full";
		// Changing shadow defines needs a material refresh, including a return to light mode.
		scene.traverse((object) => {
			if (object instanceof THREE.Mesh) {
				const list = Array.isArray(object.material) ? object.material : [object.material];
				list.forEach((material) => { material.needsUpdate = true; });
			}
		});
		resize();
	}
	function selectStation(id: string): void {
		const station = stations.find((item) => item.id === id);
		if (!station) return;
		selected = id;
		arrival = null;
		setFollowRover(false);
		const approach = new THREE.Vector3(station.x, 0.25, station.z);
		const inward = new THREE.Vector3(-station.x, 0, -station.z).normalize();
		approach.addScaledVector(inward, 1.64);
		velocity.set(0, 0);
		autoDestination = null;
		autoRoute = [];
		if (reducedMotion) {
			rover.position.copy(approach);
			arrival = id;
			updateRouteGuide([]);
		}
		else {
			autoRoute = planWorldRoute(rover.position, approach, stations);
			updateRouteGuide(autoRoute);
			const first = autoRoute.shift();
			if (first) autoDestination = new THREE.Vector3(first.x, 0.25, first.z);
		}
		world.updateMatrixWorld(true);
		desiredTarget.copy(world.localToWorld(new THREE.Vector3(station.x * 0.68, 0.36, station.z * 0.68)));
		desiredRadius = immersive ? 14.3 : 17;
		desiredPitch = 0.36;
		if (arrival) arrivalNotice(id);
		else if (autoDestination) options.onNotice(`正在往${station.sceneLabel}走，沿着小路慢慢来。`, `On the way to ${station.sceneLabelEn.toLowerCase()}. Follow the dotted path.`);
		else options.onNotice(`${station.sceneLabel} · 当前无法规划路线，可回起点重试`, `${station.sceneLabelEn} · No clear route. Return the rover to the start and try again.`);
		schedule();
	}
	function resetView(): void {
		selected = null;
		arrival = null;
		autoDestination = null;
		autoRoute = [];
		routeGuide.visible = false;
		setFollowRover(false);
		options.onOverview();
		desiredYaw = -0.06;
		desiredPitch = 0.3;
		desiredRadius = immersive ? 19 : 20;
		desiredTarget.set(0, -0.55, 0);
		defaultNotice();
		schedule();
	}
	function resetRover(): void {
		autoDestination = null;
		autoRoute = [];
		arrival = null;
		nearby = null;
		updateRouteGuide([]);
		keys.clear();
		velocity.set(0, 0);
		rover.position.set(0, 0.25, 1.35);
		rover.rotation.set(0, 0, 0);
		body.rotation.set(0, 0, 0);
		options.onNotice("小车回来了，接下来想去哪张桌？", "Back at the start. Which desk shall we visit next?");
		schedule();
	}
	function stationAt(event: PointerEvent): string | null {
		const rect = canvas.getBoundingClientRect();
		pointer.set((event.clientX - rect.left) / rect.width * 2 - 1, -(event.clientY - rect.top) / rect.height * 2 + 1);
		raycaster.setFromCamera(pointer, camera);
		return raycaster.intersectObjects(targets, false)[0]?.object.userData.station ?? null;
	}
	function onPointerDown(event: PointerEvent): void {
		if (event.button !== 0 || activePointer !== null || !event.isPrimary) return;
		activePointer = event.pointerId;
		canvas.focus({ preventScroll: true });
		dragging = true;
		dragMoved = false;
		pointerStartX = lastX = event.clientX;
		pointerStartY = lastY = event.clientY;
		canvas.setPointerCapture(event.pointerId);
		canvas.style.cursor = "grabbing";
	}
	function onPointerMove(event: PointerEvent): void {
		if (activePointer !== null && event.pointerId !== activePointer) return;
		if (dragging) {
			if (Math.hypot(event.clientX - pointerStartX, event.clientY - pointerStartY) > 6) dragMoved = true;
			if (dragMoved) {
				desiredYaw += (event.clientX - lastX) * 0.005;
				desiredPitch = THREE.MathUtils.clamp(desiredPitch + (event.clientY - lastY) * 0.003, 0.08, 0.78);
			}
			lastX = event.clientX;
			lastY = event.clientY;
		} else {
			const hit = stationAt(event);
			if (hit !== hovered) {
				hovered = hit;
				canvas.style.cursor = hit ? "pointer" : "grab";
				if (hit) noticeForStation(hit);
				else if (arrival) arrivalNotice(arrival);
				else if (nearby && !autoDestination) proximityNotice(nearby);
				else if (!selected) defaultNotice();
			}
		}
		schedule();
	}
	function onPointerUp(event: PointerEvent): void {
		if (!dragging || event.pointerId !== activePointer) return;
		dragging = false;
		activePointer = null;
		if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
		if (!dragMoved) {
			const id = stationAt(event);
			if (id) options.onSelect(id);
		}
		canvas.style.cursor = hovered ? "pointer" : "grab";
		schedule();
	}
	function onPointerCancel(event: PointerEvent): void {
		if (event.pointerId !== activePointer) return;
		dragging = false;
		dragMoved = true;
		activePointer = null;
		if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
		canvas.style.cursor = "grab";
	}
	function onPointerLeave(): void {
		if (dragging) return;
		hovered = null;
		if (arrival) arrivalNotice(arrival);
		else if (nearby && !autoDestination) proximityNotice(nearby);
		else if (!selected) defaultNotice();
		schedule();
	}
	function onWheel(event: WheelEvent): void {
		if (document.activeElement !== canvas) return;
		event.preventDefault();
		desiredRadius = THREE.MathUtils.clamp(desiredRadius + event.deltaY * 0.01, 11, 28);
		schedule();
	}
	const movementKeys = new Set(["w", "a", "s", "d", "arrowup", "arrowleft", "arrowdown", "arrowright"]);
	function onKeyDown(event: KeyboardEvent): void {
		if (event.altKey || event.ctrlKey || event.metaKey) return;
		const key = event.key.toLowerCase();
		if (movementKeys.has(key)) {
			event.preventDefault();
			autoDestination = null;
			autoRoute = [];
			arrival = null;
			routeGuide.visible = false;
			keys.add(key);
			schedule();
		} else if (key === "enter") {
			event.preventDefault();
			const nearest = [...stations].sort((a, b) => Math.hypot(a.x - rover.position.x, a.z - rover.position.z) - Math.hypot(b.x - rover.position.x, b.z - rover.position.z))[0];
			if (nearest && Math.hypot(nearest.x - rover.position.x, nearest.z - rover.position.z) < 2.2) options.onSelect(nearest.id);
			else options.onNotice("靠近装置后按 Enter，或直接点选站点", "MOVE CLOSER AND PRESS ENTER, OR SELECT A STATION");
		} else if (key === "r") { event.preventDefault(); resetRover(); }
		else if (key === "home") { event.preventDefault(); resetView(); }
	}
	function onKeyUp(event: KeyboardEvent): void { keys.delete(event.key.toLowerCase()); schedule(); }
	function clearKeys(): void {
		keys.clear();
		dragging = false;
		if (activePointer !== null && canvas.hasPointerCapture(activePointer)) canvas.releasePointerCapture(activePointer);
		activePointer = null;
	}
	function onVisibility(): void {
		previousTime = 0;
		if (document.hidden) clearKeys();
		else schedule();
	}
	function onMotionChange(event: MediaQueryListEvent): void {
		reducedMotion = event.matches;
		options.onMotionPreference(reducedMotion);
		if (reducedMotion && autoDestination) {
			const destination = autoRoute.at(-1) ?? autoDestination;
			rover.position.set(destination.x, 0.25, destination.z);
			autoDestination = null;
			autoRoute = [];
			velocity.set(0, 0);
			routeGuide.visible = false;
			if (selected) { arrival = selected; arrivalNotice(selected); }
		}
		schedule();
	}
	function onContextLost(event: Event): void {
		event.preventDefault();
		usable = false;
		if (frame) cancelAnimationFrame(frame);
		frame = 0;
		keys.clear();
		canvas.blur();
		options.onError();
	}

	function render(now: number): void {
		frame = 0;
		if (disposed || !usable || !active || document.hidden) return;
		if (quality === "light" && previousTime && now - previousTime < 30) { schedule(); return; }
		const delta = Math.min(previousTime ? (now - previousTime) / 1000 : 1 / 30, 0.05);
		previousTime = now;
		if (!reducedMotion && !animationPaused) elapsed += delta;
		movement.set(Number(keys.has("d") || keys.has("arrowright")) - Number(keys.has("a") || keys.has("arrowleft")), Number(keys.has("s") || keys.has("arrowdown")) - Number(keys.has("w") || keys.has("arrowup")));
		if (keys.size > 0) {
			const angle = yaw - world.rotation.y;
			const x = movement.x;
			const z = movement.y;
			movement.set(x * Math.cos(angle) + z * Math.sin(angle), -x * Math.sin(angle) + z * Math.cos(angle));
		}
		let requestedSpeed = 2.8;
		if (autoDestination && keys.size === 0) {
			movement.set(autoDestination.x - rover.position.x, autoDestination.z - rover.position.z);
			requestedSpeed = Math.min(2.7, movement.length() * 3);
			if (movement.length() < 0.09) {
				rover.position.copy(autoDestination);
				const next = autoRoute.shift();
				autoDestination = next ? new THREE.Vector3(next.x, 0.25, next.z) : null;
				movement.set(0, 0);
				if (!autoDestination) {
					velocity.set(0, 0);
					routeGuide.visible = false;
					if (selected) { arrival = selected; arrivalNotice(selected); }
				}
			}
		}
		movement.normalize().multiplyScalar(requestedSpeed);
		velocity.lerp(movement, 1 - Math.exp(-delta * (movement.lengthSq() > 0 ? 5.5 : 9)));
		if (velocity.length() < 0.005) velocity.set(0, 0);
		const previousX = rover.position.x;
		const previousZ = rover.position.z;
		rover.position.x += velocity.x * delta;
		rover.position.z += velocity.y * delta;
		const edge = Math.hypot(rover.position.x / 5.84, rover.position.z / 4.25);
		if (edge > 1) { rover.position.x /= edge; rover.position.z /= edge; velocity.multiplyScalar(0.25); }
		for (const station of stations) {
			const dx = rover.position.x - station.x;
			const dz = rover.position.z - station.z;
			const distance = Math.hypot(dx, dz);
			if (distance > 0 && distance < WORLD_ROVER_CLEARANCE) {
				rover.position.x = station.x + dx / distance * WORLD_ROVER_CLEARANCE;
				rover.position.z = station.z + dz / distance * WORLD_ROVER_CLEARANCE;
			}
		}
		// A plinth near the coast must never push the rover outside the walkable island.
		if (Math.hypot(rover.position.x / 5.84, rover.position.z / 4.25) > 1.001) {
			rover.position.x = previousX;
			rover.position.z = previousZ;
			velocity.set(0, 0);
		}
		const travelled = Math.hypot(rover.position.x - previousX, rover.position.z - previousZ);
		if (!autoDestination && travelled > 0) {
			const close = stations.find((station) => Math.hypot(station.x - rover.position.x, station.z - rover.position.z) < (station.id === nearby ? 2.3 : 2.1))?.id ?? null;
			if (close !== nearby) {
				nearby = close;
				if (nearby && !arrival) proximityNotice(nearby);
				else if (!nearby && !selected) defaultNotice();
			}
		}
		if (velocity.lengthSq() > 0.001) {
			const heading = Math.atan2(velocity.x, velocity.y);
			const angle = Math.atan2(Math.sin(heading - rover.rotation.y), Math.cos(heading - rover.rotation.y));
			rover.rotation.y += angle * Math.min(1, delta * 9);
			wheels.forEach((wheel) => { wheel.rotation.x += travelled / 0.15; });
			body.rotation.z = reducedMotion ? 0 : -angle * Math.min(0.04, velocity.length() * 0.012);
		} else body.rotation.z *= 0.8;
		if (!reducedMotion && !animationPaused) {
			stationAnimations.forEach((animate) => animate(elapsed));
			core.rotation.z = elapsed * 0.1;
		}
		for (const [id, halo] of stationHalos) {
			const emphasized = id === selected || id === hovered || id === nearby;
			(halo.material as THREE.MeshBasicMaterial).opacity = emphasized ? 1 : 0.4;
			halo.scale.setScalar(emphasized ? 1.045 : 1);
		}
		if (followRover) {
			world.updateMatrixWorld(true);
			desiredTarget.copy(world.localToWorld(new THREE.Vector3(rover.position.x * 0.7, 0.1, rover.position.z * 0.7)));
		}
		const smoothing = reducedMotion ? 1 : 1 - Math.exp(-delta * 5);
		yaw += (desiredYaw - yaw) * smoothing;
		pitch += (desiredPitch - pitch) * smoothing;
		radius += (desiredRadius - radius) * smoothing;
		target.lerp(desiredTarget, smoothing);
		const horizontal = Math.cos(pitch) * radius;
		camera.position.set(target.x + Math.sin(yaw) * horizontal, target.y + Math.sin(pitch) * radius + 4.2, target.z + Math.cos(yaw) * horizontal);
		camera.lookAt(target);
		renderer.render(scene, camera);
		if (!ready) { ready = true; options.onReady(); }
		const movingCamera = Math.abs(radius - desiredRadius) + Math.abs(yaw - desiredYaw) + Math.abs(pitch - desiredPitch) + target.distanceTo(desiredTarget) > 0.001;
		if ((!reducedMotion && !animationPaused) || dragging || keys.size > 0 || velocity.lengthSq() > 0 || autoDestination || movingCamera) schedule();
	}

	canvas.addEventListener("pointerdown", onPointerDown);
	canvas.addEventListener("pointermove", onPointerMove);
	canvas.addEventListener("pointerup", onPointerUp);
	canvas.addEventListener("pointercancel", onPointerCancel);
	canvas.addEventListener("lostpointercapture", onPointerCancel);
	canvas.addEventListener("pointerleave", onPointerLeave);
	canvas.addEventListener("wheel", onWheel, { passive: false });
	canvas.addEventListener("keydown", onKeyDown);
	canvas.addEventListener("keyup", onKeyUp);
	canvas.addEventListener("blur", clearKeys);
	canvas.addEventListener("webglcontextlost", onContextLost);
	window.addEventListener("blur", clearKeys);
	document.addEventListener("visibilitychange", onVisibility);
	motionPreference.addEventListener("change", onMotionChange);
	const resizeObserver = new ResizeObserver(resize);
	resizeObserver.observe(stage);
	const visibilityObserver = new IntersectionObserver(([entry]) => {
		active = entry?.isIntersecting ?? true;
		previousTime = 0;
		if (!active) clearKeys();
		else schedule();
	});
	visibilityObserver.observe(stage);
	setQuality("light");
	defaultNotice();
	schedule();

	return {
		selectStation,
		resetView,
		resetRover,
		setQuality,
		setAnimationPaused(paused) { animationPaused = paused; previousTime = 0; schedule(); },
		setFollowRover,
		setLocale(next) { locale = next; labelUpdates.forEach((update) => update()); if (arrival) arrivalNotice(arrival); else if (nearby && !autoDestination) proximityNotice(nearby); else if (selected) noticeForStation(selected); else defaultNotice(); schedule(); },
		setSolarHour(hour: number) {
			const celestial = computeCelestialState(hour);
			sun.position.set(celestial.sunPosition[0], celestial.sunPosition[1], celestial.sunPosition[2]);
			sun.color.setHex(celestial.sunColor);
			sun.intensity = celestial.sunIntensity;
			hemi.color.setHex(celestial.hemiSkyColor);
			hemi.groundColor.setHex(celestial.hemiGroundColor);
			fill.color.setHex(celestial.fillColor);
			fill.intensity = celestial.fillIntensity;
			stage.style.background = celestial.stageBackgroundCss;
			schedule();
			return celestial;
		},
		dispose() {
			disposed = true;
			clearKeys();
			if (frame) cancelAnimationFrame(frame);
			resizeObserver.disconnect();
			visibilityObserver.disconnect();
			canvas.removeEventListener("pointerdown", onPointerDown);
			canvas.removeEventListener("pointermove", onPointerMove);
			canvas.removeEventListener("pointerup", onPointerUp);
			canvas.removeEventListener("pointercancel", onPointerCancel);
			canvas.removeEventListener("lostpointercapture", onPointerCancel);
			canvas.removeEventListener("pointerleave", onPointerLeave);
			canvas.removeEventListener("wheel", onWheel);
			canvas.removeEventListener("keydown", onKeyDown);
			canvas.removeEventListener("keyup", onKeyUp);
			canvas.removeEventListener("blur", clearKeys);
			canvas.removeEventListener("webglcontextlost", onContextLost);
			window.removeEventListener("blur", clearKeys);
			document.removeEventListener("visibilitychange", onVisibility);
			motionPreference.removeEventListener("change", onMotionChange);
			const geometries = new Set<THREE.BufferGeometry>();
			const materialSet = new Set<THREE.Material>();
			const textures = new Set<THREE.Texture>();
			scene.traverse((object) => {
				if (object instanceof THREE.Mesh || object instanceof THREE.Sprite || object instanceof THREE.Line) {
					if (object instanceof THREE.Mesh || object instanceof THREE.Line) geometries.add(object.geometry);
					const list = Array.isArray(object.material) ? object.material : [object.material];
					list.forEach((material) => { materialSet.add(material); if ("map" in material && material.map instanceof THREE.Texture) textures.add(material.map); });
				}
			});
			geometries.forEach((geometry) => geometry.dispose());
			materialSet.forEach((material) => material.dispose());
			textures.forEach((texture) => texture.dispose());
			sun.shadow.dispose();
			renderer.dispose();
		},
	};
}
