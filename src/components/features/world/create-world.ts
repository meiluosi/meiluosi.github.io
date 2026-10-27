import * as THREE from "three";
import { type CelestialState, computeCelestialState } from "@/utils/celestial";
import { planWorldRoute } from "./navigation";

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
	triggerJump: () => void;
	interactCurrentStation: () => void;
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
	onCollectOculus?: (index: number, titleZh: string, titleEn: string) => void;
	onStepPlate?: (active: boolean) => void;
	onProximityStation?: (stationId: string | null) => void;
}

/** The 3D Digital Garden & Research Island Controller */
export function createWorld(options: WorldOptions): WorldController {
	const { canvas, stage, stations, immersive } = options;
	let locale = options.locale;
	let quality: WorldQuality = "light";
	const usable = true;
	const active = true;
	let disposed = false;
	let frame = 0;
	let previousTime = 0;
	let elapsed = 0;

	// Orbit & View State
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
	let followRover = true; // Default to follow the traveler

	let yaw = -0.06;
	let pitch = 0.38;
	let radius = immersive ? 14.5 : 16.5;
	let desiredYaw = yaw;
	let desiredPitch = pitch;
	let desiredRadius = radius;

	// Navigation & Auto-Routing State
	let autoDestination: THREE.Vector3 | null = null;
	let autoRoute: { x: number; z: number }[] = [];

	const keys = new Set<string>();
	const motionPreference = window.matchMedia(
		"(prefers-reduced-motion: reduce)",
	);
	let reducedMotion = motionPreference.matches;
	options.onMotionPreference(reducedMotion);

	// Setup Renderer
	const renderer = new THREE.WebGLRenderer({
		canvas,
		alpha: true,
		antialias: true,
		powerPreference: "low-power",
	});
	renderer.setClearColor(0xf3f6f1, 0);
	renderer.outputColorSpace = THREE.SRGBColorSpace;
	renderer.toneMapping = THREE.ACESFilmicToneMapping;
	renderer.toneMappingExposure = 1.08;
	renderer.shadowMap.type = THREE.PCFShadowMap;

	const scene = new THREE.Scene();
	const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 95);
	const target = new THREE.Vector3(0, 0.3, 0);
	const desiredTarget = target.clone();

	const world = new THREE.Group();
	world.rotation.y = -0.12;
	scene.add(world);

	// Lighting Rig
	const hemi = new THREE.HemisphereLight(0xfff5e1, 0x849181, 1.4);
	scene.add(hemi);

	const sun = new THREE.DirectionalLight(0xffe4ba, 2.15);
	sun.position.set(-6, 14, 7);
	sun.castShadow = true;
	sun.shadow.mapSize.set(1024, 1024);
	sun.shadow.camera.left = -9.5;
	sun.shadow.camera.right = 9.5;
	sun.shadow.camera.top = 9.5;
	sun.shadow.camera.bottom = -9.5;
	sun.shadow.camera.near = 0.5;
	sun.shadow.camera.far = 38;
	sun.shadow.normalBias = 0.035;
	scene.add(sun);

	const fill = new THREE.DirectionalLight(0xc5dfd3, 1.0);
	fill.position.set(7, 5, -6);
	scene.add(fill);

	// Palette Materials
	const materials = {
		forest: new THREE.MeshStandardMaterial({
			color: 0x315844,
			roughness: 0.72,
		}),
		sage: new THREE.MeshStandardMaterial({ color: 0x9dbda6, roughness: 0.88 }),
		paper: new THREE.MeshStandardMaterial({ color: 0xf3ead6, roughness: 0.95 }),
		peach: new THREE.MeshStandardMaterial({ color: 0xeeb28c, roughness: 0.75 }),
		blue: new THREE.MeshStandardMaterial({ color: 0x587989, roughness: 0.7 }),
		ink: new THREE.MeshStandardMaterial({ color: 0x263c35, roughness: 0.88 }),
		wood: new THREE.MeshStandardMaterial({ color: 0xb58c61, roughness: 0.86 }),
		woodEdge: new THREE.MeshStandardMaterial({
			color: 0x8c684c,
			roughness: 0.86,
		}),
		brass: new THREE.MeshStandardMaterial({
			color: 0x9e8058,
			roughness: 0.42,
			metalness: 0.48,
		}),
		lamplight: new THREE.MeshBasicMaterial({ color: 0xffe6ad }),
		grass: new THREE.MeshStandardMaterial({ color: 0x84a984, roughness: 0.82 }),
		grassUpper: new THREE.MeshStandardMaterial({
			color: 0x96ba96,
			roughness: 0.8,
		}),
		water: new THREE.MeshStandardMaterial({
			color: 0x5a8f9c,
			roughness: 0.2,
			metalness: 0.25,
			transparent: true,
			opacity: 0.85,
		}),
		crystalGlow: new THREE.MeshBasicMaterial({ color: 0x8ef5d8 }),
		treeGlow: new THREE.MeshBasicMaterial({
			color: 0xdff5dd,
			transparent: true,
			opacity: 0.8,
		}),
		characterCoat: new THREE.MeshStandardMaterial({
			color: 0xcc6e53,
			roughness: 0.65,
		}),
		characterHat: new THREE.MeshStandardMaterial({
			color: 0x405545,
			roughness: 0.7,
		}),
		characterSkin: new THREE.MeshStandardMaterial({
			color: 0xf6d7be,
			roughness: 0.85,
		}),
		characterPants: new THREE.MeshStandardMaterial({
			color: 0x33443a,
			roughness: 0.8,
		}),
		characterPack: new THREE.MeshStandardMaterial({
			color: 0x9e734c,
			roughness: 0.8,
		}),
	};

	function mesh(
		geometry: THREE.BufferGeometry,
		mat: THREE.Material,
		parent: THREE.Object3D,
		x = 0,
		y = 0,
		z = 0,
	): THREE.Mesh {
		const obj = new THREE.Mesh(geometry, mat);
		obj.position.set(x, y, z);
		obj.castShadow = true;
		obj.receiveShadow = true;
		parent.add(obj);
		return obj;
	}

	function box(
		parent: THREE.Object3D,
		size: [number, number, number],
		pos: [number, number, number],
		mat: THREE.Material,
	): THREE.Mesh {
		return mesh(new THREE.BoxGeometry(...size), mat, parent, ...pos);
	}

	function connection(
		parent: THREE.Object3D,
		from: THREE.Vector3,
		to: THREE.Vector3,
		mat: THREE.Material,
		thickness = 0.024,
	): THREE.Mesh {
		const dir = to.clone().sub(from);
		const obj = mesh(
			new THREE.CylinderGeometry(thickness, thickness, dir.length(), 6),
			mat,
			parent,
		);
		obj.position.copy(from).add(to).multiplyScalar(0.5);
		obj.quaternion.setFromUnitVectors(
			new THREE.Vector3(0, 1, 0),
			dir.normalize(),
		);
		return obj;
	}

	// Refined Notebook Model from old island
	function notebook(
		parent: THREE.Object3D,
		x: number,
		y: number,
		z: number,
		angle = 0,
	): void {
		const book = new THREE.Group();
		book.position.set(x, y, z);
		book.rotation.y = angle;
		parent.add(book);
		box(book, [0.38, 0.035, 0.29], [0, 0, 0], materials.forest);
		box(book, [0.34, 0.035, 0.27], [0.005, 0.029, 0], materials.paper);
		for (let line = 0; line < 3; line++) {
			box(
				book,
				[0.19 - line * 0.025, 0.005, 0.008],
				[0.01, 0.05, -0.065 + line * 0.045],
				materials.woodEdge,
			);
		}
		connection(
			book,
			new THREE.Vector3(-0.22, 0.05, -0.13),
			new THREE.Vector3(-0.18, 0.05, 0.15),
			materials.peach,
			0.015,
		);
	}

	// Refined Desk Lamp Model from old island
	function deskLamp(
		parent: THREE.Object3D,
		x: number,
		y: number,
		z: number,
	): void {
		const lamp = new THREE.Group();
		lamp.position.set(x, y, z);
		parent.add(lamp);
		mesh(
			new THREE.CylinderGeometry(0.115, 0.13, 0.035, 12),
			materials.brass,
			lamp,
		);
		connection(
			lamp,
			new THREE.Vector3(0, 0.02, 0),
			new THREE.Vector3(0, 0.53, 0),
			materials.brass,
			0.018,
		);
		connection(
			lamp,
			new THREE.Vector3(0, 0.53, 0),
			new THREE.Vector3(0.22, 0.62, 0.04),
			materials.brass,
			0.021,
		);
		mesh(
			new THREE.ConeGeometry(0.14, 0.15, 16, 1, true),
			materials.brass,
			lamp,
			0.22,
			0.58,
			0.04,
		);
		mesh(
			new THREE.SphereGeometry(0.048, 8, 6),
			materials.lamplight,
			lamp,
			0.22,
			0.505,
			0.04,
		).castShadow = false;
		const glow = mesh(
			new THREE.CircleGeometry(0.26, 20),
			new THREE.MeshBasicMaterial({
				color: 0xffd694,
				transparent: true,
				opacity: 0.14,
				depthWrite: false,
			}),
			lamp,
			0.22,
			0.021,
			0.04,
		);
		glow.rotation.x = -Math.PI / 2;
		glow.castShadow = false;
	}

	// Island Base & Terraced Topography
	const rock = mesh(
		new THREE.CylinderGeometry(6.6, 4.8, 1.8, 12),
		materials.forest,
		world,
		0,
		-0.9,
		0,
	);
	rock.scale.z = 0.82;
	rock.rotation.y = 0.07;

	const ground = mesh(
		new THREE.CylinderGeometry(6.65, 6.65, 0.08, 64),
		materials.grass,
		world,
		0,
		0.18,
		0,
	);
	ground.scale.z = 0.82;
	ground.castShadow = false;

	const rim = mesh(
		new THREE.TorusGeometry(6.55, 0.03, 5, 80),
		materials.paper,
		world,
		0,
		0.23,
		0,
	);
	rim.rotation.x = -Math.PI / 2;
	rim.scale.y = 0.82;
	rim.castShadow = false;

	// Upper Hill / Terrace (North Area)
	const upperBase = mesh(
		new THREE.CylinderGeometry(3.6, 3.4, 0.8, 16),
		materials.forest,
		world,
		0,
		0.55,
		-2.5,
	);
	upperBase.scale.z = 0.78;
	const upperGrass = mesh(
		new THREE.CylinderGeometry(3.7, 3.7, 0.18, 16),
		materials.grassUpper,
		world,
		0,
		0.98,
		-2.5,
	);
	upperGrass.scale.z = 0.78;

	// Stone steps to upper hill
	for (let i = 0; i < 4; i++) {
		const step = box(
			world,
			[1.3, 0.18, 0.48],
			[0, 0.28 + i * 0.18, -0.7 - i * 0.42],
			materials.paper,
		);
		step.receiveShadow = true;
	}

	// Natural Stream Pond
	mesh(
		new THREE.CylinderGeometry(1.4, 1.4, 0.08, 16),
		materials.water,
		world,
		3.0,
		0.23,
		0.6,
	);
	for (let i = 0; i < 9; i++) {
		const angle = (i / 9) * Math.PI * 2;
		const rk = mesh(
			new THREE.DodecahedronGeometry(0.18 + (i % 3) * 0.05),
			materials.sage,
			world,
		);
		rk.position.set(
			3.0 + Math.cos(angle) * 1.45,
			0.27,
			0.6 + Math.sin(angle) * 1.45,
		);
		rk.scale.set(1.1, 0.7, 1);
	}

	// Contact Shadows texture
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
	const shadowMaterial = new THREE.MeshBasicMaterial({
		map: shadowTexture,
		transparent: true,
		depthWrite: false,
	});

	function contactShadow(
		parent: THREE.Object3D,
		width: number,
		depth: number,
		y: number,
	): void {
		const shadow = mesh(
			new THREE.PlaneGeometry(width, depth),
			shadowMaterial,
			parent,
			0,
			y,
			0,
		);
		shadow.rotation.x = -Math.PI / 2;
		shadow.castShadow = false;
		shadow.receiveShadow = false;
	}
	contactShadow(world, 17, 12, -1.82);

	// Station Labels & Targets
	const targets: THREE.Object3D[] = [];
	const stationHalos = new Map<string, THREE.Mesh>();
	const stationAnimations: ((time: number) => void)[] = [];
	const detailed = new THREE.Group();
	world.add(detailed);

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
		const sprite = new THREE.Sprite(
			new THREE.SpriteMaterial({
				map: texture,
				depthTest: false,
				transparent: true,
				toneMapped: false,
			}),
		);
		sprite.position.set(0, 2.7, 0);
		sprite.scale.set(2.5, 0.46, 1);
		sprite.userData.station = station.id;
		targets.push(sprite);
		parent.add(sprite);
	}

	// Research Stations Building (Reusing the refined models)
	for (const station of stations) {
		const group = new THREE.Group();
		group.position.set(station.x, 0, station.z);
		world.add(group);
		contactShadow(group, 3.6, 3.0, 0.22);

		mesh(
			new THREE.CylinderGeometry(1.12, 1.23, 0.19, 12),
			materials.paper,
			group,
			0,
			0.32,
			0,
		);
		const haloMaterial = new THREE.MeshBasicMaterial({
			color: station.color,
			transparent: true,
			opacity: 0.45,
		});
		const halo = mesh(
			new THREE.TorusGeometry(1.19, 0.028, 6, 64),
			haloMaterial,
			group,
			0,
			0.245,
			0,
		);
		halo.rotation.x = -Math.PI / 2;
		halo.castShadow = false;
		stationHalos.set(station.id, halo);

		const hotArea = mesh(
			new THREE.CylinderGeometry(1.35, 1.35, 2.1, 10),
			new THREE.MeshBasicMaterial({
				transparent: true,
				opacity: 0,
				depthWrite: false,
			}),
			group,
			0,
			1.18,
			0,
		);
		hotArea.castShadow = false;
		hotArea.userData.station = station.id;
		targets.push(hotArea);
		label(station, group);

		if (station.id === "llm-systems") {
			// Word Desk: Bookshelf, Token blocks, Notebook, Lamp
			box(group, [1.96, 0.12, 1.12], [0, 0.52, 0], materials.wood);
			for (const x of [-0.76, 0.76]) {
				for (const z of [-0.4, 0.4])
					box(group, [0.09, 0.22, 0.09], [x, 0.39, z], materials.woodEdge);
			}
			for (const z of [-0.36, 0.02, 0.39])
				box(group, [1.82, 0.004, 0.007], [0, 0.582, z], materials.woodEdge);
			box(group, [0.56, 0.86, 0.7], [0, 0.96, 0], materials.sage);
			for (let layer = 0; layer < 4; layer++) {
				box(
					group,
					[0.65, 0.065, 0.76],
					[0, 0.66 + layer * 0.22, 0],
					materials.forest,
				);
			}
			const tokens: THREE.Mesh[] = [];
			for (let i = 0; i < 4; i++) {
				tokens.push(
					box(
						group,
						[0.16, 0.13, 0.3],
						[-0.84 + i * 0.56, 0.64, 0],
						i < 2 ? materials.paper : materials.peach,
					),
				);
			}
			for (let i = 0; i < 3; i++) {
				box(
					group,
					[0.1, 0.18 + i * 0.13, 0.13],
					[0.7 + i * 0.14, 0.66 + i * 0.065, -0.21],
					materials.peach,
				);
			}
			notebook(group, -0.53, 0.615, 0.36, -0.18);
			deskLamp(group, -0.79, 0.605, -0.31);
			stationAnimations.push((time) => {
				for (let i = 0; i < tokens.length; i++) {
					tokens[i].position.x = ((time * 0.28 + i * 0.46) % 1.84) - 0.92;
				}
			});
		} else if (station.id === "reinforcement-learning") {
			// Game Board: Go Board, Stones, MCTS Binary Tree Graph
			box(group, [1.43, 0.12, 1.18], [0, 0.5, 0.03], materials.wood);
			for (let i = -2; i <= 2; i++) {
				box(
					group,
					[0.016, 0.01, 0.99],
					[i * 0.24, 0.566, 0.03],
					materials.forest,
				);
				box(
					group,
					[1.18, 0.01, 0.016],
					[0, 0.567, 0.03 + i * 0.2],
					materials.forest,
				);
			}
			for (const [i, [x, z]] of [
				[-0.24, 0.03],
				[0, 0.03],
				[0.24, 0.23],
				[-0.24, -0.17],
			].entries()) {
				mesh(
					new THREE.SphereGeometry(0.09, 12, 8),
					i % 2 ? materials.paper : materials.ink,
					group,
					x,
					0.62,
					z,
				).scale.y = 0.5;
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
			for (let i = 1; i < points.length; i++) {
				connection(
					group,
					points[Math.floor((i - 1) / 2)],
					points[i],
					materials.brass,
				);
			}
			for (let i = 0; i < points.length; i++) {
				mesh(
					new THREE.SphereGeometry(i === 0 ? 0.13 : 0.09, 12, 8),
					i === 0 || i === 2 || i === 6 ? materials.peach : materials.forest,
					group,
					points[i].x,
					points[i].y,
					points[i].z,
				);
			}
			const signal = mesh(
				new THREE.SphereGeometry(0.065, 10, 8),
				materials.paper,
				group,
			);
			mesh(
				new THREE.CylinderGeometry(0.14, 0.12, 0.11, 14),
				materials.woodEdge,
				group,
				-0.84,
				0.47,
				0.48,
			);
			mesh(
				new THREE.SphereGeometry(0.075, 10, 8),
				materials.paper,
				group,
				-0.84,
				0.53,
				0.48,
			).scale.y = 0.45;
			notebook(group, 0.66, 0.435, 0.71, 0.28);
			stationAnimations.push((time) => {
				const phase = (time * 0.6) % 4;
				const route = [points[0], points[2], points[6], points[2], points[0]];
				signal.position.lerpVectors(
					route[Math.floor(phase)],
					route[Math.floor(phase) + 1],
					phase % 1,
				);
			});
		} else {
			// Observation Deck: DAG causal graph, arrows, do-intervention ring
			box(group, [1.55, 0.07, 0.94], [0, 0.455, 0.08], materials.wood);
			box(group, [1.36, 0.015, 0.74], [0, 0.505, 0.08], materials.paper);
			for (let line = 0; line < 4; line++) {
				box(
					group,
					[0.31 + line * 0.07, 0.006, 0.007],
					[0.15, 0.519, 0.29 - line * 0.065],
					materials.sage,
				);
			}
			connection(
				group,
				new THREE.Vector3(-0.61, 0.53, -0.22),
				new THREE.Vector3(-0.49, 0.53, 0.4),
				materials.peach,
				0.018,
			);
			const points = [
				new THREE.Vector3(0, 1.99, -0.15),
				new THREE.Vector3(-0.67, 1.03, 0.1),
				new THREE.Vector3(0.67, 1.03, 0.1),
			];
			for (const [a, b] of [
				[0, 1],
				[0, 2],
				[1, 2],
			]) {
				connection(group, points[a], points[b], materials.blue, 0.035);
				const direction = points[b].clone().sub(points[a]).normalize();
				const arrow = mesh(
					new THREE.ConeGeometry(0.086, 0.2, 8),
					materials.blue,
					group,
				);
				arrow.position.copy(points[b]).addScaledVector(direction, -0.29);
				arrow.quaternion.setFromUnitVectors(
					new THREE.Vector3(0, 1, 0),
					direction,
				);
			}
			for (let i = 0; i < points.length; i++) {
				mesh(
					new THREE.SphereGeometry(0.2, 20, 14),
					i === 1 ? materials.peach : materials.blue,
					group,
					points[i].x,
					points[i].y,
					points[i].z,
				);
			}
			const intervention = mesh(
				new THREE.TorusGeometry(0.31, 0.036, 6, 40),
				materials.peach,
				group,
				-0.67,
				1.03,
				0.1,
			);
			intervention.rotation.y = 0.3;
			stationAnimations.push((time) => {
				intervention.rotation.y = 0.3 + Math.sin(time * 0.8) * 0.24;
			});
		}
	}

	// Closed-loop Inter-station Tube Tracks
	for (let i = 0; i < stations.length; i++) {
		const from = stations[i];
		const to = stations[(i + 1) % stations.length];
		const curve = new THREE.CatmullRomCurve3([
			new THREE.Vector3(from.x * 0.72, 0.245, from.z * 0.7),
			new THREE.Vector3((from.x + to.x) * 0.31, 0.25, (from.z + to.z) * 0.38),
			new THREE.Vector3(to.x * 0.72, 0.245, to.z * 0.7),
		]);
		mesh(
			new THREE.TubeGeometry(curve, 36, 0.025, 5, false),
			materials.paper,
			world,
		).castShadow = false;
		const signal = mesh(
			new THREE.SphereGeometry(0.065, 8, 6),
			materials.peach,
			world,
		);
		stationAnimations.push((time) => {
			signal.position.copy(curve.getPoint((time * 0.065 + i * 0.33) % 1));
		});
	}

	// Central Neural Tree of Knowledge (Sumeru / Irminsul)
	const treeGroup = new THREE.Group();
	treeGroup.position.set(0, 1.05, -2.6);
	world.add(treeGroup);

	const trunk = mesh(
		new THREE.CylinderGeometry(0.32, 0.52, 2.2, 8),
		materials.woodEdge,
		treeGroup,
		0,
		1.1,
		0,
	);
	trunk.castShadow = true;
	for (let i = 0; i < 3; i++) {
		const vineCurve = new THREE.CatmullRomCurve3([
			new THREE.Vector3(Math.cos(i * 2.1) * 0.45, 0, Math.sin(i * 2.1) * 0.45),
			new THREE.Vector3(
				Math.cos(i * 2.1 + 1.2) * 0.35,
				1.1,
				Math.sin(i * 2.1 + 1.2) * 0.35,
			),
			new THREE.Vector3(
				Math.cos(i * 2.1 + 2.5) * 0.5,
				2.0,
				Math.sin(i * 2.1 + 2.5) * 0.5,
			),
		]);
		mesh(
			new THREE.TubeGeometry(vineCurve, 16, 0.05, 6, false),
			materials.treeGlow,
			treeGroup,
		);
	}
	const foliageNodes = [
		[0, 2.4, 0, 1.3],
		[-0.7, 2.2, 0.3, 0.85],
		[0.6, 2.3, -0.25, 0.9],
		[0.1, 3.0, 0.15, 0.85],
	];
	for (const [fx, fy, fz, fr] of foliageNodes) {
		const fMesh = mesh(
			new THREE.DodecahedronGeometry(fr, 1),
			materials.forest,
			treeGroup,
			fx,
			fy,
			fz,
		);
		fMesh.castShadow = true;
	}
	const coreCrystal = mesh(
		new THREE.OctahedronGeometry(0.35),
		materials.crystalGlow,
		treeGroup,
		0,
		2.4,
		0,
	);

	// Scheme 2: Floating Stepping Stones & Mechanism Plate
	const platePos = new THREE.Vector3(2.6, 0.25, 1.8);
	const plateMesh = mesh(
		new THREE.CylinderGeometry(0.55, 0.6, 0.06, 16),
		materials.brass,
		world,
		...platePos.toArray(),
	);
	let plateActivated = false;
	let floatingElevation = 0;
	const floatingStones: THREE.Mesh[] = [];
	for (let i = 0; i < 3; i++) {
		const st = mesh(
			new THREE.CylinderGeometry(0.55, 0.45, 0.2, 6),
			materials.paper,
			world,
			5.2 + i * 0.9,
			0.35 + i * 0.32,
			1.4 - i * 0.7,
		);
		floatingStones.push(st);
	}

	// Scheme 3: 3 Inspiration Oculi
	const oculi = [
		{
			id: "token",
			titleZh: "词元碎片：离散符号的向量流",
			titleEn: "Token Fragment: Vector Streams of Symbols",
			pos: new THREE.Vector3(-3.8, 1.25, -1.8),
			collected: false,
			mesh: null as THREE.Group | null,
		},
		{
			id: "search",
			titleZh: "搜索星火：多想一步的博弈树",
			titleEn: "Search Spark: One Step Further in the Tree",
			pos: new THREE.Vector3(3.6, 1.25, -1.2),
			collected: false,
			mesh: null as THREE.Group | null,
		},
		{
			id: "causal",
			titleZh: "因果神瞳：干预斩断了混杂路径",
			titleEn: "Causal Oculus: Intervention Cuts Backdoors",
			pos: new THREE.Vector3(0, 3.6, -3.8),
			collected: false,
			mesh: null as THREE.Group | null,
		},
	];
	const oculiGroup = new THREE.Group();
	world.add(oculiGroup);
	for (let i = 0; i < oculi.length; i++) {
		const item = oculi[i];
		const oGroup = new THREE.Group();
		oGroup.position.copy(item.pos);
		const oct = mesh(
			new THREE.OctahedronGeometry(0.22),
			materials.crystalGlow,
			oGroup,
		);
		const rng = mesh(
			new THREE.TorusGeometry(0.35, 0.02, 6, 16),
			materials.peach,
			oGroup,
		);
		rng.rotation.x = Math.PI / 2;
		oct.castShadow = true;
		oculiGroup.add(oGroup);
		item.mesh = oGroup;
	}

	// Atmosphere Petals Particles
	const petalCount = 32;
	const petalGeo = new THREE.BufferGeometry();
	const petalPos = new Float32Array(petalCount * 3);
	for (let i = 0; i < petalCount; i++) {
		petalPos[i * 3] = (Math.random() - 0.5) * 14;
		petalPos[i * 3 + 1] = 0.5 + Math.random() * 4.2;
		petalPos[i * 3 + 2] = (Math.random() - 0.5) * 14;
	}
	petalGeo.setAttribute("position", new THREE.BufferAttribute(petalPos, 3));
	const petals = new THREE.Points(
		petalGeo,
		new THREE.PointsMaterial({
			color: 0xf5b59e,
			size: 0.11,
			transparent: true,
			opacity: 0.75,
		}),
	);
	world.add(petals);

	// The Traveler Character (Replacing Rover with Animated Low-poly Explorer)
	const traveler = new THREE.Group();
	traveler.position.set(0, 0.25, 1.5);
	world.add(traveler);
	contactShadow(traveler, 1.2, 1.2, 0.02);

	const charBody = new THREE.Group();
	traveler.add(charBody);

	// Coat & Torso
	mesh(
		new THREE.CylinderGeometry(0.18, 0.22, 0.38, 8),
		materials.characterCoat,
		charBody,
		0,
		0.38,
		0,
	);
	// Head & Hat
	mesh(
		new THREE.SphereGeometry(0.14, 12, 10),
		materials.characterSkin,
		charBody,
		0,
		0.64,
		0,
	);
	mesh(
		new THREE.CylinderGeometry(0.12, 0.16, 0.16, 8),
		materials.characterHat,
		charBody,
		0,
		0.77,
		0,
	);
	mesh(
		new THREE.CylinderGeometry(0.3, 0.3, 0.02, 12),
		materials.characterHat,
		charBody,
		0,
		0.71,
		0,
	);
	// Backpack & Cape
	mesh(
		new THREE.BoxGeometry(0.22, 0.24, 0.14),
		materials.characterPack,
		charBody,
		0,
		0.42,
		-0.14,
	);
	const cape = mesh(
		new THREE.BoxGeometry(0.26, 0.32, 0.02),
		materials.characterCoat,
		charBody,
		0,
		0.35,
		-0.18,
	);
	cape.rotation.x = 0.15;
	// Legs
	const leftLeg = mesh(
		new THREE.CylinderGeometry(0.06, 0.05, 0.22, 6),
		materials.characterPants,
		charBody,
		-0.09,
		0.12,
		0,
	);
	const rightLeg = mesh(
		new THREE.CylinderGeometry(0.06, 0.05, 0.22, 6),
		materials.characterPants,
		charBody,
		0.09,
		0.12,
		0,
	);

	// Movement & Physics Variables
	let characterVelocityY = 0;
	let isGrounded = true;
	let characterYaw = 0;
	let walkCycle = 0;

	// Dotted Route Guide Line
	const routeGuide = new THREE.Line(
		new THREE.BufferGeometry(),
		new THREE.LineDashedMaterial({
			color: 0x9b6849,
			dashSize: 0.14,
			gapSize: 0.1,
			transparent: true,
			opacity: 0.8,
		}),
	);
	routeGuide.visible = false;
	world.add(routeGuide);

	function updateRouteGuide(route: { x: number; z: number }[]): void {
		routeGuide.geometry.dispose();
		routeGuide.geometry = new THREE.BufferGeometry().setFromPoints(
			[traveler.position, ...route].map(
				(point) => new THREE.Vector3(point.x, 0.26, point.z),
			),
		);
		routeGuide.computeLineDistances();
		routeGuide.visible = route.length > 0;
	}

	function getTerrainHeight(x: number, z: number): number {
		const distToUpper = Math.hypot(x, z - -2.5);
		if (distToUpper < 3.3) return 1.05; // Upper hill
		if (distToUpper < 3.8 && z < -0.8)
			return 0.25 + (1.0 - (distToUpper - 3.3) / 0.5) * 0.8; // Steps
		if (floatingElevation > 0.4 && x > 4.6 && x < 8.0 && z > -0.6 && z < 2.2) {
			return 0.35 + floatingElevation * 0.6;
		}
		if (Math.hypot(x, z) < 6.4) return 0.25;
		return -4.0; // Void
	}

	function triggerJump(): void {
		if (isGrounded) {
			characterVelocityY = 0.145;
			isGrounded = false;
			schedule();
		}
	}

	function noticeForStation(id: string): void {
		const st = stations.find((item) => item.id === id);
		if (st)
			options.onNotice(
				`${st.sceneLabel} · 点开看看`,
				`${st.sceneLabelEn} · Take a closer look`,
			);
	}
	function defaultNotice(): void {
		options.onNotice(
			"自由漫步探索花园，按 E 或点选站点翻开手稿。",
			"Explore the garden with WASD, or select a desk to open notes.",
		);
	}
	function arrivalNotice(id: string): void {
		const st = stations.find((item) => item.id === id);
		if (st)
			options.onNotice(
				`已到达${st.sceneLabel}，可翻开记录。`,
				`Arrived at ${st.sceneLabelEn.toLowerCase()}. Ready to explore.`,
			);
	}
	function proximityNotice(id: string): void {
		const st = stations.find((item) => item.id === id);
		if (st)
			options.onNotice(
				`${st.sceneLabel} · 按 E 打开手稿`,
				`${st.sceneLabelEn} · Press E to open notes`,
			);
	}

	function setFollowRover(follow: boolean): void {
		followRover = follow;
		options.onFollowChange(follow);
		if (follow) {
			desiredRadius = immersive ? 14.5 : 16.5;
			desiredPitch = 0.38;
		}
		schedule();
	}

	function schedule(): void {
		if (!disposed && usable && active && !document.hidden && frame === 0) {
			frame = requestAnimationFrame(render);
		}
	}

	function resize(): void {
		if (!stage.clientWidth || !stage.clientHeight) return;
		const ratio = Math.min(
			window.devicePixelRatio || 1,
			quality === "full" ? (stage.clientWidth < 700 ? 1.5 : 1.85) : 1.15,
		);
		if (renderer.getPixelRatio() !== ratio) renderer.setPixelRatio(ratio);
		renderer.setSize(stage.clientWidth, stage.clientHeight, false);
		camera.aspect = stage.clientWidth / stage.clientHeight;
		camera.fov =
			stage.clientWidth < 460 ? 50 : stage.clientWidth < 800 ? 42 : 36;
		camera.updateProjectionMatrix();
		schedule();
	}

	function setQuality(value: WorldQuality): void {
		quality = value;
		renderer.setPixelRatio(
			Math.min(
				window.devicePixelRatio || 1,
				value === "full" ? (stage.clientWidth < 700 ? 1.5 : 1.85) : 1.15,
			),
		);
		renderer.shadowMap.enabled = value === "full";
		detailed.visible = value === "full";
		scene.traverse((obj) => {
			if (obj instanceof THREE.Mesh) {
				const list = Array.isArray(obj.material)
					? obj.material
					: [obj.material];
				for (const mat of list) mat.needsUpdate = true;
			}
		});
		schedule();
	}

	// Auto-route navigation towards station
	function selectStation(id: string): void {
		const station = stations.find((item) => item.id === id);
		if (!station) return;
		selected = id;
		const approach = new THREE.Vector3(station.x, 0.25, station.z);
		const inward = new THREE.Vector3(-station.x, 0, -station.z).normalize();
		approach.addScaledVector(inward, 1.6);
		autoDestination = null;
		autoRoute = [];
		if (reducedMotion) {
			traveler.position.copy(approach);
			arrival = id;
			updateRouteGuide([]);
		} else {
			autoRoute = planWorldRoute(traveler.position, approach, stations);
			updateRouteGuide(autoRoute);
			const first = autoRoute.shift();
			if (first) autoDestination = new THREE.Vector3(first.x, 0.25, first.z);
		}
		world.updateMatrixWorld(true);
		desiredTarget.copy(
			world.localToWorld(
				new THREE.Vector3(station.x * 0.7, 0.35, station.z * 0.7),
			),
		);
		desiredRadius = immersive ? 14.0 : 16.0;
		desiredPitch = 0.36;
		if (arrival) arrivalNotice(id);
		else if (autoDestination)
			options.onNotice(
				`正在沿花径前往${station.sceneLabel}…`,
				`Walking along the path to ${station.sceneLabelEn.toLowerCase()}…`,
			);
		schedule();
	}

	function resetView(): void {
		selected = null;
		arrival = null;
		autoDestination = null;
		autoRoute = [];
		routeGuide.visible = false;
		setFollowRover(true);
		options.onOverview();
		desiredYaw = -0.06;
		desiredPitch = 0.38;
		desiredRadius = immersive ? 14.5 : 16.5;
		desiredTarget.set(0, 0.3, 0);
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
		traveler.position.set(0, 0.25, 1.5);
		options.onNotice(
			"旅人回到起点，接下来想去哪张桌？",
			"Back at the start. Which desk shall we visit next?",
		);
		schedule();
	}

	function interactCurrentStation(): void {
		if (nearby) options.onSelect(nearby);
	}

	function stationAt(event: PointerEvent): string | null {
		const rect = canvas.getBoundingClientRect();
		const ptr = new THREE.Vector2(
			((event.clientX - rect.left) / rect.width) * 2 - 1,
			(-(event.clientY - rect.top) / rect.height) * 2 + 1,
		);
		const rc = new THREE.Raycaster();
		rc.setFromCamera(ptr, camera);
		return (
			rc.intersectObjects(targets, false)[0]?.object.userData.station ?? null
		);
	}

	function onPointerDown(event: PointerEvent): void {
		if (event.button !== 0 || activePointer !== null || !event.isPrimary)
			return;
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
			if (
				Math.hypot(
					event.clientX - pointerStartX,
					event.clientY - pointerStartY,
				) > 6
			)
				dragMoved = true;
			if (dragMoved) {
				desiredYaw += (event.clientX - lastX) * 0.005;
				desiredPitch = THREE.MathUtils.clamp(
					desiredPitch + (event.clientY - lastY) * 0.003,
					0.08,
					0.85,
				);
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
		if (canvas.hasPointerCapture(event.pointerId))
			canvas.releasePointerCapture(event.pointerId);
		if (!dragMoved) {
			const id = stationAt(event);
			if (id) {
				selectStation(id);
				options.onSelect(id);
			}
		}
		canvas.style.cursor = hovered ? "pointer" : "grab";
		schedule();
	}

	function onWheel(event: WheelEvent): void {
		if (document.activeElement !== canvas) return;
		event.preventDefault();
		desiredRadius = THREE.MathUtils.clamp(
			desiredRadius + event.deltaY * 0.01,
			8.5,
			26,
		);
		schedule();
	}

	const movementKeys = new Set([
		"w",
		"a",
		"s",
		"d",
		"arrowup",
		"arrowleft",
		"arrowdown",
		"arrowright",
	]);
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
		} else if (key === " " || key === "space") {
			event.preventDefault();
			triggerJump();
		} else if (key === "e" || key === "enter") {
			event.preventDefault();
			if (nearby) options.onSelect(nearby);
		} else if (key === "r") {
			event.preventDefault();
			resetRover();
		} else if (key === "home") {
			event.preventDefault();
			resetView();
		}
	}

	function onKeyUp(event: KeyboardEvent): void {
		keys.delete(event.key.toLowerCase());
		schedule();
	}

	// Physics & Animation Loop
	function render(timestamp: number): void {
		frame = 0;
		if (disposed || !usable || !active || document.hidden) return;
		if (previousTime === 0) previousTime = timestamp;
		const delta = Math.min((timestamp - previousTime) * 0.001, 0.05);
		previousTime = timestamp;
		if (!animationPaused) elapsed += delta;

		// 1. Controls Gathering (Manual or Auto-routing)
		let moveX = 0;
		let moveZ = 0;
		if (keys.has("w") || keys.has("arrowup")) moveZ -= 1;
		if (keys.has("s") || keys.has("arrowdown")) moveZ += 1;
		if (keys.has("a") || keys.has("arrowleft")) moveX -= 1;
		if (keys.has("d") || keys.has("arrowright")) moveX += 1;

		const manualInput = Math.hypot(moveX, moveZ);
		const speed = 4.2;

		if (manualInput > 0.05) {
			autoDestination = null;
			autoRoute = [];
			routeGuide.visible = false;
			const normX = moveX / Math.max(1, manualInput);
			const normZ = moveZ / Math.max(1, manualInput);

			const cosY = Math.cos(yaw);
			const sinY = Math.sin(yaw);
			const worldMoveX = normX * cosY - normZ * sinY;
			const worldMoveZ = normX * sinY + normZ * cosY;

			traveler.position.x += worldMoveX * speed * delta;
			traveler.position.z += worldMoveZ * speed * delta;

			const targetYaw = Math.atan2(worldMoveX, worldMoveZ);
			let diff = targetYaw - characterYaw;
			while (diff < -Math.PI) diff += Math.PI * 2;
			while (diff > Math.PI) diff -= Math.PI * 2;
			characterYaw += diff * Math.min(1, 14 * delta);

			walkCycle += delta * 12;
		} else if (autoDestination) {
			// Auto walking towards next path point
			const toDest = new THREE.Vector2(
				autoDestination.x - traveler.position.x,
				autoDestination.z - traveler.position.z,
			);
			const dist = toDest.length();
			if (dist < 0.2) {
				const next = autoRoute.shift();
				if (next) {
					autoDestination.set(next.x, 0.25, next.z);
					updateRouteGuide(autoRoute);
				} else {
					autoDestination = null;
					updateRouteGuide([]);
					if (selected) {
						arrival = selected;
						arrivalNotice(selected);
					}
				}
			} else {
				toDest.normalize();
				traveler.position.x += toDest.x * 3.6 * delta;
				traveler.position.z += toDest.y * 3.6 * delta;
				characterYaw = Math.atan2(toDest.x, toDest.y);
				walkCycle += delta * 11;
			}
		} else {
			walkCycle = 0;
		}

		// 2. Vertical Physics & Terrain Height
		const groundY = getTerrainHeight(traveler.position.x, traveler.position.z);
		characterVelocityY -= 0.38 * delta;
		traveler.position.y += characterVelocityY;

		if (traveler.position.y <= groundY) {
			traveler.position.y = groundY;
			characterVelocityY = 0;
			isGrounded = true;
		} else {
			isGrounded = false;
		}

		if (traveler.position.y < -3.5) resetRover();

		// Character Mesh Update
		traveler.rotation.y = characterYaw;
		if ((manualInput > 0.05 || autoDestination) && isGrounded) {
			charBody.position.y = Math.abs(Math.sin(walkCycle)) * 0.04;
			leftLeg.rotation.x = Math.sin(walkCycle) * 0.6;
			rightLeg.rotation.x = -Math.sin(walkCycle) * 0.6;
			cape.rotation.x = 0.35 + Math.sin(walkCycle) * 0.1;
		} else {
			charBody.position.y = Math.sin(elapsed * 3) * 0.015;
			leftLeg.rotation.x = 0;
			rightLeg.rotation.x = 0;
			cape.rotation.x = 0.15;
		}

		// 3. Pressure Plate & Floating Stones
		const distToPlate = traveler.position.distanceTo(platePos);
		const onPlate = distToPlate < 0.65;
		if (onPlate !== plateActivated) {
			plateActivated = onPlate;
			if (options.onStepPlate) options.onStepPlate(onPlate);
		}
		const targetElev = plateActivated ? 1.0 : 0.0;
		floatingElevation +=
			(targetElev - floatingElevation) * Math.min(1, 5 * delta);
		plateMesh.position.y = platePos.y - (plateActivated ? 0.03 : 0);
		for (let i = 0; i < floatingStones.length; i++) {
			floatingStones[i].position.y =
				0.35 +
				i * 0.32 +
				floatingElevation * 0.72 +
				Math.sin(elapsed * 2 + i) * 0.03;
		}

		// 4. Oculi Collection Check
		for (let i = 0; i < oculi.length; i++) {
			const item = oculi[i];
			if (item.mesh) {
				if (item.collected) {
					item.mesh.scale.multiplyScalar(0.9);
					if (item.mesh.scale.x < 0.02) item.mesh.visible = false;
				} else {
					item.mesh.position.y = item.pos.y + Math.sin(elapsed * 3 + i) * 0.1;
					item.mesh.rotation.y += delta * 1.5;
					if (traveler.position.distanceTo(item.pos) < 1.1) {
						item.collected = true;
						if (options.onCollectOculus)
							options.onCollectOculus(i, item.titleZh, item.titleEn);
					}
				}
			}
		}

		// 5. Stations Proximity
		let nearestId: string | null = null;
		let minDistance = 2.0;
		for (const st of stations) {
			const d = Math.hypot(
				st.x - traveler.position.x,
				st.z - traveler.position.z,
			);
			if (d < minDistance) {
				nearestId = st.id;
				minDistance = d;
			}
		}
		if (nearestId !== nearby) {
			nearby = nearestId;
			if (options.onProximityStation) options.onProximityStation(nearby);
			if (nearby && !autoDestination) proximityNotice(nearby);
		}

		// 6. Camera Smooth Follow
		if (followRover && !dragging) {
			world.updateMatrixWorld(true);
			desiredTarget.copy(
				world.localToWorld(
					traveler.position.clone().add(new THREE.Vector3(0, 0.45, 0)),
				),
			);
		}
		target.lerp(desiredTarget, Math.min(1, 7 * delta));
		yaw += (desiredYaw - yaw) * Math.min(1, 8 * delta);
		pitch += (desiredPitch - pitch) * Math.min(1, 8 * delta);
		radius += (desiredRadius - radius) * Math.min(1, 7 * delta);

		const cosPitch = Math.cos(pitch);
		camera.position.set(
			target.x + Math.sin(yaw) * cosPitch * radius,
			target.y + Math.sin(pitch) * radius,
			target.z + Math.cos(yaw) * cosPitch * radius,
		);
		camera.lookAt(target);

		// 7. Ambient Animations
		if (!animationPaused) {
			for (let i = 0; i < stationAnimations.length; i++) {
				stationAnimations[i](elapsed);
			}
			coreCrystal.rotation.y += delta * 0.8;
			coreCrystal.rotation.x += delta * 0.5;
			const pArray = petalGeo.attributes.position.array as Float32Array;
			for (let i = 0; i < petalCount; i++) {
				pArray[i * 3 + 1] -= delta * 0.4;
				pArray[i * 3] += Math.sin(elapsed + i) * delta * 0.25;
				if (pArray[i * 3 + 1] < 0.2) pArray[i * 3 + 1] = 4.2;
			}
			petalGeo.attributes.position.needsUpdate = true;
		}

		renderer.render(scene, camera);
		schedule();
	}

	// Event Listeners
	canvas.addEventListener("pointerdown", onPointerDown);
	window.addEventListener("pointermove", onPointerMove);
	window.addEventListener("pointerup", onPointerUp);
	canvas.addEventListener("wheel", onWheel, { passive: false });
	window.addEventListener("keydown", onKeyDown);
	window.addEventListener("keyup", onKeyUp);
	window.addEventListener("resize", resize);
	motionPreference.addEventListener("change", (e) => {
		reducedMotion = e.matches;
		options.onMotionPreference(reducedMotion);
	});

	resize();
	schedule();
	options.onReady();

	return {
		selectStation,
		resetView,
		resetRover,
		triggerJump,
		interactCurrentStation,
		setQuality,
		setAnimationPaused: (paused: boolean) => {
			animationPaused = paused;
			schedule();
		},
		setFollowRover,
		setLocale: (next: WorldLocale) => {
			locale = next;
			for (let i = 0; i < labelUpdates.length; i++) {
				labelUpdates[i]();
			}
			schedule();
		},
		setSolarHour: (hour: number) => {
			const celestial = computeCelestialState(hour);
			hemi.color.setHex(celestial.period === "night" ? 0x243240 : 0xfff5e1);
			hemi.groundColor.setHex(
				celestial.period === "night" ? 0x141e26 : 0x849181,
			);
			sun.color.setHex(
				celestial.period === "night"
					? 0x6185aa
					: celestial.period === "sunset"
						? 0xf59868
						: 0xffe4ba,
			);
			sun.intensity =
				celestial.period === "night"
					? 0.45
					: celestial.period === "sunset"
						? 1.6
						: 2.15;
			renderer.setClearColor(
				celestial.period === "night" ? 0x111c26 : 0xf3f6f1,
				0,
			);
			schedule();
			return celestial;
		},
		dispose: () => {
			disposed = true;
			if (frame) cancelAnimationFrame(frame);
			canvas.removeEventListener("pointerdown", onPointerDown);
			window.removeEventListener("pointermove", onPointerMove);
			window.removeEventListener("pointerup", onPointerUp);
			canvas.removeEventListener("wheel", onWheel);
			window.removeEventListener("keydown", onKeyDown);
			window.removeEventListener("keyup", onKeyUp);
			window.removeEventListener("resize", resize);
			renderer.dispose();
		},
	};
}
