import * as THREE from "three";
import {
	type CustomLinkLengths,
	DEFAULT_CRANK,
	type JointName,
	type LinkPoint,
	STRANDBEEST_LINKS,
	solveStrandbeest,
} from "@/utils/strandbeest-geometry";

export type BeastCameraPreset = "isometric" | "side" | "axial" | "top";

export interface Beast3DOptions {
	canvas: HTMLCanvasElement;
	initialAngle?: number;
	crank?: number;
	customLengths?: CustomLinkLengths;
	explode?: number;
}

export interface Beast3DController {
	updateState(params: {
		angle: number;
		crank: number;
		customLengths?: CustomLinkLengths;
		explode: number;
		windSpeed: number;
	}): void;
	setCameraPreset(preset: BeastCameraPreset): void;
	resize(): void;
	dispose(): void;
}

interface RodVisual {
	mesh: THREE.Mesh;
	from: JointName;
	to: JointName;
	zOffset: number;
}

interface LegBayVisual {
	phaseOffset: number;
	baseZ: number;
	leftRods: RodVisual[];
	rightRods: RodVisual[];
	leftCrankRod: THREE.Mesh;
	rightCrankRod: THREE.Mesh;
	leftJoints: Map<JointName, THREE.Mesh>;
	rightJoints: Map<JointName, THREE.Mesh>;
	crankPin: THREE.Mesh;
	crankWebLeft: THREE.Mesh;
	crankWebRight: THREE.Mesh;
	leftFootRing: THREE.Mesh;
	rightFootRing: THREE.Mesh;
	sailWing: THREE.Group;
}

const BAY_PHASES = [0, 120, 240] as const;
const BAY_BASE_Z = [-22, 0, 22] as const;
const JOINT_NAMES: readonly JointName[] = [
	"G",
	"O",
	"C",
	"K",
	"L",
	"M",
	"N",
	"F",
];

// Assign slight Z-plane separation between front and rear linkage triangles so tubes don't intersect
const ROD_Z_LAYER: Record<string, number> = {
	"G-K": 1.4,
	"C-K": 1.4,
	"G-L": 1.4,
	"K-L": 1.4,
	"L-N": 0.7,
	"G-M": -1.4,
	"C-M": -1.4,
	"M-N": -0.7,
	"M-F": -1.0,
	"N-F": 0.4,
};

const upAxis = new THREE.Vector3(0, 1, 0);
const tempVecA = new THREE.Vector3();
const tempVecB = new THREE.Vector3();
const tempDir = new THREE.Vector3();
const tempMid = new THREE.Vector3();

function orientCylinder(
	mesh: THREE.Mesh,
	a: THREE.Vector3,
	b: THREE.Vector3,
	radius: number,
): void {
	tempDir.subVectors(b, a);
	const len = tempDir.length();
	if (len < 1e-4) {
		mesh.visible = false;
		return;
	}
	mesh.visible = true;
	tempMid.addVectors(a, b).multiplyScalar(0.5);
	mesh.position.copy(tempMid);
	mesh.scale.set(radius, len, radius);
	tempDir.normalize();
	mesh.quaternion.setFromUnitVectors(upAxis, tempDir);
}

export function createBeast3D(options: Beast3DOptions): Beast3DController {
	const { canvas } = options;

	const scene = new THREE.Scene();
	const renderer = new THREE.WebGLRenderer({
		canvas,
		antialias: true,
		alpha: true,
		powerPreference: "high-performance",
	});
	renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
	renderer.shadowMap.enabled = false;

	const camera = new THREE.PerspectiveCamera(36, 1.6, 1, 1200);

	// Warm studio + beach lighting
	const ambient = new THREE.AmbientLight(0xfff8ee, 1.35);
	scene.add(ambient);

	const sunLight = new THREE.DirectionalLight(0xfff2dc, 1.65);
	sunLight.position.set(95, 160, 110);
	scene.add(sunLight);

	const fillLight = new THREE.DirectionalLight(0xcfe0d8, 0.75);
	fillLight.position.set(-110, 50, -90);
	scene.add(fillLight);

	// Shared geometries & materials
	const unitCylinderGeo = new THREE.CylinderGeometry(1, 1, 1, 12, 1, false);
	const unitSphereGeo = new THREE.SphereGeometry(1, 14, 12);
	const ringGeo = new THREE.RingGeometry(1.8, 3.8, 24);
	ringGeo.rotateX(-Math.PI / 2);

	const pvcLeftMat = new THREE.MeshStandardMaterial({
		color: 0xe3b448, // Classic Theo Jansen yellow PVC conduit
		roughness: 0.36,
		metalness: 0.08,
	});
	const pvcRightMat = new THREE.MeshStandardMaterial({
		color: 0xd09b38, // Slightly deeper amber for opposite leg pair
		roughness: 0.38,
		metalness: 0.08,
	});
	const crankMat = new THREE.MeshStandardMaterial({
		color: 0xc25d3a, // Terracotta crank accent
		roughness: 0.35,
		metalness: 0.18,
	});
	const spineMat = new THREE.MeshStandardMaterial({
		color: 0x3b453b, // Dark structural spine rail
		roughness: 0.45,
		metalness: 0.25,
	});
	const jointMat = new THREE.MeshStandardMaterial({
		color: 0x2a2925, // Cable-tie black collar
		roughness: 0.5,
		metalness: 0.2,
	});
	const footMat = new THREE.MeshStandardMaterial({
		color: 0xc25d3a,
		roughness: 0.3,
		metalness: 0.15,
	});
	const sailMat = new THREE.MeshStandardMaterial({
		color: 0xf7f1e3,
		roughness: 0.6,
		metalness: 0.02,
		side: THREE.DoubleSide,
		transparent: true,
		opacity: 0.86,
	});

	const rootGroup = new THREE.Group();
	// Center the beast horizontally around x=38 (the central crankshaft O.x = 38)
	rootGroup.position.set(-38, 18, 0);
	scene.add(rootGroup);

	// Ground plane & scrolling sand grid lines
	const groundGroup = new THREE.Group();
	rootGroup.add(groundGroup);

	const groundPlaneGeo = new THREE.PlaneGeometry(240, 160);
	groundPlaneGeo.rotateX(-Math.PI / 2);
	const groundPlaneMat = new THREE.MeshBasicMaterial({
		color: 0xd8cdb8,
		transparent: true,
		opacity: 0.22,
		side: THREE.DoubleSide,
	});
	const groundPlane = new THREE.Mesh(groundPlaneGeo, groundPlaneMat);
	groundPlane.position.set(38, -84, 0);
	groundGroup.add(groundPlane);

	const gridHelper = new THREE.GridHelper(220, 22, 0xb88b6a, 0x9ca598);
	gridHelper.position.set(38, -83.9, 0);
	if (Array.isArray(gridHelper.material)) {
		for (const m of gridHelper.material) {
			m.transparent = true;
			m.opacity = 0.28;
		}
	} else {
		gridHelper.material.transparent = true;
		gridHelper.material.opacity = 0.28;
	}
	groundGroup.add(gridHelper);

	// Longitudinal spine rails connecting GL (x=0), O (x=38), GR (x=76)
	const spineRailLeft = new THREE.Mesh(unitCylinderGeo, spineMat);
	const spineRailRight = new THREE.Mesh(unitCylinderGeo, spineMat);
	const mainCrankAxis = new THREE.Mesh(unitCylinderGeo, spineMat);
	rootGroup.add(spineRailLeft, spineRailRight, mainCrankAxis);

	// Cross-frame struts at front and rear
	const frontCrossStrut = new THREE.Mesh(unitCylinderGeo, spineMat);
	const rearCrossStrut = new THREE.Mesh(unitCylinderGeo, spineMat);
	rootGroup.add(frontCrossStrut, rearCrossStrut);

	// Build 3 bays
	const bays: LegBayVisual[] = BAY_PHASES.map((phaseOffset, idx) => {
		const baseZ = BAY_BASE_Z[idx] ?? 0;

		const makeRods = (mat: THREE.Material): RodVisual[] =>
			STRANDBEEST_LINKS.map((spec) => {
				const mesh = new THREE.Mesh(unitCylinderGeo, mat);
				rootGroup.add(mesh);
				const key = `${spec.from}-${spec.to}`;
				return {
					mesh,
					from: spec.from,
					to: spec.to,
					zOffset: ROD_Z_LAYER[key] ?? 0,
				};
			});

		const leftRods = makeRods(pvcLeftMat);
		const rightRods = makeRods(pvcRightMat);

		const leftCrankRod = new THREE.Mesh(unitCylinderGeo, crankMat);
		const rightCrankRod = new THREE.Mesh(unitCylinderGeo, crankMat);
		const crankPin = new THREE.Mesh(unitCylinderGeo, crankMat);
		const crankWebLeft = new THREE.Mesh(unitCylinderGeo, spineMat);
		const crankWebRight = new THREE.Mesh(unitCylinderGeo, spineMat);
		rootGroup.add(
			leftCrankRod,
			rightCrankRod,
			crankPin,
			crankWebLeft,
			crankWebRight,
		);

		const leftJoints = new Map<JointName, THREE.Mesh>();
		const rightJoints = new Map<JointName, THREE.Mesh>();
		for (const name of JOINT_NAMES) {
			const mat = name === "F" ? footMat : name === "C" ? crankMat : jointMat;
			const jmL = new THREE.Mesh(unitSphereGeo, mat);
			const jmR = new THREE.Mesh(unitSphereGeo, mat);
			rootGroup.add(jmL, jmR);
			leftJoints.set(name, jmL);
			rightJoints.set(name, jmR);
		}

		const leftFootRingMat = new THREE.MeshBasicMaterial({
			color: 0xc25d3a,
			transparent: true,
			opacity: 0.65,
			side: THREE.DoubleSide,
		});
		const rightFootRingMat = new THREE.MeshBasicMaterial({
			color: 0xc25d3a,
			transparent: true,
			opacity: 0.65,
			side: THREE.DoubleSide,
		});
		const leftFootRing = new THREE.Mesh(ringGeo, leftFootRingMat);
		const rightFootRing = new THREE.Mesh(ringGeo, rightFootRingMat);
		rootGroup.add(leftFootRing, rightFootRing);

		// Wind sail wing mounted above O (x=38, y=7.8)
		const sailWing = new THREE.Group();
		const mast = new THREE.Mesh(unitCylinderGeo, pvcLeftMat);
		orientCylinder(
			mast,
			new THREE.Vector3(0, 0, 0),
			new THREE.Vector3(0, 42, 0),
			0.75,
		);
		sailWing.add(mast);

		const sailShape = new THREE.Shape();
		sailShape.moveTo(0, 6);
		sailShape.lineTo(18, 16);
		sailShape.lineTo(4, 41);
		sailShape.lineTo(0, 41);
		sailShape.closePath();
		const sailGeo = new THREE.ShapeGeometry(sailShape);
		const sailMesh = new THREE.Mesh(sailGeo, sailMat);
		sailWing.add(sailMesh);

		rootGroup.add(sailWing);

		return {
			phaseOffset,
			baseZ,
			leftRods,
			rightRods,
			leftCrankRod,
			rightCrankRod,
			leftJoints,
			rightJoints,
			crankPin,
			crankWebLeft,
			crankWebRight,
			leftFootRing,
			rightFootRing,
			sailWing,
		};
	});

	// Camera spherical state
	let targetAzimuth = 0.62;
	let targetElevation = 0.32;
	let currentAzimuth = 0.62;
	let currentElevation = 0.32;
	let targetDistance = 245;
	let currentDistance = 245;

	let isDragging = false;
	let lastPointerX = 0;
	let lastPointerY = 0;
	let groundScrollX = 0;

	let currentState: {
		angle: number;
		crank: number;
		customLengths?: CustomLinkLengths;
		explode: number;
		windSpeed: number;
	} = {
		angle: options.initialAngle ?? 205,
		crank: options.crank ?? DEFAULT_CRANK,
		customLengths: options.customLengths,
		explode: options.explode ?? 1,
		windSpeed: 5.5,
	};

	function updateGeometry(): void {
		const { angle, crank, customLengths, explode } = currentState;
		const zSpan = 22 * explode;

		// Position longitudinal spine rails along Z
		orientCylinder(
			spineRailLeft,
			new THREE.Vector3(0, 0, -zSpan - 8),
			new THREE.Vector3(0, 0, zSpan + 8),
			1.15,
		);
		orientCylinder(
			spineRailRight,
			new THREE.Vector3(76, 0, -zSpan - 8),
			new THREE.Vector3(76, 0, zSpan + 8),
			1.15,
		);
		orientCylinder(
			mainCrankAxis,
			new THREE.Vector3(38, 7.8, -zSpan - 8),
			new THREE.Vector3(38, 7.8, zSpan + 8),
			0.95,
		);
		orientCylinder(
			frontCrossStrut,
			new THREE.Vector3(0, 0, zSpan + 6),
			new THREE.Vector3(76, 0, zSpan + 6),
			0.95,
		);
		orientCylinder(
			rearCrossStrut,
			new THREE.Vector3(0, 0, -zSpan - 6),
			new THREE.Vector3(76, 0, -zSpan - 6),
			0.95,
		);

		let lowestFootY = -84;
		const computedFloor: number[] = [];
		for (let s = 0; s < 72; s++) {
			const samplePose = solveStrandbeest(s * 5, crank, customLengths, false);
			if (samplePose.valid && samplePose.points.F) {
				computedFloor.push(samplePose.points.F.y);
			}
		}
		if (computedFloor.length > 0) {
			lowestFootY = Math.min(...computedFloor);
		}
		groundPlane.position.y = lowestFootY - 0.5;
		gridHelper.position.y = lowestFootY - 0.4;

		for (const bay of bays) {
			const bayAngle = (angle + bay.phaseOffset) % 360;
			const zCenter = bay.baseZ * explode;
			const pairGap = 3.2 * explode;
			const zLeft = zCenter - pairGap;
			const zRight = zCenter + pairGap;

			const leftPose = solveStrandbeest(bayAngle, crank, customLengths, false);
			const rightPose = solveStrandbeest(bayAngle, crank, customLengths, true);

			// Update Left Leg
			for (const rod of bay.leftRods) {
				const a = leftPose.points[rod.from];
				const b = leftPose.points[rod.to];
				if (a && b) {
					const zOff = rod.zOffset * Math.min(explode, 1.8);
					tempVecA.set(a.x, a.y, zLeft + zOff);
					tempVecB.set(b.x, b.y, zLeft + zOff);
					orientCylinder(rod.mesh, tempVecA, tempVecB, 0.85);
				} else {
					rod.mesh.visible = false;
				}
			}

			// Update Right Leg (mirrored)
			for (const rod of bay.rightRods) {
				const a = rightPose.points[rod.from];
				const b = rightPose.points[rod.to];
				if (a && b) {
					const zOff = -rod.zOffset * Math.min(explode, 1.8);
					tempVecA.set(a.x, a.y, zRight + zOff);
					tempVecB.set(b.x, b.y, zRight + zOff);
					orientCylinder(rod.mesh, tempVecA, tempVecB, 0.85);
				} else {
					rod.mesh.visible = false;
				}
			}

			// Crank arms & shared crank pin
			const oPt = leftPose.points.O;
			const cPt = leftPose.points.C;
			if (oPt && cPt) {
				tempVecA.set(oPt.x, oPt.y, zLeft);
				tempVecB.set(cPt.x, cPt.y, zLeft);
				orientCylinder(bay.leftCrankRod, tempVecA, tempVecB, 1.15);

				tempVecA.set(oPt.x, oPt.y, zRight);
				tempVecB.set(cPt.x, cPt.y, zRight);
				orientCylinder(bay.rightCrankRod, tempVecA, tempVecB, 1.15);

				// Crank pin bridging left and right legs at C
				tempVecA.set(cPt.x, cPt.y, zLeft - 1.5);
				tempVecB.set(cPt.x, cPt.y, zRight + 1.5);
				orientCylinder(bay.crankPin, tempVecA, tempVecB, 1.05);

				bay.crankWebLeft.visible = false;
				bay.crankWebRight.visible = false;
			}

			// Joints
			const placeJoints = (
				jointsMap: Map<JointName, THREE.Mesh>,
				pts: Partial<Record<JointName, LinkPoint>>,
				zBase: number,
			) => {
				for (const [name, mesh] of jointsMap.entries()) {
					const pt = pts[name];
					if (!pt) {
						mesh.visible = false;
						continue;
					}
					mesh.visible = true;
					const r =
						name === "F" ? 1.85 : name === "G" || name === "O" ? 1.65 : 1.3;
					mesh.scale.setScalar(r);
					mesh.position.set(pt.x, pt.y, zBase);
				}
			};
			placeJoints(bay.leftJoints, leftPose.points, zLeft);
			placeJoints(bay.rightJoints, rightPose.points, zRight);

			// Foot ground contact ripple rings
			const updateFootRing = (
				ring: THREE.Mesh,
				foot: LinkPoint | undefined,
				zPos: number,
			) => {
				if (!foot) {
					ring.visible = false;
					return;
				}
				const clearance = foot.y - lowestFootY;
				if (clearance <= 3.2) {
					ring.visible = true;
					const contactStrength = Math.max(0.15, 1 - clearance / 3.2);
					ring.position.set(foot.x, lowestFootY - 0.25, zPos);
					ring.scale.setScalar(0.8 + (1 - contactStrength) * 0.9);
					(ring.material as THREE.MeshBasicMaterial).opacity =
						contactStrength * 0.75;
				} else {
					ring.visible = false;
				}
			};
			updateFootRing(bay.leftFootRing, leftPose.points.F, zLeft);
			updateFootRing(bay.rightFootRing, rightPose.points.F, zRight);

			// Top wind sail oscillation
			bay.sailWing.position.set(38, 7.8, zCenter);
			const rad = (bayAngle * Math.PI) / 180;
			bay.sailWing.rotation.y = Math.sin(rad) * 0.55;
			bay.sailWing.rotation.z = Math.cos(rad) * 0.12;
		}
	}

	function renderScene(): void {
		currentAzimuth += (targetAzimuth - currentAzimuth) * 0.18;
		currentElevation += (targetElevation - currentElevation) * 0.18;
		currentDistance += (targetDistance - currentDistance) * 0.18;

		const cosEl = Math.cos(currentElevation);
		camera.position.set(
			currentDistance * Math.sin(currentAzimuth) * cosEl,
			currentDistance * Math.sin(currentElevation),
			currentDistance * Math.cos(currentAzimuth) * cosEl,
		);
		camera.lookAt(0, -16, 0);

		renderer.render(scene, camera);
	}

	function onPointerDown(e: PointerEvent): void {
		isDragging = true;
		lastPointerX = e.clientX;
		lastPointerY = e.clientY;
		canvas.setPointerCapture(e.pointerId);
	}

	function onPointerMove(e: PointerEvent): void {
		if (!isDragging) return;
		const dx = e.clientX - lastPointerX;
		const dy = e.clientY - lastPointerY;
		lastPointerX = e.clientX;
		lastPointerY = e.clientY;
		targetAzimuth -= dx * 0.0085;
		targetElevation = Math.max(
			-0.25,
			Math.min(1.38, targetElevation + dy * 0.0075),
		);
		renderScene();
	}

	function onPointerUp(e: PointerEvent): void {
		isDragging = false;
		if (canvas.hasPointerCapture(e.pointerId)) {
			canvas.releasePointerCapture(e.pointerId);
		}
	}

	function onWheel(e: WheelEvent): void {
		e.preventDefault();
		targetDistance = Math.max(
			120,
			Math.min(400, targetDistance + e.deltaY * 0.18),
		);
		renderScene();
	}

	canvas.addEventListener("pointerdown", onPointerDown);
	canvas.addEventListener("pointermove", onPointerMove);
	canvas.addEventListener("pointerup", onPointerUp);
	canvas.addEventListener("pointercancel", onPointerUp);
	canvas.addEventListener("wheel", onWheel, { passive: false });

	function resize(): void {
		const rect = canvas.getBoundingClientRect();
		const width = Math.max(280, rect.width || 680);
		const height = Math.max(240, rect.height || 420);
		renderer.setSize(width, height, false);
		camera.aspect = width / height;
		camera.updateProjectionMatrix();
		renderScene();
	}

	resize();
	updateGeometry();
	renderScene();

	return {
		updateState(params) {
			const prevAngle = currentState.angle;
			currentState = { ...params };
			// Scroll ground grid subtly with crank rotation
			const dAngle = ((params.angle - prevAngle + 540) % 360) - 180;
			groundScrollX = (groundScrollX + dAngle * 0.16) % 10;
			gridHelper.position.x = 38 - groundScrollX;
			updateGeometry();
			renderScene();
		},
		setCameraPreset(preset: BeastCameraPreset) {
			if (preset === "isometric") {
				targetAzimuth = 0.62;
				targetElevation = 0.32;
				targetDistance = 245;
			} else if (preset === "side") {
				targetAzimuth = 0;
				targetElevation = 0.05;
				targetDistance = 235;
			} else if (preset === "axial") {
				targetAzimuth = Math.PI / 2;
				targetElevation = 0.16;
				targetDistance = 225;
			} else if (preset === "top") {
				targetAzimuth = 0.2;
				targetElevation = 1.28;
				targetDistance = 260;
			}
			renderScene();
		},
		resize,
		dispose() {
			canvas.removeEventListener("pointerdown", onPointerDown);
			canvas.removeEventListener("pointermove", onPointerMove);
			canvas.removeEventListener("pointerup", onPointerUp);
			canvas.removeEventListener("pointercancel", onPointerUp);
			canvas.removeEventListener("wheel", onWheel);
			unitCylinderGeo.dispose();
			unitSphereGeo.dispose();
			ringGeo.dispose();
			groundPlaneGeo.dispose();
			pvcLeftMat.dispose();
			pvcRightMat.dispose();
			crankMat.dispose();
			spineMat.dispose();
			jointMat.dispose();
			footMat.dispose();
			sailMat.dispose();
			groundPlaneMat.dispose();
			renderer.dispose();
		},
	};
}
