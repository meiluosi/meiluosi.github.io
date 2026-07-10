<script lang="ts">
	import { onMount } from "svelte";

	interface Particle {
		x: number;
		y: number;
		vx: number;
		vy: number;
		radius: number;
		opacity: number;
		pulse: number;
		pulseSpeed: number;
	}

	let canvas: HTMLCanvasElement;
	let ctx: CanvasRenderingContext2D;
	let particles: Particle[] = [];
	let animationId: number;
	let mouseX = -1000;
	let mouseY = -1000;
	let width = 0;
	let height = 0;

	const PARTICLE_COUNT = 80;
	const CONNECTION_DISTANCE = 150;
	const MOUSE_RADIUS = 120;
	const MOUSE_FORCE = 0.03;

	const hue = $derived(
		(() => {
			if (typeof document !== "undefined") {
				const style = getComputedStyle(document.documentElement);
				const raw = style.getPropertyValue("--primary-hue") || style.getPropertyValue("--theme-hue") || "165";
				return Number.parseInt(raw.trim()) || 165;
			}
			return 165;
		})(),
	);

	function initParticles() {
		particles = [];
		for (let i = 0; i < PARTICLE_COUNT; i++) {
			particles.push({
				x: Math.random() * width,
				y: Math.random() * height,
				vx: (Math.random() - 0.5) * 0.5,
				vy: (Math.random() - 0.5) * 0.5,
				radius: Math.random() * 2.5 + 1,
				opacity: Math.random() * 0.5 + 0.3,
				pulse: Math.random() * Math.PI * 2,
				pulseSpeed: Math.random() * 0.02 + 0.01,
			});
		}
	}

	function resize() {
		width = window.innerWidth;
		height = window.innerHeight;
		canvas.width = width;
		canvas.height = height;
		initParticles();
	}

	function animate() {
		ctx.clearRect(0, 0, width, height);

		const isDark = document.documentElement.classList.contains("dark");
		const baseAlpha = isDark ? 0.12 : 0.08;

		// Update & draw particles
		for (const p of particles) {
			// Mouse interaction
			const dx = mouseX - p.x;
			const dy = mouseY - p.y;
			const dist = Math.sqrt(dx * dx + dy * dy);
			if (dist < MOUSE_RADIUS) {
				const force = (MOUSE_RADIUS - dist) / MOUSE_RADIUS;
				p.vx -= (dx / dist) * force * MOUSE_FORCE;
				p.vy -= (dy / dist) * force * MOUSE_FORCE;
			}

			// Damping
			p.vx *= 0.999;
			p.vy *= 0.999;

			// Speed limit
			const speed = Math.sqrt(p.vx * p.vx + p.vy * p.vy);
			if (speed > 1.5) {
				p.vx = (p.vx / speed) * 1.5;
				p.vy = (p.vy / speed) * 1.5;
			}

			// Move
			p.x += p.vx;
			p.y += p.vy;

			// Wrap around
			if (p.x < -20) p.x = width + 20;
			if (p.x > width + 20) p.x = -20;
			if (p.y < -20) p.y = height + 20;
			if (p.y > height + 20) p.y = -20;

			// Pulse
			p.pulse += p.pulseSpeed;
			const pulseFactor = 0.5 + 0.5 * Math.sin(p.pulse);

			// Draw particle
			ctx.beginPath();
			ctx.arc(p.x, p.y, p.radius * (0.8 + pulseFactor * 0.4), 0, Math.PI * 2);
			ctx.fillStyle = `hsla(${hue}, 70%, ${isDark ? "65%" : "45%"}, ${p.opacity * (0.6 + pulseFactor * 0.4)})`;
			ctx.fill();
		}

		// Draw connections
		for (let i = 0; i < particles.length; i++) {
			for (let j = i + 1; j < particles.length; j++) {
				const a = particles[i];
				const b = particles[j];
				const dx = a.x - b.x;
				const dy = a.y - b.y;
				const dist = Math.sqrt(dx * dx + dy * dy);

				if (dist < CONNECTION_DISTANCE) {
					const alpha = (1 - dist / CONNECTION_DISTANCE) * baseAlpha;
					ctx.beginPath();
					ctx.moveTo(a.x, a.y);
					ctx.lineTo(b.x, b.y);
					ctx.strokeStyle = `hsla(${hue}, 70%, ${isDark ? "65%" : "45%"}, ${alpha})`;
					ctx.lineWidth = 0.5;
					ctx.stroke();
				}
			}
		}

		animationId = requestAnimationFrame(animate);
	}

	function handleMouseMove(e: MouseEvent) {
		mouseX = e.clientX;
		mouseY = e.clientY;
	}

	function handleMouseLeave() {
		mouseX = -1000;
		mouseY = -1000;
	}

	onMount(() => {
		ctx = canvas.getContext("2d")!;
		resize();

		window.addEventListener("resize", resize);
		window.addEventListener("mousemove", handleMouseMove);
		window.addEventListener("mouseleave", handleMouseLeave);

		animate();

		return () => {
			cancelAnimationFrame(animationId);
			window.removeEventListener("resize", resize);
			window.removeEventListener("mousemove", handleMouseMove);
			window.removeEventListener("mouseleave", handleMouseLeave);
		};
	});
</script>

<canvas
	bind:this={canvas}
	class="neural-particles"
	aria-hidden="true"
></canvas>

<style>
	.neural-particles {
		position: fixed;
		top: 0;
		left: 0;
		width: 100%;
		height: 100%;
		pointer-events: none;
		z-index: 1;
	}
</style>