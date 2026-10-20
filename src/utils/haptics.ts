/**
 * Web Audio procedural mechanical sound synthesizer.
 *
 * Synthesizes crisp physical sounds (shutter click, gear tooth tick, latch switch,
 * sliding friction) entirely in code using native oscillators and noise nodes.
 *
 * Zero external audio assets (0 KB network overhead).
 * Supports automatic muting via prefers-reduced-motion and user preference toggle.
 */

let audioCtx: AudioContext | null = null;

const STORAGE_KEY = "blog_haptics_enabled";

function getAudioContext(): AudioContext | null {
	if (typeof window === "undefined") return null;
	if (!audioCtx) {
		const AudioContextClass =
			window.AudioContext ||
			(window as unknown as { webkitAudioContext: typeof AudioContext })
				.webkitAudioContext;
		if (AudioContextClass) {
			audioCtx = new AudioContextClass();
		}
	}
	if (audioCtx && audioCtx.state === "suspended") {
		// Browser autoplay policy requires user gesture
		audioCtx.resume().catch(() => {});
	}
	return audioCtx;
}

export function isHapticsEnabled(): boolean {
	if (typeof window === "undefined") return false;
	try {
		const saved = localStorage.getItem(STORAGE_KEY);
		if (saved !== null) return saved === "true";
		// Default enabled unless user prefers reduced motion
		const reduced = window.matchMedia(
			"(prefers-reduced-motion: reduce)",
		).matches;
		return !reduced;
	} catch {
		return true;
	}
}

export function setHapticsEnabled(enabled: boolean): void {
	if (typeof window !== "undefined") {
		try {
			localStorage.setItem(STORAGE_KEY, enabled ? "true" : "false");
			window.dispatchEvent(
				new CustomEvent("blog-haptics-change", { detail: { enabled } }),
			);
		} catch {}
	}
}

/**
 * Mechanical escapement tick (clockwork gear tooth / brass dial step)
 * Duration: ~10ms, sharp metallic decay
 */
export function playMechanicalTick(pitchMultiplier = 1.0): void {
	if (!isHapticsEnabled()) return;
	const ctx = getAudioContext();
	if (!ctx) return;

	const now = ctx.currentTime;
	const osc = ctx.createOscillator();
	const gain = ctx.createGain();
	const filter = ctx.createBiquadFilter();

	// High damped click
	osc.type = "triangle";
	osc.frequency.setValueAtTime(1400 * pitchMultiplier, now);
	osc.frequency.exponentialRampToValueAtTime(
		Math.max(100, 320 * pitchMultiplier),
		now + 0.012,
	);

	filter.type = "bandpass";
	filter.frequency.setValueAtTime(1800 * pitchMultiplier, now);
	filter.Q.setValueAtTime(3.5, now);

	gain.gain.setValueAtTime(0.065, now);
	gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.014);

	osc.connect(filter);
	filter.connect(gain);
	gain.connect(ctx.destination);

	osc.start(now);
	osc.stop(now + 0.016);
}

/**
 * Solid mechanical latch toggle (two micro-clicks ~14ms apart)
 * Used when toggling tabs, dialogs, or switches
 */
export function playLatchClick(): void {
	if (!isHapticsEnabled()) return;
	const ctx = getAudioContext();
	if (!ctx) return;

	const now = ctx.currentTime;

	// First impact
	const osc1 = ctx.createOscillator();
	const gain1 = ctx.createGain();
	osc1.type = "square";
	osc1.frequency.setValueAtTime(800, now);
	osc1.frequency.exponentialRampToValueAtTime(220, now + 0.009);

	gain1.gain.setValueAtTime(0.045, now);
	gain1.gain.exponentialRampToValueAtTime(0.0001, now + 0.01);

	osc1.connect(gain1);
	gain1.connect(ctx.destination);
	osc1.start(now);
	osc1.stop(now + 0.012);

	// Second latch follow-up
	const osc2 = ctx.createOscillator();
	const gain2 = ctx.createGain();
	const t2 = now + 0.016;
	osc2.type = "sine";
	osc2.frequency.setValueAtTime(1250, t2);
	osc2.frequency.exponentialRampToValueAtTime(380, t2 + 0.012);

	gain2.gain.setValueAtTime(0.05, t2);
	gain2.gain.exponentialRampToValueAtTime(0.0001, t2 + 0.015);

	osc2.connect(gain2);
	gain2.connect(ctx.destination);
	osc2.start(t2);
	osc2.stop(t2 + 0.018);
}

/**
 * Leica-like soft camera shutter tick (used for opening search / command palette)
 */
export function playShutterTick(): void {
	if (!isHapticsEnabled()) return;
	const ctx = getAudioContext();
	if (!ctx) return;

	const now = ctx.currentTime;
	const osc = ctx.createOscillator();
	const gain = ctx.createGain();

	osc.type = "sine";
	osc.frequency.setValueAtTime(950, now);
	osc.frequency.exponentialRampToValueAtTime(180, now + 0.024);

	gain.gain.setValueAtTime(0.055, now);
	gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.028);

	osc.connect(gain);
	gain.connect(ctx.destination);
	osc.start(now);
	osc.stop(now + 0.03);
}

/**
 * Soft friction glide tick (used while dragging continuous sliders or scrubbing numbers)
 */
let lastSlideTime = 0;
export function playSlideTick(velocity = 1): void {
	if (!isHapticsEnabled()) return;
	const nowMs = Date.now();
	// Throttle to avoid audio buffer saturation (min 35ms interval)
	if (nowMs - lastSlideTime < 35) return;
	lastSlideTime = nowMs;

	const ctx = getAudioContext();
	if (!ctx) return;

	const now = ctx.currentTime;
	const osc = ctx.createOscillator();
	const gain = ctx.createGain();

	const freq = 600 + Math.min(600, Math.abs(velocity) * 120);
	osc.type = "triangle";
	osc.frequency.setValueAtTime(freq, now);
	osc.frequency.exponentialRampToValueAtTime(freq * 0.45, now + 0.008);

	gain.gain.setValueAtTime(0.032, now);
	gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.01);

	osc.connect(gain);
	gain.connect(ctx.destination);
	osc.start(now);
	osc.stop(now + 0.012);
}
