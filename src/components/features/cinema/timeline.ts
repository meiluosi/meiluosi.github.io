/**
 * timeline.ts — 代码电影的时间轴原语与全局色板。
 *
 * 核心原语：
 * - seg(t, a, b, ease)：返回 t 在 [a, b] 区间内的缓动进度 0..1。
 * - spring(t, t0, config)：解析解阻尼谐振子（underdamped / critically damped），纯时间函数。
 * - atCue(t, cueTime, options)：基于提示点（cue）的进出场状态。
 *
 * 一切动画（位置、透明度、数值计数、多头流光）都写成纯时间函数的组合，
 * 场景是 t 的纯函数，天然支持任意拖动进度条、倍速与瞬移回放。
 */

export type Ease = (p: number) => number;

export const clamp01 = (x: number): number => Math.max(0, Math.min(1, x));

export const lerp = (a: number, b: number, p: number): number =>
	a + (b - a) * p;

export const linear: Ease = (p) => p;

export const easeInOut: Ease = (p) =>
	p < 0.5 ? 4 * p * p * p : 1 - (-2 * p + 2) ** 3 / 2;

export const easeOut: Ease = (p) => 1 - (1 - p) ** 3;

export const easeOutQuart: Ease = (p) => 1 - (1 - p) ** 4;

export const easeOutBack: Ease = (p) => {
	const c1 = 1.70158;
	const c3 = c1 + 1;
	return 1 + c3 * (p - 1) ** 3 + c1 * (p - 1) ** 2;
};

/** t 在 [a, b] 内的缓动进度（区间外夹到 0 或 1）。 */
export const seg = (
	t: number,
	a: number,
	b: number,
	ease: Ease = linear,
): number => ease(clamp01((t - a) / (b - a)));

export interface SpringConfig {
	mass?: number; // 质量 m (默认 1.0)
	stiffness?: number; // 劲度系数 k (默认 100)
	damping?: number; // 阻尼系数 c (默认 15)
}

/**
 * 解析解闭式阻尼谐振子（Analytical Closed-Form Spring）。
 * m * x'' + c * x' + k * x = 0，初值 x(0) = -1, x'(0) = 0。
 * 返回从 t0 开始的 0 -> 1 纯函数响应，完全无需累积状态，支持任意倍速与逆向拖拽。
 */
export const spring = (
	t: number,
	t0: number,
	config: SpringConfig = {},
): number => {
	const dt = t - t0;
	if (dt <= 0) return 0;

	const m = config.mass ?? 1;
	const k = config.stiffness ?? 100;
	const c = config.damping ?? 15;

	// 自然圆频率
	const omega0 = Math.sqrt(k / m);
	// 阻尼比
	const zeta = c / (2 * Math.sqrt(k * m));

	if (zeta < 1) {
		// 欠阻尼 (Underdamped)
		const omegaD = omega0 * Math.sqrt(1 - zeta * zeta);
		const decay = Math.exp(-zeta * omega0 * dt);
		const envelope =
			Math.cos(omegaD * dt) +
			((zeta * omega0) / omegaD) * Math.sin(omegaD * dt);
		return 1 - decay * envelope;
	}
	if (Math.abs(zeta - 1) < 1e-5) {
		// 临界阻尼 (Critically damped)
		return 1 - Math.exp(-omega0 * dt) * (1 + omega0 * dt);
	}
	// 过阻尼 (Overdamped)
	const omegaD = omega0 * Math.sqrt(zeta * zeta - 1);
	const decay = Math.exp(-zeta * omega0 * dt);
	const envelope =
		Math.cosh(omegaD * dt) +
		((zeta * omega0) / omegaD) * Math.sinh(omegaD * dt);
	return 1 - decay * envelope;
};

export interface CueOptions {
	duration?: number;
	ease?: Ease;
	linger?: number;
	exitDuration?: number;
}

/**
 * 提示点辅助函数：计算相对于时间点 cueTime 的进退场进度。
 * - duration <= 0 时，t >= cueTime 瞬时为 1。
 * - 若指定 linger，则入场 duration 后停留 linger，再经 exitDuration 退回 0。
 */
export const atCue = (
	t: number,
	cueTime: number,
	options: CueOptions = {},
): number => {
	const {
		duration = 0.5,
		ease = easeOut,
		linger,
		exitDuration = 0.5,
	} = options;
	if (t < cueTime) return 0;
	if (duration <= 0) {
		if (linger !== undefined && t >= cueTime + linger) return 0;
		return 1;
	}
	if (linger !== undefined) {
		const tEnterEnd = cueTime + duration;
		const tExitStart = tEnterEnd + linger;
		const tExitEnd = tExitStart + exitDuration;
		if (t < tEnterEnd) {
			return ease((t - cueTime) / duration);
		}
		if (t < tExitStart) {
			return 1;
		}
		if (t < tExitEnd) {
			return 1 - ease((t - tExitStart) / exitDuration);
		}
		return 0;
	}
	return ease(clamp01((t - cueTime) / duration));
};

/** 数字格式化：固定小数位，且避免出现 "-0.00"。 */
export const fmt = (v: number, digits = 2): string => {
	const s = v.toFixed(digits);
	return /^-0\.?0*$/.test(s) ? s.slice(1) : s;
};

/** 色板：整部影片统一的高对比度深色科幻视觉语言。 */
export const PALETTE = {
	bg: "#0b1118",
	panel: "#111a24",
	ink: "#e8eef5",
	muted: "#8aa0b4",
	faint: "rgba(255,255,255,0.12)",
	hero: "#fbbf24", // 主角金黄色
	q: "#fb7185", // Query 珊瑚红
	k: "#38bdf8", // Key 天青蓝
	v: "#4ade80", // Value 鼠尾草绿
	head1: "#f43f5e", // Head 1 玫瑰粉 (语法头)
	head2: "#0ea5e9", // Head 2 湛蓝 (语义头)
	ffn: "#a855f7", // FFN 知识层 紫罗兰
	cache: "#10b981", // KV Cache 翡翠绿
	pos: "#c084fc", // 位置编码 紫色
	embed: "#38bdf8", // 词嵌入 浅天蓝
	mask: "#ef4444", // 因果掩码 危险警报红
	win: "#2dd4bf", // 胜出词元 青绿
} as const;
