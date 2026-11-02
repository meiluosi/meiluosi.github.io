---
title: 世界模型的最小骨架：状态向量、固定步长累加器与渲染解耦
published: 2026-11-01T09:00:00+08:00
description: 代码造物与显式世界模型第一卷之一：解剖 60fps 程序化动画的底层因果骨架，推导状态转移方程、解决 requestAnimationFrame 掉帧穿模的固定时间步累加器与渲染插值算法。
format: technical
tags:
  - 代码造物与显式世界模型
  - 物理仿真
  - 程序化动画
  - 数值计算
category: 代码造物与显式世界模型
series: "代码造物与显式世界模型"
seriesOrder: 1
lang: zh
---

当我们在屏幕上看到一个方块从左向右滑过，或者一个弹性小球在重力场中弹跳时，绝大多数前端开发者习惯将其称为“动画效果”。但在计算科学与控制论的语境下，这一过程有着完全不同的名字：**一个运行在离散时钟上的显式世界模型（Explicit World Model）**。

如果你用 CSS 的 `@keyframes` 或 `transition` 来定义动画，你其实并未建立世界模型——你只是在播放一段被提前刻录在时间轴上的“全息录像”。一旦外界发生意料之外的物理干预（比如在半空中施加一次侧向撞击力），录像就会瞬间脱节。

真正的程序化动画，必须拥有独立的**内部状态流形**、严格的**因果演化算子**与**无副作用的渲染投影**。

---

## 1. 显式世界模型的三位一体

任何由代码构建的物理世界，在数学上都可以精确还原为三个核心算子构成的闭环：

```
       外部交互 / 控制干预 a_t
               │
               ▼
┌──────────────────────────────┐
│  内部世界状态向量 s_t        │
│  [位置 x, 速度 v, 质量 m...] │
└──────────────┬───────────────┘
               │ 状态转移算子 f(s_t, a_t, Δt)
               ▼
┌──────────────────────────────┐
│  下一时刻世界状态 s_{t+1}    │───────┐
└──────────────┬───────────────┘       │ 时间推进
               │                       │
               ▼ 观测投影 g(s_t)       │
┌──────────────────────────────┐       │
│  屏幕像素 / SVG 渲染帧 o_t   │       │
└──────────────────────────────┘       ▼
                              (进入下一物理步)
```

### 1.1 状态向量 $s_t$ (Internal State Vector)
世界模型的第一原则是**完全可恢复性（Complete Recoverability）**。系统的所有内在记忆必须全部封装在状态向量 $s \in \mathbb{R}^d$ 中。对于一个质点，其最小完备状态为：
$$s_t = \begin{bmatrix} \mathbf{x}_t \\ \mathbf{v}_t \end{bmatrix} \in \mathbb{R}^{2d}$$
其中 $\mathbf{x}$ 为广义坐标，$\mathbf{v} = \dot{\mathbf{x}}$ 为广义速度。如果系统还包含温度、阻尼、电荷或质量，它们同样以参数形式编入状态。

> **反模式警示**：永远不要从 DOM 树或 Canvas 画布反向读取元素的 `offsetLeft` 或 `transform` 来参与下一帧的物理计算。画面是计算的**单向投影产物**，绝不是状态存储介质。

### 1.2 状态转移算子 $f$ (State Transition Operator)
世界在时间维度上的推进，是由微分方程控制的状态转移函数：
$$s_{t+1} = f(s_t, a_t, \Delta t)$$
其中 $a_t$ 为当前时间步注入的外部行动（如用户鼠标拖拽施加的外力、键盘输入的冲量），$\Delta t$ 为离散时间步长。函数 $f$ 封装了该世界所有的物理法则（牛顿运动定律、碰撞排斥、摩擦阻尼）。

### 1.3 观测投影算子 $g$ (Observation / Rendering Operator)
屏幕上的画面只是高维世界状态在二维视网膜上的投影：
$$o_t = g(s_t)$$
$g$ 是一个纯函数（Pure Function）。它负责将米制单位转换为屏幕像素坐标、将标量能量映射为颜色亮度。$g$ 的执行**严禁修改 $s_t$ 的任何值**。

---

## 2. 致命陷阱：为什么 `requestAnimationFrame(dt)` 会摧毁世界？

许多教程会给出如下看似轻巧的动画循环代码：

```ts
// 典型错误示例：可变时间步长直接物理步进
let lastTime = performance.now();
let x = 0;
let v = 100; // 像素/秒

function tick(now: number) {
	const dt = (now - lastTime) / 1000; // 动态帧间隔（秒）
	lastTime = now;

	// 直接根据变化的 dt 计算位移
	x += v * dt;
	render(x);

	requestAnimationFrame(tick);
}
```

这段代码在 60Hz 刷新率的显示器上表现良好（$\Delta t \approx 16.6\text{ms}$）。然而，只要用户切换浏览器标签页 2 秒、或者遭遇 GPU 瞬间卡顿，$\Delta t$ 就会突然飙升到 $2.0\text{s}$。

此时会发生灾难性的**穿模与数值爆炸**：
1. 原本厚度为 10 像素的刚体墙壁，由于物体在单步内直接跃迁了 $100 \times 2.0 = 200$ 像素，碰撞检测器完全被跳过，小球直接穿墙消失。
2. 如果存在弹簧阻尼力 $F = -kx - cv$，当 $\Delta t$ 超过系统的固有周期倒数时，计算出的加速度会直接翻倍，导致整个系统在下一帧炸向无穷大。

**世界的物理演化因果，绝不能随着渲染硬件的快慢而发生改变！**

---

## 3. 固定物理步长累加器 (Fixed-Timestep Accumulator)

解决这一矛盾的标准工程范式，由著名游戏物理学家 Glenn Fiedler（*Fix Your Timestep!*）提出：**将连续时间划分为严格恒定的微元时间步，渲染循环只负责消耗累积的时间。**

### 3.1 累加器算法数学模型
设固定的物理演化步长为 $\Delta t_{fixed}$（通常取 $1/60\text{s} \approx 0.01667\text{s}$ 或更高精度的 $1/120\text{s}$）。
当渲染帧间隔 $\Delta t_{real}$ 到来时，我们将其灌入时间累加器池（Accumulator）：
$$\text{accumulator} \leftarrow \text{accumulator} + \Delta t_{real}$$

只要累加器中的剩余时间大于等于 $\Delta t_{fixed}$，我们就强制世界模型执行一次标准状态演化，并在累加器中扣除对应时间：
$$\text{while } (\text{accumulator} \ge \Delta t_{fixed}): \quad s \leftarrow f(s, a, \Delta t_{fixed}), \quad \text{accumulator} \leftarrow \text{accumulator} - \Delta t_{fixed}$$

### 3.2 防范“死亡螺旋”（Spiral of Death）
如果设备计算能力不足，执行一次物理步进消耗的实际时间大于 $\Delta t_{fixed}$，累加器就会越积越多，物理循环陷入无限迭代，导致网页彻底卡死崩溃。
因此，必须为单帧的最大累积时间设立硬上限（Clamping）：
$$\Delta t_{real} \leftarrow \min(\Delta t_{real}, \Delta t_{max})$$
通常设 $\Delta t_{max} = 0.25\text{s}$。即使发生极限卡顿，系统也至多补齐 15 个物理步，宁可让时间变慢，也绝不让程序失去响应。

---

## 4. 消除微抖动：渲染插值系数 $\alpha$

当物理循环跳出时，累加器中往往还残留着一小段不足 $\Delta t_{fixed}$ 的时间碎屑 $\text{accumulator} < \Delta t_{fixed}$。
此时如果直接渲染最新状态 $s_{current}$，画面会在屏幕刷新率与物理时钟不严格对齐时产生微小的视觉顿挫（Stuttering）。

高阶物理引擎的解法是：**保留上一物理步的状态 $s_{previous}$，利用残余时间比率计算线性插值（Hermite / Lerp）：**
$$\alpha = \frac{\text{accumulator}}{\Delta t_{fixed}} \in [0, 1)$$
$$o_t = g\big((1 - \alpha) s_{previous} + \alpha s_{current}\big)$$

通过这一步投影插值，即使显示器是 144Hz 或可变刷新率（G-Sync / FreeSync），视觉画面永远如同丝绸般顺滑，而底层的物理微分计算依然死死锚定在绝对稳定的 60Hz 网格上。

---

## 5. 完整自包含 TypeScript 参考实现

以下是开箱即用的工业级显式世界模型骨架，满足上述全部因果一致性与插值平滑契约：

```ts
/**
 * MinimalWorldModel.ts
 * 具备固定物理步长累加器与渲染插值的显式世界模型最小骨架
 */

export interface WorldState {
	x: number; // 标量位置
	v: number; // 标量速度
}

export class WorldModelEngine {
	// 恒定物理微元时间步长 (60Hz)
	private readonly fixedDt: number = 1 / 60;
	// 防止死亡螺旋的最大单帧耗时限制
	private readonly maxFrameTime: number = 0.25;

	// 状态流形双缓冲 (用于渲染插值)
	private currentState: WorldState = { x: 0, v: 0 };
	private previousState: WorldState = { x: 0, v: 0 };

	private accumulator: number = 0;
	private lastTimestamp: number = 0;
	private isRunning: boolean = false;
	private rafId: number = 0;

	// 外部行动冲量注入队列
	private pendingImpulse: number = 0;

	constructor(initialState?: Partial<WorldState>) {
		if (initialState) {
			this.currentState = { ...this.currentState, ...initialState };
			this.previousState = { ...this.currentState };
		}
	}

	/** 外部干预接口：非侵入式向系统施加动量扰动 */
	public applyImpulse(dv: number): void {
		this.pendingImpulse += dv;
	}

	/** 状态转移算子 f(s, a, dt)：纯物理演化 */
	private stepPhysics(state: WorldState, dt: number): WorldState {
		// 1. 结算外部控制干预
		let v = state.v + this.pendingImpulse;
		this.pendingImpulse = 0;

		// 2. 简谐恢复力与阻尼算子: a = -k*x - c*v
		const k = 15.0; // 弹性系数
		const c = 0.8;  // 阻尼系数
		const a = -k * state.x - c * v;

		// 3. 半隐式欧拉积分更新速度与位置
		v += a * dt;
		const x = state.x + v * dt;

		return { x, v };
	}

	/** 启动主时钟循环 */
	public start(renderCallback: (interpolatedState: WorldState, alpha: number) => void): void {
		if (this.isRunning) return;
		this.isRunning = true;
		this.lastTimestamp = performance.now();

		const tick = (now: number) => {
			if (!this.isRunning) return;

			let frameTime = (now - this.lastTimestamp) / 1000;
			this.lastTimestamp = now;

			// 截断异常长帧，防止死亡螺旋
			if (frameTime > this.maxFrameTime) {
				frameTime = this.maxFrameTime;
			}

			this.accumulator += frameTime;

			// 恒定步长物理步进消耗池
			while (this.accumulator >= this.fixedDt) {
				this.previousState = { ...this.currentState };
				this.currentState = this.stepPhysics(this.currentState, this.fixedDt);
				this.accumulator -= this.fixedDt;
			}

			// 计算两步之间的插值比率
			const alpha = this.accumulator / this.fixedDt;
			const interpolatedState: WorldState = {
				x: this.previousState.x * (1 - alpha) + this.currentState.x * alpha,
				v: this.previousState.v * (1 - alpha) + this.currentState.v * alpha,
			};

			// 执行无副作用的渲染投影
			renderCallback(interpolatedState, alpha);

			this.rafId = requestAnimationFrame(tick);
		};

		this.rafId = requestAnimationFrame(tick);
	}

	/** 停止物理循环 */
	public stop(): void {
		this.isRunning = false;
		if (this.rafId) cancelAnimationFrame(this.rafId);
	}
}
```

---

## 6. 为什么它是理解 AI 世界模型的钥匙？

现代强化学习中的世界模型（例如 Ha & Schmidhuber 的 *World Models (2018)* 或 Danijar Hafner 的 *DreamerV3*），其内部核心 Recurrent State-Space Model (RSSM) 做的事情与我们上述的代码骨架在概念上完全一致：

1. **确定性循环状态（Deterministic Path）**：对应我们物理模型中精确由微分方程推进的隐藏状态更新；
2. **随机转换与外部行动（Stochastic Prior & Action $a_t$）**：对应外部施加的控制冲量与环境噪声；
3. **解码读出头（Decoder Head $p(o_t \mid s_t)$）**：对应我们将内部坐标插值渲染为画面像素的投影算子 $g$。

理解了代码世界模型中的状态封装、固定步长推进与渲染解耦，你就拥有了一把极度清醒的解剖刀：**无论一个智能体的内部状态是由人工编写的微分方程维持，还是由数十亿参数的循环张量隐式记忆，世界模型的精髓永远在于——它能否在脱离感官输入的漆黑虚空中，自主、稳定且合乎因果地推演下一个时间步。**
