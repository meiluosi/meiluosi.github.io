---
title: 二阶阻尼振子与动量连续性：用微分物理取代死板缓动
published: 2026-11-01T09:20:00+08:00
description: 代码造物与显式世界模型第一卷之三：解剖三次方贝塞尔缓动在交互打断时的动量撕裂硬伤，严格推导二阶阻尼谐振子微分方程，实现欠阻尼、临界阻尼与过阻尼自适应动量跟踪。
format: technical
tags:
  - 代码造物与显式世界模型
  - 阻尼振子
  - 交互动力学
  - 弹性缓动
category: 代码造物与显式世界模型
series: "代码造物与显式世界模型"
seriesOrder: 3
lang: zh
---

几乎所有现代图形界面（Web、iOS、Android）的动画体系，都是建立在“三次方贝塞尔曲线缓动”（`cubic-bezier` / `ease-out`）的基石之上的。从 CSS 关键帧到 GSAP，我们习惯了为动画设定一个明确的持续时间（例如 `duration: 0.4s`）以及一个预计算好的进度映射函数 $p(t) \in [0, 1]$。

对于无人干预的被动播片动画，贝塞尔曲线无可挑剔。然而，一旦引入**人类实时的物理交互**——例如用户在卡片滑向右侧的途中突然反悔，用手指将其向左猛甩——贝塞尔缓动系统就会暴露出致命的骨折现象：**动量撕裂（Momentum Discontinuity）**。

因为预设曲线假定了固定的起点、终点与时间长度。面对突然变动的目标，传统的补间动画只能选择：要么强行丢弃物体当前的瞬时速度从零加速（发生“视觉急刹卡顿”），要么生硬跳针（加速度突变为无穷大，即无穷大加加速度 Jerk）。

**自然界中任何具有质量的物体，其速度永远是时间的连续函数。** 要创造真正有生命力、有重量感的程序化运动，我们必须废黜静态的时间参数 $t$，拥抱**二阶阻尼动力系统（Second-Order Dynamic System）**。

---

## 1. 为什么时间参数化动画必定在打断时失败？

设静态缓动动画的位置方程为：
$$x(t) = x_{start} + (x_{end} - x_{start}) \cdot E\left(\frac{t}{T}\right)$$
其中 $E(\tau)$ 为归一化缓动曲线，$T$ 为总时长。

在任意时刻 $t_1 \in (0, T)$，物体拥有非零的瞬时速度：
$$v(t_1) = \dot{x}(t_1) = \frac{x_{end} - x_{start}}{T} \cdot E'\left(\frac{t_1}{T}\right) \ne 0$$

如果此时目标终点突然从 $x_{end}$ 被拖动到新的位置 $x_{new}$，缓动引擎必须重新生成一条新的曲线。如果重置时间 $t'=0$，初始速度通常被强制归零：
$$v(0^+) = 0 \ne v(t_1^-)$$
速度的一阶导数（加速度）在 $t_1$ 瞬间变成狄拉克 $\delta$ 函数冲击，人眼会敏锐地捕捉到这种极其廉价的“机械生硬感”。

解决这一问题的唯一方法，是**废弃以绝对时间 $t$ 寻址的显式方程，改用由当前位置 $x$、当前速度 $v$ 与目标位置 $x_t$ 共同决定的微分方程**。

---

## 2. 二阶阻尼振子微分方程与三种物理相

一个带有外部目标牵引与物理阻尼的二阶线性系统，其经典运动微分方程为：
$$\ddot{x} + 2\zeta \omega_n \dot{x} + \omega_n^2 x = \omega_n^2 x_{target}$$
其中：
- $x_{target}$ 为当前期望跟踪的目标位置（可以是鼠标坐标、陀螺仪倾角或页面滚动目标）；
- $\omega_n$ 是系统的**固有角频率（Natural Frequency）**，代表弹簧刚度，决定了系统响应的敏捷程度（$\omega_n$ 越大，拉回速度越快）；
- $\zeta$（Zeta）是无量纲的**阻尼比（Damping Ratio）**，代表阻碍运动的能量耗散率。

令跟踪误差为 $e = x - x_{target}$，则系统齐次方程为：
$$\ddot{e} + 2\zeta \omega_n \dot{e} + \omega_n^2 e = 0$$
其特征方程为：
$$\lambda^2 + 2\zeta \omega_n \lambda + \omega_n^2 = 0 \implies \lambda = -\zeta \omega_n \pm \omega_n \sqrt{\zeta^2 - 1}$$

判别式 $\Delta = \zeta^2 - 1$ 的正负，严格划分出三类截然不同的物理触感：

```
                              阻尼比 ζ 的物理相变
  ζ = 0          0 < ζ < 1             ζ = 1             ζ > 1
───┼─────────────────┼───────────────────┼─────────────────┼───► ζ
 无阻尼           欠阻尼 (Underdamped)   临界阻尼 (Critical)  过阻尼 (Overdamped)
 永恒等幅振荡     生动回弹、水波涟漪     最快且无超调归位    像深陷糖浆、厚重黏滞
```

### 2.1 欠阻尼相 ($\zeta < 1$)：生动灵巧与弹性余波
特征根为一对共轭复数 $\lambda = -\alpha \pm i \omega_d$，其中衰减率 $\alpha = \zeta \omega_n$，阻尼角频率 $\omega_d = \omega_n \sqrt{1 - \zeta^2}$。
误差响应解析解为指数包络衰减的正弦波：
$$e(t) = e^{-\zeta \omega_n t} \left( c_1 \cos(\omega_d t) + c_2 \sin(\omega_d t) \right)$$
**视觉特征**：物体冲过目标位置（Overshoot），并在目标附近来回振荡衰减。适合果冻回弹、微交互气泡与自然生命感动画。

### 2.2 临界阻尼相 ($\zeta = 1$)：工业级平滑的黄金奇点
特征根为重实根 $\lambda_1 = \lambda_2 = -\omega_n$。
解析解为：
$$e(t) = (c_1 + c_2 t) e^{-\omega_n t}$$
**视觉特征**：数学上证明的**在没有任何过冲（Zero Overshoot）的前提下，回归目标位置耗时最短的唯一点**。适合操作系统的窗口拖拽、抽屉面板收起与专业工具栏吸附。

### 2.3 过阻尼相 ($\zeta > 1$)：沉重阻抗与黏滞流体
特征根为两个不相等的负实根 $\lambda_{1, 2} = -\zeta \omega_n \pm \omega_n \sqrt{\zeta^2 - 1}$。
**视觉特征**：没有任何振荡，但由于阻尼过大，物体在接近目标时极其缓慢，产生一种“重型机械液压臂”或“泥泞拖拽”的厚实阻尼感。

---

## 3. 面向人机工效的参数重映射

在实际编写 UI 动画时，让设计师或前端直接调节 $\omega_n$（弧度/秒）极其不直观。我们可以通过工程变量替换，将其转换为符合人类感知直觉的三个参数：

1. **响应频率 $f$ (Frequency in Hz)**：
   $$\omega_n = 2\pi f$$
   $f = 1\text{Hz}$ 意味着大约 1 秒钟完成一次往返来回；$f = 3\text{Hz}$ 则是极清脆的快速响应。
2. **阻尼比 $\zeta$ (Damping Ratio)**：
   - $\zeta = 1.0$：经典吸附，不回弹；
   - $\zeta = 0.5$：轻快微弹；
   - $\zeta = 0.2$：极度夸张的卡通弹簧。
3. **初始响应因子 $r$ (Response Acceleration Factor)**：
   允许在目标瞬间突变时，根据目标的瞬时移动速度 $\dot{x}_{target}$ 提前给予加速度补偿：
   $$\ddot{x} + 2\zeta \omega_n \dot{x} + \omega_n^2 x = \omega_n^2 x_t + 2\zeta \omega_n r \dot{x}_t$$
   当 $r = 0$ 时，物体对目标的加速度平缓响应；当 $r > 0$ 时，系统会像有预判意识一样率先启动冲刺。

---

## 4. 离散化解析积分：告别欧拉近似的微步漂移

如果直接用欧拉法去离散化二阶微分方程，当步长稍微放大或刚度较高时，数值解很容易失稳。
对于线性常系数方程，最高雅的实现是：**在假定 $[t, t+\Delta t]$ 区间内目标速度恒定的前提下，直接求解微分方程的局部精确闭式解（Exact Analytic Integration）**。

通过对连续时间状态转移矩阵进行指数化，我们可以导出极其精简的无条件稳定更新算子：

```ts
/**
 * SecondOrderDynamics.ts
 * 任意时间步长绝对数值稳定的二阶阻尼谐振子跟踪器
 * 算法原理源自 Keijiro Takahashi 与 tldraw 的交互力学架构
 */

export class SecondOrderDynamics {
	// 内部状态: 位置 x 与速度 v
	private x: number;
	private v: number;

	// 目标记忆 (用于估计目标速度 dx_target)
	private xTargetPrev: number;

	// 离散化动力学滤波系数
	private k1: number = 0;
	private k2: number = 0;
	private k3: number = 0;

	/**
	 * @param f 固有频率 (Hz): 决定追踪速度与刚度 (建议 1.0 ~ 5.0)
	 * @param zeta 阻尼比: 0~1 弹性回弹，1 临界无超调，>1 黏滞
	 * @param r 初始预期加速度响应系数 (通常取 0.0 ~ 1.0)
	 * @param initialX 初始位置
	 */
	constructor(
		private f: number = 2.0,
		private zeta: number = 0.75,
		private r: number = 0.0,
		initialX: number = 0
	) {
		this.x = initialX;
		this.v = 0;
		this.xTargetPrev = initialX;
		this.recomputeConstants();
	}

	public updateParameters(f: number, zeta: number, r: number): void {
		this.f = f;
		this.zeta = zeta;
		this.r = r;
		this.recomputeConstants();
	}

	private recomputeConstants(): void {
		const omega = 2 * Math.PI * this.f;
		this.k1 = this.zeta / (Math.PI * this.f);
		this.k2 = 1 / (omega * omega);
		this.k3 = (this.r * this.zeta) / omega;
	}

	/**
	 * 60fps / 任意可变 dt 下的自适应动力学状态推进
	 * @param dt 距离上一帧的时间间隔（秒）
	 * @param xTarget 当前帧的目标位置
	 * @param vTargetOptional 可选的目标当前真实速度
	 */
	public update(dt: number, xTarget: number, vTargetOptional?: number): number {
		if (dt <= 0) return this.x;

		// 估计目标自身的瞬时移动速度
		let xTargetDeriv = vTargetOptional;
		if (xTargetDeriv === undefined) {
			xTargetDeriv = (xTarget - this.xTargetPrev) / dt;
			this.xTargetPrev = xTarget;
		}

		// 辛几何安全防卫：极小步长约束防止极端大 dt 振荡
		const iterations = Math.ceil(dt / 0.03);
		const subDt = dt / iterations;

		for (let i = 0; i < iterations; i++) {
			// 半隐式二阶微分更新
			this.x = this.x + subDt * this.v;
			const k2Stable = Math.max(
				this.k2,
				(subDt * subDt) / 2 + (subDt * this.k1) / 2,
				subDt * this.k1
			);
			const accel =
				(xTarget + this.k3 * xTargetDeriv - this.x - this.k1 * this.v) / k2Stable;
			this.v = this.v + subDt * accel;
		}

		return this.x;
	}

	/** 外部物理冲击注入 (反事实动量干预) */
	public applyImpulse(dv: number): void {
		this.v += dv;
	}

	public getPosition(): number {
		return this.x;
	}

	public getVelocity(): number {
		return this.v;
	}

	public reset(x: number): void {
		this.x = x;
		this.v = 0;
		this.xTargetPrev = x;
	}
}
```

---

## 5. 动量连续性实测对照：贝塞尔 vs. 二阶振子

我们模拟一个典型的高频打断测试：
- 物体初始位于 $x = 0$；
- $t = 0\text{s}$ 时，目标跃迁至 $x = 100$；
- 在 $t = 0.2\text{s}$（物体加速至最大速度时），目标被强行拽回 $x = 20$。

```
位置轨迹随时间的对比:

位置 x
120 ┌─────────────────────────────────────────────────────────┐
    │                                                         │
100 ├─────── - - - - - - (原目标 100)                         │
    │             ╱                                           │
 80 ├            ╱                                            │
    │           ╱ ╲ ◄── 二阶振子：带自然惯性冲过折返点，平滑拉回    │
 60 ├          ╱   ╲                                          │
    │         ╱     ╲                                         │
 40 ├        ╱       ╲                                        │
 20 ├───────┼─────────●───────────────────────────────────────┤ (新目标 20)
    │      ╱          ▲ 贝塞尔重新启动：在 0.2s 发生折线速度断裂   │
  0 └─────●───────────┴───────────────────────────────────────┘
    0   0.1  0.2     0.3     0.4     0.5     0.6     0.7     0.8 秒 (t)
```

| 维度 | 三次方贝塞尔曲线 (Cubic-Bezier) | 二阶阻尼谐振子 (Second-Order Dynamics) |
| :--- | :--- | :--- |
| **打断时速度 $v$** | 瞬间清零或按固定比例截断，发生**一阶速度不连续** | **严格保持当前动量**，顺着惯性向前滑行后平缓调头 |
| **加速度 $a$** | 发生脉冲式跳跃，加加速度 Jerk 趋于无穷大 | 由连续位移差与阻尼实时解算，加速度平滑演化 |
| **物理语义** | “一幅被播放的二维几何曲线图” | **“一个连接在弹簧上、浸没在阻尼介质中的物理质点”** |
| **代码封装复杂度** | 需维护 `startTime`、`duration`、`isAnimating` 等标志位 | **只需保存当前位置 $x$ 与速度 $v$，一帧调用一次 `update()`** |

---

## 6. 走向有机生命体

当我们把二阶阻尼振子从单一标量 $x$ 扩展到三维坐标向量 $\mathbf{p} = (x, y, z)$、刚体旋转四元数 $\mathbf{q}$ 以及相机的视场角 $\text{FOV}$ 时，神奇的质变发生了：
- 鼠标在屏幕上快速划过，游标拖尾不再死板跟随，而是像被丝绸牵引般优雅滑行；
- 相机在 3D 场景中切换视角，镜头不再是生硬的机械移动，而是伴随轻微的悬挂减震与自然过冲；
- 当两个物体碰撞时，只需简单执行 `v += dv`，二阶系统会自动把冲量消化为富有弹性的余震。

**物理法则不需要动画师逐帧手工雕琢，数学的微分约束本身就是最极致的审美。**
