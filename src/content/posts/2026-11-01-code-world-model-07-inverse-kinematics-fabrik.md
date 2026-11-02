---
title: 前向后向逆运动学（FABRIK）：程序化多足生物步态合成
published: 2026-11-01T10:00:00+08:00
description: 代码造物与显式世界模型第二卷之三：剖析传统雅可比矩阵伪逆在逆运动学中的奇异性与矩阵求逆算力痛点，推导无三角函数与无矩阵的极简 FABRIK 几何迭代算法，实现零美术关键帧的程序化多足蜘蛛地形自适应步态机。
format: technical
tags:
  - 代码造物与显式世界模型
  - 逆运动学
  - FABRIK
  - 程序化步态
  - 骨骼动画
category: 代码造物与显式世界模型
series: "代码造物与显式世界模型"
seriesOrder: 7
lang: zh
---

在游戏工业与影视动画制作中，最昂贵的资产之一是**关键帧动画切片（Baked Animation Clips）**：美术师手工为游戏角色制作“向前走”、“向后跑”、“上楼梯”等数十段动作循环。

然而，一旦将这些预先烘焙的动画放入真实不规则的物理世界，尴尬的破绽随处可见：
- 当角色走在一个 $30^\circ$ 的碎石斜坡上时，双脚要么诡异地悬空踩在空气中，要么膝盖以下直接陷进地面模型里（“脚滑”与“穿模”）；
- 当角色的手臂伸手去抓一个动态移动的把手时，如果把手不在预录动画的精确坐标上，手臂只能生硬地抓在虚空中。

要让机械生物像真实的活物一样，无论地面多么崎岖、目标如何闪转腾挪，足端都能精准踏在岩石凸起上，我们必须赋予代码**逆运动学（Inverse Kinematics, IK）**与**程序化步态合成（Procedural Locomotion）**的能力。

---

## 1. 运动学的两面：顺理成章的正向 vs. 充满歧义的逆向

一条由 $N$ 节刚性骨骼构成的机械臂，各节长度为 $d_i$，各关节旋转角为 $\theta_i$。

```
                    正向与逆向运动学问题映射
 正向运动学 (Forward Kinematics, FK): [已知 θ₁, θ₂, θ₃] ──► [唯一求出 末端手掌位置 P]
 逆向运动学 (Inverse Kinematics, IK): [已知 目标位置 Target] ──► [反求 关节角 θ₁, θ₂, θ₃] (病态非线性多解)
```

1. **正向运动学（FK）**：已知所有关节的旋转角 $\theta$，求解末端执行器（End-Effector）的空间坐标。这是一个平庸的正向投影问题，只需从根骨骼开始依次做向量平移与旋转累加即可，解是**绝对唯一**的。
2. **逆向运动学（IK）**：已知手掌必须到达目标点 $\mathbf{p}_{target}$，反求各个关节的弯曲角度。
   这是一个高维非线性方程组反解问题。通常，当自由度大于 3 时，满足条件的解有**无穷多个**（想想你的手掌贴在桌面上，手肘可以随意上下晃动）；当目标超出骨骼总长度时，方程**完全无解**；而在某些特殊姿态下，系统会陷入**奇异位形（Singularity）**，导致计算出的角速度趋于无穷大。

---

## 2. 传统雅可比矩阵求逆的痛点

在 2011 年之前，工业界最主流的数值 IK 方案基于**雅可比矩阵（Jacobian Matrix）**：
末端位移微分与关节角微分满足线性映射：
$$\Delta \mathbf{x} = \mathbf{J}(\boldsymbol{\theta}) \Delta \boldsymbol{\theta}, \quad \text{其中 } J_{ij} = \frac{\partial x_i}{\partial \theta_j}$$

为了反求 $\Delta \boldsymbol{\theta}$，我们必须对矩阵 $\mathbf{J}$ 求逆。由于骨骼自由度通常不等于空间坐标维度，必须使用摩尔-彭罗斯伪逆（Moore-Penrose Pseudo-Inverse）：
$$\Delta \boldsymbol{\theta} = \mathbf{J}^\dagger \Delta \mathbf{x} = \mathbf{J}^T (\mathbf{J} \mathbf{J}^T)^{-1} \Delta \mathbf{x}$$

这套方案在工程实践中令人痛苦不堪：
- **算力开销巨大**：在每一帧动画中，必须计算大量三角函数（$\sin, \cos$）填充雅可比矩阵，并执行耗时的矩阵相乘与求逆；
- **奇异性爆炸**：当手臂完全伸直时，$\mathbf{J} \mathbf{J}^T$ 变得不可逆，求出的关节角速度会暴增至无穷大，导致手臂在空中剧烈癫狂抖动；
- **阻尼调参地狱**：为了抑制抖动，工程师不得不引入阻尼最小二乘法（DLS），而阻尼因子 $\lambda$ 的大小几乎无法在全空间取得完美平衡。

---

## 3. FABRIK 算法：无三角函数、无矩阵求逆的几何奇迹

2011 年，剑桥大学学者 Andreas Aristidou 与 Joan Lasenby 提出了 **FABRIK（Forward And Backward Reaching Inverse Kinematics）** 算法。

FABRIK 完全摒弃了旋转角度与矩阵微分，**直接在三维关节点的位置坐标上，通过类似拉直晾衣绳的直觉，交替执行前向与后向两次拉扯迭代**：

```
                FABRIK 算法单轮迭代几何过程
 初始骨骼链:    P0(根) ── P1 ── P2 ── P3(末端)         ★ Target
                                                      
 阶段 1: 前向拉伸 (Forward Reaching)
 (强制令末端贴住目标，从末端向根部反向投影连线)
                P0' ◄── P1' ◄── P2' ◄── P3' = Target
                (此时根部 P0' 发生漂移离位)

 阶段 2: 后向拉伸 (Backward Reaching)
 (强制将根部钉回原地，从根部向末端正向投影连线)
                P0(钉死) ──► P1'' ──► P2'' ──► P3'' (极大逼近 Target)
```

### 3.1 极限检查：目标不可达判定（Unreachable Target）
在开始迭代前，先计算根节点到目标点的欧氏距离：
$$D = \|\mathbf{p}_{target} - \mathbf{p}_0\|$$
如果 $D \ge \sum_{i=0}^{N-1} d_i$（目标超出了整条链的最大延伸长度）：
此时链条完全无法弯曲，直接将每个关节沿着从 $\mathbf{p}_0$ 指向 $\mathbf{p}_{target}$ 的直线拉直即可：
$$\mathbf{p}_{i+1} = \mathbf{p}_i + d_i \frac{\mathbf{p}_{target} - \mathbf{p}_i}{\|\mathbf{p}_{target} - \mathbf{p}_i\|}$$
完全不需要任何复杂的约束解算。

### 3.2 阶段一：前向拉伸（从末端倒推回根部）
当目标可达时，我们强制命令末端执行器瞬间瞬移至目标位置：
$$\mathbf{p}_N' = \mathbf{p}_{target}$$
然后从倒数第二个关节开始，沿着它与上一关节的连线，强制拉回骨骼原始长度 $d_i$：
$$\mathbf{p}_i' = \mathbf{p}_{i+1}' + d_i \frac{\mathbf{p}_i - \mathbf{p}_{i+1}'}{\|\mathbf{p}_i - \mathbf{p}_{i+1}'\|}, \quad \text{for } i = N-1, N-2, \dots, 0$$
迭代到达根部后，我们惊奇地发现所有骨骼的长度都被完美保留，但代价是**整个骨骼的根部 $\mathbf{p}_0'$ 被拽离了它原本在身体上的锚定点**。

### 3.3 阶段二：后向拉伸（从根部正推回末端）
为了修正根部的位移，我们强制将根节点钉死回原本的真实底座：
$$\mathbf{p}_0'' = \mathbf{p}_{origin}$$
然后从第一个活动关节开始，正向推算后续节点：
$$\mathbf{p}_{i+1}'' = \mathbf{p}_i'' + d_i \frac{\mathbf{p}_{i+1}' - \mathbf{p}_i''}{\|\mathbf{p}_{i+1}' - \mathbf{p}_i''\|}, \quad \text{for } i = 0, 1, \dots, N-1$$

### 3.4 收敛速度
完成一次前向+后向拉扯，仅仅需要几行简单的向量加减法与归一化。
数学上证明，**FABRIK 在两到三次迭代内，末端误差就能收敛至 $10^{-4}$ 米以内**，收敛速度与平滑度在绝大多数场景下完胜传统雅可比求逆数个数量级！

---

## 4. 程序化步态合成：多足生物的零动画自适应

有了 FABRIK，我们要如何让一只六足机械蜘蛛（Hexapod）在完全没有美术关键帧的前提下，自然在起伏的地形上爬行？

整个步态状态机的架构极其优美清晰：

```
                      程序化多足步态闭环
                用户输入 / 路径导航速度 v_body
                             │
                             ▼
                    [身体躯干中心移动]
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   [左侧足组 (腿 1, 3, 5)]          [右侧足组 (腿 2, 4, 6)]
   处于【支撑相 (Stance)】          处于【摆动相 (Swing)】
   • 足端牢牢钉在世界地面坐标       • 抬离地面，沿抛物线跨步
   • 身体向前移动时，足端相对       • 寻找前瞻最佳着陆点 (Target)
     躯干向后运动，FABRIK 反算        落地后切换为支撑相，交替循环
     各腿关节弯曲姿态
```

### 4.1 步态触发器（Step Trigger Condition）
每条腿在身体局部坐标系中，都有一个由解剖学决定的“理想自然静止点” $\mathbf{p}_{rest}$。
随着身体向前平移，身体坐标系在世界中移动，导致理想落地点投影到世界地面的期望位置为：
$$\mathbf{p}_{desired} = \mathbf{x}_{body} + \mathbf{R}_{body} \mathbf{p}_{rest} + \tau_{lead} \mathbf{v}_{body}$$
其中 $\tau_{lead}$ 是前瞻预判时间因子（身体跑得越快，步子邁得越远）。

足端当前在世界中固定的接触点为 $\mathbf{p}_{foot}$。当两者的距离超过了步幅阈值：
$$\|\mathbf{p}_{foot} - \mathbf{p}_{desired}\| > R_{step}$$
系统触发该腿的**迈步事件（Step Event）**！

### 4.2 摆动抛物线与交替三足步态（Alternating Tripod Gait）
在昆虫六足行走中，为了保持静力学稳定，腿被严格分为两组（三足交替步态）：
- 组 A：左前、右中、左后；
- 组 B：右前、左中、右后。
同一时刻只有一组迈步，另一组三足构成稳定的三角支撑面。
迈步过程中，足端沿着一条三次方贝塞尔曲线在空中划过一条优雅的抛物线：
$$\mathbf{p}_{foot}(t) = (1-t) \mathbf{p}_{start} + t \mathbf{p}_{desired} + \mathbf{n}_{ground} \cdot h_{step} \cdot 4t(1-t)$$
每一帧将该空间坐标输入给 FABRIK，机械腿的三节关节就会随着地形的高低起伏，自动弯曲出天衣无缝的攀爬姿态。

---

## 5. 完整 TypeScript FABRIK 与骨骼步态实现

以下代码实现了纯粹的 2D/3D 通用 FABRIK 逆运动学链条与多足迈步状态机：

```ts
/**
 * FabrikKinematics.ts
 * 纯几何无矩阵 FABRIK 逆运动学解算器与程序化步态生成
 */

export interface Vector2D {
	x: number;
	y: number;
}

export class FabrikChain {
	public joints: Vector2D[] = [];
	public lengths: number[] = [];
	public totalLength: number = 0;

	// 容差与最大迭代次数
	public tolerance: number = 0.5; // 像素级精度
	public maxIterations: number = 10;

	constructor(jointCoordinates: Vector2D[]) {
		this.joints = jointCoordinates.map((p) => ({ ...p }));
		this.totalLength = 0;
		for (let i = 0; i < this.joints.length - 1; i++) {
			const d = Math.hypot(
				this.joints[i + 1].x - this.joints[i].x,
				this.joints[i + 1].y - this.joints[i].y
			);
			this.lengths.push(d);
			this.totalLength += d;
		}
	}

	/** 执行 FABRIK 逆运动学求解，使末端贴合 targetPosition */
	public solve(target: Vector2D, rootOrigin: Vector2D): void {
		const n = this.joints.length;
		if (n === 0) return;

		// 检查目标可达性
		const distToTarget = Math.hypot(target.x - rootOrigin.x, target.y - rootOrigin.y);

		// 情况 1: 目标超出范围，整条链条沿直线拉伸
		if (distToTarget >= this.totalLength) {
			this.joints[0] = { ...rootOrigin };
			for (let i = 0; i < n - 1; i++) {
				const dx = target.x - this.joints[i].x;
				const dy = target.y - this.joints[i].y;
				const d = Math.hypot(dx, dy);
				const ratio = this.lengths[i] / d;
				this.joints[i + 1] = {
					x: this.joints[i].x + dx * ratio,
					y: this.joints[i].y + dy * ratio,
				};
			}
			return;
		}

		// 情况 2: 目标可达，执行前向-后向交替迭代
		let iter = 0;
		let lastEnd = this.joints[n - 1];
		let error = Math.hypot(target.x - lastEnd.x, target.y - lastEnd.y);

		while (error > this.tolerance && iter < this.maxIterations) {
			// 阶段 1: 前向拉伸 (Forward Reaching)
			this.joints[n - 1] = { ...target };
			for (let i = n - 2; i >= 0; i--) {
				const dx = this.joints[i].x - this.joints[i + 1].x;
				const dy = this.joints[i].y - this.joints[i + 1].y;
				const d = Math.hypot(dx, dy);
				const ratio = this.lengths[i] / d;
				this.joints[i] = {
					x: this.joints[i + 1].x + dx * ratio,
					y: this.joints[i + 1].y + dy * ratio,
				};
			}

			// 阶段 2: 后向拉伸 (Backward Reaching)
			this.joints[0] = { ...rootOrigin };
			for (let i = 0; i < n - 1; i++) {
				const dx = this.joints[i + 1].x - this.joints[i].x;
				const dy = this.joints[i + 1].y - this.joints[i].y;
				const d = Math.hypot(dx, dy);
				const ratio = this.lengths[i] / d;
				this.joints[i + 1] = {
					x: this.joints[i].x + dx * ratio,
					y: this.joints[i].y + dy * ratio,
				};
			}

			lastEnd = this.joints[n - 1];
			error = Math.hypot(target.x - lastEnd.x, target.y - lastEnd.y);
			iter++;
		}
	}
}

/** 单条程序化步态机械腿 */
export class ProceduralLeg {
	public ikChain: FabrikChain;
	public footPosition: Vector2D;
	public isStepping: boolean = false;

	private stepProgress: number = 0;
	private stepStartPos: Vector2D = { x: 0, y: 0 };
	private stepTargetPos: Vector2D = { x: 0, y: 0 };

	constructor(jointCoords: Vector2D[]) {
		this.ikChain = new FabrikChain(jointCoords);
		this.footPosition = { ...jointCoords[jointCoords.length - 1] };
	}

	/** 检查是否需要迈步并在空中插值 */
	public update(
		dt: number,
		rootWorldPos: Vector2D,
		desiredLandingPos: Vector2D,
		stepTriggerRadius: number = 40.0,
		stepDurationSec: number = 0.2
	): void {
		// 1. 如果正在摆动相中，沿抛物线更新足端在空中的位置
		if (this.isStepping) {
			this.stepProgress += dt / stepDurationSec;
			if (this.stepProgress >= 1.0) {
				this.stepProgress = 1.0;
				this.isStepping = false;
				this.footPosition = { ...this.stepTargetPos };
			} else {
				const t = this.stepProgress;
				// 线性水平插值 + 垂直抛物线高度提升
				const liftHeight = 25.0 * 4 * t * (1 - t);
				this.footPosition = {
					x: this.stepStartPos.x * (1 - t) + this.stepTargetPos.x * t,
					y: this.stepStartPos.y * (1 - t) + this.stepTargetPos.y * t - liftHeight,
				};
			}
		} else {
			// 2. 处于支撑相：检查当前落地脚与期望落地脚之间的拉扯距离
			const dist = Math.hypot(
				this.footPosition.x - desiredLandingPos.x,
				this.footPosition.y - desiredLandingPos.y
			);
			if (dist > stepTriggerRadius) {
				this.isStepping = true;
				this.stepProgress = 0;
				this.stepStartPos = { ...this.footPosition };
				this.stepTargetPos = { ...desiredLandingPos };
			}
		}

		// 3. 将最新的根底座与足端输入 FABRIK 完成逆运动学求解
		this.ikChain.solve(this.footPosition, rootWorldPos);
	}
}
```

---

## 6. 从死板皮套到鲜活生物

传统计算机图形学常常把动画当成一种“自顶向下的提线木偶戏”：每一帧旋转几度都是导演在时间轴上画死的。
而 FABRIK 与程序化步态展示的是一种**自底向上的生命协调策略（Biological Coordination Policy）**：

- **躯干只负责确定前进意图**（宏观目标：向北移动）；
- **步态机负责仲裁步调节奏**（中观相位：交替离地）；
- **FABRIK 负责完成几何保真**（微观约束：关节自然弯曲适应地面）。

这种分层解耦的运动控制结构，与生物脊椎动物的神经控制体系惊人同构：大脑皮层负责发出方向指令，脊髓中央模式发生器（CPG）维持摆腿律动，而骨骼本体的逆运动学与韧带几何约束，则将地面的一切崎岖不平默默消化于无形之中。
