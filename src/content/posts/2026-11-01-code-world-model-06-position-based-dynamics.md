---
title: 位置动力学（PBD）：跳过力求逆的几何约束投影与布料形变
published: 2026-11-01T09:50:00+08:00
description: 代码造物与显式世界模型第二卷之二：解剖传统质点弹簧力学在刚性极端下的刚性微分爆炸，推导位置动力学（Position-Based Dynamics, PBD）的非线性拉格朗日乘子约束投影公式，编写自包含绳索与布料高斯-赛德尔松弛解算器。
format: technical
tags:
  - 代码造物与显式世界模型
  - 位置动力学
  - PBD
  - 布料仿真
  - 约束投影
category: 代码造物与显式世界模型
series: "代码造物与显式世界模型"
seriesOrder: 6
lang: zh
---

在计算机图形学早期，要想在屏幕上模拟一条甩动的绳索、一面随风飘荡的旗帜或一件柔顺的衣服，唯一的学术正统是**质点-弹簧系统（Mass-Spring Systems）**。

在质点弹簧模型中，两个质点之间的连线被视作一个理想胡克弹簧。质点受力为：
$$\mathbf{F}_{ij} = -k (\|\mathbf{x}_i - \mathbf{x}_j\| - d_0) \frac{\mathbf{x}_i - \mathbf{x}_j}{\|\mathbf{x}_i - \mathbf{x}_j\|}$$
然而，几乎所有亲手写过质点弹簧代码的人，都经历过一场噩梦：
如果你把弹簧弹性系数 $k$ 设得很小，布料会像劣质橡胶一样被拉扯得无限长；但只要你把 $k$ 调大（希望它表现得像一条无法拉伸的刚性绳索），常微分方程就会瞬间陷入**刚性病态（Stiff ODE）**。显式欧拉积分器会在下一帧直接把质点弹射到外太空；即使改用无条件稳定的隐式积分器（Backward Euler），也必须在每一帧求解庞大且稀疏的高维刚度切线矩阵逆，不仅极其笨重，而且计算量随质点数量呈立方级飙升。

2006 年，Matthias Müller、Bruno Heidelberger 与 Marcus Hennix 提出了著名的**位置动力学（Position-Based Dynamics, PBD）**，彻底改写了实时软体仿真的规则。

它的哲学极其激进：**既然我们最终关心的是物体的空间几何位置，为什么还要绕一大圈去计算力、力矩与加速度？何不直接在位置层面上执行刚性几何投影？**

---

## 1. PBD 的核心哲学：先做预测，再拉回流形

传统基于力的动力学算法走的是“受力 $\to$ 加速度 $\to$ 速度 $\to$ 位置”的正向演化管线；而 PBD 将物理模拟优雅地切分为两个阶段：

```
                      PBD 核心演化管线
 初始状态 (x_t, v_t)
         │
         ▼ 1. 显式外力无约束预测 (Unconstrained Prediction)
 临时位置 p* = x_t + v_t·Δt + a_ext·Δt²
         │
         ▼ 2. 约束投影多轮松弛 (Gauss-Seidel Constraint Projection)
 修正位置 p_new = p* + Δp  (使得 C(p_new) ≡ 0 严格满足几何约束)
         │
         ▼ 3. 速度隐式反解更新 (Implicit Velocity Derivation)
 最新速度 v_{t+1} = (p_new - x_t) / Δt
 最新位置 x_{t+1} = p_new
```

1. **第一阶段：无约束重力抛射**：每个质点完全无视任何结构连接，只在重力、风力等外力作用下自由飞向临时预测位置 $\mathbf{p}_i^*$；
2. **第二阶段：几何流形投影**：此时物体结构肯定被拉伸撕裂了。PBD 通过迭代松弛，直接将质点位置强制拉扯回满足几何长度约束的有效流形上；
3. **第三阶段：速度修正反算**：最终的速度不再通过加速度积分得到，而是直接根据“修正后的位置与上一帧位置的位移差”隐式反解出来。

这种架构带来了不可思议的绝对稳定性：**系统永远不会发生数值爆炸，无论外部拉扯力多么巨大，几何约束始终以厘米级精度死死锁死构件距离！**

---

## 2. 约束投影数学推导：非线性标量拉格朗日乘子

设系统包含 $N$ 个质点，每个质点的空间坐标为 $\mathbf{p}_i \in \mathbb{R}^3$，质量为 $m_i$，定义质量的倒数为逆质量（Inverse Mass）$w_i = 1 / m_i$（如果质点被固定在天花板上不可移动，则设 $w_i = 0$）。

定义一个标量约束函数 $C(\mathbf{p}_1, \dots, \mathbf{p}_k) \in \mathbb{R}$。我们期望找到一组微小的位移修正量 $\Delta \mathbf{p}_i$，使得修正后的位置严格满足约束：
$$C(\mathbf{p} + \Delta \mathbf{p}) \equiv 0$$

对约束函数进行一阶泰勒展开：
$$C(\mathbf{p} + \Delta \mathbf{p}) \approx C(\mathbf{p}) + \sum_{i} \nabla_{\mathbf{p}_i} C(\mathbf{p}) \cdot \Delta \mathbf{p}_i = 0$$

为了保证系统在修正几何形状时，**不产生任何虚假的虚功、且严格守恒线动量与角动量**，修正位移 $\Delta \mathbf{p}_i$ 必须沿着约束梯度方向，并按质点的逆质量加权分配：
$$\Delta \mathbf{p}_i = - \lambda w_i \nabla_{\mathbf{p}_i} C(\mathbf{p})$$
其中 $\lambda$ 为待求解的标量拉格朗日乘子。

将位移形式代回泰勒展开式中：
$$C(\mathbf{p}) - \sum_{i} \nabla_{\mathbf{p}_i} C(\mathbf{p}) \cdot \left( \lambda w_i \nabla_{\mathbf{p}_i} C(\mathbf{p}) \right) = 0$$
提取公因式 $\lambda$，立刻得到拉格朗日乘子的闭式解：
$$\lambda = \frac{C(\mathbf{p})}{\sum_{j} w_j \|\nabla_{\mathbf{p}_j} C(\mathbf{p})\|^2}$$

最终导出每个质点的**通用位置投影修正算子**：
$$\Delta \mathbf{p}_i = - \frac{w_i}{\sum_{j} w_j \|\nabla_{\mathbf{p}_j} C(\mathbf{p})\|^2} C(\mathbf{p}) \nabla_{\mathbf{p}_i} C(\mathbf{p})$$

---

## 3. 距离约束（Distance Constraint）的具体闭式展开

在模拟绳索、布料或软体网格时，最核心的约束是两个质点之间的**刚性距离约束**：
$$C(\mathbf{p}_1, \mathbf{p}_2) = \|\mathbf{p}_1 - \mathbf{p}_2\| - d_0 = 0$$

现在我们用上述通用公式直接手算其梯度：
1. **关于 $\mathbf{p}_1$ 的梯度向量**：
   $$\nabla_{\mathbf{p}_1} C(\mathbf{p}_1, \mathbf{p}_2) = \frac{\mathbf{p}_1 - \mathbf{p}_2}{\|\mathbf{p}_1 - \mathbf{p}_2\|} = \hat{\mathbf{n}}_{12}$$
   这正好是质点 2 指向质点 1 的单位方向向量 $\hat{\mathbf{n}}_{12}$！
2. **关于 $\mathbf{p}_2$ 的梯度向量**：
   $$\nabla_{\mathbf{p}_2} C(\mathbf{p}_1, \mathbf{p}_2) = -\frac{\mathbf{p}_1 - \mathbf{p}_2}{\|\mathbf{p}_1 - \mathbf{p}_2\|} = -\hat{\mathbf{n}}_{12}$$
3. **梯度的平方范数之和**：
   $$\sum_{j=1}^2 w_j \|\nabla_{\mathbf{p}_j} C\|^2 = w_1 \|\hat{\mathbf{n}}_{12}\|^2 + w_2 \|-\hat{\mathbf{n}}_{12}\|^2 = w_1 + w_2$$

将它们代入通用修正算子，我们得到了令无数图形学工程师击节赞叹的极简位移公式：
$$
\begin{aligned}
\Delta \mathbf{p}_1 &= -\frac{w_1}{w_1 + w_2} \big(\|\mathbf{p}_1 - \mathbf{p}_2\| - d_0\big) \frac{\mathbf{p}_1 - \mathbf{p}_2}{\|\mathbf{p}_1 - \mathbf{p}_2\|} \\
\Delta \mathbf{p}_2 &= +\frac{w_2}{w_1 + w_2} \big(\|\mathbf{p}_1 - \mathbf{p}_2\| - d_0\big) \frac{\mathbf{p}_1 - \mathbf{p}_2}{\|\mathbf{p}_1 - \mathbf{p}_2\|}
\end{aligned}
$$

### 物理直觉解构
- $(\|\mathbf{p}_1 - \mathbf{p}_2\| - d_0)$ 就是当前的误差标量：如果被拉长了，误差为正，两个质点相互靠近；如果被压缩了，误差为负，相互排斥；
- 分母上的权重分配 $\frac{w_1}{w_1 + w_2}$ 严格符合牛顿第三定律反作用力：
  - 如果质点 1 质量极小、质点 2 质量极大（$w_1 \gg w_2$），则几乎全部误差位移都由轻质点 1 独自承受（$\Delta \mathbf{p}_1 \approx -\text{error}$，$\Delta \mathbf{p}_2 \approx 0$）；
  - 如果质点 1 是锚定在墙上的天花板钉子（$w_1 = 0$），则钉子纹丝不动，全部修正量由质点 2 承担。

---

## 4. 高斯-赛德尔迭代（Gauss-Seidel Relaxation）

一条长绳通常由数十个质点和数十个距离约束首尾相连构成；一张布料网格则由数千个三角形网格约束交错纵横。
当修正第 1 个约束时，质点 2 的位置被挪动了；这随即破坏了第 2 个约束原本的距离平衡。

解决这一多体约束耦合的方法，正是经典的**高斯-赛德尔投影法（Gauss-Seidel Iterations）**：
- 在单个时间步内，依次遍历并执行每一个约束投影；
- 每次投影完成后，**立刻将更新后的坐标写入全局数组，供下一个约束读取**；
- 将整套约束列表循环迭代 $K$ 次（通常 $K = 4 \sim 16$）。

随着迭代次数增加，局部的微小几何修正像多米诺骨牌一样迅速传导至整张布料网格，整个系统在数毫秒内迅速收敛至毫米级误差。

---

## 5. 完整 TypeScript 绳索与网格布料解算器

以下代码实现了纯粹、零第三方依赖的完整 2D PBD 解算器，支持多质点悬挂链与布料网格：

```ts
/**
 * PositionBasedDynamics.ts
 * 基于位置动力学 (PBD) 的 2D 绳索与布料高斯-赛德尔约束解算器
 */

export interface Particle {
	x: number;       // 当前确认位置
	y: number;
	prevX: number;   // 历史位置 (用于速度推导)
	prevY: number;
	predX: number;   // 临时预测位置
	predY: number;
	invMass: number; // 逆质量 w = 1/m (0 表示固定钉死不可移动)
}

export interface DistanceConstraint {
	p1Index: number;
	p2Index: number;
	restLength: number;
	stiffness: number; // 刚度系数 (0.0 ~ 1.0)
}

export class PBDSimulation {
	public particles: Particle[] = [];
	public constraints: DistanceConstraint[] = [];

	// 全局物理参数
	public gravity: { x: number; y: number } = { x: 0, y: 980 }; // 像素/秒²
	public damping: number = 0.99; // 空气动量衰减
	public solverIterations: number = 8; // 高斯-赛德尔松弛轮数

	/** 添加质点 */
	public addParticle(x: number, y: number, mass: number = 1.0): number {
		const invMass = mass <= 0 ? 0 : 1 / mass;
		this.particles.push({
			x,
			y,
			prevX: x,
			prevY: y,
			predX: x,
			predY: y,
			invMass,
		});
		return this.particles.length - 1;
	}

	/** 在两质点间建立距离约束 */
	public addDistanceConstraint(i1: number, i2: number, stiffness: number = 1.0): void {
		const p1 = this.particles[i1];
		const p2 = this.particles[i2];
		const d = Math.hypot(p1.x - p2.x, p1.y - p2.y);
		this.constraints.push({
			p1Index: i1,
			p2Index: i2,
			restLength: d,
			stiffness,
		});
	}

	/** 核心物理推进单步 */
	public step(dt: number): void {
		if (dt <= 0) return;

		// 阶段 1: 无约束外力预测 (Verlet 速度与重力积分)
		for (const p of this.particles) {
			if (p.invMass === 0) {
				p.predX = p.x;
				p.predY = p.y;
				continue;
			}

			// 隐式提取上一帧速度并施加阻尼
			const vx = ((p.x - p.prevX) / dt) * this.damping;
			const vy = ((p.y - p.prevY) / dt) * this.damping;

			// 外力加速度积分预测
			p.predX = p.x + vx * dt + this.gravity.x * dt * dt;
			p.predY = p.y + vy * dt + this.gravity.y * dt * dt;
		}

		// 阶段 2: 高斯-赛德尔约束投影松弛
		for (let iter = 0; iter < this.solverIterations; iter++) {
			for (const c of this.constraints) {
				const p1 = this.particles[c.p1Index];
				const p2 = this.particles[c.p2Index];

				const dx = p1.predX - p2.predX;
				const dy = p1.predY - p2.predY;
				const currentDist = Math.hypot(dx, dy);

				if (currentDist < 1e-6) continue;

				// 计算约束误差: C = distance - restLength
				const deltaDist = currentDist - c.restLength;
				const wSum = p1.invMass + p2.invMass;
				if (wSum === 0) continue;

				// 沿法线方向的位移修正因子
				const normalX = dx / currentDist;
				const normalY = dy / currentDist;
				const impulse = (deltaDist / wSum) * c.stiffness;

				// 按逆质量比例实时投影拉回
				if (p1.invMass > 0) {
					p1.predX -= p1.invMass * impulse * normalX;
					p1.predY -= p1.invMass * impulse * normalY;
				}
				if (p2.invMass > 0) {
					p2.predX += p2.invMass * impulse * normalX;
					p2.predY += p2.invMass * impulse * normalY;
				}
			}
		}

		// 阶段 3: 状态确认与速度隐式反解
		for (const p of this.particles) {
			if (p.invMass === 0) continue;

			// 记住当前确认位置作为下一帧的历史前值
			p.prevX = p.x;
			p.prevY = p.y;

			// 更新最终位置
			p.x = p.predX;
			p.y = p.predY;
		}
	}
}
```

---

## 6. 从物理引擎到生成式设计

位置动力学（PBD）的深远意义远不止于游戏里角色的斗篷摆动。它开创了一种全新的计算范式：**将复杂的物理动力学问题，降维为一个纯几何最优化问题**。

现代的三维建模工具、参数化建筑设计以及蛋白质分子折叠模拟，都在大量汲取 PBD 的思想：
- **碰撞与自穿透防护**：直接给质点加一条“两球距离不得小于直径”的单侧不等式约束；
- **体积与不可压缩性**：给四面体网格加一条行列式体积约束 $C(\mathbf{p}) = \det(J) - V_0 \equiv 0$；
- **机器人防撕裂电缆**：为机械手臂的线缆设定最大弯曲曲率约束。

当我们摆脱了对力的沉迷，转而用**几何投影**与**因果约束**来编织代码时，虚拟世界不再是一堆脆弱易碎的微分方程火药桶，而是一张任由狂风骤雨撕扯、却永远紧密自洽的数字丝绸。
