---
title: 有向距离场（SDF）与冲量动力学：连续碰撞检测与动量守恒
published: 2026-11-01T10:10:00+08:00
description: 代码造物与显式世界模型第二卷之四：从显式网格迈入隐式场论，推导有向距离场（SDF）解析几何原语与数值梯度法线，解构高速运动穿墙（Tunneling）与牛顿恢复系数刚体冲量物理碰撞闭环。
format: technical
tags:
  - 代码造物与显式世界模型
  - 有向距离场
  - SDF
  - 碰撞检测
  - 刚体动力学
  - 动量守恒
category: 代码造物与显式世界模型
series: "代码造物与显式世界模型"
seriesOrder: 8
lang: zh
---

在绝大多数三维引擎中，物体是用**显式多边形网格（Explicit Polygonal Meshes）**来表示的：成千上万个顶点 $(x, y, z)$ 加上定义三角形面的索引列表。

当物体静止不动时，多边形网格直观而清晰；但一旦涉及**物理碰撞检测**，显式网格就会暴露出极其繁重的几何负担：
- 为了判断一个小球是否碰到了一个复杂的障碍物，算法必须用射线去和成百上千个三角形面片逐一求交（Ray-Triangle Intersection）；
- 当小球速度极快时，在一个时间步内可能直接从墙的这一侧跃迁到了另一侧，发生著名的**穿墙穿模（Tunneling Effect）**；
- 当碰撞发生后，由于网格由离散的多边形拼凑而成，顶点法线往往不连续，导致计算出的反弹力发生离散的跳变与卡死。

在计算机图形学与现代机器人学中，一种更高雅、更连续的几何数学语言是有向距离场——**SDF（Signed Distance Fields）**。

它不再把几何看作空间中空洞孤立的面片，而是将整个空间变成一个**连续的物理标量场**。结合刚体冲量动力学，十行紧凑的代码就能构建出绝不穿模、严格守恒动量与角动量的物理碰撞世界。

---

## 1. 隐式几何之美：有向距离场（SDF）

一个连续的标量函数 $d(\mathbf{p}): \mathbb{R}^n \to \mathbb{R}$，如果对空间中任意点 $\mathbf{p}$，其函数值恰好代表该点到物体表面 $\partial \Omega$ 的最短欧氏距离，且用正负号区分内外：
$$
d(\mathbf{p}) = \begin{cases} 
+\text{dist}(\mathbf{p}, \partial \Omega) & \mathbf{p} \text{ 位于物体外部} \\
0 & \mathbf{p} \text{ 恰好位于物体表面 (零等值面)} \\
-\text{dist}(\mathbf{p}, \partial \Omega) & \mathbf{p} \text{ 位于物体内部 (浸没深度)}
\end{cases}
$$
则称 $d(\mathbf{p})$ 为物体的**有向距离场（Signed Distance Field）**。

```
                    SDF 标量场等值线剖析
       d = +30 (远离外部)
          d = +10 (接近表面)
 ──────────── d = 0 ────────────► 物体表面真实边界 (Zero Iso-surface)
          d = -10 (内部渗透)
       d = -30 (深层内部)
```

### 1.1 常见解析几何 SDF 原语
感谢图形学大师 Inigo Quilez（ShaderToy 奠基人）的数学梳理，绝大多数基础几何体都拥有极其紧凑的闭式 SDF 解析解：

1. **二维圆（Circle, 中心在原点，半径为 $r$）**：
   $$d_{circle}(\mathbf{p}) = \|\mathbf{p}\| - r$$
2. **胶囊体 / 线段（Capsule, 从点 $\mathbf{a}$ 到点 $\mathbf{b}$，半径为 $r$）**：
   过点 $\mathbf{p}$ 作到线段 $AB$ 的投影截断比率 $h \in [0, 1]$：
   $$h = \text{clamp}\left(\frac{(\mathbf{p} - \mathbf{a}) \cdot (\mathbf{b} - \mathbf{a})}{\|\mathbf{b} - \mathbf{a}\|^2}, 0, 1\right)$$
   $$d_{capsule}(\mathbf{p}) = \|\mathbf{p} - (\mathbf{a} + h(\mathbf{b} - \mathbf{a}))\| - r$$
3. **圆角矩形（Rounded Box, 半宽半高向量 $\mathbf{b}$，圆角半径 $r$）**：
   $$\mathbf{q} = |\mathbf{p}| - \mathbf{b}$$
   $$d_{box}(\mathbf{p}) = \|\max(\mathbf{q}, 0)\| + \min(\max(q_x, q_y), 0) - r$$

### 1.2 免费赠送的高精度表面法向量 $\mathbf{n}$
在显式网格中，计算表面法线往往需要预先烘焙法线贴图或对相邻三角面求面积加权平均；
而在 SDF 场中，**表面的法向量自然就是距离标量场的梯度（Gradient）**！
$$\mathbf{n}(\mathbf{p}) = \frac{\nabla d(\mathbf{p})}{\|\nabla d(\mathbf{p})\|}$$
因为由纯几何距离构成的 SDF 满足程函方程（Eikonal Equation）：$\|\nabla d(\mathbf{p})\| \equiv 1$。
在代码中，我们只需在点 $\mathbf{p}$ 附近施加一个极微小的微元 $\epsilon$（如 $10^{-4}$），利用中心有限差分法，即可在任意位置以极高精度瞬时求出绝对平滑的法线：
$$\nabla d(\mathbf{p}) \approx \begin{bmatrix} \frac{d(\mathbf{p} + \epsilon \hat{\mathbf{x}}) - d(\mathbf{p} - \epsilon \hat{\mathbf{x}})}{2\epsilon} \\ \frac{d(\mathbf{p} + \epsilon \hat{\mathbf{y}}) - d(\mathbf{p} - \epsilon \hat{\mathbf{y}})}{2\epsilon} \end{bmatrix}$$

---

## 2. 连续碰撞检测（CCD）与球体追踪（Sphere Tracing）

为什么高速运动的小球会穿透薄墙？
因为离散碰撞检测只检查了时间步两端的状态 $d(\mathbf{x}_t)$ 与 $d(\mathbf{x}_{t+1})$。如果墙壁厚度为 5 像素，而单步位移为 20 像素，小球直接跃过了 $d \le 0$ 的区域。

利用 SDF，我们可以实现极其健壮的**连续碰撞检测（Continuous Collision Detection, CCD）**：
设质点沿速度方向在时间间隔 $\Delta t$ 内的运动射线为 $\mathbf{x}(\tau) = \mathbf{x}_t + \mathbf{v} \tau$（$\tau \in [0, \Delta t]$）。

```
           球体追踪 (Sphere Tracing / Raymarching) 步进
     起点 X_t
        ●─── r₁ ───►●── r₂ ──►●
        │           │         │
        └─安全气泡─┘         └──► 触碰表面 (d < 1e-4)
```

1. 在当前点计算安全距离 $r_1 = d(\mathbf{x}_t)$。这意味着以 $\mathbf{x}_t$ 为圆心、半径为 $r_1$ 的“安全气泡”内**绝对没有任何障碍物**！
2. 质点可以安全地沿着速度方向瞬间向前推进 $r_1$ 的距离；
3. 在新位置重新采样安全半径 $r_2 = d(\mathbf{x}_{new})$，继续向前跳跃；
4. 只要当前距离 $d < \epsilon$，系统在极短的微秒内精准锁定了发生碰撞的**精确碰撞时刻 $t_{impact} \in [0, \Delta t]$ 与空间接触点**。穿模现象被从数学根源上彻底消灭。

---

## 3. 刚体冲量动力学（Impulse Dynamics）

当碰撞点 $\mathbf{p}_c$ 与法向量 $\mathbf{n}$ 被精准锁定后，两个刚体如何反弹？

在物理学中，碰撞发生的时间极其短暂（毫秒级），碰撞力巨大且呈脉冲状。因此，我们不能再去积分加速度，而是直接求解**瞬时冲量标量 $J$（Impulse Magnitude）**：
$$\mathbf{J} = J \hat{\mathbf{n}}$$

### 3.1 相对法向速度与恢复系数
计算接触点处两物体的相对法向接近速度：
$$v_{rel} = (\mathbf{v}_{1, contact} - \mathbf{v}_{2, contact}) \cdot \hat{\mathbf{n}}$$
- 若 $v_{rel} \ge 0$：两物体正在相互远离，无需施加任何碰撞冲量；
- 若 $v_{rel} < 0$：两物体正在相互接近穿刺，必须施加排斥冲量。

设碰撞的**恢复系数（Restitution Coefficient）**为 $e \in [0, 1]$：
- $e = 1.0$：完全弹性碰撞，动能零损失（如理想钢球）；
- $e = 0.0$：完全非弹性碰撞，两物体贴在一起共同运动（如两团橡皮泥）。

碰撞后的期望法向速度满足牛顿碰撞定律：
$$v_{rel}^+ = -e \cdot v_{rel}^-$$

### 3.2 冲量标量 $J$ 的严密推导
质点 1 与质点 2 的接触点相对于其各自质心的杠杆力臂向量为：
$$\mathbf{r}_1 = \mathbf{p}_c - \mathbf{x}_1, \quad \mathbf{r}_2 = \mathbf{p}_c - \mathbf{x}_2$$
根据经典刚体力学角动量定理，冲量 $\mathbf{J}$ 会同时改变线速度与角速度 $\omega$：
$$
\begin{aligned}
\Delta \mathbf{v}_1 &= +\frac{\mathbf{J}}{m_1}, \quad \Delta \omega_1 = +\frac{\mathbf{r}_1 \times \mathbf{J}}{I_1} \\
\Delta \mathbf{v}_2 &= -\frac{\mathbf{J}}{m_2}, \quad \Delta \omega_2 = -\frac{\mathbf{r}_2 \times \mathbf{J}}{I_2}
\end{aligned}
$$
其中 $I_1, I_2$ 为物体关于转轴的转动惯量（Moment of Inertia）。

联立上述角动量与线动量方程，可直接导出著名的**刚体冲量解析公式**：
$$J = \frac{-(1 + e) v_{rel}}{\frac{1}{m_1} + \frac{1}{m_2} + \frac{(\mathbf{r}_1 \times \hat{\mathbf{n}})^2}{I_1} + \frac{(\mathbf{r}_2 \times \hat{\mathbf{n}})^2}{I_2}}$$

分母上的每一项都有着极其优美的物理意义：
- $\frac{1}{m_1} + \frac{1}{m_2}$：平动有效惯性的倒数；
- 包含杠杆力臂的项：转动有效惯性的倒数。如果碰撞点远离质心（杠杆力臂极大），系统更容易通过剧烈旋转来泄去法向动能，产生的线性反冲冲量就会相应减小。

---

## 4. 完整 TypeScript SDF 与刚体冲量引擎

以下自包含代码构建了一个完整的 2D 隐式物理仿真系统，支持任意解析 SDF 障碍物、表面法线自动差分与冲量反弹：

```ts
/**
 * SDFPhysicsEngine.ts
 * 基于有向距离场 (SDF) 与牛顿冲量定理的刚体碰撞引擎
 */

export interface Vector2D {
	x: number;
	y: number;
}

export interface RigidBody {
	// 质心位置与速度
	pos: Vector2D;
	vel: Vector2D;
	mass: number;
	radius: number; // 质点碰撞包络半径

	// 旋转状态: 角度 theta 与角速度 omega
	theta: number;
	omega: number;
	inertia: number; // 转动惯量

	// 物理材质
	restitution: number; // 弹性恢复系数 e (0.0 ~ 1.0)
	friction: number;    // 摩擦因数
}

export type SDFScene = (p: Vector2D) => number;

export class SDFPhysicsEngine {
	public bodies: RigidBody[] = [];
	public gravity: Vector2D = { x: 0, y: 980 };

	/** 核心算子：有限中心差分法提取任意 SDF 的精确单位表面法向量 */
	public static computeNormal(sdf: SDFScene, p: Vector2D, eps: number = 0.001): Vector2D {
		const dx = sdf({ x: p.x + eps, y: p.y }) - sdf({ x: p.x - eps, y: p.y });
		const dy = sdf({ x: p.x, y: p.y + eps }) - sdf({ x: p.x, y: p.y - eps });
		const len = Math.hypot(dx, dy);
		if (len < 1e-6) return { x: 0, y: -1 };
		return { x: dx / len, y: dy / len };
	}

	/** 常见 SDF 原语库 */
	public static primitives = {
		// 1. 水平地面: y = groundY
		plane: (p: Vector2D, groundY: number): number => p.y - groundY,

		// 2. 静态圆柱: 中心 center，半径 r
		circle: (p: Vector2D, center: Vector2D, r: number): number => {
			return Math.hypot(p.x - center.x, p.y - center.y) - r;
		},

		// 3. 有向矩形盒
		box: (p: Vector2D, center: Vector2D, halfSize: Vector2D): number => {
			const dx = Math.abs(p.x - center.x) - halfSize.x;
			const dy = Math.abs(p.y - center.y) - halfSize.y;
			const outsideDist = Math.hypot(Math.max(dx, 0), Math.max(dy, 0));
			const insideDist = Math.min(Math.max(dx, dy), 0);
			return outsideDist + insideDist;
		},

		// 4. 光滑布尔交并 (Smooth Union, Inigo Quilez 经典多项式平滑)
		smoothUnion: (d1: number, d2: number, k: number = 20.0): number => {
			const h = Math.max(k - Math.abs(d1 - d2), 0) / k;
			return Math.min(d1, d2) - (h * h * k) * 0.25;
		},
	};

	/** 物理单步推进与 SDF 冲量碰撞结算 */
	public step(dt: number, sceneSdf: SDFScene): void {
		for (const b of this.bodies) {
			// 1. 半隐式欧拉外力积分
			b.vel.x += this.gravity.x * dt;
			b.vel.y += this.gravity.y * dt;
			b.pos.x += b.vel.x * dt;
			b.pos.y += b.vel.y * dt;
			b.theta += b.omega * dt;

			// 2. SDF 表面渗透深度探测
			const dist = sceneSdf(b.pos) - b.radius;

			// 如果发生碰撞穿透 (dist <= 0)
			if (dist < 0) {
				// 获取接触表面法线
				const normal = SDFPhysicsEngine.computeNormal(sceneSdf, b.pos);

				// 几何位置硬拉回 (投影出表面，消除微小穿模震颤)
				b.pos.x -= normal.x * dist;
				b.pos.y -= normal.y * dist;

				// 计算接触法向相对速度: v_rel = v · n
				const vRel = b.vel.x * normal.x + b.vel.y * normal.y;

				// 仅当物体正在向内侵入时施加反弹冲量
				if (vRel < 0) {
					// 冲量计算: J = -(1 + e) * vRel / (1/m)
					const impulse = -((1 + b.restitution) * vRel) * b.mass;

					// 更新法向速度
					b.vel.x += (impulse / b.mass) * normal.x;
					b.vel.y += (impulse / b.mass) * normal.y;

					// 切向摩擦阻尼
					const tangent = { x: -normal.y, y: normal.x };
					const vTangent = b.vel.x * tangent.x + b.vel.y * tangent.y;
					const frictionImpulse = -vTangent * b.mass * b.friction;

					b.vel.x += (frictionImpulse / b.mass) * tangent.x;
					b.vel.y += (frictionImpulse / b.mass) * tangent.y;

					// 摩擦力产生的自旋角动量更新
					b.omega += (b.radius * frictionImpulse) / b.inertia;
				}
			}
		}
	}
}
```

---

## 5. 场论与现代 AI 几何的汇流

当我们用有向距离场取代传统顶点三角形时，我们实际上完成了一次深刻的认知范式升级：**世界不再是散落的离散多边形面片，而是空间中处处平滑可微的连续场流形。**

这种思想在今天的尖端 AI 图形学中扮演着统治级地位：
- **神经辐射场与隐式几何（NeRF & NeuS / Instant-NGP）**：不再用 3D 网格建模，而是训练一个神经网络多层感知机（MLP）直接预测空间的 SDF 标量值与颜色，以极高的保真度渲染任意复杂的拓扑结构；
- **机器人抓取规划（SDF-Based Motion Planning）**：机械臂利用障碍物环境的全局 SDF 梯度向量 $\nabla d$，像水流顺着地势滑落一样，自发避开一切碰撞壁障；
- **可微物理仿真（Differentiable Physics）**：由于 SDF 的法线可以通过反向传播自动求导，强化学习智能体能够直接获得“撞墙时动量梯度的解析导数”，在几分钟内学会如何在复杂的流体力学迷宫中自适应穿行。

第二卷至此告一段落。我们从连杆的非线性全纯几何约束，走过了位置动力学的多体松弛投影，最终在有向距离场的隐式连续场中收束了刚体碰撞。
下一卷，我们将离开孤立的机械构件，跃入由群体智能、偏微分方程与连续场交织涌现的宏观心智图景。
