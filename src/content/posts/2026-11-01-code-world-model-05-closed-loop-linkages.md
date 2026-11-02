---
title: 闭环连杆与解析几何约束：圆交点闭式解与仿生步态
published: 2026-11-01T09:40:00+08:00
description: 代码造物与显式世界模型第二卷之一：解剖闭环机械连杆的运动学代数流形，推导格吕布勒（Grübler）自由度判据、两圆交点向量闭式解与死点奇异性，用代码复刻 Theo Jansen 仿生风力兽十一圣数足端轨迹。
format: technical
tags:
  - 代码造物与显式世界模型
  - 连杆机构
  - 解析几何
  - 仿生步态
  - Strandbeest
category: 代码造物与显式世界模型
series: "代码造物与显式世界模型"
seriesOrder: 5
lang: zh
---

在机器人学与三维动画中，绝大多数人接触到的都是**开链运动学（Open Kinematic Chains）**：手臂连接前臂，前臂连接手掌，每一节骨骼相对上一节旋转一个角度 $\theta_i$。计算末端手指的位置极其简单，只需将一系列局部的旋转平移齐次变换矩阵 $\mathbf{T}_1 \mathbf{T}_2 \cdots \mathbf{T}_n$ 依次相乘即可。

然而，一旦骨骼的末端被强制用铰链锁死在地面上，或者多个部件相互成环形成**闭环连杆机构（Closed-Loop Linkage Mechanisms）**时，开链运动学的简单级联会瞬间坍塌。

闭环连杆不再允许每个关节自由随意旋转。各个杆件的长度构成了极其严苛的**非线性全纯代数约束（Holonomic Algebraic Constraints）**。一个曲柄在中央旋转，远端十几根互相咬合的刚性杆件就必须在瞬间完成联立几何方程的自洽解算。

荷兰艺术家兼工程师泰奥·扬森（Theo Jansen）创造的风力仿生兽（Strandbeest），正是闭环连杆将简单单自由度圆周旋转转化为复杂四足生命平底步态的工业奇迹。

---

## 1. 机构自由度：格吕布勒公式（Grübler's Formula）

在用代码构建连杆世界前，首要任务是判断这个连杆机构究竟能不能动？会不会被多余的死锁约束锁死成毫无弹性的刚架？

对于平面连杆机构，其自由度 $M$（Mobility / Degrees of Freedom）由**切比雪夫-格吕布勒-库茨巴赫公式（Chebyshev–Grübler–Kutzbach Criterion）**严格判定：
$$M = 3(N - 1) - 2J_1 - J_2$$
其中：
- $N$ 为机构中的总构件数（包括固定不动的机架 Ground Frame，故可动构件数为 $N-1$）；
- $J_1$ 为单自由度低副数目（在平面中通常为旋转铰链 Revolve Joint，每个铰链剥夺 2 个平移自由度，保留 1 个转动自由度）；
- $J_2$ 为两自由度高副数目（如齿轮啮合、凸轮接触，平面连杆中通常为 0）。

```
        格吕布勒自由度判定分类
   M = 0 (静定桁架)          M = 1 (单自由度机构)          M < 0 (过约束超静定)
        ▲                         ▲                           ▲
       ╱ ╲                       ╱ ╲                         ╱|╲
      ╱   ╲                     ╱   ╲                       ╱ | ╲
     ●─────●                   ●─────●                     ●──┼──●
   三根杆件铰接: 锁死        四杆机构: 唯一确定运动轨迹    加斜撑: 几何超静定
```

以经典的平面四杆机构为例：$N = 4$（1 个机架 + 3 根活动连杆），铰链数 $J_1 = 4$。
代入公式：
$$M = 3(4 - 1) - 2(4) - 0 = 9 - 8 = 1$$
**自由度恰好为 1！** 这意味着我们只需要一个电机（或者一根驱动曲柄）提供单一的输入角度 $\theta_{input}$，整套机构中所有构件的位置、姿态和角速度就会被全纯代数约束**百分之百唯一确定**。

---

## 2. 闭环连杆解算的核心：两圆交点向量闭式解

在代码中解闭环连杆，绝对不要使用低效的数值优化或多项式求根算法。对于由刚性杆构成的三角形与四边形环路，其最底层的几何原子是：**已知空间中两个定点 $A$ 和 $B$，以及第三点 $C$ 分别到 $A$ 与 $B$ 的刚性杆长 $r_a, r_b$，求 $C$ 点的坐标。**

这在代数上等价于联立求解两个圆的交点：
$$
\begin{aligned}
(x_c - x_a)^2 + (y_c - y_a)^2 &= r_a^2 \\
(x_c - x_b)^2 + (y_c - y_b)^2 &= r_b^2
\end{aligned}
$$

```
                两圆交点解析几何向量三角解
                       C_up (左手分支 branch = +1)
                      ▲
                     ╱ ╲
                 r_a╱   ╲ r_b
                   ╱  h  ╲
                  ●───┼───●
                 A    P    B
                      │
                     d│
                      ▼
                       C_down (右手分支 branch = -1)
```

### 向量闭式推导步骤
1. **基底向量与中心距**：
   计算 $A$ 到 $B$ 的位移向量与欧氏距离：
   $$\mathbf{d}_{ab} = \mathbf{p}_b - \mathbf{p}_a, \quad d = \|\mathbf{d}_{ab}\|$$
2. **奇异性死点判定（Singularity & Lockup）**：
   如果连杆尺寸不匹配，两圆可能无法相交：
   - 若 $d > r_a + r_b$：两连杆太短，构件被拉扯撕裂；
   - 若 $d < |r_a - r_b|$：一连杆包含在另一连杆内部，无法闭合；
   - 若 $d = 0$：两基点重合，方程退化。
   **此时系统陷入奇异死点（Singular Dead Point），代码必须抛出约束失效异常或触发机构卡死保护。**
3. **投影距离 $a$ 与垂直弦长 $h$**：
   过 $C$ 作 $AB$ 连线的垂足 $P$，设 $AP$ 长度为 $a$。由勾股定理：
   $$r_a^2 - a^2 = h^2 = r_b^2 - (d - a)^2$$
   展开消元解得标量投影：
   $$a = \frac{r_a^2 - r_b^2 + d^2}{2d}, \quad h = \sqrt{\max(0, r_a^2 - a^2)}$$
4. **单位基底与正交旋转向量**：
   定义沿 $AB$ 方向的单位基底 $\hat{\mathbf{u}}$ 以及逆时针旋转 90 度的正交法向量 $\hat{\mathbf{u}}_\perp$：
   $$\hat{\mathbf{u}} = \frac{\mathbf{d}_{ab}}{d} = \begin{bmatrix} d_x / d \\ d_y / d \end{bmatrix}, \quad \hat{\mathbf{u}}_\perp = \begin{bmatrix} -d_y / d \\ d_x / d \end{bmatrix}$$
5. **双分支闭式解**：
   $$C = \mathbf{p}_a + a \hat{\mathbf{u}} \pm h \hat{\mathbf{u}}_\perp$$
   其中 $\pm$ 符号对应机构的**装配模式分支（Branch Mode）**。同一个机构存在“向上拱起”与“向下弯折”两种解，在连续动画中必须保持分支符号不变，否则构件会瞬间穿透自身发生翻折断裂。

---

## 3. 解构泰奥·扬森（Theo Jansen）风力兽腿部机构

Strandbeest 的腿部机构由一个旋转曲柄驱动，配合固定在机架上的两个支点，驱动由 8 根连杆构成的闭环运动学链。Jansen 通过遗传算法在 Atari 电脑上迭代数月，筛选出了一组能够走出平底直线的**“十一圣数”（11 Holy Numbers）**。

```
              Strandbeest 单腿运动学拓扑依赖图
                      [机架支点 (0, 0)] (定点)
                              │
                    a (曲柄)  │ 旋转角 θ
                              ▼
                        [曲柄末端点 M]
                        ╱            ╲
                   b ╱                ╲ j
                    ▼                  ▼
           [铰链节点 K] ◄──── c ──── [铰链节点 L]
               │                         │
             d │                       k │
               ▼                         ▼
         [中间三角顶点 N]               [小腿顶点 P]
               ╲                         ╱
                e ╲                     ╱ l
                   ▼                   ▼
                     [足端落地接触点 FOOT]
```

### 为什么足端轨迹（Footpath）如此神奇？
一个纯圆周旋转（曲柄匀速转动 $\theta(t) = \omega t$），通过上述 8 杆闭环联立解算后，足端轨迹在地面上呈现出**三角形或熨斗形封闭环线**：
1. **支撑相（Stance Phase）**：在底部贴近地面处，轨迹是一条几乎绝对平直的水平线段，并且以近乎恒定的速度向后移动。这保证了仿生兽行走时重心高度没有任何上下颠簸，宛如行驶在铁轨上的车轮；
2. **摆动相（Swing Phase）**：在行至末端后，小腿迅速抬起跨过障碍物，以极快的速度向前伸展后重新踏实地面。

通过巧妙的几何尺度配比，纯粹的刚性木条与塑料管在没有任何电子元件控制的情况下，在物理层面内嵌了高级动物的“抬腿-迈步-平稳支撑”程序。

---

## 4. 完整 TypeScript 连杆解析器实现

以下代码实现了严苛的无奇异圆交点向量求解器，并完整复刻了 Strandbeest 腿部运动学闭式解：

```ts
/**
 * ClosedLoopLinkage.ts
 * 闭环连杆机构解析求解器与 Strandbeest 十一圣数仿生步态生成
 */

export interface Vector2D {
	x: number;
	y: number;
}

export interface StrandbeestLinkageLengths {
	a: number; // 曲柄长度 (15.0)
	b: number; // (41.5)
	c: number; // (39.3)
	d: number; // (40.1)
	e: number; // (55.8)
	f: number; // (39.4)
	g: number; // (36.7)
	h: number; // (65.7)
	i: number; // (49.0)
	j: number; // (50.0)
	k: number; // (61.9)
	l: number; // (7.8)
	m: number; // (15.0)
}

// 泰奥·扬森标准十一圣数比例 (乘以缩放系数 2.0 便于 Canvas 绘制)
export const JANSEN_HOLY_NUMBERS: StrandbeestLinkageLengths = {
	a: 38.0,
	b: 41.5 * 2,
	c: 39.3 * 2,
	d: 40.1 * 2,
	e: 55.8 * 2,
	f: 39.4 * 2,
	g: 36.7 * 2,
	h: 65.7 * 2,
	i: 49.0 * 2,
	j: 50.0 * 2,
	k: 61.9 * 2,
	l: 7.8 * 2,
	m: 15.0 * 2,
};

export class LinkageKinematics {
	/**
	 * 核心向量算子：两圆相交解算第三铰链点坐标
	 * @param pA 基点 A 坐标
	 * @param pB 基点 B 坐标
	 * @param rA 杆件 AC 长度
	 * @param rB 杆件 BC 长度
	 * @param branch +1 取正分支 (左手系)，-1 取负分支 (右手系)
	 */
	public static circleIntersection(
		pA: Vector2D,
		pB: Vector2D,
		rA: number,
		rB: number,
		branch: 1 | -1 = 1
	): Vector2D {
		const dx = pB.x - pA.x;
		const dy = pB.y - pA.y;
		const d = Math.hypot(dx, dy);

		// 奇异性防线：两连杆长度无法闭合
		if (d > rA + rB || d < Math.abs(rA - rB) || d === 0) {
			throw new Error(`Linkage singularity: distance ${d.toFixed(2)} incompatible with lengths ${rA}, ${rB}`);
		}

		// 投影距离 a 与垂线高 h
		const a = (rA * rA - rB * rB + d * d) / (2 * d);
		const hSq = rA * rA - a * a;
		const h = Math.sqrt(Math.max(0, hSq));

		// 单位向量与正交单位向量
		const ux = dx / d;
		const uy = dy / d;
		const perpX = -uy * branch;
		const perpY = ux * branch;

		// 向量加权合成第三点
		return {
			x: pA.x + a * ux + h * perpX,
			y: pA.y + a * uy + h * perpY,
		};
	}

	/**
	 * 解算 Strandbeest 单腿在给定曲柄角 theta 下的所有关节空间坐标
	 * @param crankAngleRad 曲柄旋转角 (弧度)
	 */
	public static solveStrandbeestLeg(
		crankAngleRad: number,
		lens: StrandbeestLinkageLengths = JANSEN_HOLY_NUMBERS
	): {
		crankCenter: Vector2D;
		fixedPivot: Vector2D;
		crankPin: Vector2D;
		jointK: Vector2D;
		jointL: Vector2D;
		foot: Vector2D;
	} {
		// 机架两个固定基点: 原点与固定悬挂轴心
		const crankCenter: Vector2D = { x: 0, y: 0 };
		const fixedPivot: Vector2D = { x: -38.0, y: -7.8 };

		// 1. 曲柄末端销钉位置 M
		const crankPin: Vector2D = {
			x: crankCenter.x + lens.a * Math.cos(crankAngleRad),
			y: crankCenter.y + lens.a * Math.sin(crankAngleRad),
		};

		// 2. 解算第一闭环三角形节点 K (由 fixedPivot 和 crankPin 约束)
		const jointK = this.circleIntersection(fixedPivot, crankPin, lens.b, lens.j, -1);

		// 3. 解算第二闭环三角形节点 L (由 fixedPivot 和 crankPin 约束)
		const jointL = this.circleIntersection(fixedPivot, crankPin, lens.c, lens.d, 1);

		// 4. 解算小腿下部三角形连接点，最终交汇于足端接触点 (Foot)
		const jointBottom = this.circleIntersection(jointK, jointL, lens.e, lens.f, -1);
		const foot = this.circleIntersection(jointL, jointBottom, lens.g, lens.h, -1);

		return {
			crankCenter,
			fixedPivot,
			crankPin,
			jointK,
			jointL,
			foot,
		};
	}
}
```

---

## 5. 形态即计算（Morphology is Computation）

闭环连杆机构对现代 AI 与具身智能（Embodied AI）提供了极其深刻的反思：

在传统经典控制理论中，若要让一个机械足走出一根水平直线，中央处理器必须以 1000Hz 的频率实时计算每个电机的逆运动学雅可比矩阵，消耗数千次矩阵乘法浮点运算（FLOPs）。

而泰奥·扬森的仿生兽**没有微处理器，没有角度传感器，没有控制代码**。
它的控制律被直接浇铸在塑料管的物理长度之中。当海风吹动风帆，机械骨骼的几何约束自动滤掉了高频抖动，将混乱的风力能量转化为高度有序、甚至具备被动避障自稳定性的生命步态。

**智能不一定居于神经元中。当几何尺度契合了物理世界的因果律，物质与形态本身就是在以零功耗执行最精密的神奇计算。**
