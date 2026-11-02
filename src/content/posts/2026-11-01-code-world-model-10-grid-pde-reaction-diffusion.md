---
title: 网格上的偏微分方程：波动方程、Gray-Scott 反应扩散与图灵斑图
published: 2026-11-01T10:30:00+08:00
description: 代码造物与显式世界模型第三卷之二：从常微分方程迈入偏微分方程（PDE）空间连续场，推导五点拉普拉斯算子离散卷积与 CFL 稳定性判据，用代码重现阿兰·图灵形态发生学与 Gray-Scott 细胞分裂/斑马纹化学计算。
format: technical
tags:
  - 代码造物与显式世界模型
  - 偏微分方程
  - 反应扩散
  - 图灵斑图
  - 形态发生学
  - 波动方程
category: 代码造物与显式世界模型
series: "代码造物与显式世界模型"
seriesOrder: 10
lang: zh
---

在之前的九章中，我们处理的变量无论是位置 $x$ 还是角度 $\theta$，本质上都是随单一时间变量 $t$ 演化的**常微分方程（Ordinary Differential Equations, ODEs）**。质点无论如何运动，它始终是空间中离散存在的点。

然而，大自然中更深邃的物理媒介——水面的涟漪波纹、空气中的声音震荡、热量的扩散渗透，以及斑马身上的条纹、贝壳上的螺旋几何——并不是由孤立的质点构成的，而是分布在空间每一个坐标上的**连续介质场（Continuous Fields）**。

场在时间 $t$ 和空间坐标 $(x, y)$ 上同时演化。描述这一宇宙底层的数学工具，是**偏微分方程（Partial Differential Equations, PDEs）**。

1952 年，计算机科学之父阿兰·图灵（Alan Turing）在自杀前发表了他一生中最后一篇划时代的科学论文《形态发生的化学基础》（*The Chemical Basis of Morphogenesis*）。图灵用纯粹的偏微分反应扩散方程证明了：**生命的有机形态不需要上帝或DNA逐个细胞手工雕刻，两种简单化学物质在空间网格上的反应与扩散竞赛，就能自动长出豹纹、斑马线与细胞分裂图景。**

---

## 1. 连续场的空间导数：拉普拉斯算子（Laplacian Operator）

在标量场 $u(x, y, t)$（例如温度场、水面高度场）中，最核心的空间微分算子是**拉普拉斯算子（Laplacian $\nabla^2$）**：
$$\nabla^2 u = \Delta u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}$$

很多开发者对这个符号望而生畏，但它的**物理几何直觉极其平易近人**：
> **拉普拉斯算子量化的是：当前点的值，与它周围邻居的平均值之间的差距。**

```
              五点离散拉普拉斯卷积核 (5-Point Stencil)
                        ┌───────┐
                        │ +0.25 │  (上方邻居)
                ┌───────┼───────┼───────┐
 (左方邻居)     │ +0.25 │ -1.00 │ +0.25 │  (右方邻居)
                └───────┼───────┼───────┘
                        │ +0.25 │  (下方邻居)
                        └───────┘
  ∇²u ≈ (邻居平均值 u_neighbors) - (自身当前值 u_center)
```

在离散二维网格（宽高为 $W \times H$，网格步长 $\Delta x = 1$）上，中心有限差分离散格式为：
$$\nabla^2 u_{i, j} \approx u_{i+1, j} + u_{i-1, j} + u_{i, j+1} + u_{i, j-1} - 4 u_{i, j}$$

- 如果 $\nabla^2 u_{i, j} > 0$：说明自身比周围邻居低（处于低谷），外围介质会向内涌入填补；
- 如果 $\nabla^2 u_{i, j} < 0$：说明自身比周围邻居高（处于顶峰），自身介质会向四周扩散宣泄；
- 如果 $\nabla^2 u_{i, j} = 0$：局部达成调和平衡（Harmonic Equilibrium）。

---

## 2. 2D 波动方程与 CFL 稳定性分岔

最直观的连续场是**二维波动方程（2D Wave Equation）**，它控制着水面波纹与鼓膜振动：
$$\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u - \gamma \frac{\partial u}{\partial t}$$
其中 $c$ 为波速，$\gamma$ 为介质阻尼。

### 2.1 离散递推格式推导
将时间二阶导数用中心有限差分离散化：
$$\frac{\partial^2 u}{\partial t^2} \approx \frac{u_{i, j}^{t+1} - 2u_{i, j}^t + u_{i, j}^{t-1}}{\Delta t^2}$$
代入波动方程并移项，直接解出下一时刻的水面高度 $u_{i, j}^{t+1}$：
$$u_{i, j}^{t+1} = 2u_{i, j}^t - u_{i, j}^{t-1} + c^2 \Delta t^2 \nabla^2 u_{i, j}^t - \gamma \Delta t (u_{i, j}^t - u_{i, j}^{t-1})$$

每一帧我们只需对整张网格执行一次简单的卷积，点击鼠标在水面注入一个脉冲点，动能就会以完美的正弦同心圆波纹向四周扩散，遇到墙壁时产生完美的反射、相干与驻波干涉条纹。

### 2.2 库朗-弗里德里希斯-列维（CFL）稳定性条件
在数值求解偏微分方程时，存在一条不可逾越的物理红线——**CFL 条件（Courant–Friedrichs–Lewy Condition）**：
$$C = c \frac{\Delta t}{\Delta x} \le \frac{1}{\sqrt{2}} \approx 0.7071$$
**物理含义**：在一个离散时间步 $\Delta t$ 内，波向前传播的物理距离 $c \Delta t$，绝对不能超过网格自身的空间间距 $\Delta x$。如果波速传播太快、或者时间步长过大，信息还没来得及传给相邻网格就被强制跳过，整个画面会在 3 帧之内发生黑白棋盘式的剧烈高频震荡并彻底崩溃成 NaN。

---

## 3. 图灵形态发生学与 Gray-Scott 反应扩散模型

在生命的化学胚胎发育中，为什么均匀对称的受精卵能自发破缺对称性，长出豹子的斑点与斑马的条纹？

图灵提出了**反应扩散系统（Reaction-Diffusion Systems）**。后来，科学家将其形式化为著名的 **Gray-Scott 模型**。该模型假设空间中存在两种化学物质：**激活剂（Activator $U$）**与**抑制剂（Inhibitor $V$）**。

控制方程为一对高度非线性的偏微分方程组：
$$
\begin{aligned}
\frac{\partial u}{\partial t} &= D_u \nabla^2 u - u v^2 + F(1 - u) \\
\frac{\partial v}{\partial t} &= D_v \nabla^2 v + u v^2 - (F + k)v
\end{aligned}
$$

```
                Gray-Scott 化学反应机制解剖
       扩散输入 (F(1 - u))                   自催化反应 (u + 2v -> 3v)
   ┌───────────────────────┐            ┌────────────────────────────────┐
   │ 外部源源不断向系统补充   │            │ 两个 V 遇到一个 U，迅速将 U 吞噬 │
   │ 原料底物 U (进料率 F)  │            │ 并转化繁殖为一个新的 V (uv²)   │
   └───────────┬───────────┘            └───────────────┬────────────────┘
               │                                        │
               ▼                                        ▼
      【底物浓度 U】 ────────────────────────────► 【产物浓度 V】
                                                        │
                                                        ▼ 衰变消亡 (-(F + k)v)
                                                ┌───────────────────────┐
                                                │ 抑制剂 V 自然衰变死亡   │
                                                │ 并移出系统 (消亡率 k)  │
                                                └───────────────────────┘
```

### 图灵斑图涌现的核心秘密：长程抑制与短程激活
为什么系统不会演化为死寂的均匀混合溶液？
图灵发现的关键分岔条件是：**抑制剂的扩散速率必须远远大于激活剂的扩散速率**：
$$D_v \approx 2 D_u$$
- 当局部产生微量的 $V$ 时，它会迅速吞噬周围的 $U$ 自我繁殖；
- 但由于 $V$ 扩散得太快，它迅速扩散到了外围，而在外围区域，过高的 $V$ 浓度又抑制了新的斑块生成；
- 这种“局部自我强化、外围强力压制”的博弈，最终在宏观尺度上凝固成了稳定的几何花纹！

调节两个微小的标量参数——**进料率 $F$** 与 **消亡率 $k$**，系统会自发涌现出自然界全部的生命形态图谱：
- $F=0.030, k=0.062$：**细胞分裂与有丝分裂（Mitosis）**，斑点生长到一定大小后自发分裂为两个子细胞；
- $F=0.038, k=0.061$：**斑马与指纹条纹（Fingerprint Meanders）**；
- $F=0.022, k=0.051$：**深海珊瑚礁迷宫（Coral Reef Labyrinths）**；
- $F=0.014, k=0.054$：**孤立子脉冲粒子（Soliton Waves）**。

---

## 4. 完整 TypeScript / TypedArray 反应扩散引擎实现

在网格偏微分方程求解中，直接使用普通 JavaScript 对象数组会产生巨大的垃圾回收（GC）停顿。工业级做法是使用底层的 `Float32Array` 双缓冲（Ping-Pong Buffers），直接在连续内存上进行高效卷积：

```ts
/**
 * GrayScottReactionDiffusion.ts
 * 基于 Float32Array 连续平铺内存的高性能 2D 图灵反应扩散 PDE 解算器
 */

export class GrayScottSimulation {
	public readonly width: number;
	public readonly height: number;
	private readonly size: number;

	// 扩散系数: 抑制剂 V 扩散速度必须显著高于激活剂 U
	public Du: number = 0.2097;
	public Dv: number = 0.105;

	// 进料率 F 与消亡率 k (决定斑图相变的关键分岔参数)
	public F: number = 0.037; // 斑马纹/珊瑚经典参数
	public k: number = 0.060;

	// 双缓冲内存数组 (Ping-Pong Buffers)
	private uCurrent: Float32Array;
	private vCurrent: Float32Array;
	private uNext: Float32Array;
	private vNext: Float32Array;

	constructor(width: number = 160, height: number = 160) {
		this.width = width;
		this.height = height;
		this.size = width * height;

		this.uCurrent = new Float32Array(this.size);
		this.vCurrent = new Float32Array(this.size);
		this.uNext = new Float32Array(this.size);
		this.vNext = new Float32Array(this.size);

		this.reset();
	}

	/** 重置场：全局填满 U=1，中央小区域注入 V=1 作为生命种子 */
	public reset(): void {
		this.uCurrent.fill(1.0);
		this.vCurrent.fill(0.0);

		// 在中央注入 20x20 的初始扰动核
		const cx = Math.floor(this.width / 2);
		const cy = Math.floor(this.height / 2);
		const r = 10;

		for (let y = cy - r; y <= cy + r; y++) {
			for (let x = cx - r; x <= cx + r; x++) {
				const idx = y * this.width + x;
				this.uCurrent[idx] = 0.5 + (Math.random() - 0.5) * 0.05;
				this.vCurrent[idx] = 0.25 + (Math.random() - 0.5) * 0.05;
			}
		}
	}

	/** 核心偏微分推进单步: 五点离散拉普拉斯卷积与反应项结算 */
	public step(dt: number = 1.0): void {
		const { width, height, Du, Dv, F, k } = this;
		const uCur = this.uCurrent;
		const vCur = this.vCurrent;
		const uNxt = this.uNext;
		const vNxt = this.vNext;

		// 遍历内部网格 (边界执行环形包裹 Torus)
		for (let y = 0; y < height; y++) {
			const yTop = (y === 0 ? height - 1 : y - 1) * width;
			const yMid = y * width;
			const yBot = (y === height - 1 ? 0 : y + 1) * width;

			for (let x = 0; x < width; x++) {
				const xLeft = x === 0 ? width - 1 : x - 1;
				const xRight = x === width - 1 ? 0 : x + 1;

				const idx = yMid + x;
				const u = uCur[idx];
				const v = vCur[idx];

				// 五点有限差分拉普拉斯算子
				const lapU =
					uCur[yTop + x] + uCur[yBot + x] + uCur[yMid + xLeft] + uCur[yMid + xRight] - 4 * u;
				const lapV =
					vCur[yTop + x] + vCur[yBot + x] + vCur[yMid + xLeft] + vCur[yMid + xRight] - 4 * v;

				// 反应项结算: uv²
				const uvv = u * v * v;

				// 偏微分方程推进
				uNxt[idx] = u + (Du * lapU - uvv + F * (1.0 - u)) * dt;
				vNxt[idx] = v + (Dv * lapV + uvv - (F + k) * v) * dt;
			}
		}

		// 交换双缓冲指针 (零内存分配)
		const tempU = this.uCurrent;
		this.uCurrent = this.uNext;
		this.uNext = tempU;

		const tempV = this.vCurrent;
		this.vCurrent = this.vNext;
		this.vNext = tempV;
	}

	/** 提取用于 Canvas ImageData 的 RGBA 像素缓冲 */
	public renderToImageData(imageData: ImageData): void {
		const data = imageData.data;
		const vCur = this.vCurrent;

		for (let i = 0; i < this.size; i++) {
			const val = Math.min(1.0, Math.max(0.0, vCur[i] * 3.0));
			const pixelIdx = i * 4;

			// 调色盘映射: 深铜墨色到薄荷青绿
			data[pixelIdx + 0] = Math.floor(15 + val * (16 - 15));    // R
			data[pixelIdx + 1] = Math.floor(24 + val * (185 - 24));   // G
			data[pixelIdx + 2] = Math.floor(20 + val * (129 - 20));   // B
			data[pixelIdx + 3] = 255;                                // A
		}
	}
}
```

---

## 5. 代码造物的深水区：形态学是物质与法则的凝固

偏微分反应扩散方程为我们揭示了一个哲学层面的终极震撼：

在传统的机械论看来，要造出一个有条纹的生命体，必须在“蓝图”里精确标出每一道斑纹的绝对像素坐标。
而图灵告诉我们：**自然界从来不需要一张像素级的蓝图。大自然只需要在空间里释放两种具有不同扩散速度与反应活性的化学分子，物理与数学法则便会替它完成剩下的全部图样绘制。**

代码世界模型也是如此。当我们掌握了从局部微分网格推演全局形态的偏微分方程，我们就不再是在屏幕上绘制一幅死板图画的画匠，而是向数字虚空中播撒物理因果法则的造物工匠。
