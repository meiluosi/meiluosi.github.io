---
title: 局部规则与群体涌现：Boids 三定律、空间哈希与流场寻路
published: 2026-11-01T10:20:00+08:00
description: 代码造物与显式世界模型第三卷之一：从还原论迈入复杂系统涌现，推导 Craig Reynolds Boids 仿生群体三定律，构建 O(N) 均匀网格空间哈希（Spatial Hash Grid）与势能拓扑流场寻路算法。
format: technical
tags:
  - 代码造物与显式世界模型
  - 群体智能
  - Boids
  - 空间哈希
  - 复杂系统
  - 流场寻路
category: 代码造物与显式世界模型
series: "代码造物与显式世界模型"
seriesOrder: 9
lang: zh
---

在前两卷中，我们的目光始终聚焦在孤立的质点、刚体连杆与受力构件上。这种“把大机械拆成小零件、把复杂运动拆成单体受力”的方法，是经典的**笛卡尔还原论（Cartesian Reductionism）**。

然而，大自然中最震撼人心的视觉奇迹，往往无法通过拆解单个构件来解释：
- 成千上万只椋鸟在暮色中形成遮天蔽日的黑色风暴，没有任何一只头鸟在发号施令，鸟群却能在千分之一秒内集体变向、穿梭避障，仿佛一个拥有统一意志的流体生命；
- 鱼群在遭遇捕食者袭击时瞬间分裂为两个环形涡流，随后又在捕食者身后天衣无缝地闭合。

如果让你用代码在屏幕上渲染 5,000 只鸟的群体漫游，你该如何编写程序？
你绝不可能为每一只鸟手工编写复杂的路线规划脚本。

1986 年，计算机图形学先驱克雷格·雷诺兹（Craig Reynolds）提出了震撼业界的 **Boids 模型**。它向世人证明了一个划时代的复杂系统真理：**极致复杂的宏观群体智能，仅仅源于三条极其微小的局部几何规则。**

---

## 1. 雷诺兹 Boids 三定律：操纵力与局部感知

在 Boids 模型中，每只个体（称为一个 Boid）没有任何全局大局观。它只能感知自身周围半径为 $R$ 的局部视野邻居集合 $\mathcal{N}_i$。

```
                       Boids 局部感知三定律
     1. 分离 (Separation)        2. 对齐 (Alignment)        3. 凝聚 (Cohesion)
          (排斥过近者)               (与邻居同频同向)            (向群体中心聚集)
             ▲                          ▲                          ▲
           ╱   ╲                      ▲ ▲ ▲                      ╱   ╲
          ● ◄─► ●                     │ │ │                     ● ──►●◄── ●
         (保持防撞间距)              (速度向量平均化)            (防止个体掉队孤独)
```

雷诺兹将物理学中的力抽象为**操纵力向量（Steering Force）**：
$$\mathbf{F}_{steer} = \text{truncate}\left(\mathbf{v}_{desired} - \mathbf{v}, F_{max}\right)$$
即期望速度与当前速度之间的转向差值，并施加最大转向加速度截断 $F_{max}$。

### 1.1 分离力（Separation）：排斥过近的拥挤
为了防止彼此发生物理踩踏与碰撞，个体受到来自所有过近邻居的反向排斥力，排斥力大小与距离成反比：
$$\mathbf{F}_{sep} = \sum_{j \in \mathcal{N}_i, d_{ij} < R_{sep}} \frac{\mathbf{x}_i - \mathbf{x}_j}{\|\mathbf{x}_i - \mathbf{x}_j\|^2}$$

### 1.2 对齐力（Alignment）：群体速度同频共振
个体倾向于与邻域内同伴的平均飞行航向与航速保持一致：
$$\mathbf{v}_{align\_target} = \frac{1}{|\mathcal{N}_i|} \sum_{j \in \mathcal{N}_i} \mathbf{v}_j$$
$$\mathbf{F}_{align} = \mathbf{v}_{align\_target} - \mathbf{v}_i$$

### 1.3 凝聚力（Cohesion）：向质心靠拢防掉队
个体渴望待在群体之中，避免落单被捕食者猎杀。它会向邻居们的几何质心 $\mathbf{c}_{local}$ 产生一个吸引力：
$$\mathbf{c}_{local} = \frac{1}{|\mathcal{N}_i|} \sum_{j \in \mathcal{N}_i} \mathbf{x}_j$$
$$\mathbf{F}_{coh} = \mathbf{c}_{local} - \mathbf{x}_i$$

最终，每个个体的总合力仅仅是三者的简单线性加权：
$$\mathbf{F}_{total} = w_{sep} \mathbf{F}_{sep} + w_{align} \mathbf{F}_{align} + w_{coh} \mathbf{F}_{coh}$$

仅仅依靠这三行初中几何级别的加减法，原本散落无序的数千个点在屏幕上运行几秒钟后，令人屏息的宏观图景自发涌现了：它们自动聚合成流动的缎带、在转角处形成层流、在遭遇鼠标时如同被子弹击穿的水滴般炸开并重新愈合。

---

## 2. 算力悬崖：从 $O(N^2)$ 到 $O(N)$ 空间哈希（Spatial Hashing）

尽管 Boids 算法在数学上极其优美，但如果采用朴素的暴力双重循环去寻找邻居：

```ts
// 致命性能陷阱: O(N^2) 暴力成对距离检查
for (let i = 0; i < N; i++) {
	for (let j = 0; j < N; j++) {
		if (i === j) continue;
		const dist = distance(boids[i], boids[j]);
		if (dist < R) {
			/* 处理邻居 */
		}
	}
}
```

当个体数量从 100 增加到 5,000 时，计算量将从 $10^4$ 次暴增到 $25,000,000$ 次！在 60fps 的时间窗口下，每帧只有极其宝贵的 16 毫秒，2500 万次距离开方运算会在瞬间让浏览器主线程彻底卡死瘫痪。

解决这一瓶颈的关键数据结构是**均匀网格空间哈希（Uniform Spatial Hash Grid）**。

```
              均匀网格空间哈希邻域检索原理
     ┌───────┬───────┬───────┐
     │ (0,2) │ (1,2) │ (2,2) │      每个网格桶大小 CellSize = 探测半径 R
     ├───────┼───────┼───────┤      个体位于 (1,1) 网格中
     │ (0,1) │ (1,1) │ (2,1) │ ◄──  只需检索周围九宫格内的常数个个体！
     ├───────┼───────┼───────┤      彻底告别全图扫描
     │ (0,0) │ (1,0) │ (2,0) │
     └───────┴───────┴───────┘
```

### 2.1 空间哈希算法原理
1. 将二维连续空间切分为边长等于感知半径 $R$ 的均匀方格网格（Cell）；
2. 空间中任意连续坐标 $\mathbf{p} = (x, y)$，其所在的离散整数网格索引为：
   $$\text{cellX} = \lfloor x / R \rfloor, \quad \text{cellY} = \lfloor y / R \rfloor$$
3. 在每一帧开始时，遍历所有个体，根据其整数坐标将其放入对应的哈希桶中。构建哈希表的时间复杂度为纯线性的 $O(N)$；
4. 当某个个体寻找邻居时，由于其感知半径为 $R$，潜在的邻居**只可能存在于当前网格以及相邻的 8 个邻居网格（九宫格）之中**！
5. 平均每个网格内的个体数是常数 $C$。查询每个个体邻居的耗时恒定为 $9C \sim O(1)$。

**总时间复杂度从悬崖般的 $O(N^2)$ 骤降为绝对平直的 $O(N)$！**
在现代普通笔记本电脑上，JavaScript 可以在 60fps 下毫不费力地并行解算 10,000 只个体的涌现动画。

---

## 3. 流场寻路（Flow Field Pathfinding）：万人集群的势能导航

如果群体不仅要自组织飞行，还必须在充满迷宫墙壁的世界中集体寻路到达某个遥远的目标，传统为每个个体独立运行 A* 寻路算法的做法会导致 CPU 算力瞬间耗尽。

更强大的世界模型方法是**势能流场（Flow Field）**：
1. **反向 Dijkstra 距离场**：从目标终点出发，在空间网格上执行广度优先搜索（BFS），计算出每个网格到终点的最短测地线距离标量 $\Phi(x, y)$；
2. **离散梯度向量场**：每个网格的流动方向，即为势能下降最快的负梯度方向：
   $$\mathbf{v}_{flow}(x, y) = -\frac{\nabla \Phi}{\|\nabla \Phi\|}$$
3. **个体的导航极简更新**：个体只需像顺流而下的小舟一样，在每一帧读取当前脚下网格预先烘焙好的流场向量 $\mathbf{v}_{flow}$，并将其作为一个额外的目标驱动力融入 Boids 体系中。

一万只个体共用同一张全局流场底图，在没有中央指挥官的条件下，群体会自发像水流绕过岩石一样，顺滑地分流、绕开复杂死胡同并精准汇聚到终点。

---

## 4. 完整 TypeScript 空间哈希与 Boids 群体引擎

以下自包含代码实现了极速的 2D 空间哈希表与完整的 Boids 涌现群控逻辑：

```ts
/**
 * SpatialBoidsEngine.ts
 * 具备 O(N) 均匀网格空间哈希加速的 Boids 群体涌现引擎
 */

export interface Boid {
	x: number;
	y: number;
	vx: number;
	vy: number;
	ax: number;
	ay: number;
}

export class SpatialHashGrid {
	private cellSize: number;
	// 网格桶存储个体索引: key = "cellX,cellY"
	private buckets: Map<string, number[]> = new Map();

	constructor(cellSize: number = 40) {
		this.cellSize = cellSize;
	}

	public clear(): void {
		this.buckets.clear();
	}

	private getKey(cellX: number, cellY: number): string {
		return `${cellX},${cellY}`;
	}

	/** 注册个体进入网格桶 */
	public insert(index: number, x: number, y: number): void {
		const cx = Math.floor(x / this.cellSize);
		const cy = Math.floor(y / this.cellSize);
		const key = this.getKey(cx, cy);

		let list = this.buckets.get(key);
		if (!list) {
			list = [];
			this.buckets.set(key, list);
		}
		list.push(index);
	}

	/** 查询点 (x, y) 周围九宫格内的所有潜在邻居索引 */
	public queryNeighbors(x: number, y: number): number[] {
		const cx = Math.floor(x / this.cellSize);
		const cy = Math.floor(y / this.cellSize);
		const neighbors: number[] = [];

		for (let ox = -1; ox <= 1; ox++) {
			for (let oy = -1; oy <= 1; oy++) {
				const key = this.getKey(cx + ox, cy + oy);
				const bucket = this.buckets.get(key);
				if (bucket) {
					for (let i = 0; i < bucket.length; i++) {
						neighbors.push(bucket[i]);
					}
				}
			}
		}

		return neighbors;
	}
}

export class BoidsFlockSimulation {
	public boids: Boid[] = [];
	public grid: SpatialHashGrid;

	// 感知与动力学超参数
	public perceptionRadius: number = 45;
	public separationDistance: number = 20;
	public maxSpeed: number = 180; // 像素/秒
	public maxForce: number = 350; // 最大转向加速度

	// 权重配比
	public wSep: number = 1.5;
	public wAlign: number = 1.0;
	public wCoh: number = 1.0;

	constructor(count: number, width: number, height: number) {
		this.grid = new SpatialHashGrid(this.perceptionRadius);
		for (let i = 0; i < count; i++) {
			const angle = Math.random() * Math.PI * 2;
			const speed = 80 + Math.random() * 80;
			this.boids.push({
				x: Math.random() * width,
				y: Math.random() * height,
				vx: Math.cos(angle) * speed,
				vy: Math.sin(angle) * speed,
				ax: 0,
				ay: 0,
			});
		}
	}

	public update(dt: number, width: number, height: number): void {
		// 1. 重建空间哈希索引 O(N)
		this.grid.clear();
		for (let i = 0; i < this.boids.length; i++) {
			this.grid.insert(i, this.boids[i].x, this.boids[i].y);
		}

		// 2. 局部规则结算
		for (let i = 0; i < this.boids.length; i++) {
			const b = this.boids[i];
			const neighbors = this.grid.queryNeighbors(b.x, b.y);

			let sepX = 0, sepY = 0;
			let alignX = 0, alignY = 0;
			let centerX = 0, centerY = 0;
			let neighborCount = 0;

			for (const otherIdx of neighbors) {
				if (otherIdx === i) continue;
				const other = this.boids[otherIdx];

				const dx = b.x - other.x;
				const dy = b.y - other.y;
				const distSq = dx * dx + dy * dy;

				if (distSq < this.perceptionRadius * this.perceptionRadius && distSq > 1e-4) {
					const dist = Math.sqrt(distSq);

					// 分离力 (距离越近排斥力越强)
					if (dist < this.separationDistance) {
						sepX += dx / dist;
						sepY += dy / dist;
					}

					// 对齐累加
					alignX += other.vx;
					alignY += other.vy;

					// 质心累加
					centerX += other.x;
					centerY += other.y;

					neighborCount++;
				}
			}

			// 合成并应用加速度
			let totalAx = 0;
			let totalAy = 0;

			if (neighborCount > 0) {
				// 凝聚力: 指向平均质心
				const avgCenterX = centerX / neighborCount;
				const avgCenterY = centerY / neighborCount;
				const toCenterX = avgCenterX - b.x;
				const toCenterY = avgCenterY - b.y;
				const centerDist = Math.hypot(toCenterX, toCenterY);
				if (centerDist > 0) {
					totalAx += (toCenterX / centerDist) * this.maxSpeed * this.wCoh;
					totalAy += (toCenterY / centerDist) * this.maxSpeed * this.wCoh;
				}

				// 对齐力: 拟合平均速度
				const avgVx = alignX / neighborCount;
				const avgVy = alignY / neighborCount;
				totalAx += (avgVx - b.vx) * this.wAlign;
				totalAy += (avgVy - b.vy) * this.wAlign;
			}

			// 加上分离力
			totalAx += sepX * this.maxSpeed * this.wSep;
			totalAy += sepY * this.maxSpeed * this.wSep;

			// 截断加速度
			const forceLen = Math.hypot(totalAx, totalAy);
			if (forceLen > this.maxForce) {
				totalAx = (totalAx / forceLen) * this.maxForce;
				totalAy = (totalAy / forceLen) * this.maxForce;
			}

			b.ax = totalAx;
			b.ay = totalAy;
		}

		// 3. 速度与位置积分 (辛欧拉)
		for (const b of this.boids) {
			b.vx += b.ax * dt;
			b.vy += b.ay * dt;

			// 速度限幅
			const speed = Math.hypot(b.vx, b.vy);
			if (speed > this.maxSpeed) {
				b.vx = (b.vx / speed) * this.maxSpeed;
				b.vy = (b.vy / speed) * this.maxSpeed;
			}

			b.x += b.vx * dt;
			b.y += b.vy * dt;

			// 环形世界边界包裹 (Torus Wrap)
			if (b.x < 0) b.x += width;
			if (b.x > width) b.x -= width;
			if (b.y < 0) b.y += height;
			if (b.y > height) b.y -= height;
		}
	}
}
```

---

## 5. 涌现论对心智与世界模型的启示

Boids 模型不仅是群体动画的经典教案，更是对哲学上**心身二元论（Cartesian Dualism）**的致命一击。

长期以来，人们直觉上认为：如果一个系统展现出高度智慧的整体协同，其背后必然隐藏着一个高高在上的中央司令部（“笛卡尔剧场”里的微型小人 Homunculus）。
然而复杂系统科学告诉我们：
- 黏菌没有单个脑细胞，却能在营养网络中以超越人工算法的最短拓扑路线连通整个东京地铁地图；
- 大脑中的每一个神经元只是遵循电位阈值被动放电的微小生化器件，但当千亿个神经元通过突触局部连接在一起时，宏观的心智、意识与自我感知便轰然涌现。

**世界模型的高阶智慧，并不在于构建多么庞大臃肿的集中式控制器，而在于设计出足够纯粹、自洽且支持局部交互的基础物理律。复杂性是由简单性在时间的长河中自然编织而成的。**
