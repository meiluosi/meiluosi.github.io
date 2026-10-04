---
title: 十三个数字怎么让腿走起来：Jansen 连杆的几何
format: technical
description: 用两圆交点把 Jansen 连杆算出来，看脚的轨迹为什么是 D 形，以及多条腿如何错开相位。
published: 2026-10-04T00:00:00.000Z
draft: true
tags:
  - Strandbeest
  - 机构学
  - 运动学
category: Strandbeest
lang: zh
---

## 一条腿，一根曲柄

Strandbeest 的腿叫 Jansen 连杆：曲柄加上七个运动刚体，用十个转动副连起来，只有一个自由度（据上面提到的论文的描述），转一圈曲柄，脚就走出一条封闭轨迹。它的 13 个长度是（单位未说明）：

$$
a=38.0,\ b=41.5,\ c=39.3,\ d=40.1,\ e=55.8,\ f=39.4,\ g=36.7,\ h=65.7,\ i=49.0,\ j=50.0,\ k=61.9,\ l=7.8,\ m=15.0
$$

其中 $m$ 是曲柄。这些数字我对照了 [Wikipedia 的 Jansen's linkage 条目](https://en.wikipedia.org/wiki/Jansen%27s_linkage)，但该条目没有写出"哪根杆接哪两个关节"。拓扑来自一篇 arXiv 论文（2606.22129）。我没读全文；另外我独立做了一次暴力搜索，看哪种接法能产生"平底"的脚轨迹，这一种也在候选里。

## 用两圆交点求每个关节

把固定铰 $G$ 放在原点，曲柄铰在 $(a, l)$，曲柄端点 $C$ 随角度 $\theta$ 转动。其余关节都是"到两个已知点的距离已知"，于是每个都是两个圆的交点：

$$
\begin{aligned}
K &= \mathrm{circ}(G,b;\ C,j) & L &= \mathrm{circ}(G,d;\ K,e) \\
M &= \mathrm{circ}(G,c;\ C,k) & N &= \mathrm{circ}(L,f;\ M,g) \\
F &= \mathrm{circ}(M,i;\ N,h)
\end{aligned}
$$

$F$ 是脚。两圆有两个交点，选哪个决定装配分支；我取的符号让脚始终是整条腿的最低点。代码里每个关节只是一条数据：

```ts
{ id: "K", centers: ["G", "C"], radii: ["b", "j"], side: 1 }
```

求解器按顺序走一遍就行，所以 GA 改的只是长度，拓扑不用动。

![Jansen 连杆的脚轨迹](/strandbeest/jansen-leg.gif)

## 为什么是 D 形

脚的轨迹有一段几乎是直线，落在地面上；回程抬起来，划一道弧。在我的实现里：

- 着地段占一圈的约 43%
- 抬腿高度约为轨迹宽度的三分之一（22.5 对 67.9）

着地段平，意味着脚踩着地的时候身体不会上下颠，这就是它走得稳的原因。我用一个测试把这件事钉住：脚轨迹必须有平的着地段，而且抬腿高度、连续性都在范围内。

## 一条腿不够，要多条错开

一条腿有半圈在空中，身体撑不住。Strandbeest 把很多条腿装在同一根曲轴上，相位错开。在代码里，$n$ 条腿就是同一条轨迹的 $n$ 个时间平移：

```ts
const feet = multiLegFeet(spec, evenPhases(8), 120);
```

![8 条腿相位错开行走（示意）](/strandbeest/walker.gif)

图是示意：腿画成了髋到脚的直线，着地的脚（橙色）相对地面不动，身体在上面滑过去。我检查了一件事：12 条腿时，每个时刻都至少有脚着地，支撑稳定的时间占比超过九成；只有 1 条腿时则不行。

## 小结

- 13 个长度加一个拓扑，就是一条完整的腿；关节求解只要两圆交点。
- D 形轨迹是连杆比例的结果，不是巧合。
- 想自己拖着滑块玩，或者让它自己演化，代码和演示在 [strandbeest-evolution](https://github.com/meiluosi/strandbeest-evolution)。<!-- TODO: 启用 GitHub Pages 后在这里补在线演示链接 -->

下一篇：我用遗传算法从随机长度出发，看能不能重新演化出它。
