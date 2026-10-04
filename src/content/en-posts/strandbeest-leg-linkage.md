---
sourceId: 2026-10-04-strandbeest-leg-linkage
slug: strandbeest-leg-linkage
title: Thirteen Numbers That Make a Leg Walk
editionLabel: ENGLISH EDITION · STRANDBEEST SERIES 2/3
published: 2026-10-04T00:00:00.000Z
draft: true
description: Solving the Jansen linkage with circle intersections, why the foot path is D-shaped, and how several legs are phased.
tags:
  - Strandbeest
  - Kinematics
category: Strandbeest
lang: en
---

## One leg, one crank

The Strandbeest leg is the Jansen linkage: a crank plus seven moving rigid bodies joined by ten revolute joints, one degree of freedom (per the arXiv paper cited below). Turn the crank once and the foot traces a closed loop. The 13 lengths (unit unstated):

$$
a=38.0,\ b=41.5,\ c=39.3,\ d=40.1,\ e=55.8,\ f=39.4,\ g=36.7,\ h=65.7,\ i=49.0,\ j=50.0,\ k=61.9,\ l=7.8,\ m=15.0
$$

with $m$ the crank. I checked the numbers against [Wikipedia's Jansen's linkage article](https://en.wikipedia.org/wiki/Jansen%27s_linkage), which does not say which link joins which joints. The topology comes from an arXiv paper (2606.22129) that I did not read in full; independently, a brute-force search for a flat-bottomed foot path also produced this assignment among its candidates.

## Every joint is a circle intersection

Put the fixed pivot $G$ at the origin, the crank pivot at $(a, l)$, and the crank tip $C$ at angle $\theta$. Every other joint is at known distances from two known points:

$$
\begin{aligned}
K &= \mathrm{circ}(G,b;\ C,j) & L &= \mathrm{circ}(G,d;\ K,e) \\
M &= \mathrm{circ}(G,c;\ C,k) & N &= \mathrm{circ}(L,f;\ M,g) \\
F &= \mathrm{circ}(M,i;\ N,h)
\end{aligned}
$$

$F$ is the foot. Two circles meet in two points; the choice fixes the assembly branch, and I pick signs so the foot is the lowest point of the leg. In code each joint is one line of data:

```ts
{ id: "K", centers: ["G", "C"], radii: ["b", "j"], side: 1 }
```

The solver walks the list in order, so the GA only changes lengths and never the topology.

![Foot path of the Jansen linkage](/strandbeest/jansen-leg.gif)

## Why a D

The foot path has a nearly straight stretch on the ground and an arc through the air on the return. In my implementation the ground stroke is about 43% of a revolution and the lift is about a third of the path width (22.5 vs 67.9). A flat stroke means the body does not bob while a foot is down; a test pins this property down.

## Several legs, staggered

One leg spends half a turn in the air. The Strandbeest puts many legs on one crankshaft with staggered phases; in code, $n$ legs are $n$ time shifts of the same path:

```ts
const feet = multiLegFeet(spec, evenPhases(8), 120);
```

![8 legs, phased (schematic)](/strandbeest/walker.gif)

The picture is schematic: legs are drawn as hip-to-foot lines, planted feet (orange) stay fixed relative to the ground while the body glides over them. I checked that with 12 legs at least one foot is always down and support is stable over 90% of the cycle, while with one leg it is not.

## Summary

- 13 lengths plus a topology make a complete leg, solved with circle intersections.
- The D-shaped path is a result of the proportions, not a coincidence.
- To drag the sliders or evolve it yourself: [strandbeest-evolution](https://github.com/meiluosi/strandbeest-evolution).

Next: evolving it from random lengths.
