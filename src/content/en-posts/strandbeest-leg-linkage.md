---
sourceId: 2026-10-04-strandbeest-leg-linkage
slug: strandbeest-leg-linkage
title: Thirteen Numbers That Make a Leg Walk
editionLabel: ENGLISH EDITION · STRANDBEEST SERIES 2/3
published: 2026-10-04T00:00:00.000Z
draft: false
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

with $m$ the crank. I checked the numbers against [Wikipedia's Jansen's linkage article](https://en.wikipedia.org/wiki/Jansen%27s_linkage), which does not say which link joins which joints. The topology comes from an arXiv paper ([2606.22129](https://arxiv.org/abs/2606.22129), on durability-aware multi-objective optimisation of the Jansen linkage) that I did not read in full; independently, a brute-force search for a flat-bottomed foot path also produced this assignment among its candidates.

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

**Evidence level: E2.** These three numbers were later checked against a published test-rig dataset. Pop, Dolga and Pop published the foot trajectory of their own small leg test rig in 2014 in *Applied Mechanics and Materials* (CCD camera tracking); I digitised it and compared it item by item with the analytical kinematics in this repository:

| | literature (Pop 2014) | our analytical kinematics | relative error |
|---|---|---|---|
| stroke length | 57.1 mm | 57.081 mm | 0.033 % |
| step lift height | 22.5 mm | 22.457 mm | 0.191 % |
| duty factor (ground stroke) | 0.43 | 0.433 | 0.77 % |
| foot loop width | ~65.55 mm (digitised) | 67.899 mm | 3.46 % |

Over the ground stroke the point-to-curve distance is 0.00–0.54 mm, below the ±0.80 mm digitisation uncertainty of the paper itself (line thickness and a 300 dpi raster grid). So "the path is flat and the lift is about 22.5" is not only self-consistent inside my own model: it agrees with a published measurement, which is what E2 means. Source: [*Applied Mechanics and Materials* 658: 684–689](https://doi.org/10.4028/www.scientific.net/AMM.658.684). The normalised mean shape distance to the digitised literature trajectory is 0.0295 (2.95 % of the path width).

## Several legs, staggered

One leg spends half a turn in the air. The Strandbeest puts many legs on one crankshaft with staggered phases; in code, $n$ legs are $n$ time shifts of the same path:

```ts
const feet = multiLegFeet(spec, evenPhases(8), 120);
```

![8 legs, phased (schematic)](/strandbeest/walker.gif)

The picture is schematic: legs are drawn as hip-to-foot lines, planted feet (orange) stay fixed relative to the ground while the body glides over them. I checked that with 12 legs (6 per side) at least one foot is always down and support is stable over 90% of the cycle (that is a hard constraint in the wind objective). With one leg it fails: it is in the air for half the turn.

## Summary

- 13 lengths plus a topology make a complete leg, solved with circle intersections.
- The D-shaped path is a result of the proportions, not a coincidence; it agrees with a published 2014 test-rig trajectory (E2).
- To drag the sliders or evolve it yourself: code in [strandbeest-evolution](https://github.com/meiluosi/strandbeest-evolution), or open the [live demo](https://meiluosi.github.io/strandbeest-evolution/) in your browser (nothing to install).

Next: evolving it from random lengths.
