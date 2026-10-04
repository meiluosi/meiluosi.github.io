---
sourceId: 2026-10-04-re-evolving-the-strandbeest-leg
slug: re-evolving-the-strandbeest-leg
title: Can We Re-Evolve Jansen's Leg From Random Lengths?
editionLabel: ENGLISH EDITION · STRANDBEEST SERIES 3/3
published: 2026-10-04T00:00:00.000Z
draft: true
description: A seeded genetic algorithm starts from random linkage lengths to see whether it can recover the shape and the numbers of the Jansen linkage, and how it games the fitness function.
tags:
  - Strandbeest
  - Genetic Algorithm
  - Experiment
category: Strandbeest
lang: en
---

The question: **if a GA is told only what counts as a good leg and starts from messy lengths, can it find Jansen's leg?** Jansen writes on his website that the rod-length ratios were calculated with a genetic algorithm on his Atari computer, but his fitness function and algorithm details are not on record, so this GA reconstructs the idea, not his method.

## Setup

- Genes: the 13 lengths; topology fixed to Jansen's.
- Algorithm: seeded GA (tournament selection, blend crossover, Gaussian mutation, elitism). All randomness comes from the seed, so every run is reproducible.
- Start: a "blind" box, each length uniform in [5, 80], knowing only the order of magnitude.
- Legs that cannot assemble get a graded penalty (more crank angles that assemble = higher score); otherwise random lengths almost never assemble and the GA has nothing to climb.

## The GA fooled my fitness three times

I wanted "flat ground stroke plus enough lift". It found loopholes immediately:

1. **Foot through the body**: lift was uncapped, so the foot swung above the hip. Rule added: the foot stays below the hip.
2. **Foot that only slides**: 100% ground time, a straight line, not walking. Rules added: lift at least 15% of width, ground time credited up to 0.6.
3. **"Flat" got looser as lift grew**: I first defined flat as within 5% of lift height. Now it is relative to path width.

These thresholds are my design choices, not Jansen's. The old lesson holds: the criterion you write is what gets optimised, not the one in your head.

## Experiment 1: match Jansen's foot path

20 seeds, population 150, 500 generations, default GA settings. Fitness is shape distance to Jansen's loop (normalised to width 1; 0 means identical).

| | result |
|---|---|
| median shape distance | 0.0155 |
| seeds with distance < 0.02 | 16 / 20 |
| seeds recovering the lengths | **0 / 20** |

The shape is recoverable, the numbers are not: very different length sets draw nearly the same loop, so shape to lengths is many-to-one. I did not check whether the solutions are different mechanisms or the same one under a symmetry I did not normalise.

## Experiment 2: blind search with my "flat walking" objective

Same setup, my flat-walking fitness.

| | result |
|---|---|
| Jansen's own score | 0.481 |
| median best score | 0.370 |
| seeds beating Jansen | 2 / 20 |
| median shape distance to Jansen | 0.087 |
| seeds within 0.05 of Jansen | 0 / 20 |

With default settings, blind search usually falls short of Jansen's score and never lands on his loop. Started from Jansen's lengths, the same objective reaches 0.71, so his leg is not this objective's optimum.

I suspected the GA was stuck in local optima and reran the same 20 seeds with a higher mutation rate (0.5, scale 0.15, 3 elites). It did **not** help: the median score was 0.337, no seed beat Jansen, and shape-matching seeds dropped from 16 to 11. The "higher mutation is better" impression came from a 6-seed trial and was noise. So the mutation rate is not the simple culprit; the structure of the search space and my objective are both candidates, and I have not separated them.

## What can be concluded

Can say: with a fixed topology, a D-shaped, flat-stroke foot path is findable by blind search. Cannot say: that the GA recovered Jansen's numbers, or that my objective reproduces his method.

## Will the wind carry it?

The repository also has a dynamics model: body mass, slope and drag, with a geared sail driving the crank. Crank torque comes straight from energy conservation. Legs are massless and feet do not slip; sail size, gearing, drag and rotor inertia are illustrative assumptions, so **absolute numbers mean nothing: only comparisons between designs, or between the model's own two algorithms, do**.

**An objective that only looks at the foot path does not make legs more efficient.** Under these assumptions Jansen's leg and my evolved "flat" and "high-step" legs differ little in torque and speed: a shorter-stride leg needs less torque but goes less far per turn.

**Putting wind speed into the objective makes evolution find something different.** I used steady speed in a 6 m/s wind as fitness, started from Jansen's lengths, 5 seeds, one run on flat ground and one on a 10° slope. The best legs were 20%–37% faster than Jansen's, but mostly because the **legs got bigger** (longer stride). Rescaled to Jansen's size the gain is only 8%–17% on flat ground and 7%–15% on the slope, and one seed found nothing better than Jansen. The price of larger legs is a higher peak torque (14.7 → 16.9–20.2 N·m on flat ground). This is a result inside the model; it was not checked against a real walker.

**A flywheel does not help the start.** I expected a flywheel to carry the crank over the torque peak, so I wrote a time-domain model with inertia. The test failed, and the reason is clear: a drag sail's torque falls to zero as it approaches wind speed, so the crank can never spin fast enough to store useful energy before the peak arrives. The time-domain start wind (0.733 m/s) is within about 1% of the quasi-static estimate (0.727 m/s) and is the same for rotor inertia from 0.05 to 300 kg·m². Inertia only smooths the speed ripple a little and lets the walker coast a bit when the wind stops: even a very heavy rotor (300 kg·m²) coasts about two thirds of a revolution.

The model has no foot slip, no impact loss and no soft ground, and the legs have no mass.

## Try it

Code, scripts and results: [strandbeest-evolution](https://github.com/meiluosi/strandbeest-evolution).

```bash
pnpm install
pnpm reproduce 20 500 150 default
```

Corrections welcome: better fitness functions, dynamics models, or primary sources on Jansen's method.
