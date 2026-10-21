---
sourceId: 2026-10-04-re-evolving-the-strandbeest-leg
slug: re-evolving-the-strandbeest-leg
title: Can We Re-Evolve Jansen's Leg From Random Lengths?
editionLabel: ENGLISH EDITION · STRANDBEEST SERIES 3/3
published: 2026-10-04T00:00:00.000Z
draft: false
description: A seeded genetic algorithm starts from random linkage lengths to see whether it can recover the shape and the numbers of the Jansen linkage, and how it games the fitness function.
tags:
  - Strandbeest
  - Genetic Algorithm
  - Experiment
category: Strandbeest
lang: en
---

This is the third post of the Strandbeest series. The first two covered the [evolution history](/en/posts/strandbeest-evolution-history/) and the [geometry of the leg](/en/posts/strandbeest-leg-linkage/). This one asks a plain question: **if a GA is told only what counts as a good leg and starts from messy lengths, can it find Jansen's leg?** Jansen writes on his website that the rod-length ratios were calculated with a genetic algorithm on his Atari computer, but his fitness function and algorithm details are not on record, so this GA reconstructs the idea, not his method.

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
| median shape distance | 0.0148 |
| seeds with distance < 0.02 | 19 / 20 |
| seeds recovering the lengths | **0 / 20** |

The shape is recoverable, the numbers are not: very different length sets draw nearly the same loop, so shape to lengths is many-to-one. I did not check whether the solutions are different mechanisms or the same one under a symmetry I did not normalise.

## Experiment 2: blind search with my "flat walking" objective

Same setup, my flat-walking fitness.

| | result |
|---|---|
| Jansen's own score | 0.481 |
| median best score | 0.390 |
| seeds beating Jansen | 0 / 20 |
| median shape distance to Jansen | 0.087 |
| seeds within 0.05 of Jansen | 0 / 20 |

With default settings, blind search usually falls short of Jansen's score and never lands on his loop. Started from Jansen's lengths, the same objective reaches 0.72, so his leg is not this objective's optimum.

I suspected the GA was stuck in local optima and reran the same 20 seeds with a higher mutation rate (0.5, scale 0.15, 3 elites). It did **not** help: the median score was 0.384 (0.390 with the default settings), only 1 seed beat Jansen, and seeds within 0.02 of his loop dropped from 19 to 10. The "higher mutation is better" impression came from a 6-seed trial and was noise. So the mutation rate is not the simple culprit; the structure of the search space and my objective are both candidates, and this round on its own did not separate them.

**With a mature optimiser, those two causes partly separate.** Same objective, same 20 seeds, CMA-ES spending only 12 000 evaluations per seed (16 % of the GA's budget): median best score 0.529, and 13 of 20 seeds beat Jansen. It still never landed on his loop shape (0 / 20 within 0.05). So "the optimiser was stuck" was indeed a large part of it, while "my objective happens to make Jansen's leg optimal" is still wrong.

## What can be concluded

Can say: with a fixed topology, a D-shaped, flat-stroke foot path is findable by blind search. Cannot say: that the GA recovered Jansen's numbers, or that my objective reproduces his method.

## Will the wind carry it?

The repository also has a dynamics model: body mass, slope and drag, with a geared sail driving the crank. Crank torque comes straight from energy conservation. Legs are massless and feet do not slip; sail size, gearing, drag and rotor inertia are illustrative assumptions, so **absolute numbers mean nothing: only comparisons between designs, or between the model's own two algorithms, do**.

**An objective that only looks at the foot path does not make legs more efficient.** Under these assumptions Jansen's leg and my evolved "flat" and "high-step" legs differ little in torque and speed: a shorter-stride leg needs less torque but goes less far per turn.

**Putting wind speed into the objective makes evolution find something different.** I used steady speed in a 6 m/s wind as fitness, started from Jansen's lengths, 5 seeds, one run on flat ground and one on a 10° slope. The best legs were about 15 %–27 % faster than Jansen's, but mostly because the **legs got bigger** (longer stride). Rescaled to Jansen's size the gain is only about 7 %–16 % on flat ground and 4 %–14 % on the slope, and all five seeds found something better than Jansen's own lengths. The price of larger legs is a higher peak torque on most seeds (Jansen 14.7; the five seeds land between 14.1 and 18.6 N·m on flat ground). This is a result inside the model; it was not checked against a real walker.

**A flywheel does not help the start.** I expected a flywheel to carry the crank over the torque peak, so I wrote a time-domain model with inertia. The test failed, and the reason is clear: a drag sail's torque falls to zero as it approaches wind speed, so the crank can never spin fast enough to store useful energy before the peak arrives. The time-domain start wind (0.733 m/s) is within about 1% of the quasi-static estimate (0.727 m/s) and is the same for rotor inertia from 0.05 to 300 kg·m². Inertia only smooths the speed ripple a little and lets the walker coast a bit when the wind stops: even a very heavy rotor (300 kg·m²) coasts about two thirds of a revolution.

The model has no foot slip, no impact loss and no soft ground, and the legs have no mass.

## With a higher-fidelity simulator, the torque numbers above do not hold

The wind results above come from a simplified model (quasi-static plus energy conservation). I then built a multibody dynamics simulator on MuJoCo and compared it on the same 12-leg walking scenario. **A correction first:** my initial comparison used a simulator with a bug: every foot carried two coincident collision spheres that collided with each other, adding spurious contacts to all results. I found it while building the replay viewer, because the "feet in contact" flags were true at every frame. The numbers below are after the fix.

- **Stride agrees**; the simulator is about 5% to 6% higher.
- **Torque shape and amplitude now roughly match**: with the default settings the correlation of the crank-torque curve is about 0.59 and peak-to-peak is 7.7 vs 8.1 N·m. With much softer contact or low friction the correlation falls to −0.47 and 0.10.
- **Mean torque is still about 8 N·m higher**: on flat ground it is almost entirely energy lost to foot slip, which the simple model cannot see. So the **start wind and peak torque above are not validated**.
- **The simulator's own torque level depends on how friction is handled numerically**: changing the friction cone, the friction impedance ratio or the no-slip iterations moves mean torque by up to a factor of 2.4. Numerical settings (timestep within 0.5 ms, solver iterations, loop-constraint stiffness) converge on the default design — but that only holds for the default design: halving the time step from 0.2 ms to 0.1 ms was later needed to get all 16 randomly perturbed walkers through the invariant audit, and on 64 walkers the audit still fails 19 % of the time (worst opening 13.19 mm). "It converges" has to be read together with the audit.
- **The truly physical parameters have to be measured**: contact stiffness, friction coefficient, and one effect I still do not understand (heavier legs raise mean torque even though total mass is fixed). I also fixed a unit: I had labelled contact stiffness as N/m, but it is a mass-normalised stiffness (1/s²) and cannot be compared directly with a measured pad stiffness.

So I now report torque as a **range**: the same scenario is run under four friction treatments and the minimum and maximum over valid runs are shown. That is model uncertainty, not measurement error.

**An interesting finding: the start wind.** I added a "sail drive" to the simulator (no motor, a sail turns the crank in the wind). The small walker starts in a wind of about 0.49 m/s (strict criterion: one full revolution from rest within 40 s; the search bracket is (0.479, 0.500] m/s), about half of the 0.99 m/s predicted from its motor-driven peak torque: that peak contains dynamic loads at 2 rad/s and overestimates the load at standstill. The simple model's start criterion also uses the peak torque, so it is conservative in the same way. (The "about 0.45 m/s" I noted earlier came from a looser criterion; it is slightly under the 40 s threshold.)

**A cross-check from a second, independent solver.** The repository now also has K2, an independent planar rigid-body solver with true Coulomb dry friction that shares no contact code with MuJoCo. On slow flat-ground walking the two agree on stride per revolution to 0.58 %, and on start wind to 4.4 % (MuJoCo 0.692 m/s against K2 0.7226 m/s; the criterion there is one revolution within 15 s, stricter than the 40 s criterion above, so the numbers are larger). That is E1: two independent solvers agreeing, so these numbers are not an artefact of one engine. It is still not a measurement: nothing here is measured. Next is a small physical build to measure real torque and stride and fit contact stiffness and friction.

## Try it

Code, scripts and results: [strandbeest-evolution](https://github.com/meiluosi/strandbeest-evolution).

```bash
pnpm install
pnpm reproduce 20 500 150 default
```

Corrections welcome: better fitness functions, dynamics models, or primary sources on Jansen's method.
