---
sourceId: 2026-10-06-strandbeest-design-to-print-platform
slug: strandbeest-design-to-print-platform
title: "From Design to Print: An Open Platform for Designing and Validating Strandbeest Walkers"
editionLabel: ENGLISH EDITION · STRANDBEEST SERIES
published: 2026-10-06T00:00:00.000Z
draft: false
description: "Leg geometry, a genetic algorithm, MuJoCo simulation, 3D-print export and test-rig measurements wired into one loop, with what works and what is still an assumption."
tags:
  - Strandbeest
  - Open Source
  - Simulation
  - 3D Printing
category: Strandbeest
lang: en
---

<!-- The repository's docs/ROADMAP.md is the source of truth for status; numbers follow docs/EXPERIMENTS.md. Re-checked against the upstream repository meiluosi/strandbeest-evolution on 2026-10-22. -->

The first three posts covered the history, the geometry of the leg and re-evolving it. This one is about what I built from those pieces: an open-source platform around one loop.

**design → simulate → 3D-print → measure → calibrate → redesign**

Code: [meiluosi/strandbeest-evolution](https://github.com/meiluosi/strandbeest-evolution), MIT licensed. It targets developers, professionals and hobbyists. The backend has no login and is not publicly hosted: it runs locally or with `docker compose`. There is also a public **static** demo (no backend needed — open it and drag the linkages, or run the browser-side simulation): [live demo](https://meiluosi.github.io/strandbeest-evolution/).

## The parts

- **Designer** (in the browser): drag the 13 linkage lengths and watch the foot path, with labelled joints; let a genetic algorithm evolve it in the browser.
- **Fabrication**: from one design file, printable STLs for bars, crank arms (a gripping bore at the shaft end, a loose hole at the crank pin), a phase jig, spacers, standoffs, a motor bracket and a shaft coupler, plus a bill of materials, assembly notes and printability checks. Bars are assigned to layers (parts sharing a pin, or sweeping through each other as the crank turns, must not share a layer); I verified with a finer angular step than the planner uses that no two bars in a layer collide over the whole crank cycle. The web UI has a moving 3D assembly preview and a single-part STL view.
- **Simulation**: a planar MuJoCo multibody simulator driven by a scenario file. Parameter sweeps, a persistent job queue (jobs survive restarts), stored and reproducible runs. Beside it sits an **independent** solver, K2 (planar rigid bodies, true Coulomb dry friction, sharing no contact code with MuJoCo): on slow flat-ground walking the two agree on stride per revolution to 0.58 % and on start wind to 4.4 %, which is E1 (two independent solvers agree).
- **Lab**: import raw test-rig data as a measurement with a quality report, overlay it on a simulation, and fit pad stiffness and friction.
- **Test rig** (`hardware/`): rig design, calibration procedures, five experiments (E0 to E4), ESP32 firmware, a notebook template; the ArUco video tracking lives in `packages/rig`.
- **Community data intake** (`docs/HOW-TO-MEASURE.md`): a bilingual "how to measure a walker" guide, a Measurement submission format and a validator, so a stranger can follow the guide and hand in a submission that passes validation.

## What is real and what is not

- **Nothing has touched real hardware.** The firmware's logic core is unit-tested on the host, but the Arduino glue was never compiled; part recommendations are classes, not verified models; the exported parts contain assumptions (motor hole spacing, coupler, press-fit offset) and the first print will need adjusting.
- **Calibration has only been checked on synthetic data.** That shows the import, compare and fit chain is sound, not that real data will give physically meaningful values. One step further: on synthetic data **contact stiffness and friction are not identifiable** — the posterior intervals cover most of the prior, the point estimates must not be used, and the intervals themselves are the result.
- **Torque level depends on how friction is treated numerically** (previous post), so the UI reports a range, not a single number.
- The simple model's start wind and peak torque are **not validated**. The model's own start wind is now 0.489 m/s (bracket 0.479–0.500; criterion: one revolution from rest within 40 s), but the thing most likely to spoil that number is not a wrong model — it is **pin friction**: at just 0.5 mN·m per joint the required wind rises to 1.02–1.07 m/s (+110 % to +160 %). Pin friction is the biggest unknown here, and only a measurement can settle it.

## Try it

```bash
./scripts/setup-python.sh && source .venv/bin/activate
strandbeest-api                                   # backend
pnpm install && pnpm --filter @strandbeest/web-demo dev   # web UI
# or all at once: docker compose up --build
```

Issues and PRs welcome, and real measurements most of all: they are what is missing. The repository has a written guide, "How to measure a walker" (`docs/HOW-TO-MEASURE.md`, bilingual), so that a submission following it passes validation — what is missing now is not the channel but the data.
