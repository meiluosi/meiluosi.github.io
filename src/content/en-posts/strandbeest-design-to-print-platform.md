---
sourceId: 2026-10-06-strandbeest-design-to-print-platform
slug: strandbeest-design-to-print-platform
title: From Design to Print, an Open Platform for Strandbeest Walkers
editionLabel: ENGLISH EDITION · STRANDBEEST SERIES
published: 2026-10-06T00:00:00.000Z
draft: true
description: "Leg geometry, a genetic algorithm, MuJoCo simulation, 3D-print export and test-rig measurements wired into one loop, with what works and what is still an assumption."
tags:
  - Strandbeest
  - Open Source
  - Simulation
  - 3D Printing
category: Strandbeest
lang: en
---

<!-- Draft. The repository's docs/ROADMAP.md is the source of truth for status; re-check before publishing. -->

The first three posts covered the history, the geometry of the leg and re-evolving it. This one is about what I built from those pieces: an open-source platform around one loop.

**design → simulate → 3D-print → measure → calibrate → redesign**

Code: [meiluosi/strandbeest-evolution](https://github.com/meiluosi/strandbeest-evolution), MIT licensed. It targets developers, professionals and hobbyists: no login, no public hosting, runs locally or with `docker compose`.

## The parts

- **Designer** (in the browser): drag the 13 linkage lengths and watch the foot path, with labelled joints; let a genetic algorithm evolve it in the browser.
- **Fabrication**: from one design file, printable STLs for bars, crank arms (a gripping bore at the shaft end, a loose hole at the crank pin), a phase jig, spacers, standoffs, a motor bracket and a shaft coupler, plus a bill of materials, assembly notes and printability checks. Bars are assigned to layers (parts sharing a pin, or sweeping through each other as the crank turns, must not share a layer); I verified with a finer angular step than the planner uses that no two bars in a layer collide over the whole crank cycle. The web UI has a moving 3D assembly preview and a single-part STL view.
- **Simulation**: a planar MuJoCo multibody simulator driven by a scenario file. Parameter sweeps, a persistent job queue (jobs survive restarts), stored and reproducible runs.
- **Lab**: import raw test-rig data as a measurement with a quality report, overlay it on a simulation, and fit pad stiffness and friction.
- **Test rig** (`hardware/`): rig design, calibration procedures, five experiments (E0 to E4), ESP32 firmware, ArUco video tracking, a notebook template.

## What is real and what is not

- **Nothing has touched real hardware.** The firmware's logic core is unit-tested on the host, but the Arduino glue was never compiled; part recommendations are classes, not verified models; the exported parts contain assumptions (motor hole spacing, coupler, press-fit offset) and the first print will need adjusting.
- **Calibration has only been verified on synthetic data.** That shows the import, compare and fit chain is sound, not that real data will give physically meaningful values.
- **Torque level depends on how friction is treated numerically** (previous post), so the UI reports a range, not a single number.
- The simple model's start wind and peak torque are **not validated**.

## Try it

```bash
./scripts/setup-python.sh && source .venv/bin/activate
strandbeest-api                                   # backend
pnpm install && pnpm --filter @strandbeest/web-demo dev   # web UI
# or all at once: docker compose up --build
```

Issues and PRs welcome, and real measurements most of all: they are what is missing.
