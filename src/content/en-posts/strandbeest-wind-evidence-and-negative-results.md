---
sourceId: 2026-10-28-strandbeest-wind-evidence-and-negative-results
slug: strandbeest-wind-evidence-and-negative-results
title: "How Much Wind Does a Wind-Only Strandbeest Need? Evidence Tiers, Two Solvers, and Negative Results"
editionLabel: ENGLISH EDITION · STRANDBEEST SERIES
published: 2026-10-28T00:00:00.000Z
draft: false
description: "Two independent physics solvers (MuJoCo and K2), E0–E3 evidence grading, a 30-cell scale study on start wind speed, and our hardest negative results: why pin friction shifts start wind by 2.59x, why kinematic screening anti-correlates with dynamics among top candidates, and why robust optimization flips across solvers."
tags:
  - Strandbeest
  - Simulation
  - Dynamics
  - Evidence Grading
category: Strandbeest
lang: en
---

<!-- Mirrors docs/REPORT.md and docs/EXPERIMENTS.md (§1–§40) in meiluosi/strandbeest-evolution. -->

> **The single most important sentence**: **Every number in this report comes from our simulators (MuJoCo, plus an independent planar rigid-body solver K2); not a single one comes from a physical walker.** The E3 tier (traceable physical measurement) is currently empty. Every claim below carries an explicit E0–E3 evidence tier.

This is Part 5 of the Strandbeest series. After covering Theo Jansen's twelve evolutionary periods, planar leg geometry, re-evolving the linkage with a genetic algorithm, and the open design-to-print platform ([meiluosi/strandbeest-evolution](https://github.com/meiluosi/strandbeest-evolution), [live demo](https://meiluosi.github.io/strandbeest-evolution/)), this report addresses our flagship research question:

> **For a walker driven by wind alone — sail for energy, gravity for help, no electricity and no pre-tensioned springs — how much wind does it need to start, and below what size does it stop working?**

---

## 1. Why the Answer Is Not Obvious

Wind force on a sail scales with area ($L^2$), while weight scales with volume ($L^3$ for solid bodies, or between $L$ and $L^3$ for thin-walled tubes), and pin friction depends on normal load or clearance fits. **The sign of the scaling exponent decides everything**: a larger machine can be either easier or harder for the wind to start. Intuition cannot settle this; only simulation and measurement can — and any simulation rests on physical parameters nobody has measured yet.

To keep ourselves honest, every numerical claim is graded on a four-tier evidence ladder (`docs/VALIDATION.md`):

| Tier | Meaning | Examples in this report |
|---|---|---|
| **E0** | Single-model simulation only | Flagship scale table (MuJoCo cells), multifidelity rank correlation, robust optimization rankings |
| **E1** | At least two **independent** solvers agree within pre-agreed tolerance | Slow flat-ground stride (0.58 % gap); **order of magnitude** of start wind (4.2 %–13.9 % gap) |
| **E2** | Matches **published data or literature** within uncertainty | Jansen foot trajectory vs. Pop et al. (2014) experimental rig |
| **E3** | Validated by **traceable physical measurement** | **Empty (0 records).** |

**Verification $\neq$ Validation**: two solvers agreeing proves we solved our equations correctly (*verification*), not that our equations match physical reality (*validation*).

---

## 2. What Holds Up (E2 and E1)

### 2.1 Kinematics Matches Published Experimental Trajectories to Within a Millimetre (E2)

We compared our closed-form Jansen foot trajectory against the experimental CCD-tracked trajectory published by Pop, Dolga & Pop (2014) (*Applied Mechanics and Materials* 658: 684–689):

| Metric | Literature (Pop 2014) | Our Analytic Kinematics | Difference | Relative Error |
|---|---|---|---|---|
| Stroke length $L_{\text{stroke}}$ | 57.1 mm | 57.081 mm | −0.019 mm | **0.033 %** |
| Step lift $H_{\text{lift}}$ | 22.5 mm | 22.457 mm | −0.043 mm | **0.191 %** |
| Duty factor $D$ | 0.43 | 0.433 | +0.003 | **0.77 %** |
| Shape distance | — | 0.0295 (2.95 % of path width) | Stance phase < 0.54 mm | Below digitization uncertainty (±0.80 mm) |

This lifts our kinematics layer to **E2**, though it says nothing about dynamic stride, torque, or start wind.

### 2.2 Two Independent Physics Solvers Agree on Stride (E1)

Start-wind thresholds sit right in the low-speed stick-slip regime where MuJoCo's softened friction regularization is least trustworthy. Behind our `PhysicsBackend` boundary, we built a second, completely independent planar rigid-body solver **K2**: zero MuJoCo imports, 4-DOF reduced coordinates with analytic dyad projection (**loop-closure error is 0.00 mm by construction**), Moreau-style velocity time-stepping, and **unregularized Coulomb dry friction** with unilateral Signorini contact.

On slow flat-ground walking ($\omega = 2.0\text{ rad/s}$) for the 6-leg small Jansen walker:

| Quantity | MuJoCo | K2 (Independent) | Relative Diff | Evidence Tier |
|---|---|---|---|---|
| **Stride per revolution** | 0.2760 m | 0.2776 m | **+0.58 %** | **E1** |
| Mean body height | 0.1694 m | 0.1700 m | +0.35 % | **E1** |
| Start wind (15 s criterion) | 0.6920 m/s | 0.7226 m/s | **+4.4 %** | **E1 (magnitude)** |
| Mean drive torque | 2.16 mN·m | 2.54 mN·m | **+17.6 %** | **E0** |

Stride reaches **E1**; torque differs by 17.6 % and remains sensitive to friction regularization (**E0**).

---

## 3. Flagship Experiment: Start Wind vs. Scale (E0)

We define **start wind speed** as the minimum steady wind required to turn the crank one full revolution from rest within 40 s of simulated time (bracketed via binary search to 5 % relative tolerance, always reported as an interval `(lo, hi]`).

### 3.1 One Untested Pin-Friction Assumption Dominates Everything

At baseline scale ($L = 1$) with ideal frictionless pins, the 6-leg walker starts at **(0.479, 0.500] m/s** (estimate 0.489 m/s). One-at-a-time sensitivity sweeps across 9 assumption parameters revealed:

- **`joint_friction` (Coulomb friction per printed pin) is the sole dominant parameter**: raising it from $0$ to $0.0005\text{ N}\cdot\text{m}$ more than doubles the start wind to **(1.02, 1.07] m/s**; at $0.002\text{ N}\cdot\text{m}$ it reaches **(1.27, 1.33] m/s** (**a 2.59× shift**).
- **The energy account explains why**: under motor drive on flat ground, 96.5 % of input work is dissipated in foot-ground slip; under sail drive at 4 m/s wind, **pin friction and damping consume 82 % of input work** (ground contact takes only 17 %).
- **The "minimum viable size" boundary is drawn by friction, not scale**: under a $4.17\text{ m/s}$ (~15 km/h) reference ceiling with pin friction at $0.002\text{ N}\cdot\text{m}$, $L = 0.02$ cannot start even in a $30\text{ m/s}$ gale (`(30, inf]`) and $L = 0.03$ needs $18.4\text{ m/s}$, bounding the minimum viable scale to **$L \in (0.03, 0.1]$**. With ideal pins, the smallest tested scale starts at just $0.80\text{ m/s}$.

### 3.2 The Scale Curve Is Non-Monotonic

We pre-registered our scaling predictions *before* running the 30-cell grid (sail torque $\propto L^3 v^2$, gravity load torque $\propto L^{p+1}$, predicting $v \propto L^{(p-2)/2}$):

- **Thin-walled scaling ($p = 1$)**: pre-registered slope $-0.5$; measured small-scale slope **$-0.513$** (global $-0.337$). Bigger machines need less wind ($0.80\text{ m/s}$ at $L = 0.35$ down to $0.39\text{ m/s}$ at $L = 3$).
- **Critical scaling ($p = 2$)**: pre-registered slope $0.000$; measured global slope **$-0.033$**.
- **Solid scaling ($p = 3$)**: the curve is a **non-monotonic V**, dropping from $0.529\text{ m/s}$ at $L = 0.35$ to a minimum of **$0.415\text{ m/s}$ near $L \approx 0.6$**, then rising with slope **$+0.595$** at large scales ($0.865\text{ m/s}$ at $L = 3$, matching the pre-registered $+0.5$). This **refutes** our earlier blanket claim in `WIND.md` §3 that "larger bodies are more favourable to the wind".

---

## 4. Five Hard Negative Results

1. **Kinematic screening anti-correlates with dynamic performance among top candidates ($\rho_s = -0.80$)**:
   While kinematic evaluation is 10,000× cheaper than dynamics, within the top 5 candidates selected by kinematics, the rank correlation with dynamic stride is **$-0.8000$ (K2) / $-0.7000$ (MuJoCo)**. The #1 kinematic candidate ranks 5th in dynamics, while the #3 dynamic walker is ranked 9th by kinematics and eliminated. **Rule: use kinematic screening only to discard the bottom 25 %, never to rank or pick a winner.**
2. **Robust optimization (CVaR) flips its winner across solvers**:
   Under K2, a CVaR-optimized design beats the nominal winner by $+0.273\text{ m}$ on the worst-case wind/slope scenario; re-evaluated in MuJoCo, it **loses by $-0.271\text{ m}$**. Per our validation rules, rankings that fail cross-solver stability are mechanism demonstrations, not conclusions.
3. **Adding sail area and gear ratio to the genome bought nothing**:
   At equal weights and compute budget, the **fixed sail beat the evolved sail on all three metrics** (start wind 4.4 % lower, gust CVaR 11.8 % higher, nominal distance 29.4 % higher). Furthermore, the cheap dynamometer proxy inside the loop correlated poorly with true start-wind search ($\rho = 0.303$) and inverted the ranking of the top pair.
4. **Contact stiffness and ground friction are unidentifiable together**:
   Bayesian calibration on synthetic data recovers intervals that contain the truth, but the posteriors span most of the prior — point estimates are unusable; only the interval is honest.
5. **A silent `NaN` once scrambled genetic-algorithm rankings**:
   Linkages that assembled at 24 probe angles but failed on the 90-point fine grid returned `NaN` in TypeScript vs. `-5.05` in Python, causing arbitrary sort order in JS. Fixing this and locking it in `contracts/fitness/` required regenerating our historical GA baselines.

---

## 5. Call for the First E3 Measurement

Our kinematics is backed by literature (**E2**) and our stride by two independent solvers (**E1**), but every statement about torque and start wind remains inside the model (**E0**).

If you have a 3D-printed or Lego Strandbeest, check out our bilingual guide [`docs/HOW-TO-MEASURE.md`](https://github.com/meiluosi/strandbeest-evolution/blob/main/docs/HOW-TO-MEASURE.md). Contributed data defaults to **CC0 1.0 (public domain + scientific attribution)**, and a single traceable measurement of crank torque $T(\theta)$ or start wind speed will become this project's very first **E3** evidence record.
