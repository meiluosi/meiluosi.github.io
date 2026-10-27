---
name: interactive-explainer
description: >-
  Design and implement Bret Victor, Bartosz Ciechanowski, and Theo Jansen (Strandbeest) style
  interactive physics, math, and cognitive science explainers in Svelte 5 for articles, route
  monographs, and Living Books. Use this skill whenever the user asks to add an animation,
  interactive diagram, simulation, cognitive lens, or visual explainer to a post, route, or book.
---

# Interactive Explainer & Living Book Widget Skill

This skill codifies the design philosophy, mathematical rigor, and Svelte 5 engineering standards for creating interactive explainers across the site (`/routes/[id]/`, `/books/[id]/`, `/lab/*`, and `/world/`).

---

## 1. Core Design Philosophy (The Ciechanowski / Victor Standard)

1. **First-Principles Mathematical Core (No Fake Animations)**
   - Never create purely decorative CSS keyframe loops that ignore user input.
   - Every visual element must be driven by an explicit mathematical equation, physical recurrence relation, linkage geometry, or probability projection computed in Svelte 5 `$derived` / `$derived.by`.
   - Examples:
     - *Embodied Mechanics*: Four-bar linkage cosine law or McGeer passive walker limit-cycle differential phase.
     - *Causal Inference*: Pearl's backdoor adjustment formula $P(Y \mid do(X)) = \sum_z P(Y \mid X, z) P(z)$.
     - *Information Physics*: Tishby's Information Bottleneck frontier $I(X;T)$ vs. $I(T;Y)$ parameterized by $\beta$.

2. **Bifurcation & Phase-Transition Knobs**
   - Provide **1 to 3 interactive controls** (range sliders, graph surgery toggles, or preset regime pills).
   - Every slider must cross at least one **counter-intuitive critical threshold (bifurcation point)** where the system qualitatively changes state (e.g., passive walker falling into chaotic collapse above $5^\circ$, Global Neuronal Workspace igniting at $65\%$, Cultural Ratchet slipping below $75\%$ fidelity).

3. **30-Second Epiphany Structure**
   - Every explainer widget must contain four distinct visual zones:
     1. **Header & Proposition**: Kicker tag + Title + 1-sentence theoretical proposition naming the key thinker(s).
     2. **Control Deck**: Clean sliders/toggles with live monospace readouts and "Golden Regime" hints.
     3. **Interactive Stage (SVG / Canvas)**: Crisp geometric diagrams, vector fields, or linkage kinematics + live metric cards.
     4. **Theoretical Takeaway Callout**: A bottom `.insight-box` summarizing *why* the phase transition matters philosophically or mathematically.

---

## 2. Engineering & Performance Rules

1. **Svelte 5 Runes Only**
   - Always use `let x = $state(...)` and `const y = $derived.by(() => ...)`.
   - Use **tab indentation** and **double quotes** to satisfy Biome (`pnpm format`).

2. **60fps Continuous Physics Loop & Accessibility**
   - When an explainer includes continuous motion (e.g., pendulums, oscillators, reaction-diffusion vesicles):
     - Drive the phase variable inside `onMount` via `requestAnimationFrame`.
     - Check `window.matchMedia("(prefers-reduced-motion: reduce)").matches` to default `isPlaying = false` for motion-sensitive users.
     - Always provide a visible `⏸ 暂停律动 / ▶ 开启律动` toggle button in the header.
     - Clean up `cancelAnimationFrame(rafId)` on component destroy.

3. **Automatic Global Bilingual Sync (`ZH / EN`)**
   - Include a reactive `let english = $state(false)` synced with `document.documentElement.lang` via `MutationObserver`:
     ```ts
     $effect(() => {
     	const syncLang = () => {
     		english = document.documentElement.lang.toLowerCase().startsWith("en");
     	};
     	syncLang();
     	const obs = new MutationObserver(syncLang);
     	obs.observe(document.documentElement, {
     		attributes: true,
     		attributeFilter: ["lang"],
     	});
     	return () => obs.disconnect();
     });
     ```

4. **Build-Time Static Code-Splitting (Zero Monolithic Bundles)**
   - When mounting explainers into dynamic Astro routes (such as `src/pages/routes/[id].astro` or `src/pages/books/[id].astro`), **never** import 10+ explainers into a single client-side Svelte dispatcher component.
   - Instead, mount them conditionally in the `.astro` template where `id` is known at SSG build time:
     ```astro
     {route.id === "embodied-dynamical" && <Lens12EmbodiedDynamics client:visible />}
     ```
   - This keeps each page's client JS chunk between **4 KB and 8 KB**.

---

## 3. Reference Blueprint

Before creating a new explainer, inspect [ExplainerBlueprint.svelte](./resources/ExplainerBlueprint.svelte) for the canonical layout, CSS custom properties (`--brand-paper`, `--brand-ink`, `--brand-clay`, `--brand-accent`), dark mode support (`:global(.dark)`), and 60fps loop lifecycle.
