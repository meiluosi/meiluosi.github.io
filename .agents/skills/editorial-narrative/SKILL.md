---
name: editorial-narrative
description: >-
  Structure and write high-density, skimmable technical essays, research notes, and Living Book
  monograph chapters using the "10-Second Pre-Flight Briefing + 4-Beat Section Rhythm" standard.
  Use this skill whenever authoring, refactoring, or polishing blog posts, literature reviews,
  architecture deep-dives, or monograph chapters.
---

# Editorial Narrative & Structural Density Skill

This skill codifies the narrative pacing and information architecture for long-form technical and research writing on this site, blending Distill-grade academic rigor with high-efficiency engineering skimmability.

---

## 1. The 10-Second Pre-Flight Briefing (文首导读契约)

Readers should build a complete mental model of a 5,000+ word article within the first 10 seconds before scrolling into the body.

1. **Frontmatter Briefing Fields**
   Whenever authoring or upgrading a substantive post (`src/content/posts/**` or `src/content/en-posts/**`), populate the structured briefing metadata in frontmatter so `AiSummary.astro` renders the unified Executive Brief & Concept Outline card:
   - `aiSummary`: 1–2 crisp sentences stating the core problem, mechanism, and boundary.
   - `corePapers`: Array of primary literature or specification anchors (`{ title, url, note }`).
   - `targetAudience`: Who benefits most and what prerequisite framing is assumed.
   - `takeaways`: 3 to 5 concrete `✅` items the reader will be able to distinguish, derive, or build after reading.
   - `verifiedNote`: Explicit currency & verification stamp (e.g., `"2026-11 审校：已对齐最新协议与可复现算例边界"`), turning passive "outdated" warnings into active editorial trust.

2. **Opening Hook (正文起手式)**
   - Never open with textbook clichés ("With the rapid development of AI...").
   - Open with a **concrete failure case, paradox, or minimal diagnostic task** (e.g., an inventory lookup where fluent text hides an unexecuted tool call, or a naive scale-up that confounds network size with edge density).
   - Explicitly state the **evidence boundary** upfront (what is run locally, what is a constructed teaching fixture, and what comes from external papers).

---

## 2. The 4-Beat Section Rhythm (四拍小节推进法)

Avoid walls of uninterrupted prose (never write 4+ consecutive paragraphs of plain text without structural anchors). Every major `##` section should progress through four beats:

### Beat 1 — Naive Baseline & Breakdown (朴素做法与失效边界)
- Start the section by asking *why the simplest approach fails*.
- Example: Why isn't a single prompt call enough? Why doesn't a raw scalar reward prevent reward hacking?

### Beat 2 — Multi-Dimensional Comparison Table (结构化对照表)
- Whenever decomposing components, comparing paradigms, or listing failure modes, use a **3-to-4 column Markdown table** instead of bullet-point prose.
- Recommended column patterns:
  - **Architecture / Module Breakdown**: `模块 / 机制 | 核心作用 | 缺失或退化时的表现 | 真实案例 / 边界条件`
  - **Paradigm Trade-offs**: `范式 / 方案 | 状态与控制流 | 优势与适用场景 | 代价与失效模式`
  - **Experimental / Audit Matrix**: `检查项 | 预期不变量 | 破坏方式（反例） | 诊断结论`

### Beat 3 — GitHub Alert Core Tension Callout (一句话核心矛盾收束)
- Immediately after a comparison table or critical derivation, crystallize the takeaway inside a single GitHub-style alert (`> [!IMPORTANT]` or `> [!NOTE]`).
- State the **core contradiction or invariant** in 1–2 sentences, optionally anchored by a compact structural equation:
  ```markdown
  > [!IMPORTANT]
  > 核心矛盾在于：真实任务需要的是「观测 $\to$ 动作 $\to$ 环境反馈」的闭环验证，而单次生成只提供了「提示 $\to$ 文本」的开环映射；缺少状态锚定的流利回答无法证明操作真实发生。
  ```
- Do not place two alert blocks back-to-back.

### Beat 4 — Visual or Executable Verification (机制图或可交互算例)
- Close the loop with a concrete artifact:
  - A clean **Mermaid diagram** (`flowchart LR`, `sequenceDiagram`, or `stateDiagram-v2` with quoted labels),
  - A minimal **teaching log / runnable Python snippet** (`CodeSandbox`), or
  - An embedded **Svelte 5 interactive explainer** (`interactive-explainer` skill).

---

## 3. Mathematical & Evidentiary Hygiene

1. **Zero Bare Unicode Math**
   - All variables, parameters, distributions, and complexity bounds ($\theta, \pi_\theta, \mathcal{L}, \mathbb{E}, Q, A_{\text{rupture}}$) must use standard inline `$...$` or display `$$...$$` LaTeX.
   - Wrap Chinese annotations inside math mode with `\text{...}` so `InteractiveMathEnhancer` links cross-equation symbols without KaTeX warnings.

2. **Reading Depth & Citation Precision**
   - Distinguish between full-text verification, specific section/appendix inspection, and abstract-only awareness.
   - Anchor key claims to exact section numbers, equations, or verifiable code fixtures.
