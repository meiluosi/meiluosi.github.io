---
sourceId: 2026-10-21-fly-connectome-10-review-checklist
slug: fly-connectome-10-review-checklist
title: "Twelve Ways a Project Can Fool Itself: A Review Checklist Built from Real Errors"
editionLabel: ENGLISH OVERVIEW · REFLECTION NOTE
published: 2026-10-21T00:00:00.000Z
description: "Twelve self-deception patterns that actually occurred in this project, how review caught each one, and the seven practices that followed."
tags:
  - Fly Connectome
  - Review Checklist
  - Pre-registration
  - Negative Results
category: Fly Connectome
series: Fly Connectome Research
seriesOrder: 10
lang: en
---

> **Scope note.** This is an English overview of the tenth article in the Fly Connectome Research series, a Chinese reflection note. The source collects twelve concrete self-deception patterns that occurred in this project, each with the stage it appeared in, the review that caught it, and the correction that followed. It also reports the project's own multiplicity check, its independent re-implementations, and the limits of same-model review. It is a checklist of method failures, not a set of new results, and it contains no claim that the real wiring is useful.

## Twelve patterns, each from an error that actually happened

Every stage of the project was reviewed by an independent sub-agent with no context, given only the code, the pre-registration documents, and the results, and required to verify each finding before reporting it. Every round changed part of the written conclusion. The source assembles them into a table.

The reviews were not a formality. They ran the code, recomputed quantities the text described, and compared what the pre-registration had promised against what the results contained. The table below lists the twelve patterns in the order they were caught, with the stage each belongs to and the correction that followed.

| # | Practice | Where it appeared | How it was found | What changed |
| --- | --- | --- | --- | --- |
| 1 | Positive control used the same feature source, so it was circular | stage 0 | Review: the teacher and the real student shared the same wiring | Replaced with a non-circular teacher (30% reshuffle, newly drawn shuffled wiring) |
| 2 | The metric's implementation did not match its name | stage 0 | Review ran the code: "after 50 trials" was reading trial 40, and "silencing" had become maximum suppression | Fixed and re-run, the old column marked void |
| 3 | Writing "was surpassed" as "equivalent" | continual-learning audit | Review checked the data | Wording corrected |
| 4 | Reading out a general property and treating it as a test of the wiring | stage 2 | Review: sparse coding plus error-driven updating alone can produce it | Counted only as a normality check, not as evidence |
| 5 | Attributing several differences that were mixed together to one cause | stage 2 | Review: the unit-weight learner's encoding is unstable (the same odour's two encodings overlap only 0.61), not "synapse strength" | The mechanistic attribution removed |
| 6 | Pass or fail decided under a fixed learning rate | stage 2 | Review re-ran it: after changing the learning rate, a learner that had originally failed also passed | Marked as learning-rate dependent |
| 7 | "Matched" meant count, not effective dose | stage 5 | Review computed it: silencing 818 high-frequency Kenyon cells removes 93% of the encoding positions, while random silencing removes only 40% | Conclusion changed to "not detected"; 4 of the main conclusion family are near-tautologies |
| 8 | A ranking metric confounded with channel activity | stage 5b | Review ranked by the most active channel, independently of the wiring; the excess loss was larger | The hub effect attributed to activity; the "bridger" (participation coefficient 0.50 to 0.65) is almost not a bridge at all |
| 9 | Computing rankings separately per wiring and then comparing them | stage 5b | Review: the shuffled wiring's fan-out is almost flat and the hub set overlaps the real one by only 1 to 4, so a null effect is nearly inevitable | Renamed to "the comparison is between different channel sets" |
| 10 | The best point of the tuning grid sits at the edge | stage 6b | All three compartment learners' best learning rates were on the grid's upper bound | The grid was relaxed and the run repeated |
| 11 | Updates have no upper bound | stage 6c | Review checked the synapse values: at learning rate 4.0 they land in [−6.9, +9.5]; in the control clipped to [0, 1] the advantage disappears (roughly 8 against 54) | Bounded version re-run, the advantage goes to zero |
| 12 | Writing "not significant" as "no stable advantage or disadvantage" | stage 6c | Review: the intervals [+2.4, +14.6] and [−14.9, −3.7] both exclude 0, with opposite signs | Wording changed to "cannot conclude" |

Two of these deserve a second reading. Item 4 is the most consequential for the series: a property that a sparse code with error-driven updating can produce on its own is not evidence about wiring, and it was demoted to a normality check. Item 12 is the wording trap the project had already hit once, in which a real difference was softened into "no stable advantage or disadvantage" even though the two intervals excluded zero in opposite directions.

The corrections are as instructive as the errors. Most of them did not require new data: they required restating the existing measurement with the right control, the right wording, or the right magnitude. The one that required a new run was item 10, where the best tuning point sat on the edge of the grid and the grid had to be widened before the comparison could be read at all.

The pattern across the table is that the errors cluster on the side of a comparison rather than on the side of a computation. The arithmetic was usually right and the label attached to it was wrong: a metric that did not compute what its name promised, a verdict written past the data, a mechanism named for an effect that had more than one cause. The pipeline itself was sound enough for the corrections to be expressible as corrections to the write-up, and in most cases as corrections to the write-up alone.

## Three patterns running behind the twelve

- **Apparent advantages mostly come from mismatched controls.** Items 1, 5, 7, 9, and 11 are all cases where the control differed from the object being compared along a second axis: feature source, encoding stability, effective dose, channel set, or the numerical range of the update. In each case the comparison would have looked clean and the conclusion would have been about the wiring, when in fact it was about something the control had changed. Before each comparison, the source's advice is to write down what else differs between the two sides besides the one being tested.
- **A reviewer has to be able to run the code.** Items 2, 7, and 11 were only found after the reviewer executed a few lines of code. Reading the documentation would not have surfaced them, because a document can describe a metric and the code can implement a different one. This is also why the reviewer is given the code rather than only the results.
- **"No effect" depends on whether the design can see one.** The positive control added after item 1 is the precondition for every later null result in the series. A comparison that never demonstrated sensitivity cannot support a null, and the series had already produced several nulls before that demonstration existed.

## Multiplicity, re-implementation, and the literature

- **Global multiplicity.** Each stage corrected within its own family first. Putting the 67 headline comparisons into a single family and applying a Holm correction using a normal approximation back-derived from the intervals leaves 42 still significant, and the 16 judged "meaningful (strict)" within stages are all among them. Among results judged meaningful or small but real within a stage, only two stop being significant after the global correction. A uniform correction does not change any stage's headline conclusion; the main uncertainty comes from effect size and design, not from multiplicity. The correction rests on a normal approximation back-derived from the intervals, so its coverage is approximate rather than exact.
- **Independent re-implementation.** Three stages were re-implemented by another sub-agent from the written specification, without seeing the original code or results. Stage 0b and stage 4 match the original values; stage 3 has the same direction but a magnitude 2 to 4 times smaller, mainly because the Kenyon cell count differs.
- **Literature checking.** Five research agents read the relevant literature, and anything read only as an abstract or from memory was marked unverified. As a result, claims such as "Kenyon cell wiring is close to random", "the dopamine signal is a prediction error", and "consistent with Lin 2014" were all narrowed.

The pattern across all three self-checks is the same. None of them produced a reason to discard the project's headline conclusions, and each of them did change a specific sentence, a specific number, or the confidence attached to it. The multiplicity correction left the stage-level conclusions intact; the re-implementation confirmed stage 0b and stage 4 and shrank stage 3 by a factor of 2 to 4; the literature pass narrowed three claims that had been stated too broadly.

## The limits of same-model review

- **The reviewers are the same model family.** Cross-model research indicates that same-family models correlate more on errors and tend to agree, so they are not an independent error source. The practice here is to have reviewers run the code, verify each finding, and describe the activity as same-model, context-isolated review rather than peer review.
- **The re-implementer reads the original implementer's written specification.** Flaws in the specification itself are not discovered this way.
- **No human expert review.** Neuroscientific judgements, such as the PAM/PPL1 valence rule, have not been checked by a domain expert.
- **The 0.03 minimum meaningful effect was chosen in advance.** There is no independent argument that it sits above the test-set noise floor, so strictly "equivalent" means "not detected within this bound".
- **The literature was read shallowly.** The retrieval tooling returned model-summarised pages, and only a few papers were read mostly in full.

These limits are the reason the source names the practice rather than implying it. Same-model, context-isolated review is a real check on code and on arithmetic, and it caught the twelve patterns above. It is not a second opinion about whether the premise of the experiment is right, and it does not substitute for someone who can judge whether the mushroom body wiring means what the model assumes.

## Seven practices that can be taken directly

1. Before every comparison, write down what else differs between the two sides.
2. A positive control must be non-circular: the teacher cannot be the instance the student already has.
3. Plot where the best point sits on the tuning grid; if it is at the edge, relax the grid and re-run.
4. Learning rules should be checked for numerical range — whether synapse values go out of bounds.
5. Reviewers must run the code, and every finding must be verified before it is reported.
6. When the control and the object are not matched, state what "matched" means: count, dose, or learning level.
7. After a null result, write "not detected", not "does not exist".

The seventh is the one the series needed most, and it is the reason the closing article states its null result in the conditional: no evidence that the real wiring helps, rather than proof that it does not.

The first and the seventh are the pair the series ends on. The first says that before any comparison, write down the second axis along which the two sides differ, because a difference found there is a difference that will be misread as a difference in the wiring. The seventh says that once the comparison has been made honestly, the honest report of a null is a bounded non-detection. Neither of the two requires a new experiment, and between them they cover most of what went wrong in the twelve cases above.

## Where the Chinese original lives

The Chinese original is [What the Reviews Caught: A Checklist for Not Fooling Yourself](/posts/2026-10-21-fly-connectome-10-review-checklist/). It carries the review reports themselves — the stage 0, stage 2, stage 4, stage 5, and stage 6 documents, the replication directory, the multiplicity analysis, and the literature review — together with the pointers into the project repository at <https://github.com/meiluosi/fly-connectome-study> (directory `fly-v4/verification/`).

Its main limit is the one stated above: this is a record of method failures and corrections, produced by reviewers drawn from the same model family as the implementers, with no human expert review and only shallow literature verification. The checklist is transferable; the assurances it gives about this project's own conclusions are not independent.

The Chinese original also carries the artefacts behind the table rather than only the table: the five review reports themselves, the replication directory with the independent re-implementations, the multiplicity analysis document, and the literature review with its unverified markers. Those are the files a reader should consult before reusing any of the twelve corrections as a general practice, because each of them is tied to a specific measurement in this project.

The checklist's own scope is worth stating plainly, because it is easy to read a list of twelve failures as a claim that the project's conclusions are correspondingly weak. The failures were found and corrected, and the corrections are the point: what the series demonstrates is a procedure that produces a bounded null result and says which parts of the bound are still soft.

One consequence is that the checklist is reusable and the specific findings are not. A "how to fool yourself" list transfers to a different project; the 0.61 encoding overlap, the 818 silenced Kenyon cells and the [−6.9, +9.5] synapse range belong to the measurements they came from and mean nothing outside them.
