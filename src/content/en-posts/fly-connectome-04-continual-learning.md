---
sourceId: 2026-10-21-fly-connectome-04-continual-learning-audit
slug: fly-connectome-04-continual-learning
title: "A Fly-Inspired Continual Learner Under Fair Comparison"
editionLabel: ENGLISH OVERVIEW · RESEARCH NOTE
published: 2026-10-21T00:00:00.000Z
description: "An audit of a fly-inspired continual learning method: rehearsal-free SLDA exceeds it, and the sparse advantage came from the dense control."
tags:
  - Continual Learning
  - Class-Incremental Learning
  - Baselines
  - Null Result
category: Fly Connectome
series: Fly Connectome Research
seriesOrder: 4
lang: en
---

> **Scope note.** This is an English overview of the fourth Chinese article of the Fly Connectome Research series. It audits one published result — FlyModel (Shen, Dasgupta, and Navlakha, arXiv 2107.07617, later in Neural Computation) — and adds no new method. The audit reproduced the method on MNIST-20 and did not reproduce its CIFAR-100 numbers, so the CIFAR-100 comparison is reported as two feature regimes used only for relative comparison. Every method in the comparison used the same features, the same task order, and 5 seeds. The audit found that a rehearsal-free SLDA baseline exceeded FlyModel on all three benchmarks, and that most of the reported sparse-over-dense advantage came from the dense control rather than from the representation. It does not show that FlyModel is useless, and it does not speak for the mushroom body circuit in general; it speaks for one comparison, with the missing opponents and the feature pipeline stated.

## What FlyModel actually is

FlyModel applies the mushroom body idea — random sparse projection plus keeping only the strongest 5% of units — to class-incremental continual learning, and reports that forgetting drops sharply, beating EWC, GEM, and BI-R. The question the audit asks is whether the conclusion survives when the same data and tuned opponents are used.

The update rule is simple. Each class owns one weight vector; while learning a class, that class's sparse features are added onto it and the other classes' weights are untouched; at prediction time the class with the largest dot product wins. So FlyModel is a class-prototype classifier on random sparse features.

Two consequences follow.

1. Its "old classes are not overwritten" property is largely constructed. The class weights are mutually independent, so learning a new class never touches an old one.
2. The paper's opponents are all backpropagation networks — EWC, GEM, BI-R, and a plain network. Not one of them is a rehearsal-free prototype baseline on the same features, such as nearest class mean or SLDA with a shared covariance. The most natural opponent is missing.

The audit's line of attack is therefore to add the missing opponent on top of the same features.

## The audit design: benchmarks, reproduction, and opponents

- **Two benchmarks.** MNIST-20 is MNIST plus FashionMNIST in 10 tasks of 2 classes each, with features from a LeNet5 trained on KMNIST, 84-dimensional. CIFAR-100 is 25 tasks of 4 classes each, with features from an ImageNet-pretrained ResNet18, 512-dimensional.
- **Reproduce first.** MNIST-20 reproduced, with final accuracy 0.707 against about 0.75 in the paper, and 0.817 against about 0.86 at 10 classes. CIFAR-100 did not reproduce: 32 × 32 input gives 0.340, 224 × 224 gives 0.508, and the paper reports about 0.41, which falls between them. Because the paper does not specify how the features were handled, the two are treated as a weak-feature and a strong-feature regime and used only for relative comparison.
- **Everything held fixed.** All methods use the same features, the same task order, and 5 seeds. Tuning uses a validation split of the training set and the test set is not touched. Every method gets the same tuning budget, and the grid is written into the registration in advance.
- **The opponents.** Nearest class mean in Euclidean and cosine forms, SLDA, the plain class-and-dot-product rule, a dense control, EWC, fine-tuning, small-buffer replay with 200 and 1000 samples, and an offline upper bound.

Hidden widths were implemented at 40 times the input dimension, giving 3360 and 20480, where the paper used 3200 and 20000. The difference is recorded in the deviations log.

## Four findings

Final accuracy across the three benchmarks:

| Method | MNIST-20 | CIFAR-100 weak features | CIFAR-100 strong features |
| --- | --- | --- | --- |
| FlyModel (paper setting) | 0.707 | 0.340 | 0.508 |
| FlyModel (validation-tuned) | 0.715 | 0.340 | 0.540 |
| SLDA (rehearsal-free) | 0.760 | 0.400 | 0.639 |
| Nearest class mean (Euclidean) | 0.601 | 0.273 | 0.533 |
| Replay ER-1000 (interleaved) | 0.761 | 0.215 | 0.432 |
| Offline upper bound | 0.872 | 0.437 | 0.677 |

The best backpropagation baseline, selected on the validation set, is ER-200 for MNIST-20 at 0.622, EWC for CIFAR-100 with weak features at 0.069, and ER-200 for strong features at 0.162.

### It is exceeded by a rehearsal-free SLDA

The result is consistent across the three benchmarks, with gaps of 0.054, 0.060, and 0.131. Even after validation tuning, FlyModel at 0.715, 0.340, and 0.540 does not catch up.

One cost has to be stated: SLDA must estimate a shared covariance and invert it, so its cost scales with the square of the feature dimension and it is not cheaper than FlyModel. It is, however, also rehearsal-free, needs no backpropagation, and is independent of sample order. The simpler nearest class mean sits below FlyModel on weak features (0.60 against 0.71, 0.28 against 0.34) and matches or slightly exceeds it on strong features (0.53 against 0.51).

### "Sparse beats dense by a lot" mostly came from the control

The paper used a dense control to argue that sparse encoding matters. The audit found the following.

- Under the registered dense control, dense features plus the class-and-dot-product rule, FlyModel's advantage is +0.56, +0.13, and +0.07.
- The paper-style alternative dense formulation, min-max, collapses to chance level on all three benchmarks at 0.049, 0.010, and 0.010. Weight saturation was the first suspect, but on MNIST-20 removing the clipping still left the dense features bad at about 0.11, while FlyModel itself without clipping fell to 0.48. The problem is not saturation: the class-and-dot-product scoring rule simply does not work well on dense features.
- In an exploratory control, added after the results were seen and labelled as such, the classifier is fixed to the same nearest class mean and only the sparse versus dense representation is compared. The advantage shrinks to about +0.03 on MNIST-20, about +0.02 on CIFAR with weak features, and -0.01 on strong features.

Sparse representation on its own matters much less than that. Most of the paper's sparse advantage came from mixing the representation with the classifier rule.

### It does beat the backpropagation baselines, with discounts

Under the registered criterion, FlyModel at least 0.05 above the best backpropagation baseline, the comparison holds for both data streams and all three benchmarks:

| Benchmark | Within-class sequential stream | Interleaved stream |
| --- | --- | --- |
| MNIST-20 | +0.084 | +0.094 |
| CIFAR-100 weak features | +0.271 | +0.228 |
| CIFAR-100 strong features | +0.347 | +0.273 |

That win needs to be read in layers.

- **EWC and fine-tuning have collapsed.** Under class-incremental single-pass training they reach 0.98 accuracy after a task and end at 0.05 to 0.10 after all tasks, remembering only the last one. This matches recent literature on EWC failing in class-incremental settings, so "better than EWC" is nearly a comparison against a failed baseline. The paper's own EWC reached 0.58 at 10 classes, and no such baseline was reproduced here.
- **The backpropagation baselines are weak on CIFAR.** ER-200 reaches only 0.11 to 0.24 with no extra tuning; a very wide hidden layer (20480) and single-pass training may contribute. This should be read as a baseline under this budget and setting, not the upper bound of backpropagation.
- **On MNIST-20, ER-1000 beats FlyModel.** Replaying 1000 samples gives 0.757 and 0.761, SLDA gives 0.760, and FlyModel gives 0.707. FlyModel only wins at the small-buffer tier.

### The forgetting metric cannot separate these methods

FlyModel, nearest class mean, and SLDA have weight forgetting of exactly 0 by construction: each class's statistics are independent, so the result is independent of task order, which was verified with tests. Their memory loss, roughly 0.09 to 0.14, comes entirely from competition in the decision space as the class count grows, not from weights being overwritten.

Memory loss therefore cannot say which of these methods remembers better. Accuracy at learning time and final accuracy are more discriminating.

## Errors the registration and the implementation made

- **The registered data stream broke the baselines.** It handed a task's two classes as one sequential block, and the backpropagation baselines collapsed on the first class within the task, with learning accuracy exactly at chance for two classes, which the review found. An interleaved stream was added and the learning-rate grid extended downward; the original results were kept.
- **"Exceeded" was written as "equivalent".** The data showed FlyModel being exceeded, but the document first said prototype equivalence held, and the equivalence test was never run. The conclusion was changed to "exceeded by rehearsal-free prototype baselines".
- **The result cache lost updates.** A single large file written by several processes overwrote itself, and it was replaced with one file per result plus a concurrency test.
- **The source of the sparse advantage was guessed wrong once.** Weight saturation was the first suspect and the exploratory control refuted it.

## What the audit does not establish

- Only one paper was audited, and the CIFAR-100 numbers were not reproduced.
- Only one feature extractor was used per benchmark: LeNet5 trained once, and a pretrained ResNet18 whose training data overlaps semantically with CIFAR-100.
- The intervals reflect seed variance only; deterministic methods have no interval.
- The paper's other two opponents, GEM and BI-R, are not included, and the exploratory control was added after the results were seen.

None of this shows FlyModel is useless. It is cheap, needs no backpropagation, stores no old samples, and remains better against small-buffer replay and against EWC. What can be said is that the fly-inspired part is not the reason it performs well: nearest class mean or SLDA is often better.

Four practices come out of it: test the controls first, since the most striking number came from a control that had collapsed; add the missing opponents, which was the most valuable step of the audit; tune the baselines properly and look at their absolute level, because a baseline that stops at chance indicates a problem with the comparison itself; and separate the representation from the classifier, or it is impossible to tell which one is doing the work.

## Where the Chinese original lives

The Chinese original is [Does a Fly-Inspired Continual Learning Method Hold Up Under Fair Comparison?](/posts/2026-10-21-fly-connectome-04-continual-learning-audit/). It holds the FlyModel structure diagram, the accuracy and sparse-versus-dense figures, the per-method result tables for both data streams, the retrospective on the registration and implementation mistakes, the boundary section, and the file locations inside the `ml-cl-audit/` directory. Its limit is the one stated above: this is an audit of one paper on two benchmarks with one feature extractor per benchmark, and the absence of a measured advantage for the fly-inspired component is a statement about that comparison, not about the mushroom body circuit.

The next note in this series covers [evolving the learning rule](/en/posts/fly-connectome-05-meta-learning/).
