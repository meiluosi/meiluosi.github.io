---
sourceId: 2026-10-21-fly-connectome-02-protocol-and-review
slug: fly-connectome-02-preregistration
title: "Pre-Registration, Hash Locking, and Context-Free Review"
editionLabel: ENGLISH OVERVIEW · METHODS NOTE
published: 2026-10-21T00:00:00.000Z
description: "How pre-registration, sha256 hash locking, and context-free sub-agent review changed stated conclusions across four projects in one series."
tags:
  - Pre-Registration
  - Reproducibility
  - Research Protocol
  - Null Result
category: Fly Connectome
series: Fly Connectome Research
seriesOrder: 2
lang: en
---

> **Scope note.** This is an English overview of the second Chinese article of the Fly Connectome Research series. It documents a process, not a new experiment: the seven-step protocol that every experiment in the series shares, the problems an independent review caught in four earlier projects, and the blind spots the protocol does not cover. The numbers quoted here are the ones the source reports as review findings or corrections, mostly from the continual-learning audit and the traveling-salesman rounds, and they are reported here as changes to earlier statements, not as new measurements. The article does not claim that the protocol produced the series' negative results, and it does not claim that it prevented them: it states that a pre-registration guarantees no post-hoc redefinition of the criterion, not that a setup can detect an effect.

## The failure that motivated the protocol

In the continual-learning audit, the fly-inspired method and the best simple baseline differed by very little. A sentence was written into the working document saying that prototype equivalence held.

The review report pointed out that the data showed the fly-inspired method being exceeded, by 0.054, 0.060, and 0.131 across the three benchmarks, and that the equivalence test had never been run. The wording had been rounded off along the threshold instead of tested against it.

The source is explicit about why this class of error is hard to catch from the inside: nothing about writing it felt like cheating. A personal project has no peer review, so it has to build something that argues back.

## The seven-step chain

1. **Write the pre-registration.** Before running anything, record the hypothesis, what will be compared against, the decision criteria, and the boundary that would count as equivalence, for example plus or minus 0.5 percentage points.
2. **Lock the hashes.** Record the sha256 of the code and of the registration document, so that any later edit of either file is visible.
3. **Separate tuning from testing.** Tuning uses one block of seeds, testing uses a fresh block, and no parameter is revisited after the test has been read.
4. **Run the test once.** Results are stored as one file per key, so a run can be resumed rather than restarted.
5. **Produce numbers with the registered script.** The analysis script generates the numbers; the interpretation is written separately.
6. **Review independently.** A context-free sub-agent is started, given only the code and the documents, and asked for leakage, control defects, and statistical problems. The project produced 9 reports in total: 4 on compartments, 2 on the continual-learning audit, and 3 on the traveling salesman.
7. **Correct and record.** When a conclusion changes it goes into a deviation log, and the original results are not deleted.

The statistical rules are fixed in advance as well: bootstrap over instances, and an equivalence claim requires the whole confidence interval to lie inside the boundary. If an interval touches the boundary, the only permitted description is "uncertain".

Steps 1, 2, and 6 are the ones that fail silently. A registration written after the fact looks identical to one written before it, a hash added later looks identical to a hash added earlier, and a reviewer that has already read the results cannot be unbriefed of them. In practice the source treats the deviation log, rather than the paperwork, as the record that carries the truth about which is which.

## What the review caught, project by project

### Compartments and forgetting

The initial result was attractive: gated compartments forgot less, and the overlap of KC inputs within a compartment behaved like a knob — raise the overlap, forget less. The review broke that story into four pieces.

- **The NaN cases were the same 5 seeds.** They appeared only at the two highest overlap levels, and by the rule the whole partition was excluded, leaving 1 or 2 partitions at the high-overlap end to support the regression line. Dropping those 5 seeds consistently leaves a FlyWire slope of 0.012 with interval [-0.007, 0.033], which crosses 0.
- **Overlap was entangled with in-degree.** The correlation between overlap and the variance of KC in-degree inside a compartment is -0.93. The knob was in fact adjusting which MBONs, and how many nearly input-free MBONs, got gated together.
- **The "multi-head" baseline was a sham.** It is mathematically equivalent to the single-head delta rule and produced bit-identical results, so the statement that gating itself does not reduce forgetting is void.
- **Different amounts of learning created a floor effect.** The ceiling on forgetting is learning accuracy minus 0.5, so anything that learns less naturally forgets less.

The registered matched-learning test then gave 20 equivalences and 25 negative means out of 26 comparable cells. The conclusion became that compartments appear to reduce forgetting mainly because they learn less.

### The continual-learning audit

The registered within-class sequential stream made every backpropagation baseline collapse on the first class of a task, with learning accuracy exactly at chance for two classes. An interleaved stream was added and the learning-rate grid extended; the original results were kept. The paper's dense control sat at chance level on CIFAR because of weight saturation and argmax ties across the board; after swapping to the same classifier, the sparse advantage shrank from +0.56, +0.13, and +0.07 to within about 3 percentage points. The result cache was one large file that concurrent processes overwrote, and it was changed to one file per key with a concurrency test.

### The foraging task

With 20 worlds and with 60 worlds, a hint appeared that the real wiring was slightly stronger. It vanished on a re-run: within the same batch of worlds, net reward fluctuates by about 5 purely because the agent's internal random number stream differs, which cannot separate a difference of about 10. The protocol was changed to pre-register first and then run 200 new worlds. There, real was about equal to shuffled (-0.4, interval [-3.6, +2.8]), and random sparse was better by 6.9 to 8.3.

### The traveling salesman

Each of the three rounds had its conclusion downgraded.

- **Round one, 100 instances.** "No feature mapping beats plain ant colony optimization by 0.5 percentage points" became "uncertain". The round-two advantage seen on the tuning set, about 4.84 against 2.7, disappeared on the test set; picking the best of the configurations on a tuning set overestimates, which is the winner's curse.
- **Round two, 300 instances.** The edge features correlated with distance at 0.85, so each circuit was "distance plus a little smoothing bias" and their equivalence was a necessary consequence of the design. The analysis script had been modified once after registration, the hash no longer matched, and that was listed as well.
- **Round three, turn context.** The residual of the learned value after binning by turn angle was only plus or minus 0.01 to 0.02 with no trend, meaning the learner had not learned the turn.

Three failure modes recur across the nine reports. A criterion is phrased loosely enough that a result can be rounded into it, as when "exceeded" became "equivalent" after the equivalence test had been skipped. A control degrades without anyone checking its absolute level, so a collapse is read as a large effect. And a boundary moves after the numbers are visible, as when the registered round-four setting saturated every arm and could only be recorded afterwards as a deviation. The protocol's answer to each is procedural rather than technical: a written criterion, a tested control, and a deviation log.

## Five practices worth taking

1. **Run a positive control before claiming no effect.** To write "no effect", first show that the design can detect an effect when there is one.
2. **Let intervals speak for equivalence.** Write the boundary down, and let the interval fall inside it or not.
3. **Test the controls and the baselines themselves.** Look at their absolute level, not only at the difference.
4. **Get a context-free reviewer.** A report takes minutes and every one of them changed a sentence. It is not an authority; key findings have to be re-checked by hand before anything changes.
5. **Keep the original results and write a deviation log.** Record what the registration said, what it was changed to, and why.

## What the protocol cannot fix

- **Hashes can come after tuning.** In the traveling-salesman third round the hash was locked after tuning, freezing only the tuning result; it cannot show that the design was fixed before tuning.
- **The registration itself can be wrong.** The registered data stream made the baselines degrade, and the registered round-four setting made every arm saturate. Both could only be recorded as deviations afterwards.
- **A reviewer catches only what it reads.**
- **The protocol cannot guarantee detection.** It can only guarantee that the criterion was not redefined after the fact.

## Where the Chinese original lives

The Chinese original is [How Not to Fool Yourself: Pre-Registration, Hash Locking, and Sub-Agent Review](/posts/2026-10-21-fly-connectome-02-protocol-and-review/). It holds the protocol diagram, the before-and-after figure of how the claims changed after review, the per-project review findings with their numbers, the five practices, and the file locations: the `PREREG_*.md`, `*.sha256`, `DEVIATIONS*.md`, and `verification/` entries inside each experiment directory. Its limit is the one stated above: this process produces a record and prevents post-hoc redefinition, and it does not make a weak design able to detect the effect it is looking for.

The experiment records live in one public repository, `fly-connectome-study`, with the pre-registrations, hash files, deviation logs, and review reports filed per experiment. This note is the second of eleven articles in the series; the first sets up the circuit and the shared comparison frame that every later experiment reuses.

The next note in this series covers [the foraging and swarm experiments](/en/posts/fly-connectome-03-foraging-and-swarm/).
