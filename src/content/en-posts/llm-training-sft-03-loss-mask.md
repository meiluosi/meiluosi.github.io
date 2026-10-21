---
sourceId: 2026-10-20-llm-training-sft-03-loss-mask
slug: llm-training-sft-03-loss-mask
title: "Which Tokens Should Carry the Loss: Answer Supervision and Label Boundaries"
editionLabel: ENGLISH OVERVIEW · TECHNICAL NOTE
published: 2026-10-20T00:00:00.000Z
description: "An overview of label masks, the first answer token, multi-turn supervision, and loss normalization in supervised fine-tuning."
tags:
  - Supervised Fine-Tuning
  - Loss Mask
  - Instruction Data
category: LLM Training Research
series: LLM Training Research Notes · SFT Volume
seriesOrder: 3
lang: en
---

> **Scope note.** This is an English overview of a Chinese article on supervision scope and label masks in supervised fine-tuning (SFT). The token sequence, the label table, the numeric weighting example, and the Python snippet are all constructed teaching material written by the source author: they use a hand-made token sequence rather than a real tokenizer output, and no model was trained. The article cites fixed-version library documentation to describe interfaces; it reports no training run, no loss curve, and no task measurement. A label check that passes confirms an internal convention holds, not that a trainer, a dataset, or a model behaves correctly.

## The label mask decides what is scored, not what is visible

A training sequence contains task rules, user material, and a demonstration answer. All of it participates in the computation, but not all of it should be treated as output the model is asked to imitate. Consider a long experiment record that must be read before producing a short JSON object. If every position carries a loss, training also demands that the model predict the experiment record itself. That is not a programming error; it changes how the supervision signal is distributed.

**The loss mask determines which targets enter the loss. It does not determine which inputs the model can see.**

Ignoring framework parameter names for a moment, the objectives are:

| Supervision scope | What is required to be predicted | What to watch |
| --- | --- | --- |
| Full sequence | Prompt, material, answer, and chosen control markers | Large amounts of input text also contribute loss |
| All assistant answers | Selected assistant output in every turn | Historically wrong answers may also be imitated |
| Last answer only | Selected output in the final turn | Earlier answers are context only; fewer supervised targets |

There is also a "completion-only" framing. It describes a prompt/completion segmentation, and it is not always equivalent to "all assistant messages": if the completion contains several roles, or if historical assistant replies sit inside the prompt, the two rules select different positions. So define what should be trained first, then map that definition onto the trainer's parameters. Reading a switch name is not a substitute for inspecting the actual labels.

### Eight positions to see where the first answer token goes

Below is a custom minimal token sequence. Each bracketed item is assumed to be exactly one token, for the sake of checking item by item; this is not real tokenizer output.

```text
0 [BOS]
1 [USER]
2 [material]
3 [END]
4 [ASSISTANT]
5 [answer A]
6 [answer B]
7 [END]
```

The intent is to learn answer A, answer B, and the END that terminates the answer, but not the role header. The labels for the whole sequence are:

```text
input:  BOS USER material END ASSISTANT answer A answer B END
labels:  -100 -100 -100 -100 -100   answer A answer B END
```

The article keeps the convention that the model is shifted by one internally. The actual scoring pairs are:

| Output position | Context already seen at that position | Target it predicts |
| ---: | --- | --- |
| 4 | ASSISTANT | answer A |
| 5 | answer A | answer B |
| 6 | answer B | END |

The key point: **keeping answer A as the label at position 5 is what makes the output at position 4 learn to predict it.** Excluding the output at position 4 because it is a role header would drop the first answer token from supervision. Similarly, the last input END has no following target, but END itself is still a label that the previous position predicts. These are two different senses of "last". If an implementation already separates inputs and targets externally, the labels above must be converted into that external shifted form; marking ranges on the raw sequence and then shifting them a second time by output position is a double shift.

## Ignoring prompt loss does not hide the prompt

Setting the user material's labels to the ignore value does not remove the corresponding input embeddings. At the answer positions, causal attention can still read the material.

Moreover, backpropagation from the answer loss can still pass through the representations at prompt positions and the parameters that produce them. "The prompt does not carry loss" therefore does not mean "the prompt did not participate in learning", and certainly not "the prompt's embeddings receive no gradient".

Three things need to be kept apart:

- **Input present** — is the text fed to the model at all?
- **Attention visible** — can the current output read it?
- **Target scored** — is the model directly asked to predict the token at that position?

Supervising only the answer typically preserves the first two and selectively keeps the third. Blocking the whole prompt with an attention mask does not achieve the same effect: it would leave an extraction task without the evidence it needs.

### Keeping wrong history in a multi-turn record

Suppose the first assistant turn wrongly filled an unrecorded step count with 10000, the user then says "the material does not state the step count, do not guess", and the last assistant turn corrects it to `null`. If the goal is to train correction, the wrong history can stay as visible context while only the correct final answer is supervised. That does not directly require the model to reproduce the wrong answer.

Under a rule where every assistant message is scored, the erroneous earlier reply enters the imitation target, which may conflict with the current training intent. On the other hand, if the historical answers were correct demonstrations, supervising all of them provides more valid targets. But if each turn is also split into its own training sample while the full dialogue is supervised again, early replies are counted twice. No rule fits every dialogue automatically. Label each sample with its role and turn provenance, state which replies are trusted and which are only erroneous background, and then check the resulting target positions.

## End markers, framework switches, and truncation

The extraction task needs the model to stop once the JSON is complete. If label filtering deletes every real end marker, the model loses direct supervision for stopping. But "put all EOS back in" is not a safe answer either. Some templates distinguish the end of a message from the end of a turn; some role headers are supplied by the prompt while other control tokens belong to the generated range. The decision should follow the target model's template and inference stopping rule. If padding and EOS share an ID, handle them by their actual positions rather than ignoring every occurrence of that ID. The most direct acceptance check is to print the last few valid labels in readable form and confirm that both the answer content and the expected end marker are present, with no padding mixed in.

Using TRL 0.23.1 as a fixed reference, `assistant_only_loss=True` applies to assistant replies in conversational data, and the documentation requires the template to be able to return the corresponding token mask. Prompt-completion data has a separate `completion_only_loss` segmentation convention; the two can be combined, but not every input format can be dropped into either ([TRL SFT Trainer](https://huggingface.co/docs/trl/v0.23.1/en/sft_trainer)). The Transformers 4.57.1 template interface also offers `return_assistant_tokens_mask`, whose support depends on generation-region markers in the template ([tokenizer template interface](https://huggingface.co/docs/transformers/v4.57.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template)). These documents express intent; they do not replace output inspection. Even with a switch set to true, spot-check the supervision scope of every turn and confirm that role headers, end markers, truncation, and the first answer token behave as intended. The article uses these fixed versions to explain semantics and does not claim that any local training project installed them or ran the corresponding configuration.

A convenient-looking alternative is to find the `assistant` marker in the text and treat everything after it as supervised. It runs into several problems: multiple boundaries in a multi-turn record; user material that quotes the same words; template markers that are hard to distinguish from ordinary prose; and the fact that a string tokenized on its own may not have the same boundaries as the same string inside its full context. A more reliable starting point is structured messages plus the generation-region information the template provides, or an explicit position mapping maintained in a custom pipeline. Whichever mechanism is used, it needs to be tested with material that contains boundary-style text.

Truncation can also leave a sample with no valid label at all. Suppose the maximum length is short, long material fills the window, and the whole answer is cut away: the input tensor still holds many tokens, but the loss mask is all zeros. Such a sample cannot be averaged in with the rest as ordinary training data; it produces ineffective updates, or non-finite loss if handled carelessly. It should be excluded explicitly, resegmented, or handled by another defined policy, with the count recorded. A second case is harder to notice: only half the answer survives, so valid targets exist but the end marker is gone. A program that checks "at least one label exists" cannot know the sample no longer matches the complete task. Checks therefore have to cover both the number of valid targets and the completeness of the target sequence.

## Sample weights and a minimal label check

Suppose two samples have effective answer lengths of 2 and 8, with mean token losses of 1 and 3. Averaging over valid tokens gives

$$
L_{\text{token}}=\frac{2\times1+8\times3}{10}=2.6,
$$

while averaging over samples gives

$$
L_{\text{sample}}=\frac{1+3}{2}=2.
$$

The first gives every target position equal weight, so longer answers contribute more; the second gives every record's average loss equal weight. Which one applies should match the task and the comparison being asked. If the goal is to compare full-sequence supervision with answer-only supervision, the two loss values cannot simply be compared: they score different sets of positions, so the numbers are not the same measurement. A better approach fixes the data and the training budget, computes validation metrics on the same answer positions, and then looks at the extraction task result.

The snippet below checks only the teaching sequence and the internal shift convention. It does not depend on a training framework and does not generate a real chat template:

```python
# Each item is a custom teaching token ID, not real tokenization output.
ids = [0, 1, 2, 3, 4, 5, 6, 3]
answer_mask = [0, 0, 0, 0, 0, 1, 1, 1]
labels = [token if keep else -100
          for token, keep in zip(ids, answer_mask)]

# Internal shift: the output at position t predicts labels[t + 1].
pairs = [(t, labels[t + 1])
         for t in range(len(ids) - 1)
         if labels[t + 1] != -100]
assert pairs == [(4, 5), (5, 6), (6, 3)]
```

It verifies the position relationship between the first answer and the end marker. Real data must also cover multi-turn records, left and right padding, empty answers, templates that do not support masking, truncation, and packing. Passing this small check is not evidence that a whole trainer is correct.

## The Chinese original and the limits of this note

The full Chinese article carries the eight-position token listing, the label table, the pairing table, the Python snippet, the multi-turn and truncation walkthroughs, and the reference list with the TRL 0.23.1 and Transformers 4.57.1 links — material summarized but not reproduced here. The original is at [the Chinese source article](/posts/2026-10-20-llm-training-sft-03-loss-mask/); the next note covers [how instruction data is selected and generated](/en/posts/llm-training-sft-04-data-selection/). The main limit is unchanged: the label conventions described here were checked against a hand-made teaching sequence, and no training run establishes that any supervision scope performs better.
