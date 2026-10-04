"""Small local datasets only; tokenizer fitting is isolated to the training split."""
from __future__ import annotations
import json
import os
from pathlib import Path
import torch
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from .planning import digest, dump

SPECIALS = ["<pad>", "<eos>", "<user>", "<assistant>"]
PAD, EOS, USER, ASSISTANT = range(4)


def rows(path: Path):
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("In-memory baseline accepts at most 32 MiB per JSONL file")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_prepared(path: Path):
    manifest = json.loads((path / "manifest.json").read_text())
    for split in ("train", "validation", "test"):
        if digest((path / f"{split}.jsonl").read_bytes()) != manifest["splits"][split]["sha256"]:
            raise ValueError(f"Prepared {split} data differs from manifest")
    return manifest


def fit_tokenizer(prepared: Path, output: Path, vocab_size: int):
    verify_prepared(prepared)
    if vocab_size < 260:
        raise ValueError("Byte BPE needs 256 byte symbols plus four special tokens")
    if output.exists():
        raise ValueError(f"Refusing to overwrite tokenizer: {output}")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=SPECIALS,
                                 initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tokenizer.train_from_iterator((row["text"] for row in rows(prepared / "train.jsonl")), trainer)
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output))
    result = {"kind": "train-only-byte-bpe", "requested_vocab": vocab_size,
              "actual_vocab": tokenizer.get_vocab_size(), "special_tokens": SPECIALS,
              "tokenizer_sha256": digest(output.read_bytes()),
              "train_sha256": digest((prepared / "train.jsonl").read_bytes())}
    output.with_suffix(".meta.json").write_text(dump(result))
    return result


def verify_tokenizer(path: Path, prepared: Path):
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    if meta["tokenizer_sha256"] != digest(path.read_bytes()):
        raise ValueError("Tokenizer differs from recorded hash")
    if meta["train_sha256"] != digest((prepared / "train.jsonl").read_bytes()):
        raise ValueError("Tokenizer was fitted on different training data")
    tokenizer = Tokenizer.from_file(str(path))
    if [tokenizer.token_to_id(t) for t in SPECIALS] != list(range(4)):
        raise ValueError("Unexpected special token IDs")
    return tokenizer


def lm_examples(records, tokenizer, context):
    examples = []
    for row in records:
        tokens = tokenizer.encode(row["text"]).ids + [EOS]
        # Each document starts a fresh sequence. Padding has no loss; no split crossing.
        for start in range(0, len(tokens) - 1, context):
            chunk = tokens[start:start + context + 1]
            examples.append((chunk[:-1], chunk[1:]))
    if not examples:
        raise ValueError("No language-model examples")
    return examples


def conversation(prompt, answer, tokenizer, context):
    if not isinstance(prompt, str) or not prompt.strip() or not isinstance(answer, str) or not answer.strip():
        raise ValueError("Non-empty string prompt and completion required")
    prefix = [USER] + tokenizer.encode(prompt).ids + [ASSISTANT]
    completion = tokenizer.encode(answer).ids + [EOS]
    tokens = prefix + completion
    if len(tokens) - 1 > context:
        raise ValueError("Post-training record exceeds context; increase context or shorten data explicitly")
    labels = [-100] * len(prefix) + completion
    return tokens[:-1], labels[1:]


def post_examples(records, tokenizer, context, stage, split):
    valid_splits = {"train", "validation", "test"}
    prompts, hashes = {}, {}
    for row in records:
        if row.get("split") not in valid_splits:
            raise ValueError("Each post-training row requires train/validation/test split")
        for field in ("id", "source", "license", "prompt"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Missing post-training {field}")
        normalized = " ".join(row["prompt"].split())
        if normalized in prompts and prompts[normalized] != row["split"]:
            raise ValueError("Same prompt appears in different post-training splits")
        prompts[normalized] = row["split"]
        for field in (("answer",) if stage == "sft" else ("chosen", "rejected")):
            key = digest((normalized + "\n" + str(row.get(field))).encode())
            if key in hashes and hashes[key] != row["split"]:
                raise ValueError("Cross-split post-training duplicate")
            hashes[key] = row["split"]
        if stage == "dpo" and row.get("chosen") == row.get("rejected"):
            raise ValueError("Chosen and rejected completions must differ")
    selected = [row for row in records if row["split"] == split]
    if not selected:
        raise ValueError(f"No {split} post-training rows")
    if stage == "sft":
        return [conversation(row["prompt"], row["answer"], tokenizer, context) for row in selected]
    return [(conversation(row["prompt"], row["chosen"], tokenizer, context),
             conversation(row["prompt"], row["rejected"], tokenizer, context)) for row in selected]


def batch(examples, device):
    width = max(len(x) for x, _ in examples)
    ids = torch.full((len(examples), width), PAD, dtype=torch.long)
    labels = torch.full((len(examples), width), -100, dtype=torch.long)
    for index, (x, y) in enumerate(examples):
        ids[index, :len(x)] = torch.tensor(x)
        labels[index, :len(y)] = torch.tensor(y)
    return ids.to(device), labels.to(device)


def audit_post_isolation(prepared, paths):
    """Exact normalized checks across all supplied stages, not semantic decontamination."""
    normalize = lambda text: " ".join(text.split())
    pretrain_splits = {}
    for split in ("train", "validation", "test"):
        for row in rows(prepared / f"{split}.jsonl"):
            pretrain_splits[normalize(row["text"])] = split
    prompts, completions = {}, {}
    for path in set(paths):
        for row in rows(path):
            split = row.get("split")
            if split not in ("train", "validation", "test"):
                raise ValueError("Each post-training row requires a valid split")
            prompt = normalize(row["prompt"])
            if prompt in prompts and prompts[prompt] != split:
                raise ValueError("Cross-stage prompt crosses dataset splits")
            prompts[prompt] = split
            for key in ("answer", "chosen", "rejected"):
                if key not in row:
                    continue
                completion = normalize(row[key])
                combined = normalize(row["prompt"] + " " + row[key])
                for text in (prompt, completion, combined):
                    other = pretrain_splits.get(text)
                    if other and (other == "train") != (split == "train"):
                        raise ValueError("Exact post-training text crosses pretraining held-out boundary")
                identity = (prompt, completion)
                if identity in completions and completions[identity] != split:
                    raise ValueError("Cross-stage completion crosses dataset splits")
                completions[identity] = split
    return {"post_files_checked": len(set(paths)), "method": "Whitespace-normalized exact prompt/pair and whole-document matches",
            "remaining": ["near duplicates", "semantic overlap", "external benchmark contamination", "unlisted files"]}
