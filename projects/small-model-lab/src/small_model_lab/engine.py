"""Real gradient updates, checkpoint recovery and explicitly scoped measurements."""
from __future__ import annotations
import copy
import hashlib
import json
import math
import platform
import random
import resource
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
import torch
from tokenizers import Tokenizer
from .data import batch, lm_examples, post_examples, rows, verify_prepared, verify_tokenizer, audit_post_isolation, USER, ASSISTANT, EOS
from .model import Decoder, masked_loss, completion_logps, dpo_loss
from .planning import digest, dump, plan

ROOT = Path(__file__).resolve().parents[2]


def source_hash():
    h = hashlib.sha256()
    for path in sorted(Path(__file__).parent.rglob("*.py")):
        h.update(str(path.relative_to(Path(__file__).parent)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def hardware():
    def command(*args):
        try:
            return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {"platform": platform.platform(), "machine": platform.machine(),
            "python": platform.python_version(), "torch": str(torch.__version__),
            "chip": command("sysctl", "-n", "machdep.cpu.brand_string") if platform.system() == "Darwin" else platform.processor(),
            "model": command("sysctl", "-n", "hw.model") if platform.system() == "Darwin" else None,
            "memory_bytes": command("sysctl", "-n", "hw.memsize") if platform.system() == "Darwin" else None,
            "mps_built": torch.backends.mps.is_built(), "mps_available": torch.backends.mps.is_available(),
            "cuda_available": torch.cuda.is_available(), "free_disk_bytes": shutil.disk_usage(Path.cwd()).free}


def device_for(name):
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS unavailable in this process; choose --device cpu explicitly or inspect the environment")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    return torch.device(name)


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def rng_state(device):
    state = {"python": random.getstate(), "cpu": torch.get_rng_state()}
    if device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    elif device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state, device):
    random.setstate(state["python"])
    torch.set_rng_state(state["cpu"])
    if device.type == "mps":
        torch.mps.set_rng_state(state["mps"])
    elif device.type == "cuda":
        torch.cuda.set_rng_state_all(state["cuda"])


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    return checkpoint


def load_model(path, device):
    checkpoint = load_checkpoint(path)
    model = Decoder(checkpoint["architecture"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    tokenizer = Tokenizer.from_str(checkpoint["tokenizer"])
    return model, tokenizer, checkpoint


def dataset_fingerprint(prepared, post_data, related):
    verify_prepared(prepared)
    return {"prepared_manifest_sha256": digest((prepared / "manifest.json").read_bytes()),
            "post_data_sha256": digest(post_data.read_bytes()) if post_data else None,
            "related_post_sha256": sorted(digest(p.read_bytes()) for p in related)}


def train(args):
    if args.steps < 1 or args.batch < 1 or args.accumulation < 1 or args.threads < 1 or args.save_every < 1:
        raise ValueError("steps, batch, accumulation and threads must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.beta) or args.beta <= 0:
        raise ValueError("lr and beta must be finite and positive")
    if not math.isfinite(args.max_state_gib) or args.max_state_gib <= 0:
        raise ValueError("max-state-gib must be finite and positive")
    if args.output.exists():
        raise ValueError("Refusing to overwrite run output; resume into a new directory")
    if args.stage != "pretrain" and not args.post_data:
        raise ValueError("SFT and DPO require --post-data")
    if args.resume and args.parent:
        raise ValueError("Use either resume or parent")
    if args.stage != "pretrain" and not (args.parent or args.resume):
        raise ValueError("Post-training requires a parent checkpoint")
    if args.stage == "pretrain" and args.parent:
        raise ValueError("Pretraining starts randomly; use resume to continue")
    planned = plan(args.config, None)
    if planned["kind"] != "planning-estimate-not-a-training-run":
        raise ValueError("This engine requires a random-initialization decoder configuration")
    config = json.loads(args.config.read_text())
    tokenizer = verify_tokenizer(args.tokenizer, args.data)
    architecture = {**config["architecture"], "vocab_size": tokenizer.get_vocab_size()}
    # Count before allocating: full FP32 Adam state can be large on unified memory.
    d = architecture["hidden_size"]
    parameters = architecture["vocab_size"] * d + architecture["context_length"] * d + architecture["layers"] * (12*d*d + 13*d) + 2*d
    if parameters > 50_000_000 and not args.allow_large:
        raise ValueError("Above 50M requires --allow-large after reviewing sustained resource measurements")
    state_bytes = parameters * (20 if args.stage == "dpo" else 16)
    if state_bytes / 2**30 > args.max_state_gib:
        raise ValueError("Estimated optimizer/reference state exceeds --max-state-gib; this excludes activations and OS")
    # Reserve room for a full checkpoint, an atomic temporary copy, and 4 GiB free space.
    disk_needed = parameters * (32 if args.stage == "dpo" else 24) + 4 * 2**30
    if shutil.disk_usage(args.output.parent if args.output.parent.exists() else ROOT).free < disk_needed:
        raise ValueError("Insufficient disk for two checkpoint copies plus 4 GiB reserve")
    device = device_for(args.device)
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = Decoder(architecture).to(device)
    assert sum(p.numel() for p in model.parameters()) == parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, foreach=False)
    config_hash = digest(args.config.read_bytes())
    tokenizer_hash = digest(args.tokenizer.read_bytes())
    data_hash = dataset_fingerprint(args.data, args.post_data, args.related_post_data)
    isolation = audit_post_isolation(args.data, ([args.post_data] if args.post_data else []) + args.related_post_data)
    options = {"lr": args.lr, "batch": args.batch, "accumulation": args.accumulation, "beta": args.beta,
               "seed": args.seed, "threads": args.threads, "dtype": "float32", "scheduler": "constant"}
    sampler = random.Random(args.seed)
    first_step, cumulative_tokens, cumulative_loss_tokens = 0, 0, 0
    parent_hash = digest(args.parent.read_bytes()) if args.parent else None
    reference = None
    if args.resume:
        saved = load_checkpoint(args.resume)
        for key, expected in (("architecture", architecture), ("stage", args.stage), ("config_sha256", config_hash),
                              ("tokenizer_sha256", tokenizer_hash), ("data", data_hash), ("training_options", options),
                              ("backend", device.type), ("source_sha256", source_hash())):
            if saved[key] != expected:
                raise ValueError(f"Resume {key} differs; exact resume requires the same data, code, options and backend")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        first_step = saved["step"]
        cumulative_tokens = saved["processed_tokens"]
        cumulative_loss_tokens = saved["loss_tokens"]
        sampler.setstate(saved["sampler_state"])
        parent_hash = saved["parent_checkpoint_sha256"]
        if args.stage == "dpo":
            reference = copy.deepcopy(model)
            reference.load_state_dict(saved["reference"])
        restore_rng(saved["rng"], device)
    elif args.parent:
        saved = load_checkpoint(args.parent)
        required_stage = "pretrain" if args.stage == "sft" else "sft"
        if saved["stage"] != required_stage or saved["architecture"] != architecture or saved["tokenizer_sha256"] != tokenizer_hash:
            raise ValueError("Parent stage, architecture or tokenizer is incompatible")
        model.load_state_dict(saved["model"])
        if args.stage == "dpo":
            reference = copy.deepcopy(model)
    if first_step >= args.steps:
        raise ValueError("--steps is the total target; it must exceed the saved step")
    if reference is not None:
        reference.eval().requires_grad_(False)
    context = architecture["context_length"]
    examples = (lm_examples(rows(args.data / "train.jsonl"), tokenizer, context) if args.stage == "pretrain"
                else post_examples(rows(args.post_data), tokenizer, context, args.stage, "train"))
    args.output.mkdir(parents=True)
    checkpoint_path = args.output / "checkpoint.pt"

    def save_checkpoint(step):
        checkpoint = {"schema_version": 1, "stage": args.stage, "step": step, "architecture": architecture,
                      "model": cpu_tree(model.state_dict()), "optimizer": cpu_tree(optimizer.state_dict()),
                      "rng": rng_state(device), "sampler_state": sampler.getstate(),
                      "training_options": options, "backend": device.type, "source_sha256": source_hash(),
                      "config_sha256": config_hash, "tokenizer_sha256": tokenizer_hash,
                      "tokenizer": tokenizer.to_str(), "data": data_hash, "parent_checkpoint_sha256": parent_hash,
                      "processed_tokens": cumulative_tokens, "loss_tokens": cumulative_loss_tokens,
                      "reference": cpu_tree(reference.state_dict()) if reference is not None else None}
        temporary = checkpoint_path.with_suffix(".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(checkpoint_path)

    started = datetime.now(timezone.utc).isoformat()
    history = []
    warmup_steps = min(3, args.steps - first_step)
    mps_sampled = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    synchronize(device)
    wall_start = time.perf_counter()
    model.train()
    for step in range(first_step, args.steps):
        synchronize(device)
        update_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss_value, tokens, loss_tokens = 0.0, 0, 0
        microbatches = [[examples[sampler.randrange(len(examples))] for _ in range(args.batch)]
                        for _ in range(args.accumulation)]
        update_loss_tokens = (sum(sum(label != -100 for label in y) for micro in microbatches for _, y in micro)
                              if args.stage != "dpo" else None)
        for chosen_rows in microbatches:
            if args.stage == "dpo":
                chosen, rejected = zip(*chosen_rows)
                cx, cy = batch(chosen, device)
                rx, ry = batch(rejected, device)
                with torch.no_grad():
                    rc, rr = completion_logps(reference, cx, cy), completion_logps(reference, rx, ry)
                loss = dpo_loss(completion_logps(model, cx, cy), completion_logps(model, rx, ry), rc, rr, args.beta)
                tokens += sum(len(x) for x, _ in chosen) + sum(len(x) for x, _ in rejected)
                loss_tokens += int((cy != -100).sum()) + int((ry != -100).sum())
            else:
                x, y = batch(chosen_rows, device)
                loss = masked_loss(model(x), y)
                tokens += sum(len(x) for x, _ in chosen_rows)
                loss_tokens += int((y != -100).sum())
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss; stopping without saving invalid weights")
            # Variable-length LM/SFT examples are weighted by supervised tokens across
            # the entire accumulated update; DPO is equally weighted by pairs.
            weight = 1 / args.accumulation if args.stage == "dpo" else int((y != -100).sum()) / update_loss_tokens
            (loss * weight).backward()
            loss_value += loss.item() * weight
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        synchronize(device)
        cumulative_tokens += tokens
        cumulative_loss_tokens += loss_tokens
        history.append({"step": step + 1, "loss": loss_value, "processed_tokens": tokens,
                        "loss_tokens": loss_tokens, "seconds": time.perf_counter() - update_start})
        if device.type == "mps":
            mps_sampled.append(torch.mps.current_allocated_memory())
        if (step + 1) % args.save_every == 0:
            save_checkpoint(step + 1)
    loop_including_checkpoint_seconds = time.perf_counter() - wall_start
    wall_seconds = sum(row["seconds"] for row in history)
    if reference is not None and any(p.grad is not None for p in reference.parameters()):
        raise RuntimeError("Reference unexpectedly received gradients")
    save_checkpoint(args.steps)
    measured = history[warmup_steps:]
    total_tokens = sum(row["processed_tokens"] for row in history)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = rss if platform.system() == "Darwin" else rss * 1024
    report = {"schema_version": 1, "kind": "actual-training-run", "run_id": args.output.name,
              "started_utc": started, "stage": args.stage, "initialization": "random" if args.stage == "pretrain" and not args.resume else "checkpoint",
              "purpose": "Engineering baseline; not a model capability benchmark", "hardware": hardware(),
              "backend": device.type, "parameters": parameters, "architecture": architecture,
              "source_sha256": source_hash(), "config_sha256": config_hash, "tokenizer_sha256": tokenizer_hash,
              "data": data_hash, "isolation_audit": isolation, "parent_checkpoint_sha256": parent_hash,
              "resume_checkpoint_sha256": digest(args.resume.read_bytes()) if args.resume else None,
              "training_options": options, "start_step": first_step, "end_step": args.steps,
              "processed_tokens_this_invocation": total_tokens, "processed_tokens_cumulative": cumulative_tokens,
              "loss_tokens_cumulative": cumulative_loss_tokens, "training_wall_seconds": wall_seconds,
              "loop_seconds_including_periodic_checkpoint_io": loop_including_checkpoint_seconds,
              "tokens_per_second_including_warmup": total_tokens / wall_seconds,
              "warmup_steps_excluded": warmup_steps,
              "tokens_per_second_after_warmup": sum(r["processed_tokens"] for r in measured) / sum(r["seconds"] for r in measured) if measured else None,
              "timing_scope": "Synchronized optimizer loop only; excludes data/tokenizer, initialization, evaluation and checkpoint writes. DPO tokens count policy inputs once; frozen reference compute is included in elapsed time.",
              "memory": {"process_lifetime_peak_rss_bytes": rss_bytes,
                         "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
                         "mps_sampled_max_allocated_bytes": max(mps_sampled) if mps_sampled else None,
                         "mps_true_peak_bytes": None,
                         "note": "RSS is process lifetime high-water, not isolated stage peak; MPS samples after updates can miss transient peaks. Unified-memory totals must not sum overlapping metrics."},
              "checkpoint_bytes": checkpoint_path.stat().st_size, "checkpoint_sha256": digest(checkpoint_path.read_bytes()),
              "history": history, "capability_benchmark": None}
    (args.output / "report.json").write_text(dump(report))
    return report


@torch.no_grad()
def evaluate(args):
    device = device_for(args.device)
    model, tokenizer, checkpoint = load_model(args.checkpoint, device)
    manifest = verify_prepared(args.data)
    if digest((args.data / "manifest.json").read_bytes()) != checkpoint["data"]["prepared_manifest_sha256"]:
        raise ValueError("Evaluation data differs from checkpoint data manifest")
    context = checkpoint["architecture"]["context_length"]
    examples = lm_examples(rows(args.data / f"{args.split}.jsonl"), tokenizer, context)
    total_nll, total_tokens = 0.0, 0
    for item in examples:
        x, y = batch([item], device)
        count = int((y != -100).sum())
        total_nll += masked_loss(model(x), y).item() * count
        total_tokens += count
    nll = total_nll / total_tokens
    result = {"kind": "held-out-language-model-evaluation", "stage": checkpoint["stage"], "split": args.split,
              "checkpoint_sha256": digest(args.checkpoint.read_bytes()),
              "tokenizer_sha256": checkpoint["tokenizer_sha256"],
              "dataset_sha256": digest((args.data / f"{args.split}.jsonl").read_bytes()),
              "supervised_tokens": total_tokens, "mean_next_token_nll": nll, "perplexity": math.exp(nll),
              "scope": ("Tiny held-out synthetic fixtures; pipeline sanity only, not useful language ability or RSI evidence."
                        if manifest["sources"] == ["original-synthetic-fixture-v1"] else
                        "Held-out next-token statistics for this data/tokenizer only; not a general capability or RSI benchmark.")}
    if args.post_data:
        kind = args.post_kind
        records = rows(args.post_data)
        selected = [r for r in records if r["split"] == args.split]
        post = post_examples(records, tokenizer, context, kind, args.split)
        per_example = []
        for row, item in zip(selected, post):
            if kind == "sft":
                x, y = batch([item], device)
                per_example.append({"id": row["id"], "completion_nll": masked_loss(model(x), y).item(), "completion_tokens": int((y != -100).sum())})
            else:
                cx, cy = batch([item[0]], device)
                rx, ry = batch([item[1]], device)
                margin = (completion_logps(model, cx, cy) - completion_logps(model, rx, ry)).item()
                per_example.append({"id": row["id"], "chosen_minus_rejected_logp_sum": margin,
                                    "chosen_tokens": int((cy != -100).sum()), "rejected_tokens": int((ry != -100).sum())})
        result["post_training"] = {"kind": kind, "dataset_sha256": digest(args.post_data.read_bytes()), "examples": per_example,
                                   "note": "Raw DPO log-probability sums favor short answers; no reward-model or general capability claim."}
    if args.output:
        if args.output.exists():
            raise ValueError("Refusing to overwrite evaluation")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(dump(result))
    return result


@torch.no_grad()
def generate(args):
    if not 1 <= args.max_new_tokens <= 512:
        raise ValueError("max-new-tokens must be between 1 and 512")
    device = device_for(args.device)
    model, tokenizer, checkpoint = load_model(args.checkpoint, device)
    encoded = tokenizer.encode(args.prompt).ids
    tokens = [USER] + encoded + [ASSISTANT] if args.chat else encoded
    if not tokens:
        raise ValueError("Prompt has no tokens")
    output = []
    for _ in range(args.max_new_tokens):
        x = torch.tensor([tokens[-model.architecture["context_length"]:]], device=device)
        next_id = int(model(x)[0, -1].argmax())
        if next_id == EOS:
            break
        tokens.append(next_id)
        output.append(next_id)
    return {"stage": checkpoint["stage"], "prompt": args.prompt, "generated_text": tokenizer.decode(output),
            "generated_ids": output, "decoding": {"method": "greedy", "max_new_tokens": args.max_new_tokens,
            "context_overflow": "sliding window; learned positions reset per window", "chat_template": args.chat},
            "notice": "Engineering smoke weights may produce meaningless text."}
