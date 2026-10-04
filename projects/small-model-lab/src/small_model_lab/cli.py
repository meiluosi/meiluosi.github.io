"""Local CLI: no automatic model/data downloads or network services."""
from __future__ import annotations
import argparse
from pathlib import Path
from .planning import dump, plan, prepare


def parser():
    root = argparse.ArgumentParser(description="Measured from-scratch small-model laboratory")
    commands = root.add_subparsers(dest="command", required=True)
    p = commands.add_parser("plan")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--measured-tokens-per-second", type=float)
    p = commands.add_parser("prepare")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p = commands.add_parser("tokenize")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--vocab-size", type=int, default=8192)
    p = commands.add_parser("hardware")
    p = commands.add_parser("train")
    p.add_argument("--stage", choices=("pretrain", "sft", "dpo"), required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--post-data", type=Path)
    p.add_argument("--related-post-data", type=Path, action="append", default=[], help="All other SFT/DPO files, to audit cross-stage split isolation")
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--parent", type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, required=True, help="Total target optimizer steps, including restored steps")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--accumulation", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.0005)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--max-state-gib", type=float, default=2.0)
    p.add_argument("--allow-large", action="store_true")
    p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    p = commands.add_parser("evaluate")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--post-data", type=Path)
    p.add_argument("--post-kind", choices=("sft", "dpo"), default="sft")
    p.add_argument("--output", type=Path)
    p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    p = commands.add_parser("generate")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--chat", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    p = commands.add_parser("smoke")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    return root


def dispatch(args):
    if args.command == "plan":
        return plan(args.config, args.measured_tokens_per_second)
    if args.command == "prepare":
        return prepare(args.input, args.output, args.seed)
    if args.command == "tokenize":
        from .data import fit_tokenizer
        return fit_tokenizer(args.data, args.output, args.vocab_size)
    from . import engine
    if args.command == "hardware":
        return engine.hardware()
    if args.command in ("train", "evaluate", "generate"):
        return getattr(engine, args.command)(args)
    return smoke(args.output, args.device)


def smoke(output, device):
    from .engine import ROOT
    if not (ROOT / "examples/documents.jsonl").exists():
        raise ValueError("smoke needs the source checkout: use pip install -e . or scripts/lab.py")
    if output.exists():
        raise ValueError("Smoke output must be a new directory")
    output.mkdir(parents=True)
    data = output / "data"
    tokenizer = output / "tokenizer.json"
    prepare(ROOT / "examples/documents.jsonl", data, 42)
    dispatch(parser().parse_args(["tokenize", "--data", str(data), "--output", str(tokenizer), "--vocab-size", "512"]))
    common = ["--config", str(ROOT / "configs/tiny-smoke.json"), "--data", str(data), "--tokenizer", str(tokenizer), "--device", device, "--related-post-data", str(ROOT / "examples/sft.jsonl"), "--related-post-data", str(ROOT / "examples/dpo.jsonl")]
    stages, evaluations = [], []
    parent = None
    for stage, steps in (("pretrain", 12), ("sft", 8), ("dpo", 8)):
        directory = output / stage
        command = ["train", *common, "--stage", stage, "--steps", str(steps), "--output", str(directory)]
        if parent:
            command += ["--parent", str(parent), "--post-data", str(ROOT / f"examples/{stage}.jsonl")]
        stages.append(dispatch(parser().parse_args(command)))
        parent = directory / "checkpoint.pt"
        evaluate = ["evaluate", "--checkpoint", str(parent), "--data", str(data), "--split", "test", "--device", device,
                    "--output", str(directory / "evaluation.json")]
        if stage != "pretrain":
            evaluate += ["--post-data", str(ROOT / f"examples/{stage}.jsonl"), "--post-kind", stage]
        evaluations.append(dispatch(parser().parse_args(evaluate)))
    resumed = dispatch(parser().parse_args(["train", *common, "--stage", "pretrain", "--steps", "14",
                                            "--resume", str(output / "pretrain/checkpoint.pt"), "--output", str(output / "resumed")]))
    sample = dispatch(parser().parse_args(["generate", "--checkpoint", str(parent), "--prompt", "What is a token?", "--chat",
                                          "--max-new-tokens", "16", "--device", device]))
    result = {"schema_version": 1, "kind": "actual-engineering-smoke", "scope": "Synthetic, tiny, brief; no capability or sustained-throughput claim",
              "stages": stages, "evaluations": evaluations, "resume": resumed, "sample": sample}
    (output / "summary.json").write_text(dump(result))
    return {"summary": str(output / "summary.json"), "parameters": stages[0]["parameters"],
            "backend": device, "stages": [r["stage"] for r in stages], "resumed_to_step": resumed["end_step"],
            "notice": result["scope"]}


def main(argv=None):
    root = parser()
    try:
        result = dispatch(root.parse_args(argv))
        print(dump(result), end="")
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        root.exit(2, f"error: {error}\n")
