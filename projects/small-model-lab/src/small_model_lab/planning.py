"""Standard-library CLI. No downloads, training, or fabricated run metrics."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import unicodedata


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dump(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def positive_int(value: object, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def plan(config_path: Path, throughput: float | None) -> dict:
    raw = config_path.read_bytes()
    config = json.loads(raw)
    if throughput is not None and (not math.isfinite(throughput) or throughput <= 0):
        raise ValueError("Measured throughput must be finite and greater than zero")
    if config.get("initialization") != "random":
        return {"kind": "pretrained-baseline-plan", "config_sha256": digest(raw),
                "configuration": config, "parameter_estimate": None,
                "note": "The scratch GPT parameter formula does not apply to this baseline."}
    a, train = config["architecture"], config["pretraining"]
    expected = {"kind": "gpt-decoder", "position_embedding": "learned",
                "mlp_expansion": 4, "activation": "gelu", "linear_bias": True,
                "layer_norm_affine": True, "layer_norms_per_block": 2,
                "final_layer_norm": True, "tied_embeddings": True, "lm_head_bias": False}
    for key, value in expected.items():
        if a.get(key) != value:
            raise ValueError(f"Unsupported architecture option {key}: expected {value!r}")
    layers = positive_int(a["layers"], "layers")
    width = positive_int(a["hidden_size"], "hidden_size")
    vocab = positive_int(a["vocab_size"], "vocab_size")
    context = positive_int(a["context_length"], "context_length")
    heads = positive_int(a["attention_heads"], "attention_heads")
    if width % heads:
        raise ValueError("hidden_size must be divisible by attention_heads")
    batch = positive_int(train["micro_batch_size"], "micro_batch_size")
    accumulation = positive_int(train["gradient_accumulation_steps"], "accumulation")
    budget = positive_int(train["token_budget"], "token_budget")
    pilot = positive_int(train["pilot_tokens"], "pilot_tokens")
    embedding = vocab * width
    position = context * width
    blocks = layers * (12 * width * width + 13 * width)
    final_norm = 2 * width
    parameters = embedding + position + blocks + final_norm
    per_update = context * batch * accumulation
    updates = math.ceil(budget / per_update)
    scheduled = updates * per_update
    return {
        "kind": "planning-estimate-not-a-training-run", "name": config["name"],
        "config_sha256": digest(raw), "initialization": "random",
        "architecture": a,
        "parameters": {"total": parameters, "millions": parameters / 1e6,
                       "token_embedding": embedding, "position_embedding": position,
                       "transformer_blocks": blocks, "final_norm": final_norm,
                       "formula": "V*d + C*d + L*(12*d*d + 13*d) + 2*d; tied output head"},
        "fp32_adam_state": {"bytes_per_parameter": 16, "bytes": parameters * 16,
                            "gib": parameters * 16 / 2**30,
                            "excludes": ["activations", "temporary tensors", "framework", "OS", "data", "DPO reference"]},
        "token_plan": {"pilot_target": pilot, "target": budget, "tokens_per_update": per_update,
                       "optimizer_updates_rounded_up": updates, "scheduled_full_tokens": scheduled,
                       "assumption": "Full non-padding sequences; repeated data also counts as processed tokens."},
        "time_estimate": {"user_supplied_measured_tokens_per_second": throughput,
                          "net_training_hours": scheduled / throughput / 3600 if throughput else None,
                          "excludes": ["data preparation", "validation", "checkpoint IO", "interruptions"]},
        "measured_peak_memory_gib": None, "measured_loss": None,
    }


def prepare(input_path: Path, output: Path, seed: int) -> dict:
    if os.path.lexists(output):
        raise ValueError(f"Refusing to overwrite output: {output}")
    if input_path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("This first in-memory tool accepts at most 32 MiB; use a later streaming pipeline for larger corpora.")
    raw = input_path.read_bytes()
    records = []
    ids = set()
    parents: dict[str, str] = {}
    first_hash: dict[str, dict] = {}
    duplicates = []
    blank_lines = 0

    def find(group: str) -> str:
        parents.setdefault(group, group)
        original = group
        while parents[group] != group:
            group = parents[group]
        while parents[original] != original:
            parent = parents[original]
            parents[original] = group
            original = parent
        return group

    def union(left: str, right: str) -> None:
        left, right = find(left), find(right)
        if left != right:
            small, large = sorted((left, right))
            parents[large] = small

    for line_number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            blank_lines += 1
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"Line {line_number}: expected a JSON object")
        for field in ("id", "source", "license", "group", "text"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise ValueError(f"Line {line_number}: non-empty string {field} is required")
        item = {key: item[key].strip() for key in ("id", "source", "license", "group", "text")}
        if item["id"] in ids:
            raise ValueError(f"Line {line_number}: duplicate id {item['id']!r}")
        ids.add(item["id"])
        # NFC and whitespace collapse are deliberately conservative; retain case and punctuation.
        item["text"] = " ".join(unicodedata.normalize("NFC", item["text"]).split())
        if not item["text"]:
            raise ValueError(f"Line {line_number}: text is empty after normalization")
        item["text_sha256"] = digest(item["text"].encode("utf-8"))
        find(item["group"])
        previous = first_hash.get(item["text_sha256"])
        if previous:
            union(item["group"], previous["group"])
            duplicates.append({"removed_id": item["id"], "kept_id": previous["id"],
                               "group": item["group"], "source": item["source"], "license": item["license"],
                               "text_sha256": item["text_sha256"]})
        else:
            first_hash[item["text_sha256"]] = item
            records.append(item)
        if len(ids) > 100000:
            raise ValueError("This first data tool accepts at most 100,000 records")
    components = sorted({find(row["group"]) for row in records}, key=lambda value: digest(f"{seed}:{value}".encode()))
    if len(components) < 3:
        raise ValueError("At least three independent document components are required after deduplication")
    n_validation = max(1, int(len(components) * .1))
    n_test = max(1, int(len(components) * .1))
    n_train = len(components) - n_validation - n_test
    assignment = {group: "train" if i < n_train else "validation" if i < n_train + n_validation else "test"
                  for i, group in enumerate(components)}
    splits: dict[str, list] = {name: [] for name in ("train", "validation", "test")}
    for item in sorted(records, key=lambda row: row["id"]):
        component = find(item["group"])
        splits[assignment[component]].append({**item, "split_group": component})
    payloads = {name: "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
                for name, rows in splits.items()}
    manifest = {
        "schema_version": 1, "kind": "prepared-data-manifest", "input_name": input_path.name,
        "input_sha256": digest(raw), "seed": seed, "normalization": "NFC + whitespace collapse; retain case/punctuation",
        "split_policy": "Document groups; exact-duplicate groups merged transitively; hash-seeded component order; approximately 80/10/10 by component count, minimum one validation/test group.",
        "input_records": len(ids), "blank_lines": blank_lines, "retained_records": len(records),
        "removed_exact_duplicates": len(duplicates), "duplicates": duplicates,
        "component_count": len(components),
        "splits": {name: {"records": len(rows), "components": len({r['split_group'] for r in rows}),
                          "sha256": digest(payloads[name].encode("utf-8"))} for name, rows in splits.items()},
        "sources": sorted({row["source"] for row in records} | {row["source"] for row in duplicates}),
        "licenses_as_declared": sorted({row["license"] for row in records} | {row["license"] for row in duplicates}),
        "remaining_work": ["review declared permissions", "near-duplicate detection", "evaluation-set decontamination", "fit tokenizer on train only"],
        "tokenizer": None, "training_run": None,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        for name, text in payloads.items():
            (staging / f"{name}.jsonl").write_text(text, encoding="utf-8")
        (staging / "manifest.json").write_text(dump(manifest), encoding="utf-8")
        # Atomic directory creation claims the output without replacing another process's files.
        output.mkdir()
        for item in staging.iterdir():
            item.rename(output / item.name)
    finally:
        shutil.rmtree(staging)
    return {"output": str(output), "input_records": len(ids), "retained_records": len(records),
            "removed_exact_duplicates": len(duplicates), "splits": manifest["splits"]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Small-model planning/data tools; model training is not implemented yet.")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("plan", help="Calculate architecture parameters, state memory and token budget")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--measured-tokens-per-second", type=float)
    d = commands.add_parser("prepare", help="Prepare local JSONL; no downloads and no overwrite")
    d.add_argument("--input", type=Path, required=True)
    d.add_argument("--output", type=Path, required=True)
    d.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        result = plan(args.config, args.measured_tokens_per_second) if args.command == "plan" else prepare(args.input, args.output, args.seed)
        print(dump(result), end="")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"error: {error}\n")
