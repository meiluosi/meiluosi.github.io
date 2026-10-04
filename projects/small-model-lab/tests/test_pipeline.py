"""Tests for scientific/engineering invariants, with tiny real gradient updates."""
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from small_model_lab.cli import parser, dispatch
from small_model_lab.data import (audit_post_isolation, batch, conversation, fit_tokenizer, post_examples,
                                  verify_tokenizer, ASSISTANT, EOS)
from small_model_lab.engine import ROOT, load_checkpoint, source_hash
from small_model_lab.model import Decoder, masked_loss, dpo_loss
from small_model_lab.planning import prepare


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.data = cls.root / "prepared"
        prepare(ROOT / "examples/documents.jsonl", cls.data, 42)
        cls.tokenizer_path = cls.root / "tokenizer.json"
        fit_tokenizer(cls.data, cls.tokenizer_path, 512)
        cls.tokenizer = verify_tokenizer(cls.tokenizer_path, cls.data)
        cls.arch = json.loads((ROOT / "configs/tiny-smoke.json").read_text())["architecture"]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_causality_and_real_gradient(self):
        torch.manual_seed(9)
        model = Decoder(self.arch)
        x = torch.randint(4, 512, (1, 10))
        changed = x.clone(); changed[:, 6:] = (changed[:, 6:] + 1) % 512
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(x)[:, :6], model(changed)[:, :6], atol=1e-6))
        before = model.tokens.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        masked_loss(model(x[:, :-1]), x[:, 1:]).backward(); optimizer.step()
        self.assertFalse(torch.equal(before, model.tokens.weight))

    def test_completion_shift_mask_and_eos(self):
        x, y = conversation("What is a token?", "A unit.", self.tokenizer, 128)
        marker = x.index(ASSISTANT)
        self.assertTrue(all(v == -100 for v in y[:marker]))
        self.assertEqual(y[marker], self.tokenizer.encode("A unit.").ids[0])
        self.assertEqual(y[-1], EOS)
        _, padded = batch([(x, y), ([2, 3], [-100, EOS])], "cpu")
        self.assertTrue((padded[1, 2:] == -100).all())

    def test_variable_length_accumulation_matches_token_objective(self):
        logits = torch.randn(2, 4, 8, requires_grad=True)
        labels = torch.tensor([[1, -100, -100, -100], [2, 3, 4, 5]])
        combined = masked_loss(logits, labels)
        accumulated = masked_loss(logits[:1], labels[:1]) / 5 + masked_loss(logits[1:], labels[1:]) * 4 / 5
        self.assertTrue(torch.allclose(combined, accumulated, atol=1e-6))
        self.assertTrue(torch.allclose(torch.autograd.grad(combined, logits, retain_graph=True)[0],
                                      torch.autograd.grad(accumulated, logits)[0], atol=1e-6))

    def test_dpo_initial_loss_and_gradient_direction(self):
        chosen = torch.tensor([-5.], requires_grad=True)
        rejected = torch.tensor([-6.], requires_grad=True)
        loss = dpo_loss(chosen, rejected, chosen.detach(), rejected.detach(), 0.1)
        self.assertAlmostEqual(loss.item(), math.log(2), places=6)
        loss.backward()
        self.assertLess(chosen.grad.item(), 0)
        self.assertGreater(rejected.grad.item(), 0)

    def test_cross_stage_split_leakage_is_rejected(self):
        a = self.root / "audit-a.jsonl"; b = self.root / "audit-b.jsonl"
        row = {"prompt": "Same held-out question", "answer": "Answer", "split": "train"}
        a.write_text(json.dumps(row) + "\n")
        b.write_text(json.dumps({**row, "split": "test"}) + "\n")
        with self.assertRaisesRegex(ValueError, "Cross-stage"):
            audit_post_isolation(self.data, [a, b])

    def test_manifest_tampering_is_rejected(self):
        target = self.root / "tampered"
        prepare(ROOT / "examples/documents.jsonl", target, 42)
        (target / "train.jsonl").write_text('{"text":"tampered"}\n')
        with self.assertRaisesRegex(ValueError, "manifest"):
            fit_tokenizer(target, self.root / "bad-tokenizer.json", 512)

    def test_duplicate_groups_merge_before_split(self):
        raw = self.root / "duplicate.jsonl"
        rows = [{"id": str(i), "source": "fixture", "license": "CC0-1.0", "group": group, "text": text}
                for i, (group, text) in enumerate([("a", "same"), ("b", "same"), ("b", "sibling"), ("c", "third"), ("d", "fourth")])]
        raw.write_text("\n".join(json.dumps(r) for r in rows))
        destination = self.root / "dedup"
        result = prepare(raw, destination, 42)
        self.assertEqual(result["removed_exact_duplicates"], 1)
        containing = []
        for split in ("train", "validation", "test"):
            records = [json.loads(line) for line in (destination / f"{split}.jsonl").read_text().splitlines()]
            containing += [split for row in records if row["text"] in ("same", "sibling")]
        self.assertEqual(len(set(containing)), 1)

    def train(self, name, steps, resume=None, stage="pretrain", parent=None):
        command = ["train", "--stage", stage, "--config", str(ROOT / "configs/tiny-smoke.json"),
                   "--data", str(self.data), "--tokenizer", str(self.tokenizer_path), "--output", str(self.root / name),
                   "--steps", str(steps), "--save-every", "1", "--device", "cpu", "--accumulation", "2"]
        if resume:
            command += ["--resume", str(resume)]
        if parent:
            command += ["--parent", str(parent), "--post-data", str(ROOT / f"examples/{stage}.jsonl")]
        return dispatch(parser().parse_args(command))

    def test_exact_cpu_resume_and_fresh_process_generation(self):
        self.train("whole", 4)
        self.train("part", 2)
        report = self.train("continued", 4, self.root / "part/checkpoint.pt")
        whole, resumed = [load_checkpoint(self.root / f"{name}/checkpoint.pt") for name in ("whole", "continued")]
        self.assertEqual(report["start_step"], 2)
        self.assertEqual(whole["sampler_state"], resumed["sampler_state"])
        for key in whole["model"]:
            self.assertTrue(torch.equal(whole["model"][key], resumed["model"][key]), key)
        self.assertNotEqual(source_hash(), __import__("hashlib").sha256(b"").hexdigest())
        result = subprocess.run([sys.executable, str(ROOT / "scripts/lab.py"), "generate", "--checkpoint",
                                 str(self.root / "continued/checkpoint.pt"), "--prompt", "A token", "--device", "cpu",
                                 "--max-new-tokens", "2"], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)["decoding"]["method"], "greedy")

    def test_invalid_resource_budget_is_rejected(self):
        args = parser().parse_args(["train", "--stage", "pretrain", "--config", "unused", "--data", "unused",
                                    "--tokenizer", "unused", "--output", str(self.root / "invalid"), "--steps", "1",
                                    "--max-state-gib", "nan"])
        with self.assertRaisesRegex(ValueError, "finite"):
            dispatch(args)

    def test_all_stages_and_reference_frozen(self):
        self.train("stage-pt", 2)
        self.train("stage-sft", 2, stage="sft", parent=self.root / "stage-pt/checkpoint.pt")
        self.train("stage-dpo", 2, stage="dpo", parent=self.root / "stage-sft/checkpoint.pt")
        sft = load_checkpoint(self.root / "stage-sft/checkpoint.pt")
        dpo = load_checkpoint(self.root / "stage-dpo/checkpoint.pt")
        for key in sft["model"]:
            self.assertTrue(torch.equal(sft["model"][key], dpo["reference"][key]))
        self.assertTrue(any(not torch.equal(sft["model"][key], dpo["model"][key]) for key in sft["model"]))
        self.assertTrue(all(p.grad is None for p in dpo["reference"].values()))


if __name__ == "__main__":
    unittest.main()
