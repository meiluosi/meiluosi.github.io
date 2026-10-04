# small-model-lab

在个人电脑上，实际运行 **数据 → 分词 → 随机初始化预训练 → SFT → DPO → 评测 → 本地 CLI 推理**。这是可单独复制、安装和发布到 GitHub 的 Python 项目，不依赖枫语网站、Node、Astro 或外部相邻目录。

**已完成的是 138,560 参数的工程闭环，不是具有实用能力的模型。** 2026-10-04 在 M4 MacBook Air 上真实执行了 MPS 预训练 12 步、SFT 8 步、DPO 8 步，恢复预训练至 14 步，并在新进程中加载推理。合成小样本只验证数据流、反向传播、保存恢复和报告，不能支持语言能力、自我改进或大规模训练效率结论。

- [真实 MPS 逐步记录、评测和恢复](reports/mac-mps-smoke.json)
- [新进程 MPS 推理输出](reports/mac-mps-fresh-process.json)
- [复制到独立目录后的 CPU 全链路验证](reports/portable-copy.json)
- [验证清单](reports/validation.json)、[模型卡](docs/MODEL_CARD.md)、[数据卡](docs/DATA_CARD.md)
- [资源与实验方案](docs/PLAN.md)、[一手参考与许可边界](docs/REFERENCES.md)

## 快速开始

推荐创建独立 Python 环境。代码支持 Python 3.10–3.14；实测环境是 Python 3.10.18、PyTorch 2.14.0、tokenizers 0.22.1。系统默认 Python 3.14.7 未用于本次训练；是否有适合其它 Python/平台的 PyTorch wheel，应按官方安装页选择。

```bash
cd small-model-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/lab.py hardware
python -m unittest discover -s tests -v
python scripts/lab.py smoke --output runs/first-smoke --device cpu
# Apple Silicon 在 hardware 显示 mps_available=true 后可改为 --device mps
```

安装后也可使用 `small-model-lab` 命令。`smoke` 需要本项目的 `examples/` 和 `configs/`，因此使用源码目录或 editable install；其它命令通过显式路径读取数据和配置。`requirements-tested.txt` 记录实测版本，不承诺所有平台都有相同版本的二进制包。

`smoke` 不联网下载语料或权重。输出目录必须不存在；为了保护运行产物，命令不覆盖既有数据、分词器或 run。

## 逐阶段运行

```bash
python scripts/lab.py prepare --input examples/documents.jsonl --output data/fixture
python scripts/lab.py tokenize --data data/fixture --output data/tokenizer.json --vocab-size 512

python scripts/lab.py train --stage pretrain --config configs/tiny-smoke.json \
  --data data/fixture --tokenizer data/tokenizer.json --steps 12 \
  --related-post-data examples/sft.jsonl --related-post-data examples/dpo.jsonl \
  --output runs/pretrain --device cpu

python scripts/lab.py train --stage sft --config configs/tiny-smoke.json \
  --data data/fixture --tokenizer data/tokenizer.json --steps 8 \
  --parent runs/pretrain/checkpoint.pt --post-data examples/sft.jsonl \
  --related-post-data examples/dpo.jsonl --output runs/sft --device cpu

python scripts/lab.py train --stage dpo --config configs/tiny-smoke.json \
  --data data/fixture --tokenizer data/tokenizer.json --steps 8 \
  --parent runs/sft/checkpoint.pt --post-data examples/dpo.jsonl \
  --related-post-data examples/sft.jsonl --output runs/dpo --device cpu

python scripts/lab.py evaluate --checkpoint runs/dpo/checkpoint.pt \
  --data data/fixture --split test --post-data examples/dpo.jsonl --post-kind dpo \
  --output runs/dpo/evaluation.json --device cpu
python scripts/lab.py generate --checkpoint runs/dpo/checkpoint.pt \
  --prompt 'What is a token?' --chat --max-new-tokens 16 --device cpu
```

所有后训练数据文件都应列入 `--post-data` 或 `--related-post-data`，以检查跨阶段 split 隔离。未列出的文件、近重复和语义污染不在当前审计范围内。

恢复已保存的预训练（保持之前所有影响计算的参数和相关数据文件一致）：

```bash
python scripts/lab.py train --stage pretrain --config configs/tiny-smoke.json \
  --data data/fixture --tokenizer data/tokenizer.json --steps 14 \
  --related-post-data examples/sft.jsonl --related-post-data examples/dpo.jsonl \
  --resume runs/pretrain/checkpoint.pt --output runs/resumed --device cpu
```

`--steps` 是包含已恢复步骤的总目标；`--save-every` 默认 50，每到间隔及正常结束时原子替换该 run 的最新检查点。意外中断只能恢复最后一次保存，不能恢复未落盘的步骤。检查点含模型、AdamW、常量学习率配置、随机状态、数据采样状态、tokenizer、来源哈希及 DPO 冻结参考。恢复要求相同源码、配置、后端、数据和训练参数；本次证明了 CPU 同环境的逐位一致，跨后端和跨版本不作此承诺。

## 档位与当前边界

| 配置 | 按目标词表计算的参数量 | 状态 |
|---|---:|---|
| tiny-smoke | 138,560 | MPS、CPU 工程闭环已运行 |
| scratch-30m | 29,676,544 | 下一次正式资源测量的起点，未训练 |
| scratch-100m | 91,740,672 | 条件对照，未训练 |
| scratch-500m | 514,859,520 | 约 0.5B 扩展候选，未训练 |

实际模型使用**拟合后的真实词表大小**，若语料不足，BPE 未达到目标词表，参数也随之变化。30M/100M/500M 是档位名。0.5B 使用 32K 词表，其它主档使用 8K，不能将这些配置直接解释为只改变容量的严格对照。

```bash
python scripts/lab.py plan --config configs/scratch-30m.json
python scripts/lab.py plan --config configs/scratch-500m.json
```

预算命令不启动训练。`token_budget` 是规划输入；实际运行必须显式提供 `--steps`，批量/累积取 CLI 参数并写入报告。当前 FP32 全参数 AdamW 状态按 16 bytes/参数估算，DPO 冻结参考额外 4 bytes/参数；这些都不是峰值内存。超过 50M 需 `--allow-large`，状态预算默认最多 2 GiB，磁盘还检查两份检查点及 4 GiB 余量。通过检查仍不代表训练可行，详见资源方案。

实现是便于检查的 pre-norm GPT decoder：学习位置向量、因果注意力、GELU、LayerNorm、共享输入输出 embedding。训练使用 FP32、AdamW、常量学习率和梯度裁剪；变长 LM/SFT 的梯度累积按整个有效 batch 的监督 token 加权。SFT 仅计算 assistant completion（含 EOS），DPO 使用 completion log-probability **之和**及冻结 SFT 参考。

当前数据工具每个 JSONL 最多 32 MiB，预处理最多 100,000 行；训练数据驻内存，各文档独立分块，右侧 padding 不计 loss。未实现大语料 streaming、跨文档 packing、近重复去污、正式任务能力基准、混合精度、KV cache、量化或 Web 服务。CLI 使用 greedy 解码；它已经能部署为独立本地进程。GGUF、safetensors、Ollama 和 MLX 导出均属于后续工作。

## 迁移与发布

把**整个本目录**复制成新仓库根目录，再执行快速开始；不要只复制 `src/`。`docs/`、`examples/`、`configs/`、`tests/`、`.github/` 和项目配置均自包含。独立 GitHub 仓库会运行 `.github/workflows/ci.yml` 的 CPU 测试与 smoke；它不依赖网站 CI。当前尚未创建远端仓库、上传权重或发布模型。

代码使用 [MIT](LICENSE)，本项目自写合成 fixture 使用 CC0-1.0。第三方依赖、将来采用的数据和模型权重分别遵循其自身许可。数据、缓存、权重和运行目录已被 `.gitignore` 排除；仅提交经过检查的小型 JSON 报告。原相邻训练规划目录没有被修改。
