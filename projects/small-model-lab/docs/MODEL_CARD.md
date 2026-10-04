# Tiny engineering baseline 模型卡

**用途：验证实现。没有实用语言能力声明，未发布模型权重。**

架构为随机初始化、2层decoder、64宽、4头、128上下文、512词表、共享输出权重，总计138,560参数。tokenizer为train-only byte-level BPE，包含256字节alphabet和4个special token。FP32 AdamW，常量learning rate 0.0005，seed 42，microbatch1；没有加载现成预训练参数。

2026-10-04在M4 MPS完成12步预训练→8步SFT→8步DPO，另从预训练第12步恢复至14步。DPO参考冻结为第8步SFT权重。每阶段在同一个合成test fixture评估next-token NLL/PPL，后训练附逐题completion NLL或chosen/rejected log-probability差。报告见[真实MPS记录](../reports/mac-mps-smoke.json)。

这些短跑损失和持出统计只验证代码能运行。极小数据、重复访问fixture、词表和模型规模、几十次更新都不足以说明泛化、推理或RSI。一次真实greedy输出为报告中的原样文本；它并不形成可用回答。正式模型应重新提供来源、预算、独立持出评测与风险分析。

检查点内嵌tokenizer和架构，原生CLI已验证在独立MPS进程加载。无KV cache，无量化，无GGUF/MLX转换，无服务端。跨后端不保证逐位重现。代码MIT；fixture CC0-1.0；后续权重许可需要随所用语料和上游组件另行确认。
