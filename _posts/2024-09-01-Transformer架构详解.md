---
layout: post
title: "Transformer架构详解"
subtitle: "深入理解自注意力机制与位置编码"
date: 2024-09-01
author: "Feng Yu"
header-img: "img/post-bg-recitewords.jpg"
catalog: true
permalink: /2024/09/01/Transformer架构详解/
tags:
  - 大语言模型
  - Transformer
  - 注意力机制
  - 深度学习
categories:
  - Deep Learning
  - NLP
---

> Transformer架构的诞生标志着深度学习进入新纪元。2017年，Google团队在论文《Attention is All You Need》中提出Transformer，彻底颠覆了序列建模的传统范式。它不仅催生了BERT、GPT等里程碑模型，更成为现代大语言模型的基石。

---

## 一、核心架构

**Encoder块**结构：
```
Input Embedding + Positional Encoding
    ↓
Multi-Head Self-Attention
    ↓
Add & Norm
    ↓
Feed-Forward Network
    ↓
Add & Norm
```

**完整实现**：
```python
import torch.nn as nn

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # 多头自注意力
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        # 前馈网络
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 自注意力 + 残差连接 + LayerNorm
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # FFN + 残差连接 + LayerNorm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        
        return x
```

**Decoder块**增加交叉注意力：
```python
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # 三个注意力层
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # 1. Masked Self-Attention（防止看到未来）
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # 2. Cross-Attention（attend to encoder）
        attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout2(attn_output))
        
        # 3. FFN
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_output))
        
        return x
```

---

## 二、完整Transformer模型

```python
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        dropout=0.1,
        max_len=5000
    ):
        super().__init__()
        
        # Embedding层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # 输出层
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
    
    def generate_mask(self, src, tgt):
        # Encoder mask（padding）
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        # [batch, 1, 1, src_len]
        
        # Decoder mask（padding + causal）
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        # [batch, 1, tgt_len, 1]
        
        seq_len = tgt.size(1)
        nopeak_mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
        nopeak_mask = nopeak_mask.to(tgt.device)
        
        tgt_mask = tgt_mask & nopeak_mask
        
        return src_mask, tgt_mask
    
    def encode(self, src, src_mask):
        # Embedding + 位置编码
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # 通过Encoder层
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        
        return x
    
    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        # Embedding + 位置编码
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # 通过Decoder层
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        return x
    
    def forward(self, src, tgt):
        """
        Args:
            src: [batch, src_len]
            tgt: [batch, tgt_len]
        Returns:
            output: [batch, tgt_len, tgt_vocab_size]
        """
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        
        output = self.fc_out(decoder_output)
        return output
```

---

## 三、训练技巧

### Label Smoothing
```python
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        # pred: [batch * seq_len, vocab_size]
        # target: [batch * seq_len]
        
        vocab_size = pred.size(-1)
        confidence = 1.0 - self.smoothing
        
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (vocab_size - 1))
        true_dist.scatter_(1, target.unsqueeze(1), confidence)
        
        return F.kl_div(F.log_softmax(pred, dim=-1), true_dist, reduction='batchmean')
```

### Warmup Learning Rate
```python
class NoamOpt:
    def __init__(self, d_model, warmup_steps, optimizer):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.optimizer = optimizer
        self._step = 0
    
    def step(self):
        self._step += 1
        lr = self.d_model ** (-0.5) * min(
            self._step ** (-0.5),
            self._step * self.warmup_steps ** (-1.5)
        )
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.optimizer.step()
```

---

## 四、应用场景

**机器翻译**：
```python
# 训练
model = Transformer(src_vocab_size, tgt_vocab_size)
criterion = LabelSmoothingLoss()

for src, tgt in dataloader:
    output = model(src, tgt[:, :-1])  # 输入去掉最后一个token
    loss = criterion(
        output.reshape(-1, output.size(-1)),
        tgt[:, 1:].reshape(-1)  # 目标去掉第一个token
    )
    loss.backward()
    optimizer.step()
```

**文本生成**（Greedy Decoding）：
```python
@torch.no_grad()
def generate(model, src, max_len=50):
    model.eval()
    
    src_mask, _ = model.generate_mask(src, src[:, :1])
    encoder_output = model.encode(src, src_mask)
    
    ys = torch.ones(1, 1).fill_(start_token).long()
    
    for i in range(max_len):
        _, tgt_mask = model.generate_mask(src, ys)
        decoder_output = model.decode(ys, encoder_output, src_mask, tgt_mask)
        
        prob = model.fc_out(decoder_output[:, -1])
        next_word = prob.argmax(dim=-1).unsqueeze(0)
        
        ys = torch.cat([ys, next_word], dim=1)
        
        if next_word.item() == end_token:
            break
    
    return ys
```

---

## 五、总结

**Transformer的关键创新**：
1. **自注意力机制**：O(n²)复杂度换取全局视野
2. **位置编码**：巧妙注入顺序信息
3. **残差连接 + LayerNorm**：稳定深层网络训练
4. **并行化**：GPU友好，训练效率高

**局限性**：
- 序列长度的平方复杂度
- 对位置编码的依赖
- 缺乏归纳偏置（如CNN的平移不变性）

**后续改进方向**：
- **Sparse Attention**（Longformer）
- **Linear Attention**（Performer）
- **相对位置编码**（T5, DeBERTa）

Transformer彻底改变了NLP，并逐渐渗透到CV、多模态、强化学习等领域，堪称深度学习的"万能架构"！🚀
