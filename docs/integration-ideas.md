# 🔬 开源项目调研

> 可集成到博客中的有趣开源项目，按领域分类。每个项目含基本信息、集成方案、可行性评估。

---

## 目录

- [金融分析](#金融分析)
- [LLM 应用平台](#llm-应用平台)
- [数据可视化](#数据可视化)
- [AI 交互](#ai-交互)
- [其他有趣项目](#其他有趣项目)

---

## 金融分析

### FinGPT

- **仓库**：[AI4Finance-Foundation/FinGPT](https://github.com/AI4Finance-Foundation/FinGPT)
- **定位**：金融大语言模型，专注于金融领域的 LLM 应用
- **核心能力**：
  - 金融情感分析（新闻、社交媒体）
  - 金融 RAG（检索增强生成）
  - LoRA 微调（低成本适配金融子领域）
  - HuggingFace 模型发布
- **集成方式**：内容型（无需代码改动）
  - 写系列文章：《FinGPT 实战：金融情感分析》《用 LoRA 微调金融大模型》等
  - 如果将来做交互式 demo，可以调用 FinGPT 的 HuggingFace 模型 API
- **可行性**：⭐⭐⭐⭐⭐ 与博客 AI/LLM 定位完美契合，写文章即可
- **门槛**：低（只需要写文章，不需要集成代码）

---

### AKShare

- **仓库**：[akfamily/akshare](https://github.com/akfamily/akshare)
- **定位**：开源财经数据接口库，100+ 数据源
- **核心能力**：
  - A 股/港股/美股/期货/外汇/加密货币行情数据
  - 宏观经济数据（GDP、CPI、PMI 等）
  - 新闻舆情数据
  - 纯 Python，API 简单
- **集成方式**：构建时数据面板
  - 在 `pnpm build` 时运行 Python 脚本，调用 AKShare 获取数据
  - 生成 JSON 数据文件 + ECharts 图表静态 HTML
  - 实现纯静态的数据可视化面板页面
- **可行性**：⭐⭐⭐⭐ 需要构建时 Python 环境，但数据是静态的，零运行时风险
- **门槛**：中（需要 Python 环境 + 数据面板设计）

---

### OpenBB

- **仓库**：[OpenBB-finance/OpenBB](https://github.com/OpenBB-finance/OpenBB)
- **定位**：开源 Bloomberg 替代品，投资研究平台
- **核心能力**：
  - 100+ 数据提供商集成
  - 终端式命令行界面
  - MCP Server 支持（可被 Claude 等 AI 调用）
  - Python SDK / REST API
- **集成方式**：
  - 内容型：写《OpenBB 实战》系列文章
  - 技术型：通过 OpenBB MCP Server 让 AI 看板娘能回答金融数据问题
- **可行性**：⭐⭐⭐ 定位偏重，更适合投资专业人士
- **门槛**：高（需要 Python 后端服务）

---

### FinRL

- **仓库**：[AI4Finance-Foundation/FinRL](https://github.com/AI4Finance-Foundation/FinRL)
- **定位**：金融强化学习框架
- **核心能力**：
  - 股票交易、投资组合管理、加密货币交易等 RL 环境
  - 内置 A2C / PPO / DDPG / SAC / TD3 等算法
  - 基于 Stable Baselines 3 / RLlib
  - 回测框架
- **集成方式**：内容型 + 文章配套代码
  - 写系列文章：《FinRL 实战：用 PPO 训练股票交易 Agent》
  - 文章附带 Colab Notebook 链接，读者可在线运行
- **可行性**：⭐⭐⭐⭐⭐ 与博客 RL 定位完美契合
- **门槛**：低（只需要写文章，代码放 Colab）

---

## LLM 应用平台

### LangChain / LlamaIndex

- **仓库**：[langchain-ai/langchain](https://github.com/langchain-ai/langchain) / [run-llama/llama_index](https://github.com/run-llama/llama_index)
- **定位**：LLM 应用开发框架
- **集成方式**：底层工具，用于构建自定义 RAG 管线
- **可行性**：⭐⭐⭐ 需要自己写后端，不如直接用现成的 LLM 应用平台方便
- **门槛**：高

---

### FastGPT / MaxKB

- **仓库**：[labring/FastGPT](https://github.com/labring/FastGPT) / [1Panel-dev/MaxKB](https://github.com/1Panel-dev/MaxKB)
- **定位**：国产开源知识库问答平台
- **集成方式**：可嵌入 iframe
- **可行性**：⭐⭐⭐⭐ 国产、中文友好，但需要部署
- **门槛**：中（需要 Docker 部署）

---

## 数据可视化

### ECharts

- **仓库**：[apache/echarts](https://github.com/apache/echarts)
- **定位**：百度开源数据可视化库
- **集成方式**：Svelte 封装 ECharts 组件，用于数据面板、知识图谱等
- **可行性**：⭐⭐⭐⭐⭐ 纯前端，零运行时成本
- **门槛**：低

---

### D3.js

- **仓库**：[d3/d3](https://github.com/d3/d3)
- **定位**：数据驱动的文档操作库，灵活度极高
- **集成方式**：力导向图（知识图谱）、桑基图、树图等
- **可行性**：⭐⭐⭐⭐ 纯前端
- **门槛**：中（学习曲线较陡）

---

### Mermaid.js

- **仓库**：[mermaid-js/mermaid](https://github.com/mermaid-js/mermaid)
- **定位**：文本驱动图表生成
- **已有基础**：✅ 已通过 `rehype-mermaid` 插件集成在文章中
- **扩展**：可在非文章页面使用（如关于页面展示技术栈）

---

### Three.js

- **仓库**：[mrdoob/three.js](https://github.com/mrdoob/three.js)
- **定位**：3D 渲染引擎
- **集成方式**：
  - 3D 粒子背景（替代 Canvas 2D）
  - 3D 知识图谱
  - 3D 数据可视化
- **可行性**：⭐⭐⭐ 重武器，适合特殊场景
- **门槛**：高

---

## AI 交互

### predict-anything / predict-probability

- **概念**：基于 AI 的预测市场/概率估计工具
- **集成方式**：
  - 内容型：写文章介绍预测市场概念
  - 交互型：构建一个简单的"AI 预测"小工具，用户输入问题，AI 评估概率
- **可行性**：⭐⭐⭐ 概念有趣，但需要 LLM API 后端
- **门槛**：中高

---

### 数字人

- **概念**：AI 驱动的虚拟形象，可语音交互
- **开源方案**：
  - [HeyGen](https://www.heygen.com/)（商业，有免费额度）
  - [D-ID](https://www.d-id.com/)（商业）
  - [MuseTalk](https://github.com/TMElyralab/MuseTalk)（开源，实时嘴型同步）
  - [SadTalker](https://github.com/OpenTalker/SadTalker)（开源，音频驱动面部动画）
- **集成方式**：
  - 看板娘升级：用 MuseTalk 让看板娘"开口说话"
  - 视频版 AI 摘要：对每篇文章生成一个 AI 数字人讲解视频
- **可行性**：⭐⭐ 需要 GPU 算力，实时性要求高
- **门槛**：高

---

## 其他有趣项目

### Anki 风格 flashcards

- **概念**：在文章末尾自动生成知识点卡片，读者可复习
- **实现方式**：构建时用 LLM 从文章提取关键概念 → 生成 Q&A 对 → 前端 Svelte 卡片翻转组件
- **可行性**：⭐⭐⭐⭐ 与博客教学定位契合
- **门槛**：中

---

### Mermaid 思维导图自动生成

- **概念**：自动将文章结构生成 Mermaid 思维导图
- **实现方式**：构建时从 Markdown 标题层级生成 Mermaid 代码
- **可行性**：⭐⭐⭐⭐⭐ 纯文本，零依赖
- **门槛**：低

---

### 互动式决策树/因果图

- **概念**：嵌入交互式 DUCG（动态不确定性因果图）可视化
- **实现方式**：Svelte + D3.js / ECharts 实现因果图编辑器
- **可行性**：⭐⭐⭐⭐ 与博客核心主题 DUCG 高度契合
- **门槛**：中高（需要了解 DUCG 数据结构）

---

## 集成可行性总览

| 项目 | 集成方式 | 难度 | 惊艳度 | 与博客契合度 | 推荐 |
|------|---------|------|--------|------------|------|
| **AKShare** | 构建时数据面板 | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 🥈 |
| **FinGPT** | 内容型（写文章） | ⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 🥈 |
| **FinRL** | 内容型 + Colab | ⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 🥈 |
| **ECharts** | 纯前端组件 | ⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 🥈 |
| **Mermaid 思维导图** | 构建时生成 | ⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | 🥇 |
| **Anki Flashcards** | 构建时 + 前端 | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 🥉 |
| **OpenBB** | 内容型 / MCP | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | 🏅 |
| **数字人** | GPU 渲染 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | 🏅 |
| **predict-anything** | LLM 后端 | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | 🏅 |

---

## 推荐优先级

```
🥇 Mermaid 思维导图 — 纯文本，零依赖，自动生成文章结构图
🥈 AKShare 数据面板 — 构建时静态生成，有数据有图表
🥈 FinGPT / FinRL 系列文章 — 内容型，写文章即可
🥉 AI 文章摘要 + 增强搜索 — 构建时静态生成，安全
🏅 其他项目按需探索
```