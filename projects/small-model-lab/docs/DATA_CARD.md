# 合成数据卡：original-synthetic-fixture-v1

本项目为工程测试自写 16 篇短文，主要为英语，含两条中文；另有 8 条 SFT 和 8 条 DPO 示例。所有 fixture 以 CC0-1.0 提供（[法律文本](https://creativecommons.org/publicdomain/zero/1.0/legalcode)）。没有抓取外部语料，没有个人数据，没有下载他人模型输出。

预训练以 document group 为单位按固定种子 hash 排序，约80/10/10划分（本fixture为14/1/1）。精确重复合并，并把相关 document group 连通后再分组。文本做NFC及空白规范化，不改变大小写。完整输入、各split、tokenizer均保留SHA-256；tokenizer只拟合train。

后训练各6条train、1条validation、1条test；DPO使用与SFT一致的split归属。chosen为简短说明，rejected统一为无关句子，仅用于检测DPO数据流和梯度方向。该偏好分布极不真实、长度不匹配，不能作为偏好能力评测。

当前检查覆盖manifest完整性、同文件prompt跨split、显式列入related-post-data的跨阶段prompt/完整pair，以及与预训练完整文档的规范化精确匹配。不覆盖未列出的文件、片段/语义/近重复、外部基准污染。许可字段只是来源声明，不会自动赋予第三方数据的训练或再分发权利。
