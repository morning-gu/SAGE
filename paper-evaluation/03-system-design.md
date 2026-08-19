# 03 系统设计方案

> 四轮设计迭代后的最终架构：统一维度 + 微调 4B + classify-then-explain + 单帧 + 大小模型复检
> 日期：2026-08-15

---

## 1. 架构总览

单帧图像进入系统后经过两层处理：

```
单帧图像
    |
    v
Layer 1: 微调 Qwen3-VL-4B（首 token sigmoid 多标签分类）
    -> 15 个原子行为 + sigmoid 置信度
    -> 零样本 Qwen3-VL-32B（单图，语义验证，第二意见）
    |
    v
Layer 2: 小模型复检层（单帧内验证）
    -> 几何验证(ViTPose) | 深度验证 | 位置验证(10规则) | 物品验证
    -> 置信度调整：一致->提升，不一致->降低/否决
    |
    v
输出：15 个原子行为置信度 + 主导类别
```

关键设计决策：
- 模型只做单帧视觉理解，不涉及时序推理
- 15 个原子行为全部是单帧可判的视觉观察

---

## 2. 类别体系与 Token 映射

### 2.1 完整类别表（15 个原子行为）

| ID | 中文 | 英文 | 候选 token |
|---|---|---|---|
| 0 | 正常 | normal | "normal" |
| 1 | 离座 | away | "away" |
| 2 | 遮挡 | blocked | "blocked" |
| 3 | 玩玩具 | toy | "toy" |
| 4 | 玩电子设备 | phone | "phone" |
| 5 | 吃零食 | snack | "snack" |
| 6 | 闭眼 | eyesclosed | "eyesclosed" |
| 7 | 趴桌 | prone | "prone" |
| 8 | 低头 | bowed | "bowed" |
| 9 | 斜肩 | slope | "slope" |
| 10 | 仰躺 | recline | "recline" |
| 11 | 仰头 | lookup | "lookup" |
| 12 | 歪头 | tilt | "tilt" |
| 13 | 转头 | turn | "turn" |
| 14 | 托腮 | chinrest | "chinrest" |

> **主导类别判定**：不使用固定优先级排序。模型基于视觉理解自行判断最主导的行为；最终主导类别由业务逻辑层结合各类别概率与模型判断结果通过算法确定（当前框架仅输出数据，业务逻辑层暂未实现）。

### 2.2 Token 验证脚本

部署前运行，验证候选 token 是否为 Qwen 词表单 token，不是则添加特殊 token并扩展词表。输出 token_map.json。（完整脚本见旧文档 07 第 2.2 节，核心逻辑：遍历候选词列表，找单 token 映射，无则添加 <|label|> 特殊 token）

---

## 3. 训练方案

### 3.1 Classify-then-explain 训练格式（借鉴 YuFeng-XGuard）

输出 = [主导类别token] + [自然语言解释]

System Prompt 要求模型两步：选主导类别token -> 一句话推理依据。

训练样本示例：
- 输入：单张图像 + "分析学习行为"
- 输出：bowed 学生头部明显前倾，鼻尖低于肩部连线...

主导类别由模型基于视觉理解自行判断（不设固定优先级）。最终主导类别由业务逻辑层结合各类别 sigmoid 概率与模型判断的主导类别通过算法确定。当前框架仅输出两类数据（15 类概率 + 模型判断的主导类别），业务逻辑层暂未实现。

### 3.2 Loss 设计：CE + 辅助 BCE 混合

total_loss = CE(完整序列) + 0.3 * BCE(首 token 类别 logits)

- CE（主 loss）：标准自回归 CE 在完整输出序列上（分类token + 解释）。联合训练提升分类鲁棒性（YuFeng-XGuard 论证：模型学到"能自洽解释的分类"）
- BCE（辅助 loss）：首 token 位置对 15 个类别 token 的 logit 各自独立 sigmoid。提供每类别独立置信度

HybridLoss 实现：CE 用 CrossEntropyLoss(ignore_index=-100)，BCE 用 binary_cross_entropy_with_logits。两者都乘以 source_weights（人工1.0, verified_llm 0.8, llm 0.5）。

### 3.3 LoRA 配置

rank=32, alpha=64, target_modules=[q/k/v/o + gate/up/down + lm_head], dropout=0.05, lr=5e-5(cosine), epochs=5, batch=4(grad_accum=4有效16), bf16, 混入10%通用VQA防遗忘。显存~18GB，训练1-2小时。

---

## 4. 训练数据与 LLM 生成管线

### 4.1 单帧训练样本格式

每条样本：sample_id, source(manual/llm_generated/verified_llm), source_weight(1.0/0.5/0.8), image(单张), messages(system+user[image+text]+assistant[token+explanation]), labels(15二值), primary_label, metadata(user_id/scene/lighting/camera)。

### 4.2 LLM 生成管线（四阶段）

Stage 1: Qwen3-VL-32B 用8个人工种子做few-shot，对未标注单帧生成标签+解释
Stage 2: Qwen3-VL-32B 独立验证标签与图像一致性
Stage 3: 保留Generator与Verifier一致的样本，丢弃不一致的
Stage 4: 随机抽10%人工检查，通过率<85%则整批丢弃

单帧生成的优势：每张图独立，可完全并行，不需要帧对配对，质量更稳定。

### 4.3 数据集划分

| 集合 | 来源 | 样本量 | Loss 权重 |
|---|---|---|---|
| Train | 人工70% + LLM70% | ~2500-3000 | manual=1.0, verified_llm=0.8, llm=0.5 |
| Val | 人工15% + LLM15% | ~300-500 | 温度缩放+早停+阈值 |
| Test | 仅人工15% | ~100-150 | 最终评估，不被LLM数据污染 |

按 user_id 分组划分，防泄漏。Test 集只用人工标注。

---

## 5. 推理管线

### 5.1 分层推理（借鉴 YuFeng-XGuard）

| 层级 | 解码 | 延迟 | 用途 |
|---|---|---|---|
| Tier 1 | 仅首 token logits | ~150ms | 实时检测（softmax主导行为 + sigmoid每类别置信度） |
| Tier 2 | 完整生成 | ~500ms | 按需调试/审计（解释文本） |

Tier 2 触发场景：置信度灰色区间(0.3-0.7)、小模型复检冲突、线上问题排查。

### 5.2 Tier 1 推理

前向传播取首 token logits -> 限制到15个类别token -> softmax得主导行为概率 + sigmoid/T得每类别独立置信度 -> 按每类别阈值判定活跃行为。

### 5.3 Tier 2 推理

自回归生成 [token] explanation -> 解析得主导类别 + 解释文本。

---

## 6. 温度缩放与每类别阈值

### 6.1 温度缩放

验证集上用 L-BFGS 优化全局温度 T：calibrated_prob = sigmoid(logit / T)。目标 ECE < 0.05。

### 6.2 每类别独立阈值

验证集上对每个类别搜索最大化 F1 的阈值。高误报代价类别（phone/toy/snack）阈值上浮0.1。

---

## 7. 小模型复检层

### 7.1 验证器映射

| 现有模块 | 角色 | 验证行为 |
|---|---|---|
| PostureDetectionProcessor (C++) | 坐姿几何验证 | prone/bowed/tilt/slope/lookup/chinrest/recline |
| AttentionDetectionProcessor (C++) | 头部姿态验证 | eyesclosed/turn |
| position_detection_rule.py | 离座验证 | away |
| 深度服务 | 遮挡校正 | slope |
| VLM 32B 输出 | 物品验证 | phone/toy/snack |

### 7.2 置信度调整规则

confirmed -> prob * 1.2; rejected -> prob * 0.3; unavailable -> 不变; 32B也判 -> * 1.15。全部截断到1.0。

### 7.3 物品-手部空间验证（新增）

复用 VLM 32B "手持物品"输出 + ViTPose 手腕关键点(keypoint 9/10)。手腕可见(conf>0.3)且VLM判玩电子设备 -> confirmed；手腕不可用 -> 降级不否决。后续可升级YOLO bbox+手腕IoU精确验证。

---

## 8. 与现有系统集成

### C++ 侧

AnomalyDetectorNode 新增 CheckBehaviorWithFineTuned4B 方法：单图输入，调用 SyncVLMModelInferFirstToken 返回首 token logits JSON。替代现有4次VLM调用(seat/posture/attentionA/attentionB)为1次。

### Python 侧

companion_status_detector.py 的 status_detector 重构为：解析微调4B概率 -> 小模型复检 -> 置信度调整 -> 输出。

---

## 9. 延迟对比

| 方案 | VLM调用 | 输入 | 延迟 |
|---|---|---|---|
| 当前(双帧) | 4次 | 2多图+2单图 | ~2000-3000ms |
| 简化后(Tier1) | 1次首token | 1图 | ~150ms |
| 简化后(Tier1+2) | 2次 | 1图 | ~650ms |

---

## 10. 实施时间线

| 步骤 | 内容 | 时间 |
|---|---|---|
| 1 | 数据标注(800人工+LLM生成2000) | 2周 |
| 2 | Token验证+映射 | 1天 |
| 3 | 训练脚本+LoRA微调(CE+BCE) | 3天 |
| 4 | 温度缩放+每类别阈值 | 2天 |
| 5 | 推理管线(Tier1+Tier2) | 3天 |
| 6 | 集成到系统(C+++Python) | 2天 |
| 7 | 评测+消融 | 3天 |
| 合计 | | 约4周 |
