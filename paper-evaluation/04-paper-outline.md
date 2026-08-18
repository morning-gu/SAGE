# 04 论文 Outline

> 定位：Application + Technique 混合型论文
> 核心叙事：VLM 微调 + 大小模型验证范式 -> 单帧学生行为检测
> 候选投稿：Expert Systems with Applications (Elsevier, IF 8.5)

---

## 拟定标题

**英文**：A Propose-Verify Framework for Single-Frame Student Behavior Detection with Fine-tuned Vision-Language Models

**中文**：基于微调视觉语言大模型的大模型提议-小模型验证单帧学生行为检测框架

---

## Contribution List（3 项）

1. **大模型提议-小模型验证范式**：微调 Qwen3-VL-4B 输出 15 个原子行为 + 校准置信度（首 token sigmoid），骨骼几何/物品检测/深度等小模型复检校验或否决。借鉴 YuFeng-XGuard 的 classify-then-explain 训练格式。

2. **物品-手部空间重合验证**：用物品检测 + 手腕关键点的空间关系验证 VLM 的行为分类。

3. **评测数据集 + 系统性消融**：构建 800 单帧多标签数据集 + LLM 生成扩充到 3000+，设计 D0-D3 消融矩阵量化各模块贡献。

---

## Section 1: Introduction（1.5 页）

### 六段逻辑链

1. **背景**：智能伴学产品通过摄像头实时监测学习行为，检测坐姿异常和注意力分散时提醒。
2. **现有局限**：传统方法依赖专用训练模型，泛化差；纯 VLM 零样本精度不足（几何判断弱）；无校准置信度无法做验证/否决。
3. **问题本质**：如何在无需训练专用模型的前提下，实现高精度、置信度驱动的实时学生行为检测？
4. **关键挑战**：(1) 语义-精度矛盾（VLM语义强但几何弱）；(2) 遮挡处理；(3) 多行为共存的多标签分类。
5. **解决方案**：微调 4B 为首 token 多标签分类器 + classify-then-explain 训练 + 小模型复检。
6. **贡献列表**：如上 3 项。

---

## Section 2: Related Work（1 页）

- **学生行为监测**：Zaletelj 2017(Kinect), Huang 2024(CV), Su 2021(视频分析), Tang 2026(传感器)
- **坐姿识别**：Cao 2025(MVSP), Wang 2025(USSP), Li 2023(异常坐姿)
- **VLM 微调与首 token 推理**：YuFeng-XGuard 2026(classify-then-explain + 首 token 分层推理)
- **定位**：首次将微调 VLM + 大小模型验证范式应用于学生行为监测

---

## Section 3: Method（3 页）

### 3.1 系统架构（Figure 1）

两层处理：微调 4B 分类 -> 小模型复检 -> 输出。

### 3.2 微调 4B 分类器（1 页）

- Classify-then-explain 训练格式
- CE + 辅助 BCE 混合 loss
- LoRA 微调配置
- 首 token sigmoid 多标签输出

### 3.3 小模型复检层（1 页 + Algorithm 1）

- 验证器映射表（几何/深度/位置/物品）
- 置信度调整规则（confirmed/rejected/unavailable）
- 物品-手部空间验证

### 3.4 分层推理（0.5 页）

- Tier 1 首 token ~150ms + Tier 2 完整生成 ~500ms（按需）
- Tier 2 触发场景：置信度灰色区间、小模型复检冲突

### 3.5 个性化参考帧（0.5 页）

- 参考帧选择流程
- 离座检测 10 规则（cur骨架 vs 参考帧骨架）

---

## Section 4: Experiments（2-3 页）

### 4.1 数据集（Table 1）

800 单帧人工标注 + LLM 生成扩充。类别分布、Cohen's kappa、与现有数据集对比。

### 4.2 主实验（Table 2）

D0(零样本基线) vs D1(微调SFT) vs D2(微调+验证层) 的 mAP/Macro-F1/ECE/primary_accuracy/延迟对比。

### 4.3 消融实验（Table 3 + Figure 3）

D0-D3 消融矩阵。关键发现：微调 vs 零样本、验证层贡献、校准贡献。

### 4.4 置信度校准（Figure 4）

Reliability diagram，温度缩放前后 ECE 对比。

### 4.5 延迟分析（Table 4）

当前双帧(~2000ms) vs Tier1(~150ms) vs Tier1+2(~650ms)。

### 4.6 失败案例（Figure 5）

4-6 个代表性失败案例。

---

## Section 5: Discussion（0.5 页）

- VLM 在行为分类中的优势与局限
- 大小模型验证范式的有效性
- 局限性：数据规模、模型依赖、场景特定、隐私、仅单帧不涉及时序推理

---

## Section 6: Conclusion（0.3 页）

总结 3 项贡献，展望 VLM 持续进步可能进一步简化架构。

---

## 图表清单

| 编号 | 内容 |
|---|---|
| Figure 1 | 系统架构图（两层处理） |
| Figure 2 | 深度遮挡校正示意 |
| Figure 3 | 消融实验柱状图 |
| Figure 4 | 置信度校准 reliability diagram |
| Figure 5 | 失败案例可视化 |
| Table 1 | 数据集统计 |
| Table 2 | 主实验结果 |
| Table 3 | 消融实验结果 |
| Table 4 | 延迟分析 |
| Algorithm 1 | 复检伪代码 |

---

## 写作时间线

| 阶段 | 内容 | 时间 |
|---|---|---|
| 1 | 数据集构建+微调 | 2周 |
| 2 | 评测+消融 | 1周 |
| 3 | Intro+Related Work | 1周 |
| 4 | Method | 1周 |
| 5 | Experiments | 1周 |
| 6 | Discussion+Abstract+润色 | 0.5周 |
| 合计 | | 约6周 |