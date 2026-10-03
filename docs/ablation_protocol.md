# SAGE v2 消融实验协议（预注册）

> 状态：**冻结**（2026-09-21）。本文件在 v2 实验开跑前定稿，跑完前不得修改判定规则。
> 目的：回应 2026-09-21 五席评审的两项已验证 CRITICAL（DA C-1 baseline 公平性、DA C-2 MP 未消融），并以预注册方式防止 HARKing（结果出来后再改假设）。

---

## 1. 目的与范围

| 评审问题 | 本协议的回应 |
|---|---|
| DA C-1：baseline 欠训练 + 分辨率不对等，headline 对比不成立 | YOLO / ResNet / YOLO-BCE 三 family 网格搜索（val-only 模型选择）+ 选优后 3 seeds；YOLO imgsz 上探 640 |
| DA C-2：MP 损失项从未消融却写入 C2 | d1_bce_mp vs d1_bce_nomp，各 3 seeds，预注册 keep/drop 规则（§4） |
| 载体检 vs 机制混淆 | exthead 外部头对照：CE-only LoRA 特征 + 线性探针（§3.5） |
| 单 seed Major | 除 D0（§6）外全部实验 3 seeds |
| 数值口径混乱 | 统计检验与延迟测量规范固定（§5、§7） |

## 2. 命名与目录

- 命名：`<design>[_<loss>]_s<seed>[_<variant>]`（如 `d1_bce_nomp_s43`、`yolo_grid_e50_lr1e-3_i640`）
- 结果落 `results/experiments_v2/<exp_id>/`，每实验至少含 `eval_report.json` + `predictions.json`（eval_only 的 D0 除外，复用现有结果）
- 每个实验由 `configs/experiments/<exp_id>.yaml` 注册，经 `scripts/run_experiment.py` 执行（幂等；`--force` 强制重跑；`--until <hours>` 预算截断）

## 3. 实验矩阵（总预算 ≈ 60-65 GPU 时，单卡 A10 约 3 自然日）

| 组 | 实验 ID | 配置 | 目的 | GPU 时 | 状态 |
|---|---|---|---|---|---|
| MP 消融 | `d1_bce_nomp_s{42,43,44}` | `--mp-weight 0`（BCE-only） | DA C-2 方向 | 18h | 待跑 |
| MP 消融 | `d1_bce_mp_s{42,43,44}` | `--mp-weight 1.0`（完整 SAGE loss） | 同上对照组；s42 兼作 provenance 重跑（旧 ckpt args.json 不记录 loss 权重） | 18h | 待跑 |
| YOLO 网格 | `yolo_grid_e{25,50,100}_lr{1e-3,1e-2}_i{224,640}` 12 个 | yolov8s-cls | DA C-1：公平基线，val 选优 | ~8h | 待跑 |
| ResNet 网格 | `resnet_grid_lr{1e-4,1e-3}_e{25,50}_i{224,448}` 8 个 | ResNet50+sigmoid+BCE | DA C-1 | ~5h | 待跑 |
| YOLO-BCE 网格 | `yolobce_grid_` 同 ResNet 维度 8 个 | YOLOv8s backbone+sigmoid+BCE | 审稿人点名缺失基线（softmax vs sigmoid 隔离） | ~6h | 待跑（可优雅失败，不阻塞主线） |
| 外部头 | `exthead_ce-lora_s{42,43,44}` | CE-only LoRA 特征 + Linear(hidden,15) | 载体检 vs 机制分离 | ~1-2h | 待跑 |
| 外部头（可选） | `exthead_frozen_s42` | 不挂 LoRA | 载体检上限 | ~1h | 可选 |
| 零样本 | `d0_zeroshot_s42` | 复用现有 D0 结果 | 免重跑（§6） | 0 | 复用 |
| 选优补种子 | 各 family val 最优配置补 s43/s44 | — | 多 seed mean±std | ~3-6h | 依赖网格结果 |

历史 D0/D1/D1-CE/D-yolo/D-cnn/D-sup/D2 结果保留，标注"历史结果（seed=42，旧 baseline 协议）"，仅作参考不再进入 v2 主表。

## 4. MP keep/drop 预注册规则（防 HARKing，冻结）

对每对 seed（42/43/44 各自配对）计算 D1(BCE+MP) − D1(BCE-only) 的 macro-F1 配对 bootstrap 差（10000 次重采样，seed=42，双侧百分位法，`sage.metrics.paired_bootstrap`）：

- **KEEP**：≥ 2/3 seeds 的 95% CI 排除 0 **且** 池化（三 seed 样本按行拼接后）Δmacro-F1 ≥ +1pp → MP 保留进 C2 贡献声明
- **DROP**：不满足 KEEP → MP 从 C2 贡献声明移除；`train_sage_vlm.py` 默认 `--mp-weight 0`；论文如实报告"MP 修正未带来显著增益"
- 三 seed 方向不一致（部分正部分负）→ 如实报告 **inconclusive** 并执行 DROP（不允许事后挑选有利 seed）
- DROP 情形下 MP 仍作为实现细节保留在代码与附录中，附本次消融数据

## 5. 统计检验规范

- **McNemar**（Primary Acc hit 的逐样本配对二值结果）：n_disc = n01+n10 < 25 用精确二项检验，否则 χ2 连续性校正。实现单源 `sage.metrics.mcnemar_exact`（历史手写规则已固化为代码）
- **macro-F1 配对 bootstrap**：10000 次重采样、seed=42、双侧百分位法（`sage.metrics.paired_bootstrap`）
- **p 值分辨率下限**：bootstrap 的 p 值分辨率是 1/n_boot = 0.0001。报告 `p < 0.0001`，**禁止写 `p < 1/10000` 之外更精细的表述**（如 p<0.000001 超出该方法可支持的分辨率）
- **多种子报告**：mean ± std（样本 std，ddof=1）+ 各 seed 数值点列；不合并 seeds 做单一检验后代替 per-seed 报告
- 历史锚点回归：`tests/test_metrics.py` 钉住 0.12902（全正退化 macro-F1）、p=0.125（n01=4/n10=0）、p=0.3616（n01=18/n10=12）等论文数值

## 6. D0 免多种子的理由

D0 为零样本贪心解码（max_new_tokens=1、无采样随机性）：同一 checkpoint 权重下输出确定，多种子必然完全一致，跑多种子无信息量。种子方差仅来自训练初始化/数据顺序，D0 不训练，故单次评估即代表其全部方差。

## 7. 延迟测量协议（修正 372+71≠497 口径）

- 测量前丢弃 warmup 样本（前 20 个请求不入统计）
- 报告 p50 / p95（`sage.metrics.latency_stats` 单一实现）
- **D1（VLM 单独）与 D2（VLM + 复检串行端到端）分别声明**：D2 端到端延迟与 D1 + 复检开销之和不必相等（并行/串行实现细节、调度开销），论文只报告直接测得的 D2 端到端 p50/p95 与复检层单独开销 p50/p95，禁止将两者相加"推出"端到端数字
- 硬件与 batch 注明（A10 24GB，bs=1 串行推理）

## 8. 公平性协议（DA C-1）

1. **val-only 模型选择**：网格（超参/分辨率/epoch 数）只在 val(288) 上评估与选优；**禁止用 test(287) 参与任何模型选择**
2. **test 只评一次**：每 family 选出的最终配置在 test 上评估一次（各 seed 各一次），不做 test 上的二次调参
3. **early-stop**：各网格 run 监控 val macro-F1（`sage.ml_common.fit`，`--patience` 设定；网格 epoch 上限 e25/e50/e100 为预算维度）
4. **分辨率对齐**：YOLO 网格含 imgsz=640（其原生训练分辨率尺度），ResNet 网格含 448；报告时同时给 224（历史可比）与选优分辨率两行，并声明"分辨率上界"含义——VLM 输入为原生分辨率，网格已将各基线推至其合理分辨率上限
5. **3 seeds**：每 family 选优配置跑 s42/s43/s44 报 mean±std
6. 训练预算对齐声明：VLM 单次 5h47m（train_runtime=20852s）；网格 epoch 数（25/50/100）为使小模型充分收敛的计算预算维度，选优后基线的最终对比基于其**各自 val 最优**而非统一弱配置

## 9. 风险与 fallback

| 风险 | 缓解 |
|---|---|
| 预算 60-65 GPU 时不可接受 | fallback ①：复用现有 seed-42 checkpoint（论文标注历史代码版本）；② VLM 5→3 epochs + eval_loss 早停（best ckpt 在 epoch 4）省 ~14h，牺牲与历史可比性 |
| ms-swift 版本漂移 | 训练环境 pin ms-swift==4.4.2；runner 启动前 import 探针 |
| 非 Ampere GPU（T4/Turing，无原生 bf16） | **默认精度已改为 fp16**（2026-09-21 起，T4 训练机）；A10/Ampere 机器用 `--dtype bf16` 显式回退历史协议。fp16 与历史 bf16 run 数值行为可能有细微差异，同一组实验内必须统一精度并在论文复现说明中记录；T4 16GB 正式跑前先 5-step 冒烟确认不 OOM（历史 peak ~13GB 为 bf16/A10 数字） |
| ultralytics 内部 API 取 backbone 不稳定 | pin 版本 + `train_yolo_bce.py` 优雅失败（exit 2），yolobce_grid 失败不阻塞主线 |
| 测试集 287 样本 bootstrap CI 宽 | 预注册规则已含 ≥2/3 seeds 条件，允许 inconclusive 出口（§4） |
| exthead 特征提取绕开 SwiftSft | 纯 transformers+peft 实现（`scripts/extract_ext_features.py`），不受 monkey-patch 脆弱性影响；JSONL prompt 与训练记录逐字一致 |
| d1_ce_only 历史 predictions.json 概率退化（全 1.0，收集 bug） | 涉及该对照的 McNemar 从 primary label 重算；不直接使用其存储 probs |

---

## 修订日志（不改动上述冻结规则）

- **2026-09-29**：发现 e3_eval_s43/s44 的 eval 阶段配置误指向 per-seed val pool（val 同时是角度上限的冻结集），Table 4 的 s43/s44 两行与池化统计受污染。已在相同冻结常数（ecdf/tau/thresholds 均出自 val，协议不变）下对 test split 重跑 s43/s44 的 eval 阶段；s42 本就使用 test。旧 val-run 工件存档为 `metrics_valrun.json` / `cases_valrun.jsonl`（results/veto_eval_s43|s44/）与 `eval_report_valrun.json`（results/experiments/e3_eval_s43|s44/）；eval 配置已改回 test.jsonl；池化统计工件见 `results/stats_pooled_mcnemar.json`。冻结规则本身零改动。
- **2026-09-29（帧级统计补充）**：新增帧级（聚类感知）统计工件 `results/stats_frame_level.json`（帧级 McNemar 3:1、图像级 VP 9/10），作为池化统计的伴随披露。同轮确认的协议偏差一并记录：①选优配置未补 3 seeds（§8.5 预算未执行，论文以单 run 报告）；②延迟表报 mean/p95 而非 §7 的 p50/p95（测量脚本口径所致）。新增帧级（聚类感知）统计工件 `results/stats_frame_level.json`（帧级 McNemar 3:1、图像级 VP 9/10），作为池化统计的伴随披露。
