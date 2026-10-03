# §4.3 Geometric Veto Algorithm — 形式化定稿

> 状态：**本文 §4.3 的唯一权威底稿**（正文英文为 LaTeX transplant-ready）。论文写作、实验与代码实现
> 均以本文件描述的现状为准；历史迭代记录见 git log 与冻结决策表（§7）。
>
> 核心契约（冻结）：
> 1. **FVR 安全底线** δ = 0.05；τ 与角度上限都在验证集上按同一规则冻结（FVR ≤ δ 的网格最大值）。
> 2. **验证集分位数归一化**：每策略 ECDF（n_min = 30 + 点质量退化防护，单值占比 > 50% 即禁用——
>    无有效校准即无否决权）。
> 3. **normal 不在否决范围内**；策略层是**主张条件化验证器**——只在 VLM 声称某异常类时被调用，对该
>    "声称"做证据一致性核查，永不主动判定异常。
> 4. **统一门控 + 冻结阈值权威双路径**：所有否决都需通过验证集冻结的权威门——多数策略走分位门
>    （ẽ ≥ τ）；slope/tilt 的矛盾区（头接近水平）是全帧池的多数区，分位门结构性无法判别，改用 val
>    声称帧上冻结的角度上限（raw 下限 = −max_angle），替换而非叠加分位门。每策略只有一条权威路径。
> 5. **策略注册表（7 个）**：slope / tilt / bowed / prone / lookup / recline / eyesclosed，全部为
>    pose/人脸关键点几何策略并全部产出 raw_evidence（方向：值越大越反驳声称）。collect 阶段跑全部
>    策略（不按 claims 过滤）以越过校准下限。
> 6. **证据层**：tilt/lookup 优先使用 ROI 裁剪人脸关键点的眼线/眼角几何（ear_tilt_source 溯源），
>    姿态耳线仅作 face 检测失败时的回退；姿态耳点在高置信度下仍会大幅误读头部滚转，故不作主测量。
>    人脸服务按 human_detect 框 + 35% 边距裁剪推理，landmark 映射回全图坐标。
> 7. **经查证否决的设计项**：prone 几何归属门（prone_raw 在 tilt 真/假声称上分布完全重叠，任何切点
>    都会误杀真声称）——不实施；相关误报如实归入不可收割集合。

---

## 1. Notation and Decision-Authority Contract

Let $\mathcal{C} = \{c_1, \dots, c_{15}\}$ denote the behavior classes, where $c_1 = \texttt{nr}$ (normal) is the default study state and $\mathcal{A} = \mathcal{C} \setminus \{\texttt{nr}\}$ the 14 anomaly classes. For an input frame $x$:

- The VLM (Stage 1, §4.2) produces per-class probabilities $p(c \mid x) \in [0,1]$ from a single first-token forward pass. The predicted multi-label set is $\hat{Y} = \{c : p(c \mid x) \geq 0.5\}$ and the primary prediction is $\hat{y} = \arg\max_c p(c \mid x)$.
- The veto layer (Stage 2) is **claim-conditioned**: it is invoked only on the anomaly claims $c \in \hat{Y} \cap \mathcal{A}$. For each claimed class $c$, strategy $s_c$ examines frame $x$ and returns a verdict on the *claim itself*:

$$v_c(x) \in \{\texttt{contradicted},\ \texttt{consistent},\ \texttt{insufficient}\}, \qquad e_c(x) \in [0,1]$$

  where `contradicted` means the geometric or object evidence is inconsistent with the presence of $c$, `consistent` means the evidence is compatible with the claim, `insufficient` means no usable evidence (e.g., detector miss, keypoints out of view). $e_c$ is the strategy's evidence strength after cross-strategy quantile normalization (§3).

**Decision-authority contract (定义性陈述，§4.1 复用).** The VLM holds default detection authority: its predictions stand unless vetoed. The small model holds *veto-only authority* over the VLM's anomaly claims: it may remove a claimed anomaly but (i) never adds or promotes a label, (ii) never initiates an anomaly judgment of its own — strategies run only on VLM claims, and (iii) never acts on `insufficient` evidence. Normal is outside the veto scope by construction: vetoing normal could only yield an empty prediction or a circular fallback, and repairing a missed anomaly would require an add-action that the contract forbids. This boundary is stated as a design property, not a deficiency (§6.2).

## 2. Strategy Layer (§4.3.1)

7 claim-conditioned strategies (trimmed from 14 during development — see header), one per
retained anomaly class, all geometric / face-landmark based:

- **Kept** (VLM-weak classes first): slope, recline, eyesclosed, tilt — the VLM's four weakest anomaly classes (E1/E2: primary error 73–83% or multi-label precision 0.53); plus lookup, bowed, prone — reliably decisive checks.
- **Removed**: phone/snack/toy (object-detector recall too low: 0–12% decisive, 0 vetoes), turn (3% decisive), blocked (loses exactly the keypoints it needs under occlusion), chinrest (VLM already at F1 0.99).
- **Removed**: away — semantic mismatch with the annotation convention (dataset away = empty seat, strategy checked "any person standing") plus a point-mass raw pool; development diagnostics showed every veto it fired was a false kill of a class the VLM already solves at F1 0.98.

Tilt additionally carries the **face eye-line fallback**: when the pose ear geometry is missing or degenerate (`ear_tilt_unreliable`, extreme nose_ratio) but the face detector returns 106 landmarks, `ear_tilt` carries the eye-line roll angle with `ear_tilt_source = "face_eye_line"`, the ear-availability gates are bypassed, and a fixed 23° threshold band replaces the nose_ratio-adaptive one (the uncorrected eye line reads systematically larger).

Each strategy emits raw evidence $g_c(x)$ — a measured geometric quantity oriented so that **larger values contradict the claim more strongly** (e.g. EAR for eyesclosed, negated shoulder tilt for slope, torso extension ratio for recline) — compared against a class-specific threshold for the verdict.

Each strategy is an *evidence consistency checker*, not a detector: it answers "is the frame evidence compatible with the claim that $c$ is present?", conditional on the VLM having claimed $c$. It is never asked "is any anomaly present?" — that question belongs exclusively to Stage 1.

## 3. Evidence-Strength Normalization (冻结决策 2)

Raw evidence strengths are strategy-specific in both scale and distribution (angle margins in degrees vs. detector confidences in $[0,1]$). To admit a single demotion threshold $\tau$:

1. For each strategy $s_c$, collect raw evidence values $\{g_c(x)\}$ over all **validation** frames — the collect stage runs every registered strategy on every frame (not only the claimed classes) so each strategy's pool clears $n_{\min}$.
2. Map each value to its empirical quantile: $\tilde{e}_c(x) = \hat{F}_{c,\text{val}}(g_c(x)) \in [0,1]$, where $\hat{F}_{c,\text{val}}$ is the validation ECDF (piecewise-linear interpolation between order statistics).
3. Test-frame evidence is normalized by the *frozen* validation ECDF. A strategy whose pool fails either floor is disabled — its $\tilde{e}_c$ is `None` for all test frames, i.e. no veto authority: (i) fewer than $n_{\min} = 30$ validation invocations (insufficient calibration), or (ii) a **degenerate pool**: a single value holds more than 50% of the samples. A point-mass pool pins the modal value's quantile at ≈ 1.0 regardless of $\tau$, so any contradiction landing on it vetoes at every $\tau$ — the gate cannot discriminate. This is the empirically observed away failure mode (283/284 validation frames at seated_frac = 1.0; every veto it fired in development diagnostics was a false kill).

$\tilde{e}_c$ therefore carries a unified semantics: the fraction of same-strategy validation evidence values it dominates. $\tau$ is a single quantile threshold across all strategies.

## 4. Veto Decision Rule (§4.3.2, Algorithm 1)

> **Algorithm 1: Claim-Conditioned Geometric Veto**
> **Input:** frame $x$; VLM output $p(\cdot)$, $\hat{Y}$, $\hat{y}$; frozen threshold $\tau$
> **Output:** corrected set $\hat{Y}'$, corrected primary $\hat{y}'$

```text
1:  V ← ∅
2:  for each c ∈ Ŷ ∩ A do                    # claim-conditioned: VLM claims only
3:      (v_c, ẽ_c) ← s_c(x)                  # evidence consistency check
4:      if v_c = contradicted and ẽ_c ≥ τ then
5:          V ← V ∪ {c}                      # veto the claim
6:  Ŷ' ← Ŷ \ V
7:  if ŷ ∈ V then                            # primary fallback
8:      ŷ' ← argmax_{c ∈ Ŷ' ∩ A} p(c) if Ŷ' ∩ A ≠ ∅ else nr
9:  else
10:     ŷ' ← ŷ
11: return Ŷ', ŷ'
```

Design rationale (对应 §6.2 The Veto Contract):

- **Dual validation-frozen authority (line 4).** Every veto requires validation-frozen authority under the same $\delta$ rule, via one of two paths:
  * **Quantile authority** (default): normalized evidence strength $\tilde{e}_c \geq \tau$ with a computable, calibrated value. An earlier asymmetric rule (non-primary vetoes on `contradicted` alone) made FVR structurally independent of $\tau$, which the first E3 run demonstrated empirically; uniform gating restored $\tau$ as the single knob.
  * **Frozen-threshold authority**: for the interpretable-angle strategies (slope/tilt) whose contradiction region is the *majority* region of the all-frames pool, the quantile gate is structurally unable to discriminate (the pool's tilde jumps from ≈0.94 to 1.0 across the exact-0 point mass, so τ = 0.95 admits only quantization artifacts). For these, a per-strategy angle ceiling $T^*_c$ is frozen on validation **claims** (largest grid value with FVR ≤ δ among claimed-and-contradicted frames with angle ≤ $T^*_c$); the veto condition becomes $g_c(x) \geq -T^*_c$ (contradiction-oriented raw = negated angle). The floor *replaces* the quantile gate for that strategy — each strategy still has exactly one authority path, each individually frozen to satisfy FVR ≤ δ.
- **Threshold freezing rule (冻结决策 1).** $\tau$ is selected on the validation set as the largest value such that validation FVR (§5) does not exceed the pre-registered safety floor $\delta = 0.05$; $\tau$ is then frozen and applied once to the test set. The sweep grid is $\tau \in \{0.50, 0.55, \dots, 0.95\}$.
- **Normal is out of scope (冻结决策 3).** Line 2 iterates over $\hat{Y} \cap \mathcal{A}$ only; $s$ strategies are never invoked for normal, and $\hat{y} = \texttt{nr}$ is never demoted.
- **No-add invariant.** `consistent` verdicts never insert labels absent from $\hat{Y}$; the pipeline is monotone in label removal only.
- **Fallback to default state (line 8).** If the demoted primary was the only anomaly claim, the system predicts normal — the taxonomy's designated default state. This is a Stage-1 fallback rule, not a Stage-2 judgment.

**Consequences of the claim-conditioned contract (诚实边界，写入 §6.2):** the veto layer can only improve precision-side reliability (removing wrong claims). VLM false negatives — missed anomalies, including the geometric blind-spot classes — are structurally beyond its reach and require temporal or multi-view extensions (§7 future work). The semantic–geometric complementarity claim (C1) is therefore a precision-side claim.

## 5. Veto-Quality Metrics (§4.3.3)

All metrics are computed over (frame, class) pairs on the evaluation split. Let $V$ be the set of veto decisions and $Y(x)$ the ground-truth label set of frame $x$.

| Metric | Definition | Interpretation |
|---|---|---|
| Veto Precision (VP) | $\frac{|\{(x,c) \in V : c \notin Y(x)\}|}{\|V\|}$ | fraction of vetoes that remove a wrong claim |
| False-Veto Rate (FVR) | $1 - \mathrm{VP}$ | fraction of vetoes that kill a true positive — **safety floor metric** |
| True-Positive Kill rate (TPK) | $\frac{|\{(x,c) \in V : c \in Y(x)\}|}{|\{(x,c) : c \in Y(x), c \in \mathcal{A}\}|}$ | fraction of true anomaly positives killed — risk-exposure view |
| Coverage | $\frac{|\{(x,c) : c \in \hat{Y} \cap \mathcal{A},\ v_c \neq \texttt{insufficient}\}|}{|\{(x,c) : c \in \hat{Y} \cap \mathcal{A}\}|}$ | fraction of VLM anomaly claims that received an evidence check |

Primary-demotion quality is reported separately: among demoted primaries, the fraction whose corrected primary $\hat{y}'$ matches the ground-truth primary, against the un-demoted baseline. VP and FVR are always reported as a pair (写作红线 4/5)。All metrics decompose into the semantic-cued and geometry-cued groups of §3.2. Note that Coverage is now claim-dependent (it conditions on VLM output); the per-frame claim count distribution is reported alongside it.

## 6. Evaluation Script Spec — `scripts/eval_veto.py`

**Inputs (deterministic, no GPU):**

```
--annotations  PATH   ground truth:  frame_id -> {labels: [...], primary: str}
--predictions  PATH   VLM output:    frame_id -> {probs: {class: float}, primary: str}
--evidence     PATH   strategy log:  frame_id -> {class: {verdict, raw_evidence}}
                      # 仅覆盖 VLM 声称的异常类；normal 无记录
--ecdf         PATH   frozen validation ECDFs per strategy (produced by --fit step)
--tau          FLOAT  frozen demotion threshold
--delta        FLOAT  FVR safety floor (default 0.05, preregistered)
--fit          FLAG   stage A: fit validation ECDFs + sweep tau grid, emit ecdf file + sweep curve
--eval         FLAG   stage B: single frozen-tau pass on test
--groups       PATH   taxonomy grouping: class -> {semantic | geometric}
--out-dir      PATH   output directory (default results/veto_eval/)
```

Two-stage protocol (preregistrable): (A) fit ECDFs + sweep $\tau$ on validation, freeze at FVR ≤ δ; (B) single frozen-$\tau$ pass on test. Both stages emit artifacts; test metrics come only from stage B. A `freeze` stage sits between A and B: on validation records **with VLM probabilities**, freeze the per-strategy raw floors for the interpretable-angle strategies (slope/tilt) — `--stage freeze --records <val.jsonl> --out-dir <dir>` writes `thresholds.json` (`{strategy: {max_angle, raw_floor, n_pool, fvr_at_T}}`), consumed by `--stage eval --thresholds <thresholds.json>`. Freezing requires claim pools; evidence-only records (no VLM) cannot support it.

**Outputs:**

- `metrics.json` — VP / FVR / TPK / coverage, overall + per group + per class; demotion accuracy before/after; per-frame claim-count distribution.
- `cases.jsonl` — one record per veto: `frame_id, class, verdict, tilde_e, correct (bool), was_primary (bool), demotion_outcome ∈ {improved, degraded, neutral} | null`.
- `summary.md` — §5.3 tables.

Veto logic lives in `sage/pipeline.py` (single source of truth); the script is a thin CLI over it. End-to-end veto on/off comparison reuses `sage/eval_harness.py`.

## 7. Frozen Decisions Log

| # | 决策 | 冻结值 | 日期 |
|---|---|---|---|
| 1 | FVR 安全底线 δ | **0.05**（τ 取验证集上满足 FVR ≤ 0.05 的最大值） | 2026-09-22 |
| 2 | e_c 归一化 | **验证集分位数归一化**（每策略 ECDF，测试集用冻结 ECDF 插值；n_min=30 校准下限） | 2026-09-22 |
| 3 | normal 降级 | **禁止**——升级为架构决策：策略层主张条件化，仅否决 VLM 异常声称，永不主动判定异常；NormalStrategy 移除 | 2026-09-22 |
| 4 | E4 公平性 | 软融合基线与 Algorithm 1 共用同一策略层输出（contradicted→负 LR、consistent→正 LR、insufficient→0） | 2026-09-22 |
| 5 | 门控统一 | **所有否决**（含非 primary）要求 ẽ ≥ τ；无校准强度即无否决权。依据：首轮 E3 证明旧非对称门控下 FVR 不受 τ 控制 | 2026-09-26 |
| 6 | 策略注册表 | 14 → 8：删 phone/snack/toy/turn/blocked/chinrest（决断率 0–14%）；保留策略全部产出 raw_evidence（方向：值越大越反驳声称） | 2026-09-26 |
| 7 | away 删除 + 退化防护 | 注册表 8 → 7（away 语义与标注相反且点质量退化，开发期诊断中其全部否决均为误杀）；ECDF 池单值占比 > 50% 即禁用（DEGENERATE_MASS=0.5），与 n_min 并列为校准下限 | 2026-09-26 |
| 8 | 冻结阈值权威 + 眼线回退 | slope/tilt 的否决权威改为 val 声称帧冻结的角度上限（`--stage freeze`，FVR ≤ δ 规则同 τ；max_angle 常数待 T4 val 采集后冻结，禁止用 test 取证值）；tilt 增加人脸眼线回退（ear_tilt_source 溯源）；prone 归属门经查证否决（分布重叠） | 2026-09-26 |

~~Open Items~~ → 全部关闭。
