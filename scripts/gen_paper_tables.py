"""Generate LaTeX tables for the ESWA manuscript from the eval artifacts.

Every number in the paper's §5 tables is generated from the experiment
reports — no hand-transcribed values.  Run after the E1/E3 artifacts are
final:

    python scripts/gen_paper_tables.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

PAPER = Path("paper/tables")
PAPER.mkdir(parents=True, exist_ok=True)

SEEDS = (42, 43, 44)
VARIANTS = ("d1_bce_nomp", "d1_bce_mp", "d1_ce_only")
NAMES = {"d1_bce_nomp": "Dual-channel (ours)",
         "d1_bce_mp": "Dual-channel + MP",
         "d1_ce_only": "Generative-only (CE)"}

CLASS_ORDER = ["away", "lookup", "phone", "slope", "turn", "toy", "prone",
               "bowed", "eyesclosed", "blocked", "chinrest", "recline",
               "tilt", "snack", "normal"]


def load_variant(variant: str) -> dict:
    """Pooled per-class tp/fp/fn and seed-level aggregates (test split)."""
    tp: dict[str, int] = {}
    fp: dict[str, int] = {}
    fn: dict[str, int] = {}
    macro, mmap, prim = [], [], []
    prim_exact = []
    for seed in SEEDS:
        r = json.load(open(f"results/experiments/{variant}_s{seed}/eval_report.json"))
        o = r["overall"]
        macro.append(o["macro_f1"])
        mmap.append(o["mAP"])
        prim.append(o["primary_accuracy"])
        prim_exact.append(o["primary_accuracy_exact"])
        for cls, d in r["per_class"].items():
            tp[cls] = tp.get(cls, 0) + round(d["f1"] and d["support"] * d["recall"] or 0)
            fp[cls] = fp.get(cls, 0) + d["fp"]
            fn[cls] = fn.get(cls, 0) + d["fn"]
    return {"tp": tp, "fp": fp, "fn": fn,
            "macro": macro, "mAP": mmap, "prim": prim, "prim_exact": prim_exact}


def fmt_mean(values: list[float]) -> str:
    m = sum(values) / len(values)
    s = (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5
    return f"{m:.3f} $\\pm$ {s:.3f}"


def main_table() -> str:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Test-split results (3 seeds, mean $\\pm$ std). "
        "Primary accuracy uses the exact primary-label match; macro-$F_1$ "
        "and mAP are computed over the 15 multi-label classes.  The "
        "generative-only variant collapses on the auxiliary channel "
        "(macro-$F_1$ at the degenerate all-negative level) while retaining "
        "a usable primary channel.}",
        "\\label{tab:main}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Variant & macro-$F_1$ & mAP & primary & primary (exact) \\\\",
        "\\midrule",
    ]
    data = {}
    for v in VARIANTS:
        data[v] = load_variant(v)
        d = data[v]
        lines.append(
            f"{NAMES[v]} & {fmt_mean(d['macro'])} & {fmt_mean(d['mAP'])} & "
            f"{fmt_mean(d['prim'])} & {fmt_mean(d['prim_exact'])} \\\\")
    # E0 baselines (validation-selected best per family; no seeds).
    # 2026-09-29: selection restored to the frozen protocol (val macro-F1 for
    # the sigmoid families, val top-1 for yolo-cls); the previous ids were
    # test-best rather than val-best (final-panel C-1).
    e0 = {}
    for label, d in (("YOLOv8s-cls", "yolo_grid_e100_lr1e-2_i640"),
                     ("ResNet50 + sigmoid", "resnet_grid_lr1e-4_e50_i448"),
                     ("YOLOv8s + sigmoid", "yolobce_grid_lr1e-3_e25_i448")):
        o = json.load(open(f"results/experiments/{d}/eval_report.json"))["overall"]
        e0[label] = o
    d0 = json.load(open("results/experiments/d0_zeroshot_s42/eval_report.json"))["overall"]
    lines.append("\\midrule")
    for label in ("YOLOv8s-cls", "ResNet50 + sigmoid", "YOLOv8s + sigmoid"):
        o = e0[label]
        lines.append(f"  {label} & {o['macro_f1']:.3f} & {o['mAP']:.3f} & "
                     f"{o['primary_accuracy']:.3f} & "
                     f"{o['primary_accuracy_exact']:.3f} \\\\")
    lines.append(f"  Zero-shot VLM (no FT) & {d0['macro_f1']:.3f} & "
                 f"{d0['mAP']:.3f} & {d0['primary_accuracy']:.3f} & "
                 f"{d0['primary_accuracy_exact']:.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    Path(PAPER / "main_results.tex").write_text("\n".join(lines))
    return data


def perclass_table(data: dict) -> None:
    """Per-class precision/recall/F1 pooled over the 3 dual-channel seeds."""
    ours = data["d1_bce_nomp"]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Per-class results of the dual-channel model, pooled over "
        "3 seeds (test split; support = per-seed positive count).  Classes "
        "are grouped into semantics-cued and geometry-cued families "
        "(Section~\\ref{sec:dataset}).}",
        "\\label{tab:perclass}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Class & Support & Precision & Recall & $F_1$ \\\\",
        "\\midrule",
    ]
    our_support = {}
    for seed in SEEDS:
        r = json.load(open(f"results/experiments/d1_bce_nomp_s{seed}/eval_report.json"))
        for c, d in r["per_class"].items():
            our_support[c] = d["support"]
    for group, classes in (
            ("\\emph{Semantics-cued}", ["phone", "snack", "toy", "blocked",
                                        "eyesclosed", "away"]),
            ("\\emph{Geometry-cued}", ["slope", "tilt", "turn",
                                                  "prone", "bowed", "chinrest",
                                                  "lookup", "recline"]),
            ("\\emph{Default}", ["normal"])):
        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{4}}{{l}}{{{group}}} \\\\")
        for c in classes:
            tp = ours["tp"].get(c, 0)
            fp = ours["fp"].get(c, 0)
            fn = ours["fn"].get(c, 0)
            prec = tp / (tp + fp) if tp + fp else float("nan")
            rec = tp / (tp + fn) if tp + fn else float("nan")
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else float("nan")
            name = f"\\textit{{{c}}}" if c not in ("normal",) else c
            sup = our_support.get(c, 0)
            lines.append(f"  {name} & {sup} & {prec:.2f} & {rec:.2f} & {f1:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    Path(PAPER / "perclass.tex").write_text("\n".join(lines))


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def veto_table() -> None:
    """Three-seed veto-quality table from the per-seed eval reports."""
    rows = []
    tot_v = tot_ok = 0
    e2e = []
    for seed in SEEDS:
        base = "results/veto_eval" if seed == 42 else f"results/veto_eval_s{seed}"
        m = json.load(open(f"{base}/metrics.json"))
        o = m["overall"]
        tot_v += o["n_vetoes"]
        tot_ok += o["n_vetoes_correct"]
        e2e.append((m["end_to_end"]["vlm_primary_acc"],
                    m["end_to_end"]["veto_primary_acc"],
                    o["n_vetoes"], o["n_vetoes_correct"],
                    o["n_primary_demotions"]))
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Veto-decision quality and end-to-end effect of the "
        "geometric veto layer, evaluated per seed on the test split.  All "
        "constants ($\\tau$, per-strategy angle ceilings) are frozen on the "
        "validation split of the same seed; no test statistic is used for "
        "tuning.  VP carries Wilson 95\\% confidence intervals.}",
        "\\label{tab:veto}",
        "\\small",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Seed & vetoes & correct & VP & 95\\% CI & TPK & acc. (VLM) & "
        "acc. (+veto) \\\\",
        "\\midrule",
    ]
    for seed, (vlm, veto, nv, nok, _) in zip(SEEDS, e2e):
        vp = nok / nv if nv else float("nan")
        lo, hi = _wilson(nok, nv)
        lines.append(f"  {seed} & {nv} & {nok} & {vp:.3f} & "
                     f"[{lo:.2f}, {hi:.2f}] & {(nv - nok) / 213:.4f} & "
                     f"{vlm:.4f} & {veto:.4f} \\\\")
    vp_pooled = tot_ok / tot_v
    lo, hi = _wilson(tot_ok, tot_v)
    lines += [
        "\\midrule",
        f"  pooled & {tot_v} & {tot_ok} & {vp_pooled:.3f} & "
        f"[{lo:.2f}, {hi:.2f}] & & & \\\\",
        "\\bottomrule", "\\end{tabular}", "\\end{table*}", "",
    ]
    Path(PAPER / "veto.tex").write_text("\n".join(lines))


if __name__ == "__main__":
    data = main_table()
    perclass_table(data)
    veto_table()
    print(f"tables written to {PAPER}/")
