r"""Generate the paper's results figures (Figs. 3-5), redesigned.

Design contracts
----------------
fig:complementarity -- Core claim: the dual-channel VLM leads the ResNet50
    baseline on 11 of 14 classes; the baseline's residual edge concentrates
    on slope/tilt (the veto layer's target classes) and blocked.
    Paradigm: per-class dumbbells (precision | F1) -- the gap and its sign
    are readable directly, unlike 28 grouped bars; baseline-wins rows get a
    shaded band. Baseline wins: precision slope +0.11, tilt +0.26,
    blocked +0.08; F1 slope +0.16, tilt +0.04.
    Data: data/figure_data/complementarity_pooled.json (VLM values
    cross-checked against paper/tables/perclass.tex).

fig:vetoes -- Core claim: the veto layer is high-precision; the single
    false veto across 23 decisions is one slope positive (seed 42).
    VP (veto precision = correct vetoes / all vetoes) is annotated per
    seed; the false veto is labelled with its class.

fig:confusion -- Core claim: errors concentrate among the geometry-cued
    classes and normal, matching the taxonomy. Rows/columns are ordered by
    the taxonomy (normal | semantics-cued | geometry-cued) so the block
    structure is visible; cells show row-normalized percentages so colour
    and annotation carry the same unit.
    Data: data/figure_data/confusion_mainline_seed.json.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "pdf.fonttype": 42,          # editable TrueType text in PDF
    "svg.fonttype": "none",      # editable text in SVG
    "font.size": 7,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
})

# Okabe-Ito colorblind-safe palette
C_VLM = "#0072B2"      # blue       - dual-channel VLM
C_CNN = "#D55E00"      # vermillion - ResNet50 baseline / geometry-cued
C_GREY = "#9CA3AF"
FIG = REPO / "paper" / "figures"
DATA = REPO / "data" / "figure_data"

SEEDS = (42, 43, 44)
SEMANTIC = ["phone", "snack", "toy", "blocked", "eyesclosed", "away"]
GEOMETRY = ["slope", "tilt", "turn", "prone", "bowed", "chinrest",
            "lookup", "recline"]


def _save(fig, name: str, panel_alignment: bool = False, png_dpi: int = 300) -> None:
    """Export the paper PDF into paper/figures and every preview/QA
    artifact (PNG preview, alignment JSON/SVG) into paper/figures/qa, so
    the deliverable directory keeps only the files the paper includes.
    Collision audits are run separately with --json-out under qa/."""
    if panel_alignment:
        require_matplotlib_panel_alignment(
            fig,
            json_out=str(FIG / "qa" / f"{name}.alignment.json"),
            overlay_svg=str(FIG / "qa" / f"{name}.alignment.svg"),
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            strict=True,
        )
    QA = FIG / "qa"
    FIG.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(QA / f"{name}.png", dpi=png_dpi)
    print(f"saved {FIG / name}.pdf (+ qa/{name}.png)")


def fig_complementarity() -> None:
    d = json.load(open(DATA / "complementarity_pooled.json"))
    classes = SEMANTIC + GEOMETRY

    # y layout: semantics block on top, divider, geometry block below
    sem_pos = {c: -i for i, c in enumerate(SEMANTIC)}          # 0 .. -5
    geo_pos = {c: -7.5 - i for i, c in enumerate(GEOMETRY)}    # -7.5 .. -14.5
    y_div = -5.9            # divider between the two blocks
    y_sem_header = 0.55     # above the first semantics row
    y_geo_header = -6.6     # clear of divider and of the slope band
    pos = {**sem_pos, **geo_pos}

    fig, axes = plt.subplots(
        1, 2, figsize=(5.40, 2.9), sharey=True,
        gridspec_kw={"wspace": 0.08})

    for ax, metric, label in (
            (axes[0], "precision", "precision (pooled)"),
            (axes[1], "f1", "$F_1$ (pooled)")):
        vlm = [d["vlm"][metric][c] for c in classes]
        cnn = [d["resnet50"][metric][c] for c in classes]
        # shade baseline-wins rows
        for i, c in enumerate(classes):
            if cnn[i] > vlm[i]:
                ax.axhspan(pos[c] - 0.42, pos[c] + 0.42,
                           color=C_CNN, alpha=0.08, lw=0)
        for c in classes:
            p = pos[c]
            a, b = d["vlm"][metric][c], d["resnet50"][metric][c]
            if b > a:   # baseline wins -> shaded; label the gap directly
                ax.text(max(a, b) + 0.035, p, f"+{b - a:.2f}",
                        ha="left", va="center",
                        fontsize=6.2, color=C_CNN, zorder=4)
            ax.plot([a, b], [p, p], color=C_GREY, lw=0.9, zorder=1)
        ax.scatter(vlm, [pos[c] for c in classes], s=14, color=C_VLM,
                   marker="o", zorder=3, label="Dual-channel VLM (ours)")
        ax.scatter(cnn, [pos[c] for c in classes], s=16, color=C_CNN,
                   marker="s", zorder=2, label="ResNet50 baseline")
        ax.set_xlim(0, 1.04)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xlabel(label, labelpad=1.5)
        ax.tick_params(axis="y", length=0)

    axes[0].set_yticks([pos[c] for c in classes], classes)
    axes[0].set_ylim(-15.1, 0.95)

    # taxonomy headers + block divider
    axes[0].text(0.0, y_sem_header, "semantics-cued", fontsize=6.5,
                 fontweight="bold", color="#374151",
                 transform=axes[0].get_yaxis_transform())
    axes[0].text(0.0, y_geo_header, "geometry-cued", fontsize=6.5,
                 fontweight="bold", color="#374151",
                 transform=axes[0].get_yaxis_transform())
    axes[0].axhline(y_div, color="#D1D5DB", lw=0.5)

    axes[1].legend(loc="lower left", bbox_to_anchor=(0.0, 1.02),
                   ncol=2, fontsize=6.2, handletextpad=0.3,
                   columnspacing=0.9, borderaxespad=0)

    _save(fig, "complementarity", panel_alignment=True)


def fig_vetoes() -> None:
    # From paper/tables/veto.tex (seed 42: 9 vetoes, 8 correct, 1 false slope)
    rows = [(42, 8, 1), (43, 7, 0), (44, 7, 0)]
    x = np.arange(len(rows))
    ok = [r[1] for r in rows]
    bad = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(3.24, 2.15))
    ax.bar(x, ok, 0.52, label="Correct veto", color=C_VLM)
    ax.bar(x, bad, 0.52, bottom=ok, label="False veto", color=C_CNN)
    for i, (o, b) in enumerate(zip(ok, bad)):
        if o:
            ax.text(i, o / 2, str(o), ha="center", va="center",
                    fontsize=6.5, color="white")
        if b:
            ax.text(i, o + b / 2, str(b), ha="center", va="center",
                    fontsize=6.5, color="white")
        ax.text(i, o + b + 0.18, f"VP {o / (o + b):.2f}", ha="center",
                fontsize=6.5)
    ax.annotate("slope (false veto)", xy=(0.28, 8.6), xytext=(0.55, 9.45),
                fontsize=6.2, color=C_CNN, ha="left",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=C_CNN))
    ax.set_xticks(x, [f"seed {s}" for s, _, _ in rows])
    ax.set_ylabel("veto decisions (test)")
    ax.set_ylim(0, 10.6)
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    # ylim tightened to remove dead space above the bars
    ax.legend(loc="upper right", fontsize=6.2, handlelength=1.1,
              handleheight=0.9, borderaxespad=0.2)
    _save(fig, "vetoes")


def fig_confusion() -> None:
    d = json.load(open(DATA / "confusion_mainline_seed.json"))
    old = d["classes"]
    M = np.array(d["matrix"], dtype=float)
    new_order = ["normal"] + SEMANTIC + GEOMETRY
    idx = [old.index(c) for c in new_order]
    M = M[np.ix_(idx, idx)]

    rates = M / M.sum(axis=1, keepdims=True)
    n = len(new_order)

    fig, ax = plt.subplots(figsize=(5.40, 4.35), constrained_layout=True)
    # pcolormesh keeps the heatmap fully vector in the PDF (imshow embeds
    # a raster image at ~100 dpi effective)
    edges = np.arange(n + 1) - 0.5
    X, Y = np.meshgrid(edges, edges)
    im = ax.pcolormesh(X, Y, rates, cmap="Blues", vmin=0, vmax=1,
                       shading="flat")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    for i in range(n):
        for j in range(n):
            if M[i, j]:
                v = rates[i, j]
                ax.text(j, i, f"{round(v * 100)}%", ha="center", va="center",
                        fontsize=5.4,
                        color="white" if v > 0.55 else "#1F2937")
    # taxonomy block separators: normal | semantics (1-6) | geometry (7-14)
    # (per-cell white gridlines removed: adjacent same-colour cell labels
    # merge into one PDF text run and the line would cross the merged box)
    for b in (0.5, 6.5):
        ax.axhline(b, color="white", lw=2.2)
        ax.axvline(b, color="white", lw=2.2)
    ax.set_xticks(range(n), new_order, rotation=90, fontsize=6.5)
    ax.set_yticks(range(n), new_order)
    ax.set_xlabel("predicted label", labelpad=2)
    ax.set_ylabel("ground-truth label", labelpad=2)
    for s in ax.spines.values():
        s.set_visible(False)
    cbar = fig.colorbar(im, fraction=0.046, pad=0.03)
    if getattr(cbar, "solids", None) is not None:
        cbar.solids.set_rasterized(False)  # keep the gradient vector too
        cbar.solids.set_edgecolor("face")  # hide antialiasing seams
    cbar.set_label("row-normalized rate", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5)
    cbar.outline.set_linewidth(0.6)
    # group labels above the columns
    trans = ax.get_xaxis_transform()
    ax.text(0.0, 1.012, "normal", ha="center", fontsize=6.2,
            color="#374151", transform=trans)
    ax.text(3.5, 1.012, "semantics-cued", ha="center", fontsize=6.2,
            color="#374151", transform=trans)
    ax.text(11.0, 1.012, "geometry-cued", ha="center", fontsize=6.2,
            color="#374151", transform=trans)
    _save(fig, "confusion")


def fig_architecture() -> None:
    """fig:architecture -- detect-then-veto pipeline schematic (Python port
    of the former inline-TikZ figure so every paper figure shares one
    matplotlib pipeline).

    Canvas is 136.5 x 72 mm, full-bleed (set_position), and the paper
    includes it at the text width (~388 pt in elsarticle review 12pt), so
    the effective scale is ~1:1 and 7.2 pt text keeps mathtext subscripts
    at ~5.05 pt, above the 5 pt glyph floor.

    QA note: audit_figure_collisions reports one text-text overlap inside
    the T-star-c symbol: mathtext emits the superscript/subscript pair as
    separate PDF spans, so the script glyph box overlaps its base. Verified
    clean at 300 dpi -- a typographic subscript, not a collision.
    """
    from matplotlib.patches import FancyBboxPatch

    INK = "#1F2937"
    EDGE = "#374151"
    C_B1 = "#3D4DB7"
    C_B2 = "#C05F17"
    fig, ax = plt.subplots(figsize=(136.5 / 25.4, 72 / 25.4))
    ax.set_xlim(0, 136.5)
    ax.set_ylim(0, 72)
    ax.set_aspect("equal")
    ax.set_position((0, 0, 1, 1))
    ax.axis("off")

    def band(x, y, w, h, fc, ec):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=2",
            fc=fc, ec=ec, lw=0.9, ls=(0, (4, 3)), zorder=1))

    def box(x, y, w, h, fc, dashed=False):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6",
            fc=fc, ec=EDGE, lw=0.7 if dashed else 0.8,
            ls=(0, (3, 2)) if dashed else "-", zorder=3))

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=4,
                    arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=1.0,
                                    mutation_scale=8))

    def text_at(x, y, t, fs=7.2, bold=False, color=INK, ls=1.35,
                va="center"):
        ax.text(x, y, t, ha="center", va=va, fontsize=fs,
                fontweight="bold" if bold else "normal", color=color,
                zorder=5, linespacing=ls)

    box(1.25, 20, 24, 22, "#DCE4F7")     # dual-channel VLM
    box(28.75, 20, 24, 22, "#EDF0FA")    # anomaly claims
    box(56.25, 20, 24, 22, "#F7DFC8")    # checkers
    box(83.75, 20, 24, 22, "#FAEBDD")    # authority gate
    box(111.25, 20, 24, 22, "#DCEFDD")   # corrected claims
    box(56.25, 46, 24, 12, "white", dashed=True)  # services feeder
    band(0.25, 17, 54, 28, "#F0F3FB", "#A9B2DE")
    band(55.25, 15, 53.5, 46, "#FDF3EA", "#E8B48A")

    ax.text(13.25, 41.3, "dual-channel\nVLM", ha="center", va="top",
            fontsize=7.2, fontweight="bold", color=INK, linespacing=1.3,
            zorder=5)
    text_at(13.25, 28.5,
            "primary token $\\hat{y}$ +\n15-way sigmoid\n"
            "$p(\\cdot)$")
    text_at(40.75, 31,
            "anomaly claims\n$\\hat{Y} = \\{c :$\n"
            "$p(c) \\geq 0.5\\}$")
    ax.text(68.25, 42.0, "claim-\nconditioned\ncheckers", ha="center",
            va="top", fontsize=7.2, fontweight="bold", color=INK,
            linespacing=1.3, zorder=5)
    text_at(68.25, 27.3, "verdict $v_c$,\nevidence $g_c$")
    text_at(95.75, 40.0, "authority gate", bold=True)
    text_at(95.75, 33.2,
            "$\\tilde{e}_c \\geq \\tau$  or\n"
            "$g_c \\geq -T^*_c$")
    ax.text(95.75, 24.9, "(ceilings frozen\nto $\\delta$)", ha="center",
            va="center", fontsize=6.6, color=INK, linespacing=1.3,
            zorder=5)
    text_at(123.25, 31,
            "corrected claims\n$\\hat{Y}' = \\hat{Y} "
            "\\setminus V$\nprimary fallback")
    text_at(68.25, 52, "person / pose / face /\ndepth services", fs=6.6)

    arrow(25.25, 31, 28.75, 31)
    arrow(52.75, 31, 56.25, 31)
    arrow(80.25, 31, 83.75, 31)
    arrow(107.75, 31, 111.25, 31)
    arrow(68.25, 46, 68.25, 42)

    ax.text(0.25, 46.2, "Stage 1 – detection authority\n(adds claims)",
            ha="left", va="bottom", fontsize=6.6, fontweight="bold",
            color=C_B1, linespacing=1.3, zorder=2)
    ax.text(55.25, 62.2,
            "Stage 2 – veto-only authority\n(removes claims, never adds)",
            ha="left", va="bottom", fontsize=6.6, fontweight="bold",
            color=C_B2, linespacing=1.3, zorder=2)

    dash = (0, (3, 2))
    ax.plot([13.25, 13.25], [20, 10], color=EDGE, lw=0.9, ls=dash, zorder=2)
    ax.plot([13.25, 123.25], [10, 10], color=EDGE, lw=0.9, ls=dash,
            zorder=2)
    ax.annotate("", xy=(123.25, 20), xytext=(123.25, 10), zorder=2,
                arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=0.9,
                                ls=dash, mutation_scale=8))
    ax.text(68.25, 6.6, "output stands wherever no veto fires",
            ha="center", va="center", fontsize=6.8, color=INK, zorder=2)

    _save(fig, "architecture")


if __name__ == "__main__":
    fig_complementarity()
    fig_vetoes()
    fig_confusion()
    fig_architecture()
    print("figures written")
