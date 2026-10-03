r"""Dataset class-distribution figure (Fig. 1), redesigned v3.

Design contract
---------------
Core claim: the 14 anomaly classes are frequency-balanced (100-118 frames)
while 'normal' appears at its natural base rate (700 frames).
Paradigm: ONE horizontal bar axis with a marked break.  'normal' is the
first row of the shared y axis and its bar continues through the break to
its true 700-frame length; the anomaly classes keep a 0-215 frame scale so
their balance stays readable.  This replaces the earlier split-panel
design whose isolated right panel read as a separate metric and looked
thin and empty.
Data: data/sage_eval/annotations.json (unmodified).
QA: render-time panel-alignment gate (the two broken-axis segments share
one y axis; their intentional unequal widths are recorded as an
exemption).  PNG preview and alignment records go to paper/figures/qa/;
collision audits are run separately with --json-out under qa/.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

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
C_NORMAL = "#009E73"   # bluish green - default state
C_VLM = "#0072B2"      # blue         - semantics-cued
C_CNN = "#D55E00"      # vermillion   - geometry-cued
C_EDGE = "#374151"
FIG = REPO / "paper" / "figures"
QA = FIG / "qa"

ann = json.load(open("data/sage_eval/annotations.json"))
frames = ann if isinstance(ann, list) else ann.get("frames", ann)
cnt = Counter()
for _, s in (frames.items() if isinstance(frames, dict) else enumerate(frames)):
    for c, v in s.get("labels", {}).items():
        if v:
            cnt[c] += 1

SEM = ["phone", "snack", "toy", "blocked", "eyesclosed", "away"]
GEO = ["slope", "tilt", "turn", "prone", "bowed", "chinrest", "lookup", "recline"]
sem = sorted(SEM, key=cnt.get, reverse=True)
geo = sorted(GEO, key=cnt.get, reverse=True)

# shared-y row layout: normal on top, then the two taxonomy blocks
positions = [0.0] + [1.8 + i for i in range(len(sem))] \
            + [8.8 + i for i in range(len(geo))]
labels = ["normal"] + sem + geo
vals = [cnt["normal"]] + [cnt[c] for c in sem + geo]
colors = [C_NORMAL] + [C_VLM] * len(sem) + [C_CNN] * len(geo)

fig = plt.figure(figsize=(5.40, 2.75), constrained_layout=True)
gs = fig.add_gridspec(1, 2, width_ratios=[3.9, 1.0], wspace=0.05)
ax = fig.add_subplot(gs[0])
axn = fig.add_subplot(gs[1], sharey=ax)

for ypos, v, c in zip(positions, vals, colors):
    # normal is drawn to the segment edge (215), not to 700 relying on
    # clipping: an over-wide PDF path would underlie the right segment
    # and trip the collision audit
    ax.barh(ypos, min(v, 215), height=0.62, color=c)
axn.barh(0, cnt["normal"] - 655, left=655, height=0.62, color=C_NORMAL)

ax.set_xlim(0, 215)
ax.set_xticks([0, 100, 200])
ax.set_xlabel("frames", labelpad=1.5)
axn.set_xlim(655, 719)
axn.set_xticks([700])
axn.tick_params(labelleft=False, left=False)

ax.set_yticks(positions, labels)
ax.set_ylim(16.5, -0.8)          # inverted: 'normal' row on top
ax.spines["right"].set_visible(False)
axn.spines["left"].set_visible(False)

# value labels
for ypos, v in zip(positions[1:], vals[1:]):
    ax.text(v + 4, ypos, str(v), va="center", fontsize=6.5, color="#4B5563")
axn.text(cnt["normal"] + 5, 0, str(cnt["normal"]), va="center",
         fontsize=6.5, color="#4B5563")

# taxonomy group headers (same styling as fig:complementarity)
trans = ax.get_yaxis_transform()
ax.text(0.0, 0.9, "semantics-cued", fontsize=6.5, fontweight="bold",
        color="#374151", transform=trans)
ax.text(0.0, 7.8, "geometry-cued", fontsize=6.5, fontweight="bold",
        color="#374151", transform=trans)

from matplotlib.patches import Patch  # noqa: E402
ax.legend(handles=[
    Patch(color=C_NORMAL, label="Default state (normal)"),
    Patch(color=C_VLM, label="Semantics-cued"),
    Patch(color=C_CNN, label="Geometry-cued")],
    loc="lower right", fontsize=6.5, handlelength=1.1, handleheight=0.9,
    borderaxespad=0.2, labelspacing=0.35)

# axis-break marks at the junction between the two segments
d = 0.5
kw = dict(marker=[(-1, -d), (1, 0)], markersize=12, linestyle="none",
          color=C_EDGE, clip_on=False, mew=0.8)
ax.plot([1], [0], transform=ax.transAxes, **kw)
ax.plot([1], [1], transform=ax.transAxes, **kw)
kw["marker"] = [(1, -d), (-1, 0)]
axn.plot([0], [0], transform=axn.transAxes, **kw)
axn.plot([0], [1], transform=axn.transAxes, **kw)

require_matplotlib_panel_alignment(
    fig,
    json_out=str(QA / "classdist.alignment.json"),
    overlay_svg=str(QA / "classdist.alignment.svg"),
    panel_ids={ax: "anomaly-classes", axn: "default-state"},
    exemptions=[{
        "panels": ["anomaly-classes", "default-state"],
        "checks": ["panel-width"],
        "reason": ("intentional broken-axis design: the left segment covers "
                   "0-215 frames so the anomaly-class balance stays readable; "
                   "the right segment resumes at 655 and only carries the "
                   "continuation of the normal bar to its true 700-frame "
                   "length. Shared y axis keeps heights, gutters and row "
                   "positions aligned."),
    }],
    tolerance_pt=1.5,
    gutter_tolerance_pt=1.5,
    strict=True,
)

FIG.mkdir(parents=True, exist_ok=True)
QA.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG / "classdist.pdf")
fig.savefig(QA / "classdist.png", dpi=300)
print(f"saved {FIG / 'classdist'}.pdf (+ qa/classdist.png)")
