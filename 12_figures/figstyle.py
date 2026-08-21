"""
Shared figure style for the paper. Import this first in every panel script.

    from figstyle import *          # noqa
    fig = new_fig(210, 56)
    ...
    save(fig, "fig1a_hazard_maps")

Conventions fixed here so every panel matches:
  · Arial, falling back to Helvetica / Liberation Sans / DejaVu Sans — all metric-
    compatible; the whole stack is written into the svg so it resolves anywhere
  · NO panel letters and NO axes titles: the author adds those in the layout program
  · colour language is shared across all five figures
"""
import glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
for _f in glob.glob(os.path.expanduser("~/.fonts/*.ttf")):
    try: fm.fontManager.addfont(_f)
    except Exception: pass
import matplotlib.pyplot as plt

MM = 1 / 25.4
OUT = "/data/figs/out"

# ---- page width -----------------------------------------------------------
# The six scripts are written on a 183 mm design grid. Everything they place goes through R(),
# which normalises by their own W_MM/H_MM, so the canvas and the type can be scaled together and
# the layout is reproduced exactly at another width. Set this back to 183.0 to revert.
DESIGN_W_MM = 183.0
PAGE_W_MM = 210.0            # A4 width
SCALE = PAGE_W_MM / DESIGN_W_MM
os.makedirs(OUT, exist_ok=True)

# ---- canonical sizes (mm) -------------------------------------------------
A4_W = 210.0
COL2 = 183.0          # Nature double column
COL1 = 89.0           # Nature single column

# ---- colour language (shared across figures 1-5) --------------------------
C_DEM = "#B2182B"     # demand channel / stress up
C_VRE = "#2166AC"     # generation channel / net-load down
C_MIX = "#762A83"     # mixed channel
C_CHRONIC = "#01665A" # chronic reliability   (fig 3)
C_STORM = "#DFC27D"   # storm component       (fig 3)
C_NOPOL, C_IRA, C_OBBBA = "#BABABA", "#4393C3", "#D6604D"   # fig 5 policies
# Row 2 of fig 1 answers a different question from row 3 (is a slow climate state real and
# how widespread, vs how big is a hazard day and through which channel). Reusing the
# demand-red / generation-blue channel language there made one colour mean two things.
C_FIELD = "#1F6F5C"   # field-significance accent — passes the permutation hurdle
C_NS = "#BBBBBB"      # not significant
C_GRID = "#DDDDDD"
C_TXT = "#555555"

# ---- type sizes -----------------------------------------------------------
# Four sizes, and every figure uses these and only these. Nature asks for 5-7 pt.
FS_LABEL = 6.4 * SCALE        # in-panel item labels that stand in for a title
FS_ANNOT = 5.6 * SCALE        # secondary annotation
FS_TICK = 5.6 * SCALE         # EVERY tick label, on every axis of every panel
FS_AXIS = 6.4 * SCALE         # every axis label and colourbar label
FS_LEG = 5.2 * SCALE          # every legend
FS_VAL = 5.2 * SCALE          # every in-plot number

plt.rcParams.update({
    # a stack, not a name: matplotlib writes the whole list into the svg, so the file resolves on
    # a machine that has any one of them, and all of them are metric-compatible with each other
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": FS_TICK,
    "axes.linewidth": .5 * SCALE,
    "axes.labelsize": FS_AXIS,
    "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
    "xtick.major.width": .5 * SCALE, "ytick.major.width": .5 * SCALE,
    "xtick.major.size": 2 * SCALE, "ytick.major.size": 2 * SCALE,
    "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})


def new_fig(w_mm, h_mm):
    """Figure sized in millimetres, scaled to the target page width."""
    return plt.figure(figsize=(w_mm * SCALE * MM, h_mm * SCALE * MM))


def bare(ax):
    """Strip an axes to data only."""
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def despine(ax, keep=("left", "bottom")):
    for k, sp in ax.spines.items():
        sp.set_visible(k in keep)


def save(fig, stem, svg=True, pdf=False, png_dpi=0, svg_max_mb=6.0, tight=True):
    """Save svg (preferred) + pdf + png. Falls back to png-only if the svg is huge.

    tight=True trims dead margin (right for a standalone panel). Use tight=False for an
    assembled multi-panel figure, where the canvas size IS the deliverable and trimming
    would silently change the page geometry."""
    bb = "tight" if tight else None
    paths = []
    if svg:
        p = f"{OUT}/{stem}.svg"
        fig.savefig(p, bbox_inches=bb, facecolor="white")
        mb = os.path.getsize(p) / 1e6
        if mb > svg_max_mb:
            os.remove(p); print(f"  svg dropped ({mb:.1f} MB > {svg_max_mb} MB limit)")
        else:
            paths.append(f"{p} ({mb:.2f} MB)")
    if pdf:
        p = f"{OUT}/{stem}.pdf"; fig.savefig(p, bbox_inches=bb, facecolor="white")
        paths.append(f"{p} ({os.path.getsize(p)/1e6:.2f} MB)")
    if png_dpi:                      # svg is the deliverable; a raster copy is optional
        p = f"{OUT}/{stem}.png"
        fig.savefig(p, dpi=png_dpi, bbox_inches=bb, facecolor="white")
        paths.append(f"{p} ({os.path.getsize(p)/1e6:.2f} MB)")
    print("saved:"); [print("   ", x) for x in paths]


# ---- shared data paths ----------------------------------------------------
R1 = "/data/enso/r1_causal"
LO = "/data/tell_pred/future/hist_full40"
EQ = "/data/equity_cost/analysis"
CERF = "/data/cerf_out"
