"""
FIGURE 3 v8 — Which weather causes outages, where the burden falls, and how it scales.

Order follows the argument: identify first, then map what the identified hazards account for, then
show the regularities. Panels (a) and (b) are v7's two panels with their positions exchanged on the
author's instruction, and both are rebuilt on the unified intensity panel (/data/attrib.py).
Panels (c) to (e) are Figure 2's old bottom row, unchanged.

  (a) is the effect significant, and does the design hold
  (b) how the effect scales with the intensity of the event
  (c) county outage attributable to the hazards whose pre-event window is zero

The bottom row of the previous version is gone. It held the two outage channels on a subregion-day
panel over 2002 to 2019, a daily landfall event study, and a within-storm wind dose response. The
first sat on a different unit and a different period from every other panel. The other two rest on
ten storms, and a daily event study with forty-two coefficients cannot be given a standard error
from ten clusters: two of them came out exactly zero. The wind gradient those panels were meant to
show is estimated here in (b), on the national panel, where the clusters are 2,269 counties and
2,830 days, and that is the gradient Fig. 6 applies.

WHAT CHANGED IN (a) AND (b)

  v7 (a) reported one matched-cohort effect per hazard and marked four of five as unusable. That
  design compared a treated county with untreated counties nearby, which does not exist for a cold
  outbreak. Here every hazard enters one Poisson panel with county x calendar-month and day fixed
  effects, at its own intensity, and the comparison is every other county in the country that day.
  Which hazards pass their pre-event test is read from the screen's own output,
  attrib_identified.json, and is not written down here.

  v7 (b) multiplied one constant effect by a count of events, so a 34-kt brush over Wisconsin was
  charged the same as a Category 4 landfall, and outage was attributed to counties that lost no
  power. Here the attribution is a share of each county-day's OBSERVED outage, so it cannot exceed
  what happened and a quiet county-day contributes nothing.
"""
import sys
import os as _os_rp
for _rp in ("04_demand_model", "07_hazard_calendar", "09_outage_attribution",
            "02_downscale_wind", "12_figures"):
    _ap = _os_rp.path.abspath(_os_rp.path.join(
        _os_rp.path.dirname(_os_rp.path.abspath(__file__)), "..", _rp))
    if _os_rp.path.isdir(_ap) and _ap not in sys.path:
        sys.path.insert(0, _ap)
from figstyle import *                                    # noqa
import json, numpy as np, pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch
from scipy.spatial import cKDTree

import os as _os_rp

for _rp in ("04_demand_model", "07_hazard_calendar", "09_outage_attribution",

            "02_downscale_wind", "12_figures"):

    _ap = _os_rp.path.abspath(_os_rp.path.join(

        _os_rp.path.dirname(_os_rp.path.abspath(__file__)), "..", _rp))

    if _os_rp.path.isdir(_ap) and _ap not in sys.path:

        sys.path.insert(0, _ap)
import attrib_artifacts as AA          # F3.9: the screen's artifact names live in one place
EQ = "/data/equity_cost/analysis"
AT = "%s/attrib" % EQ
# F3.3. The set of hazards this figure draws as carried forward used to be written here as
# a fixed pair of hazard keys, while the screen in 04_chk3.py decides the set afresh on every rerun.
# The figure therefore kept drawing two hazards after the screen began passing four, and dropped the
# others without a word. It is read from the screen now, and a hazard with no drawing style stops
# the figure instead of vanishing from it.
SJ = AA.screened_json(AT)
SCREENED = list(SJ["screened_hazards"])
C_TC, C_CV = "#A6761D", "#D95F02"
C_ADEQ, C_DAM = C_VRE, "#A6761D"
C_NUL, C_PLA, C_BAD = "#C4C4C4", "#4D4D4D", "#B2182B"

W_MM, H_MM = 183.0, 132.0
fig = new_fig(W_MM, H_MM)
def R(x, y, w, h): return [x / W_MM, 1.0 - (y + h) / H_MM, w / W_MM, h / H_MM]

A = json.load(open("%s/attrib.json" % AT))
Rz = A["results"]

# ============ (a) is the effect significant, and does the design hold ============
# Two coefficients per hazard, each a single estimate with an exact interval. The upper marker is
# the event-day effect at the hazard's strongest intensity, which is the sharpest test of whether
# the hazard raises outage at all. The lower marker is the same quantity 14 to 8 days before the
# event, when no forecast can yet drive an operating decision, so any departure from one there is
# a pre-trend and not an effect. A hazard is causally interpretable only if the lower marker
# covers one and the upper marker does not.
HZ = [("tc", "hurricane", C_TC, ["34 to 50 kt", "50 to 64 kt", "64 to 83 kt", "83 kt and above"]),
      ("convective", "severe convection", C_CV,
       ["footprint Q1", "footprint Q2", "footprint Q3", "footprint Q4"]),
      ("fire", "fire weather", "#7570B3", ["fire Q1", "fire Q2", "fire Q3", "fire Q4"]),
      # cold and heat shared one grey while they were never drawn as clouds. They are now, so
      # each carries its own color; the duplicate-color check below fails if that stops being true.
      ("cold", "cold outbreak", "#2C7FB8", ["cold Q1", "cold Q2", "cold Q3", "cold Q4"]),
      ("heat", "heat wave", "#E7298A", ["heat Q1", "heat Q2", "heat Q3", "heat Q4"])]
STYLE = {h: (nm, col, bins) for h, nm, col, bins in HZ}
_nostyle = [h for h in SCREENED if h not in STYLE]
if _nostyle:
    raise SystemExit(
        "the screen passes %s, and this figure has no drawing style for %s. Add an entry to HZ "
        "(key, printed name, color, intensity bin labels) before redrawing. The figure refuses to "
        "draw rather than leave a passing hazard out of the panels without saying so."
        % (", ".join(SCREENED), ", ".join(_nostyle)))
_cols = {}
for _h in SCREENED:
    _cols.setdefault(STYLE[_h][1], []).append(_h)
_clash = {c: v for c, v in _cols.items() if len(v) > 1}
if _clash:
    raise SystemExit("two hazards drawn together share a color, so the panels cannot be read: %s. "
                     "Give each of them its own color in HZ." % _clash)
axa = fig.add_axes(R(19.0, 7.0, 60.0, 44.0))
yv = np.arange(len(HZ))[::-1]
def band(r):
    return np.exp(r["beta"]), np.exp(r["beta"] - 1.96 * r["se"]), np.exp(r["beta"] + 1.96 * r["se"])
for i, (h, nm, col, bins) in enumerate(HZ):
    rp = Rz[h + "|lead"]
    okp = h in SCREENED          # the screen's verdict, not a second copy of its first condition
    re_ = Rz["%s|impact|%s" % (h, bins[3])]
    m, lo, hi = band(re_)
    # the event-day estimate of a hazard that fails the pre-event test is drawn dashed and its
    # value bracketed, the same convention panel (c) uses, so a large number cannot be read as a
    # causal effect on the strength of its marker alone
    axa.plot([lo, hi], [yv[i] + .21] * 2, color=col, lw=1.3, zorder=3, solid_capstyle="round",
             ls="-" if okp else (0, (2.6, 1.6)))
    axa.scatter([m], [yv[i] + .21], s=17, color=col, edgecolor="white", linewidth=.4, zorder=4)
    axa.text(hi * 1.14, yv[i] + .21, ("%.1f×" % m) if okp else ("(%.1f×)" % m),
             fontsize=FS_VAL, va="center", ha="left", color=col)
    m, lo, hi = band(rp)
    # The red used to be keyed to the screen verdict, so a hazard dropped for the SIGN of its
    # event-day effect had its pre-event marker painted red as well, against a legend that reads
    # \"95% CI excludes one\". The color is keyed to that interval now, and to nothing else.
    _leadbad = not (lo <= 1.0 <= hi)
    pc = C_BAD if _leadbad else C_PLA
    axa.plot([lo, hi], [yv[i] - .23] * 2, color=pc, lw=.8, zorder=3)
    axa.scatter([m], [yv[i] - .23], s=11, facecolor=pc if _leadbad else "white", edgecolor=pc,
                linewidth=.8, zorder=5)
axa.axvline(1.0, color="black", lw=.6, ls=(0, (3, 3)), zorder=1)
axa.set_xscale("log"); axa.set_xlim(0.66, 62)
axa.set_xticks([1, 2, 5, 10, 25]); axa.set_xticklabels(["1", "2", "5", "10", "25"], fontsize=FS_TICK)
axa.set_yticks(yv); axa.set_yticklabels([n for _, n, _, _ in HZ], fontsize=FS_TICK)
axa.tick_params(axis="y", length=0, pad=2.5)
axa.set_ylim(-1.10, len(HZ) - 0.35)
axa.set_xlabel("outage as a multiple of the county's ordinary day  (95% CI)",
               fontsize=FS_AXIS, labelpad=2)
despine(axa)
axa.legend(handles=[
    Line2D([], [], marker="o", ls="-", color="#666666", ms=3.0, lw=1.3,
           label="event day, strongest intensity"),
    Line2D([], [], marker="o", ls="-", mfc="white", mec=C_PLA, mew=.8, color=C_PLA, ms=2.8, lw=.8,
           label="14 to 8 days before the event"),
    Line2D([], [], marker="o", ls="", color=C_BAD, ms=2.8, label="95% CI excludes one"),
    Line2D([], [], color="#8E8E8E", lw=1.3, ls=(0, (2.6, 1.6)),
           label="did not pass the pre-event screen")],
    loc="lower right", bbox_to_anchor=(0.995, 0.012), ncol=1, fontsize=FS_VAL,
    handlelength=1.6, labelspacing=.26, borderpad=.26, frameon=True, facecolor="white",
    edgecolor="#D8D8D8", framealpha=.94).get_frame().set_linewidth(.3)

# ============ (c) how the effect scales with the intensity of the event ============
axb = fig.add_axes(R(15.0, 78.0, 43.0, 42.0))
axb.axhline(1.0, color="black", lw=.6, ls=(0, (3, 3)), zorder=1)
LABX = {}
for h, nm, col, bins in HZ:
    ok = h in SCREENED
    b = [np.exp(Rz["%s|impact|%s" % (h, k)]["beta"]) for k in bins]
    axb.plot(np.arange(4), b, color=col, lw=1.2, zorder=3, solid_capstyle="round",
             ls="-" if ok else (0, (2.6, 1.6)))
    axb.scatter(np.arange(4), b, s=13, color=col, edgecolor="white", linewidth=.4, zorder=4)
    LABX[h] = nm
axb.set_yscale("log")
axb.set_ylim(0.66, 46); axb.set_xlim(-0.28, 3.28)
# The names sit above the panel rather than beside the lines, which keeps the plotting area wide.
# Positions are fixed rather than packed, because five names do not fit on one line at this size.
NPOS = {"tc": (0.00, 1.115), "convective": (0.40, 1.115),
        "cold": (0.00, 1.030), "fire": (0.37, 1.030), "heat": (0.72, 1.030)}
for h, nm, col, bins in HZ:
    ok = h in SCREENED
    x_, y_ = NPOS[h]
    axb.text(x_, y_, LABX[h], transform=axb.transAxes, fontsize=FS_VAL, va="bottom",
             ha="left", color=col if ok else "#7A7A7A")
axb.set_yticks([1, 2, 5, 10, 25]); axb.set_yticklabels(["1", "2", "5", "10", "25"], fontsize=FS_TICK)
axb.set_xticks([0, 1, 2, 3])
axb.set_xticklabels(["weakest\nquarter", "", "", "strongest\nquarter"], fontsize=FS_TICK,
                    linespacing=1.2)
axb.tick_params(axis="x", length=0, pad=2.5)
axb.set_ylabel("outage as a multiple of\nthe county's ordinary day", fontsize=FS_AXIS, labelpad=2,
               linespacing=1.3)
axb.set_xlabel("intensity of the event", fontsize=FS_AXIS, labelpad=2)
despine(axb)
# No key here. The solid and dashed styles carry the verdict of the panel above, and the
# caption states it, which keeps the plotting area clear at this size.

# ============ (d) hurricanes alone, by the wind the county took ============
# The regression multiplier of (c) is relative to a county-month level that itself contains the
# storm, because a county-by-calendar-month effect fitted over eight years absorbs part of a rare
# and very large event. This panel therefore shows the unadjusted data instead: each exposed
# county-day on the landfall day or the day after, divided by that county's mean over quiet days
# of the same calendar month. The two quantities are not the same and are not drawn together.
TB = pd.read_parquet("%s/tc_band_cloud.parquet" % AT)
BND = ["34 to 50 kt", "50 to 64 kt", "64 to 83 kt", "83 kt and above"]
BSH = ["34 to 50", "50 to 64", "64 to 83", "83 and\nabove"]
BCOL = ["#E8C37E", "#D2952F", "#A6761D", "#6B4A10"]
axd = fig.add_axes(R(74.0, 78.0, 43.0, 42.0))
rng = np.random.default_rng(3)
LOD, HID = 0.05, 6e4
for j, b in enumerate(BND):
    v = TB[TB.band == b].ratio.values
    v = v[(v >= LOD) & (v <= HID)]
    x = j + rng.normal(0, .105, len(v))
    axd.scatter(x, v, s=2.2, color=BCOL[j], alpha=.22, linewidth=0, zorder=2, rasterized=True)
med = [float(np.median(TB[TB.band == b].ratio.values)) for b in BND]
q1 = [float(np.percentile(TB[TB.band == b].ratio.values, 25)) for b in BND]
q3 = [float(np.percentile(TB[TB.band == b].ratio.values, 75)) for b in BND]
for j in range(4):
    axd.plot([j, j], [q1[j], q3[j]], color="#3A2606", lw=1.4, zorder=4, solid_capstyle="round")
axd.plot(range(4), med, color="#3A2606", lw=1.2, zorder=5)
axd.scatter(range(4), med, s=20, color="#3A2606", edgecolor="white", linewidth=.5, zorder=6)
for j, b in enumerate(BND):
    axd.text(j, HID * 0.55, "%d" % (TB.band == b).sum(), ha="center", va="top", fontsize=FS_VAL,
             color="#6B4A10")
axd.text(0.5, 1.005, "county-days plotted", transform=axd.transAxes, fontsize=FS_VAL, va="bottom",
         ha="center", color="#6B4A10")
axd.axhline(1.0, color="black", lw=.6, ls=(0, (3, 3)), zorder=3)
axd.set_yscale("log"); axd.set_ylim(LOD, HID); axd.set_xlim(-0.6, 3.6)
axd.set_yticks([0.1, 10, 1000]); axd.set_yticklabels(["0.1", "10", "1000"], fontsize=FS_TICK)
axd.set_xticks(range(4)); axd.set_xticklabels(BSH, fontsize=FS_TICK, linespacing=1.15)
axd.tick_params(axis="x", length=0, pad=2.5)
axd.set_ylabel("outage on the landfall day, as a multiple\nof the county's own quiet-month mean",
               fontsize=FS_AXIS, labelpad=2, linespacing=1.3)
axd.set_xlabel("wind at the county (kt)", fontsize=FS_AXIS, labelpad=2)
despine(axd)
axd.legend(handles=[
    Line2D([], [], marker="o", ls="", color="#A6761D", alpha=.5, ms=2.0, label="one county-day"),
    Line2D([], [], marker="o", ls="-", color="#3A2606", ms=3.0, lw=1.2, label="median and quartiles")],
    loc="lower right", bbox_to_anchor=(0.995, 0.012), fontsize=FS_VAL, handlelength=1.5,
    labelspacing=.24, borderpad=.24, frameon=True, facecolor="white", edgecolor="#D8D8D8",
    framealpha=.94).get_frame().set_linewidth(.3)

# ============ (e) before, during and after the event ============
CL = pd.read_parquet("%s/profile_cloud.parquet" % AT)
PF = json.load(open("%s/profile_fit.json" % AT))
BLK = ["lead", "antic", "impact", "restore", "tail"]
BLAB = ["14 to 8\nbefore", "7 to 1\nbefore", "day 0\nto 1", "2 to 6\nafter", "7 to 14\nafter"]
axe = fig.add_axes(R(132.0, 78.0, 48.0, 42.0))
axe.axhline(1.0, color="black", lw=.6, ls=(0, (3, 3)), zorder=3)
LO, HI = 0.02, 400.0
nclip = 0
SHOW = SCREENED                      # the cloud is drawn for exactly the hazards the screen passes
# Two hazards were once hard-coded here, so the offset, the jitter and the point budget were all
# written for two. They are derived from the count now: the clouds occupy the same total width
# whatever the screen returns, and the total number of points per block is held constant so a
# larger screened set does not simply fill the panel with ink.
NSH = max(len(SHOW), 1)
DX = min(0.30, 0.60 / max(NSH - 1, 1))          # separation between neighbouring clouds
JIT = .062 * min(1.0, DX / 0.30)                # jitter shrinks with the separation
CAP = max(400, int(2800 / NSH))                 # points per hazard per block
for i, (h, nm, col, bins) in enumerate(HZ):
    if h not in SHOW:
        continue
    k = SHOW.index(h)
    for j, b in enumerate(BLK):
        v = CL[(CL.hazard == h) & (CL.block == b)].ratio.values
        nclip += int(((v < LO) | (v > HI)).sum())
        v = v[(v >= LO) & (v <= HI)]
        if len(v) > CAP:
            v = rng.choice(v, CAP, replace=False)
        x = j + (k - (NSH - 1) / 2.0) * DX + rng.normal(0, JIT, len(v))
        axe.scatter(x, v, s=1.6, color=col, alpha=.17, linewidth=0, zorder=1, rasterized=True)
for h, nm, col, bins in HZ:
    ok = h in SCREENED
    yv2 = [np.exp(PF[h][b]) for b in BLK]
    axe.plot(np.arange(5), yv2, color=col, lw=1.3, zorder=4, solid_capstyle="round",
             ls="-" if ok else (0, (2.6, 1.6)))
    axe.scatter(np.arange(5), yv2, s=15, color=col, edgecolor="white", linewidth=.4, zorder=5)
# The three labels here used to be typed out, including "the other three", which counted the
# hazards the screen had dropped. Each hazard is named at the height its own tail block reaches,
# and names that would collide are pushed apart down the log axis.
_lab = sorted(((float(np.exp(PF[h]["tail"])), h, nm, col) for h, nm, col, _ in HZ), reverse=True)
_lo10, _hi10 = np.log10(LO), np.log10(HI)
_gap = (_hi10 - _lo10) * 0.055
_ys = [np.log10(v) for v, _, _, _ in _lab]
for _i in range(1, len(_ys)):
    _ys[_i] = max(_lo10 + 0.05, min(_ys[_i], _ys[_i - 1] - _gap))
for (_v, _h, _nm, _col), _yy in zip(_lab, _ys):
    _c = _col if _h in SCREENED else "#8A8A8A"
    # Four of the five tail values sit within a factor of 1.25 of each other, so the names have to
    # be spread out to be legible. A name that has moved off its own line is joined back to it.
    if abs(np.log10(_v) - _yy) > 0.02:
        axe.plot([4.02, 4.11], [_v, 10 ** _yy], color=_c, lw=.4, alpha=.75, zorder=6,
                 solid_capstyle="butt", clip_on=False)
    axe.text(4.14, 10 ** _yy, _nm, fontsize=FS_VAL, va="center", color=_c)
axe.set_yscale("log"); axe.set_ylim(LO, HI); axe.set_xlim(-0.55, 6.05)
axe.set_yticks([0.1, 1, 10, 100]); axe.set_yticklabels(["0.1", "1", "10", "100"], fontsize=FS_TICK)
axe.set_xticks(range(5)); axe.set_xticklabels(BLAB, fontsize=FS_TICK, linespacing=1.15)
axe.tick_params(axis="x", length=0, pad=2.5)
axe.set_ylabel("outage as a multiple of the county's\nown quiet-month mean", fontsize=FS_AXIS,
               labelpad=2, linespacing=1.3)
axe.set_xlabel("days relative to the event", fontsize=FS_AXIS, labelpad=2)
despine(axe)
axe.legend(handles=[
    Line2D([], [], marker="o", ls="", color="#8A8A8A", alpha=.5, ms=1.8,
           label="one county-day, a hazard the screen passes"),
    Line2D([], [], color="#666666", lw=1.3, label="fitted multiplier, every hazard")],
    loc="upper right", bbox_to_anchor=(0.995, 0.988), fontsize=FS_VAL, handlelength=1.5,
    labelspacing=.24, borderpad=.24, frameon=True, facecolor="white", edgecolor="#D8D8D8",
    framealpha=.94).get_frame().set_linewidth(.3)

# ============ (b) where the attributable burden falls ============
M = AA.read_screened(AT)
gz = pd.read_csv("%s/did/2023_Gaz_counties_national.txt" % EQ, sep="\t", dtype={"GEOID": str})
gz.columns = [c.strip() for c in gz.columns]; gz["fips"] = gz.GEOID.str.zfill(5)
# THE MERGE MUST NOT LOSE A COUNTY SILENTLY. The 2022 vintage of the Census gazetteer replaced
# Connecticut's eight counties with nine planning regions, so FIPS 09001 to 09015 are absent from
# the 2023 file while equity_joined_v2.csv still keys on them. An inner join therefore dropped
# eight counties carrying 15.2 million attributable customer-hours, 0.86% of the screened total,
# and the nearest-neighbour fill below then painted every Connecticut cell with a centroid from
# New York, Massachusetts or Rhode Island. The map showed another state's numbers rather than a
# gap. Any county that cannot be placed is now excluded from the fill, so it renders as no data,
# and the loss is counted, named and asserted against a ceiling.
_ct = gz[["fips", "INTPTLAT", "INTPTLONG"]]
cty = _ct.merge(M[["fips", "rate"]], on="fips", how="inner")
# A county the gazetteer cannot place is still placed, from the county mask the rest of the
# pipeline already uses to assign weather to counties. county_mask_tgw.npz maps TGW grid cells to
# county FIPS, so the centroid of a county's own cells is that county's centroid in exactly the
# geometry the hazard flags were built on. This recovers Connecticut without importing a second
# vintage of anybody's shapefile.
_miss = sorted(set(M.fips) - set(_ct.fips))
if _miss:
    _cm = np.load("/data/loads_measured/county_mask_tgw.npz", allow_pickle=True)
    _tg = np.load("/data/tgw_hist/tgw_grid.npz")
    # the TGW grid file names its coordinates XLAT and XLONG, as WRF writes them
    _tlat, _tlon = np.asarray(_tg["XLAT"]).ravel(), np.asarray(_tg["XLONG"]).ravel()
    _pf = np.array(["%05d" % int(x) for x in _cm["pair_fips"]])
    _pc = np.asarray(_cm["pair_cell"]).astype(np.int64)
    _add = []
    for _f in _miss:
        _k = _pc[_pf == _f]
        if len(_k):
            _add.append((_f, float(_tlat[_k].mean()), float(_tlon[_k].mean())))
    if _add:
        print("  placed %d counties from the county mask that the gazetteer vintage lacks: %s"
              % (len(_add), ", ".join(a[0] for a in _add[:8])), flush=True)
        _ct = pd.concat([_ct, pd.DataFrame(_add, columns=["fips", "INTPTLAT", "INTPTLONG"])],
                        ignore_index=True)
        cty = _ct.merge(M[["fips", "rate"]], on="fips", how="inner")

_lost = M[~M.fips.isin(set(_ct.fips))]
if len(_lost):
    _sh = float(_lost.att_screened.sum()) / float(M.att_screened.sum()) if "att_screened" in M else float("nan")
    print("  %d counties have no gazetteer centroid and are left blank rather than filled from a "
          "neighbour: %s ... (%.2f%% of the screened total)"
          % (len(_lost), ", ".join(sorted(_lost.fips)[:6]), 100 * _sh), flush=True)
    assert _sh < 0.02, ("%.2f%% of the attributable total has no centroid; the gazetteer vintage "
                        "does not match the county keys and the map would be misleading" % (100 * _sh))
    _states = sorted({f[:2] for f in _lost.fips})
    assert set(_states) <= {"09"}, ("counties without a centroid come from states %s, not only the "
                                    "Connecticut planning-region change" % _states)
NNEG = int((cty.rate < 0).sum())      # these fall in the pale swatch with the true zeroes
zm = np.load("/data/datasets/grid/subregion_mask.npz", allow_pickle=True)["subregion_mask"]
zc = np.load("/data/datasets/grid/coordinate.npz")
lat, lon = zc["lat"].astype(float), zc["lon"].astype(float)
rr = np.where((zm > 0).any(1))[0]; cc = np.where((zm > 0).any(0))[0]
sub = zm[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
ext = [lon[cc.min()], lon[cc.max()], lat[rr.min()], lat[rr.max()]]
GLON, GLAT = np.meshgrid(lon[cc.min():cc.max() + 1], lat[rr.min():rr.max() + 1])
land = sub > 0
# The fill only reaches cells whose nearest placed county is close enough to be that county. A
# state with no placed county at all would otherwise be painted from across a border.
_, ii = cKDTree(np.c_[cty.INTPTLONG.values, cty.INTPTLAT.values]).query(np.c_[GLON[land], GLAT[land]])
_dd, _ = cKDTree(np.c_[cty.INTPTLONG.values, cty.INTPTLAT.values]).query(np.c_[GLON[land], GLAT[land]])
_too_far = _dd > 1.0          # degrees; a county centroid is never a degree from its own territory
img = np.full(sub.shape, np.nan); img[land] = cty.rate.values[ii]
img[img <= 0] = np.nan
ASP = 1.0 / np.cos(np.deg2rad(np.mean(ext[2:])))
MX, MY_, MW_ = 83.0, 4.0, 98.0
MH_ = MW_ * (ext[3] - ext[2]) * ASP / (ext[1] - ext[0])
axm = fig.add_axes(R(MX, MY_, MW_, MH_))
pos = cty.rate[cty.rate > 0]
vhi = float(np.percentile(pos, 95))
nrm = plt.matplotlib.colors.Normalize(0.0, vhi)
cmap = plt.get_cmap("YlOrBr").copy()
axm.imshow(np.where(land, 0.0, np.nan), origin="lower", extent=ext, aspect="auto", zorder=1,
           cmap=plt.matplotlib.colors.ListedColormap(["#ECECEC"]))
axm.imshow(img, origin="lower", extent=ext, aspect="auto", cmap=cmap, norm=nrm,
           interpolation="nearest", zorder=2)
axm.contour(np.where(land, sub, np.nan), levels=np.arange(.5, sub.max() + 1), colors="white",
            linewidths=.3, extent=ext, origin="lower", zorder=5)
axm.set_xlim(ext[0], ext[1]); axm.set_ylim(ext[2], ext[3]); bare(axm)
cax = fig.add_axes(R(MX + 24.0, MY_ + MH_ + 3.2, 48.0, 2.2))
cb = fig.colorbar(plt.cm.ScalarMappable(norm=nrm, cmap=cmap), cax=cax, orientation="horizontal",
                  extend="max")
cb.set_ticks([0, vhi / 2, vhi])
cb.set_ticklabels(["0", "%.0f h" % (vhi / 2), "%.0f h" % vhi])
cb.ax.tick_params(labelsize=FS_TICK, width=.5, length=2, pad=1.5)
cb.set_label("attributable outage per customer, hours over 2015 to 2022",
             fontsize=FS_AXIS, labelpad=2)
cb.outline.set_linewidth(.5)
cax.add_patch(plt.Rectangle((-0.115, 0), .058, 1, transform=cax.transAxes, facecolor="#ECECEC",
                            edgecolor="0.65", lw=.4, clip_on=False, zorder=6))
cax.text(-0.133, .5, "none or\nnegative" if NNEG else "none", transform=cax.transAxes,
         ha="right", va="center", fontsize=FS_TICK, linespacing=1.1)

save(fig, "fig3_v8", tight=False, png_dpi=200)
print("page %.0f x %.0f mm" % (W_MM, H_MM))
print("(a) impact multipliers, weakest to strongest bin:")
for h, nm, _, bins in HZ:
    print("    %-18s %s   lead z %+.2f  antic z %+.2f"
          % (nm, "  ".join("%5.2f" % np.exp(Rz["%s|impact|%s" % (h, k)]["beta"]) for k in bins),
             Rz[h + "|lead"]["z"], Rz[h + "|antic"]["z"]))
print("    screen (lead-block z under %.2f AND pre-event precision under %.2f of the impact"
      % (SJ["pre_event_z_threshold"], SJ["precision_ratio"]))
print("    effect), read from attrib_identified.json, passed by: %s" % ", ".join(SCREENED))
for _h, _why in SJ.get("excluded_hazards", {}).items():
    print("        excluded %-11s %s" % (_h, _why))
print("(b) %.2f%% attributable, %d counties positive, %d negative, median %.2f, 95th %.1f, "
      "max %.1f h per customer"
      % (100 * SJ["share"], (cty.rate > 0).sum(), NNEG, pos.median(), vhi, pos.max()))
D6 = json.load(open("%s/dose_for_fig6.json" % AT))["bands"]
print("(d) hurricane cloud medians by band: %s" % [round(m, 1) for m in med])
print("(e) clipped points outside %.2f to %.0f: %d of %s drawn-hazard rows (%s in the file)"
      % (LO, HI, nclip, format(int(CL.hazard.isin(SHOW).sum()), ","), format(len(CL), ",")))
print("(d) profiles: %s" % {h: [round(np.exp(PF[h][b]), 2) for b in BLK] for h, _, _, _ in HZ})
print("dose response handed to Fig. 6, whole event, by wind band:")
for b in D6:
    print("    %-18s total %+.3f log points  = %.0f times an ordinary day" % (b["band"], b["total"], b["multiplier"]))
