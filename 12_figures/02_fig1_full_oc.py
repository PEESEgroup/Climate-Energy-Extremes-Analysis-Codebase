"""
FIGURE 1 — assembled at Nature double-column width.

183 mm is Nature's maximum figure width, so the page is built AT final size and is never
rescaled in production. That keeps every type size exactly as specified here: 7.4 pt item
labels, 6.8 pt axis labels, 6.2 pt annotation — all inside Nature's 5-7 pt requirement.

ROW ORDER, because the docstring used to have it backwards and a caption written from it would
mislabel the panels: row 1 is the three hazard maps, row 2 is the joint panel regression and the
running cumulative effect, row 3 is the two circulation-pattern panels.

Layout is in MILLIMETRES from the top-left. Rows 2 and 3 share one two-cell grid
(x = 24.0 and 107.5, each 73.5 mm wide, right edge 181 mm). Row 1 is deliberately off that
grid, at x = 2 / 62 / 122, each 59 mm wide; the note above MAP_X says why.
save(..., tight=False) keeps the canvas exactly 183 mm.

Rows — hazards first, upstream context last, so the reader is not bounced between the two.
  1  where each hazard bites        3 CONUS maps          (HAZARD)
  2  how much | which channel | how long                  (HAZARD — same six, same order)
  3  what modulates upstream        3 field-significance  (CLIMATE STATE)

Row 3 sits last deliberately: we tested 48 climate-state x hazard frequency pairs and 40 were
null, so the slow modes are upstream context and a largely negative result, not the lead-in.

Dropped relative to the 210 mm draft: the per-hazard exact wild-cluster-bootstrap p column
in row 3 col 1. At 51 mm it consumed 40% of the axis and squeezed the forest; the nested
confidence intervals already carry significance, and the exact p values belong in the SI
table. Reported there, not lost.

No panel letters, no sub-headings, no titles anywhere — the author adds all of those in the
layout program. Only data labels, axis labels and units remain.
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
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

W_MM, H_MM = 183.0, 173.0   # plan caps height at 200 mm
fig = new_fig(W_MM, H_MM)

def R(x, y, w, h):
    return [x / W_MM, 1.0 - (y + h) / H_MM, w / W_MM, h / H_MM]

def TX(x, y, s, **kw):
    kw.setdefault("fontsize", FS_LABEL)
    return fig.text(x / W_MM, 1.0 - y / H_MM, s, **kw)

def FY(y):
    return 1.0 - y / H_MM

def full_width_rule(y_data, ylo, yhi, top_mm, h_mm, x0=2.0, x1=181.0):
    """One hairline running across the label gutter and all three cells of a row.

    Ties the three cells together without the ink of alternating bands: the group boundary
    is a single continuous line, so the eye reads the row as one object."""
    yf = (1.0 - (top_mm + h_mm) / H_MM + (y_data - ylo) / (yhi - ylo) * h_mm / H_MM)
    fig.add_artist(Line2D([x0 / W_MM, x1 / W_MM], [yf, yf], transform=fig.transFigure,
                          color=C_GRID, lw=.7, zorder=-5))

# ---------------------------- shared hazard vocabulary ----------------------------
NICE = {"cold": "Cold outbreak", "heat": "Heat wave", "fire": "Fire weather",
        "vre_drought": "Renewable drought", "ar": "Atm. river", "tc_local": "Tropical cyclone"}
# Second label line = how many subregion-days carry each flag, and what share of every subregion-day
# cell of the panel that is; see the denominator note below. It is the only place the reader can see
# how far apart the six flags are in how often they fire, a range the coefficient plot cannot show.
#
# COUNTED, never typed in. The fire and atmospheric-river definitions have both been rebuilt since
# these numbers were first written down, and a hard-coded count outlives the flag it described
# without ever looking wrong. They are read from the same arrays the estimator is fitted on.
#
# Cold, heat, fire, renewable drought and tropical cyclone are the panel_v3 columns. The AR row is
# NOT a panel column: panel_v3 carries an ar_pub column holding the superseded shapefile flag, while
# the estimator's AR slot is the adopted ivt_p95_cov25 flag. That array is attached under the fresh
# name ar, which is the name the estimator json uses. The
# panel's ar_pub column is left alone, so one name never denotes two constructions in this file.
# Refuse an unstamped or superseded panel before any label is counted from it. 15_subregion_flags.py
# stamps exactly these four hazards; ar_pub and tc_local are carried through from panel_v2 and
# are stamped by no builder, so naming them here would raise on a correct file.
import hazard_defs as HD
PANEL = f"{R1}/panel_v3.parquet"
HD.require_stamp(PANEL, ["cold", "heat", "fire", "vre_drought"])
PAN = pd.read_parquet(PANEL)
PAN["date"] = pd.to_datetime(PAN.date)
# The adopted AR flag is stamped by 06_ar_variants.py, and ar_fix.py can still write this exact
# path with no stamp at all. Three sibling consumers refuse an unstamped or superseded
# file; this one used to read whatever was there.
# require_stamp reads parquet schema metadata, and this is an npz, so the check is the one
# 07_ar_adopt_oc.py makes: the stamp lives in a key inside the archive. ar_fix.py can still
# write this exact path with no stamp at all, and three sibling consumers refuse such a
# file while this one used to read whatever was there.
_Z = np.load("/data/enso/ar_flag_variants.npz", allow_pickle=True)
if "hazard_defs_stamp" not in _Z.files:
    raise ValueError("/data/enso/ar_flag_variants.npz carries no hazard_defs stamp; rerun 06_ar_variants.py before reading it")
_arst = json.loads(str(_Z["hazard_defs_stamp"]))
_arw, _arg = HD.definition_hash("ar"), (_arst.get("definition_hash") or {}).get("ar")
if _arg != _arw:
    raise ValueError("the atmospheric-river flag is at definition %s, current is %s; rerun 06_ar_variants.py" % (_arg, _arw))
_sub = [str(x) for x in _Z["subregions"]]
_dts = pd.to_datetime([str(x) for x in _Z["dates"]])
_A = pd.DataFrame(_Z["ivt_p95_cov25"].T, index=_dts, columns=_sub).stack()
_A.index = _A.index.set_names(["date", "subregion"])
PAN["ar"] = _A.reorder_levels(["subregion", "date"]).reindex(
    pd.MultiIndex.from_arrays([PAN.subregion.values, PAN.date.values])).fillna(False).values.astype(float)
# denominator = every subregion-day cell of the panel, a missing cell counted as unflagged. That
# is the rule the hazard occurrence table uses too, so the label on this figure and that table are
# the same number and not two conventions that happen to be close.
_CELLS = PAN.subregion.nunique() * PAN.date.nunique()
NDAY = {h: int((PAN[h] > 0.5).sum()) for h in NICE}
FRAC = {h: 100.0 * NDAY[h] / _CELLS for h in NICE}
print("flag days, counted from the arrays: "
      + "  ".join("%s %s (%.2f%%)" % (h, format(NDAY[h], ","), FRAC[h]) for h in NICE))
CHAN = {"cold": C_DEM, "heat": C_DEM, "fire": C_VRE, "vre_drought": C_VRE,
        "ar": C_VRE, "tc_local": C_MIX}
TIER3, TIER2 = ["cold", "heat", "fire", "vre_drought"], ["ar", "tc_local"]
ORDER = TIER3 + TIER2
CT, CN, GW = 2.110, 1.960, 1e-3

Y, _y = {}, 0.0
for _g, _grp in enumerate((TIER3, TIER2)):
    if _g: _y -= 0.95
    for _h in _grp:
        Y[_h] = _y; _y -= 1.0
YLO, YHI = _y + 0.45, 0.90
YLO_L = YLO - 2.00        # blank band at the foot of each row-3 cell for its own legend

M = json.load(open(f"{R1}/r1_final_main_ourchain.json"))     # AR slot = ivt_p95_cov25
DEC = json.load(open(f"{R1}/r1_decomposition_ourchain.json"))["channel_decomposition_mean"]
# The figure now calls the atmospheric-river slot "ar" everywhere, which is what the rebuilt
# estimator json calls it. A json written before the rename spells it "ar_pub", so that spelling is
# resolved onto "ar" rather than the figure being renamed back. The shim runs in one direction only:
# nothing here ever creates an "ar_pub" key, so the superseded name cannot re-enter the figure.
def _alias(d):
    if isinstance(d, dict):
        if "ar_pub" in d and "ar" not in d: d["ar"] = d["ar_pub"]
        for _v in list(d.values()): _alias(_v)
_alias(M); _alias(DEC)

# ================================= ROW 1 — maps =================================
# R1-FLAG maps. The previous file, hazard_significance.csv, composited a DIFFERENT hazard
# definition family from the joint regression in row 3: its fire rows were bit-identical to
# fire_hdw_summary.net_pct_hiHDW, top-decile HDW days at 1461 per subregion, while the regression
# fires on the fire flag carried in the panel. Those are not the same days. Rows 1 and 3 now use
# the same flags. Old file kept intact; the literature-standard per-hazard definitions move to SI
# as a robustness comparison (cold Spearman 0.95, heat 0.98, fire 0.48).
sig = pd.read_csv(f"{LO}/hazard_significance_ourchain.csv")
N_MIN = 10          # below this the cell has too few hazard days to support an estimate
MIN_SPELL = 3        # 08_fig1_row1.py needs three spells to run the Welch test on spell means


def tested(d):
    return (d.tag_days >= N_MIN) & d.p.notna()


# The choropleth carries the PERCENTAGE change; a large % on a small subregion is not a large
# effect. Overlaid circles carry the ABSOLUTE MW on one shared scale across all three maps, so
# fire's dramatic -27% is visibly a small absolute quantity next to cold's +51%.
ROW1 = ["cold", "heat", "fire"]     # the three hazards mapped in this row; the csv holds six
# The circle area is one shared scale across the three maps, so it must be set by the three
# hazards drawn here. It used to be the maximum over ALL SIX hazards in the csv, and renewable
# drought and tropical cyclone are never drawn in this row: their effects therefore set the
# scale of circles they do not appear on, shrinking every circle that does appear.
MW_MAX = float(sig.loc[sig.hazard.isin(ROW1) & tested(sig), "net_MW"].abs().max())
# day-weighted national mean, GW: each subregion's effect weighted by its own hazard-day count,
# over the subregions that clear N_MIN. It was three typed-in constants, which cannot follow the
# csv they are printed on top of, so it is recomputed from that csv. Same formula fig1_row1 prints.
# TESTED MEANS A P VALUE EXISTS. Clearing N_MIN on day count is not enough: 08_fig1_row1.py needs at
# least three hazard SPELLS to run a Welch test on spell means, so FRCC's 12 cold days in 2 spells
# carry p = NaN. Gating the fill, the cross, the circle, the colour scale and the national mean on
# the day count alone shaded FRCC, crossed it as "not significant" and gave it the third largest
# circle on the cold map, while the box above said 16 subregions were tested. One predicate now
# decides all five, so the map and its own caption cannot disagree again.
def natmw(hz):
    q = sig[(sig.hazard == hz) & tested(sig) & sig.net_MW.notna()]
    return float((q.net_MW * q.tag_days).sum() / q.tag_days.sum()) / 1e3
NATMW = {hz: natmw(hz) for hz in ROW1}
zm = np.load("/data/datasets/grid/subregion_mask.npz", allow_pickle=True)
mask = zm["subregion_mask"]; name2id = {str(n): int(i) for i, n in zm["id_to_subregion"]}
zc = np.load("/data/datasets/grid/coordinate.npz")
lat, lon = zc["lat"].astype(float), zc["lon"].astype(float)
rows = np.where((mask > 0).any(1))[0]; cols = np.where((mask > 0).any(0))[0]
r0, r1_, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
subm = mask[r0:r1_, c0:c1]
ext = [lon[c0], lon[c1 - 1], lat[r0], lat[r1_ - 1]]
LON, LAT = np.meshgrid(lon[c0:c1], lat[r0:r1_])
ASP = 1.0 / np.cos(np.deg2rad(np.mean(ext[2:])))
# Row 1 is deliberately NOT on the column grid. Maps carry no row labels, so the 24 mm
# label gutter would sit idle; column position carries no shared meaning across rows, so the
# alignment buys nothing semantic. Spending that width on the maps instead is +16% linear,
# +34% area on the densest element in the figure. Right edge still lands on 181 mm.
MAP_X, MAP_W = [2.0, 62.0, 122.0], 59.0
MAP_H = MAP_W * (ext[3] - ext[2]) * ASP / (ext[1] - ext[0])

norm = TwoSlopeNorm(vmin=-25, vcenter=0, vmax=25); cmap = plt.get_cmap("RdBu_r")
MAP_Y = 3.0
for k, hz in enumerate(ROW1):
    x = MAP_X[k]
    ax = fig.add_axes(R(x, MAP_Y, MAP_W, MAP_H))
    d = sig[sig.hazard == hz].set_index("sub")
    img = np.full(subm.shape, np.nan)
    thin = np.zeros(subm.shape, bool)
    for nm, i in name2id.items():
        if nm not in d.index:
            continue
        if not bool(tested(d.loc[[nm]]).iloc[0]):
            thin |= (subm == i)          # too few days, or too few spells to test: leave it grey
        else:
            img[subm == i] = float(d.loc[nm, "pct"])
    ax.imshow(img, origin="lower", extent=ext, cmap=cmap, norm=norm,
              interpolation="nearest", aspect="auto")
    ax.imshow(np.where(thin, 1.0, np.nan), origin="lower", extent=ext, aspect="auto",
              cmap=plt.matplotlib.colors.ListedColormap(["#D9D9D9"]), vmin=0, vmax=1)
    ax.contour(np.where(subm > 0, subm, np.nan), levels=np.arange(.5, subm.max() + 1),
               colors="white", linewidths=.3, extent=ext, origin="lower")
    for nm, i in name2id.items():
        if nm in d.index and bool(tested(d.loc[[nm]]).iloc[0]) and not bool(d.loc[nm, "fdr_sig"]):
            m = subm == i
            if m.sum():
                ax.plot(LON[m].mean(), LAT[m].mean(), marker="x", ms=2.8, mew=.8,
                        color="#1A1A1A", zorder=5)
    for nm, i in name2id.items():                       # absolute-magnitude overlay
        if nm not in d.index or not bool(tested(d.loc[[nm]]).iloc[0]):
            continue
        m = subm == i
        if not m.sum():
            continue
        mw = float(d.loc[nm, "net_MW"])
        sz = abs(mw) / MW_MAX * 42 + 1.2
        ax.scatter(LON[m].mean(), LAT[m].mean(), s=sz, facecolor="none",
                   edgecolor="white", linewidth=1.0, zorder=4, alpha=.85)
        ax.scatter(LON[m].mean(), LAT[m].mean(), s=sz, facecolor="none",
                   edgecolor="#1A1A1A", linewidth=.4, zorder=4)
    # THE DENOMINATOR IS WHAT WAS ACTUALLY TESTED. A subregion clears N_MIN on day count and can
    # still carry no p value, because 08_fig1_row1.py needs at least three hazard SPELLS to run a
    # Welch test on spell means. FRCC has 12 cold days in 2 spells: it entered this denominator,
    # could never be significant, and was drawn with a not-significant cross, which tells the
    # reader it was tested and failed. Requiring a p value makes the count honest and the marker
    # correct, and the grey "not tested" fill now covers both reasons.
    ok = d[(d.tag_days >= N_MIN) & d.p.notna()]
    lo, hi = float(ok.pct.min()), float(ok.pct.max())
    fmt = lambda v: (("%+.0f" % v).replace("-", "−")) if abs(v) >= 0.5 else "0"
    # The national GW went through a bare %+.2f while the two percentages went through fmt, so one
    # box carried a typographic minus in "−66 to +2%" and an ASCII hyphen in "-0.89 GW". Every
    # sign in the box now comes from the same substitution.
    fmt_gw = lambda v: ("%+.2f" % v).replace("-", "−")
    ax.text(0.015, 0.035,
            "%d of %d tested subregions significant  ·  %s to %s%%\nnationally, per hazard day  %s GW"
            % (int(ok.fdr_sig.sum()), len(ok), fmt(lo), fmt(hi), fmt_gw(NATMW[hz])),
            transform=ax.transAxes, fontsize=FS_VAL, color="#333333", zorder=10,
            ha="left", va="bottom", linespacing=1.45,
            bbox=dict(facecolor="white", edgecolor="#D8D8D8", linewidth=.3,
                      boxstyle="round,pad=0.28"))
    bare(ax)

CB_Y = MAP_Y + MAP_H + 4.0
cax = fig.add_axes(R(MAP_X[1], CB_Y, MAP_W, 2.3))
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                  orientation="horizontal", extend="both")
cb.set_ticks([-25, 0, 25]); cb.set_ticklabels(["≤ −25", "0", "≥ 25"])
cb.ax.tick_params(labelsize=FS_TICK, width=.5, length=2, pad=1.5)
cb.set_label("net-load change on hazard days (%)", fontsize=FS_AXIS, labelpad=2)
cb.outline.set_linewidth(.5)
axk = fig.add_axes(R(2.0, CB_Y - 2.4, 44.0, 6.4)); bare(axk)
axk.legend(handles=[
    Line2D([], [], marker="x", ls="", color="#1A1A1A", mew=.8, ms=2.8,
           label="not significant"),
    Patch(facecolor="#D9D9D9",
          # Two reasons put a subregion here and the label must name both. FRCC is grey with
          # 12 cold days, above N_MIN, because those days form only 2 spells and the Welch
          # test needs 3. Saying "under 10 hazard days" told the reader something false.
          label="under %d hazard days or under %d spells, not tested" % (N_MIN, MIN_SPELL))],
    loc="center left", fontsize=FS_ANNOT, handlelength=1.2, labelspacing=.42,
    borderpad=.3, frameon=True, facecolor="white", edgecolor="#D8D8D8",
    framealpha=.92).get_frame().set_linewidth(.3)
axl = fig.add_axes(R(139.0, CB_Y - 2.2, 42.0, 6.0)); axl.set_xlim(0, 1); axl.set_ylim(0, 1); bare(axl)
for xx, mw in [(0.06, 2000), (0.22, 8000), (0.46, 20000)]:
    sz = mw / MW_MAX * 42 + 1.2
    axl.scatter([xx], [.62], s=sz, facecolor="none", edgecolor="white", linewidth=1.0, alpha=.85)
    axl.scatter([xx], [.62], s=sz, facecolor="none", edgecolor="#1A1A1A", linewidth=.4)
    axl.text(xx, .10, "%g" % (mw / 1000), fontsize=FS_VAL, ha="center", va="bottom", color=C_TXT)
axl.text(0.60, .10, "GW", fontsize=FS_VAL, ha="left", va="bottom", color=C_TXT)
axl.text(0.0, 1.02, "circle = absolute change", fontsize=FS_ANNOT, color=C_TXT, va="top")
# ========================== ROW 2 — the supply/demand plane ==========================
# Replaces three bar cells of counts. A count of subregions clearing a permutation hurdle is a
# statistical construct; this plots the two PHYSICAL quantities the paper is about, on the same
# pair of axes as the channel decomposition above, so the seasonal layer and the daily layer are
# visibly the same dichotomy.
#
# Two cells, WINTER and SUMMER, on IDENTICAL limits, because the comparison between them is the
# result.
#
# NATIONAL GW. seasonal_field_ourchain.csv stores the mean over the 18 subregions of each subregion's own
# change; the national change is the SUM, which is that mean x 18. The sum is exact only for the
# daily MEAN (subregional peaks do not coincide), so both axes use daily means.
#
# The renewable-drought column is gone from this row. Over the 16 pattern-season cells its effect
# correlates -0.984 with the generation effect: the same signal measured as duration instead of
# level, costing a third of the row. It moves to Extended Data, and stays a hazard in row 3.
SF = pd.read_csv(f"{R1}/seasonal_field_ourchain.csv")
NSUB = 18.0
PLAB = {"El Nino vs neutral": "El Niño", "La Nina vs neutral": "La Niña",
        "ENSO + vs -": "ENSO", "PNA + vs -": "PNA", "NAO + vs -": "NAO", "AO + vs -": "AO",
        "Pacific blocking": "Pacific blocking", "Atlantic blocking": "Atlantic blocking"}
OFF = {("DJF", "El Niño"): (0, -8, "center"), ("DJF", "La Niña"): (-7, -2, "right"),
       ("DJF", "ENSO"): (8, 0, "left"), ("DJF", "PNA"): (8, 2, "left"),
       ("DJF", "NAO"): (0, -8, "center"), ("DJF", "AO"): (8, 1, "left"),
       ("DJF", "Pacific blocking"): (-9, 3, "right"), ("DJF", "Atlantic blocking"): (0, 7, "center"),
       ("JJA", "El Niño"): (0, 7, "center"), ("JJA", "La Niña"): (7, 0, "left"),
       ("JJA", "ENSO"): (0, -8, "center"), ("JJA", "PNA"): (7, -1, "left"),
       ("JJA", "NAO"): (-8, 1, "right"), ("JJA", "AO"): (7, 1, "left"),
       ("JJA", "Pacific blocking"): (0, -8, "center"),
       ("JJA", "Atlantic blocking"): (-8, 0, "right")}
BY, BH_ = 113.0, 52.0
PX, PW = [24.0, 107.5], [73.5, 73.5]      # the two-cell grid rows 2 and 3 share
XLO, XHI = -14.5, 23.5
YLO_S, YHI_S = -4.9, 1.5
SZ0, SZK = 9.0, 6.2                     # marker area = SZ0 + SZK x subregions responding
for k, (seas, sname) in enumerate([("DJF", "winter"), ("JJA", "summer")]):
    ax = fig.add_axes(R(PX[k], BY, PW[k], BH_))
    ax.axhline(0, color=C_GRID, lw=.5, zorder=1)
    ax.axvline(0, color=C_GRID, lw=.5, zorder=1)
    d = SF[SF.season == seas]
    for pat, lab in PLAB.items():
        rn = d[(d.pattern == pat) & (d.outcome == "net_mean")]
        rv = d[(d.pattern == pat) & (d.outcome == "vre_mean")]
        if not len(rn) or not len(rv):
            continue
        rn = rn.iloc[0]; rv = rv.iloc[0]
        dem = str(rn.verdict).upper() == "YES"; gen = str(rv.verdict).upper() == "YES"
        col = C_MIX if (dem and gen) else (C_DEM if dem else (C_VRE if gen else C_NS))
        nsub = max(int(rn.R_obs), int(rv.R_obs))
        x = rn.mean_diff * NSUB; y = rv.mean_diff * NSUB
        ax.scatter([x], [y], s=SZ0 + nsub * SZK, color=col, zorder=4,
                   edgecolor="white", linewidth=.5, alpha=.95)
        dx, dy, ha = OFF[(seas, lab)]
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(dx, dy), ha=ha,
                    va="center", fontsize=FS_VAL,
                    color=("#1A1A1A" if (dem or gen) else "#9A9A9A"), zorder=6)
    ax.set_xlim(XLO, XHI); ax.set_ylim(YLO_S, YHI_S)
    ax.set_xticks([-10, 0, 10, 20]); ax.set_yticks([-4, -2, 0])
    ax.tick_params(labelsize=FS_TICK, width=.4, length=1.8, pad=2)
    ax.set_xlabel("effect on national net load (GW)", fontsize=FS_AXIS, labelpad=2)
    if k == 0:
        ax.set_ylabel("effect on national wind\nand solar generation (GW)", fontsize=FS_AXIS,
                      labelpad=2)
    ax.text(0.985, 0.035, sname, transform=ax.transAxes, fontsize=FS_LABEL, ha="right",
            va="bottom", color=C_TXT)
    despine(ax)
    if k == 0:
        ax.legend(handles=[
            Line2D([], [], marker="o", ls="", color=C_DEM, ms=3.6, label="acts on demand"),
            Line2D([], [], marker="o", ls="", color=C_VRE, ms=3.6, label="acts on generation"),
            Line2D([], [], marker="o", ls="", color=C_MIX, ms=3.6, label="acts on both"),
            Line2D([], [], marker="o", ls="", color=C_NS, ms=3.6, label="neither")],
            loc="upper right", bbox_to_anchor=(1.006, 1.010), ncol=1, fontsize=FS_VAL,
            handlelength=1.0, columnspacing=1.05, labelspacing=.26, borderpad=.28,
            handletextpad=.42, frameon=True, facecolor="white", edgecolor="#D8D8D8",
            framealpha=.94).get_frame().set_linewidth(.3)

# marker-area key: the legend above names the size encoding, this one gives its scale
axs = fig.add_axes(R(PX[1] + 42.0, BY + 1.0, 30.0, 9.5))
axs.set_xlim(0, 1); axs.set_ylim(0, 1); bare(axs)
for xx, ns in [(0.10, 4), (0.34, 8), (0.66, 16)]:
    axs.scatter([xx], [.60], s=SZ0 + ns * SZK, facecolor="none", edgecolor="#8A8A8A",
                linewidth=.5)
    axs.text(xx, .06, "%d" % ns, fontsize=FS_VAL, ha="center", va="bottom", color=C_TXT)
axs.text(0.80, .06, "of 18", fontsize=FS_VAL, ha="left", va="bottom", color=C_TXT)
axs.text(0.0, 1.0, "size: subregions responding", fontsize=FS_ANNOT, color=C_TXT, va="top")

# ============ ROW 3 — how much, through which channel, and for how long ============
# d and e merged: they shared the mean net coefficient and drew it twice, on the same rows and the
# same GW axis, so they read as one chart drawn twice. Bars are the two channels, the filled circle
# on them is the net effect with its intervals, the open circle below is the daily peak. The
# per-channel intervals e used to draw are gone: colour already says whether a channel is
# significant, and they were what made the merged row crowded.
CY, CH_ = 52.0, 52.0
CX, CW_ = [24.0, 107.5], [73.5, 73.5]     # the same two cells as row 2, right edge 181 mm
full_width_rule((Y["vre_drought"] + Y["ar"]) / 2, YLO_L, YHI, CY, CH_)
mn, pk = M["main_joint"]["netload_anom_mean"], M["main_joint"]["netload_anom_peak"]

axc = fig.add_axes(R(CX[0], CY, CW_[0], CH_)); axc.patch.set_visible(False); axc.set_zorder(3)
BARY, PKY = 0.22, -0.14   # the daily-peak row sits 0.36 below its own mean row and 0.64 above
                          # the next hazard's, so the pair reads as one hazard rather than two
for h in ORDER:
    c = DEC[h]; yy = Y[h]
    parts = [(+c["beta_load"] * GW, C_DEM if abs(c["t_load"]) > CT else C_NS),
             (-c["beta_vre"] * GW, C_VRE if abs(c["t_vre"]) > CT else C_NS)]
    pos = neg = 0.0
    for val, col in parts:
        if val >= 0:
            axc.barh(yy + BARY, val, left=pos, height=.42, color=col, lw=.3,
                     edgecolor="white", zorder=3); pos += val
        else:
            axc.barh(yy + BARY, val, left=neg, height=.42, color=col, lw=.3,
                     edgecolor="white", zorder=3); neg += val
    b, se = mn[h]["beta"] * GW, mn[h]["se"] * GW
    col = CHAN[h] if abs(b / se) > CT else C_NS
    axc.plot([b - CT * se, b + CT * se], [yy + BARY, yy + BARY], color="#33333366", lw=1.9,
             solid_capstyle="butt", zorder=4)
    axc.plot([b - CN * se, b + CN * se], [yy + BARY, yy + BARY], color="#333333", lw=.7, zorder=5)
    axc.scatter([b], [yy + BARY], s=15, color=col, zorder=6, edgecolor="white", linewidth=.55)
    bp, sp = pk[h]["beta"] * GW, pk[h]["se"] * GW
    cp = CHAN[h] if abs(bp / sp) > CT else C_NS
    axc.plot([bp - CT * sp, bp + CT * sp], [yy + PKY, yy + PKY], color=cp, lw=.6, alpha=.65, zorder=3)
    axc.scatter([bp], [yy + PKY], s=8, facecolor="white", edgecolor=cp, linewidth=.6, zorder=4)
axc.axvline(0, color="black", lw=.5, ls=(0, (3, 3)), zorder=1)
axc.set_yticks([Y[h] for h in ORDER])
axc.set_yticklabels([NICE[h] for h in ORDER], fontsize=FS_TICK)
for h in ORDER:
    axc.text(-0.035, Y[h] - .40, "%s d  ·  %.2f%%" % (format(NDAY[h], ","), FRAC[h]),
             transform=axc.get_yaxis_transform(), fontsize=FS_VAL, color="#9A9A9A",
             ha="right", va="center")
    axc.text(11.35, Y[h] + BARY, ("%+.2f" % (mn[h]["beta"] * GW)).replace("-", "−"),
             fontsize=FS_VAL, ha="right", va="center", color="black")
    axc.text(11.35, Y[h] + PKY, ("%+.2f" % (pk[h]["beta"] * GW)).replace("-", "−"),
             fontsize=FS_VAL, ha="right", va="center", color="#8A8A8A")
axc.set_ylim(YLO_L, YHI); axc.tick_params(axis="y", length=0, pad=2.5)
axc.set_xlim(-3.4, 11.6)
axc.set_xticks([-2, 0, 2, 4, 6, 8])
axc.set_xlabel("effect on net load, and its two components (GW)", fontsize=FS_AXIS, labelpad=2)
despine(axc)
axc.legend(handles=[
    Patch(facecolor=C_DEM, label="demand component"),
    Patch(facecolor=C_VRE, label="generation component"),
    Patch(facecolor=C_NS, label="component not significant"),
    Line2D([], [], marker="o", ls="", color="#666666", mec="white", mew=.55, ms=3.0,
           label="net effect, daily mean"),
    Line2D([], [], color="#333333", lw=1.9, alpha=.4, label="95% CI, $t_{(17)}$"),
    Line2D([], [], marker="o", ls="", mfc="white", mec="#666666", ms=2.6, label="daily peak"),
    Line2D([], [], ls="", label="grey: days, share of all")],
    loc="lower left", bbox_to_anchor=(0.014, 0.018), ncol=2, fontsize=FS_VAL, handlelength=1.3,
    handleheight=.85, columnspacing=1.2, labelspacing=.26, borderpad=.26, handletextpad=.42,
    frameon=True, facecolor="white", edgecolor="#D8D8D8",
    framealpha=.94).get_frame().set_linewidth(.3)

G = M["gate2_distributed_lag"]["netload_anom_mean"]
# The 15 distributed-lag coefficients are drawn as ONE running cumulative curve per hazard, with
# its endpoint total and the share accumulated before day 0. Individual bars could not be read at
# this cell width, and the pre-event span is the honest placebo display.
# LAG15 below is the only lag key list the figure uses; two earlier lists sat here unread, so an
# edit to the lag window changed nothing visible. Edit LAG15.

axe = fig.add_axes(R(CX[1], CY, CW_[1], CH_)); axe.patch.set_visible(False); axe.set_zorder(3)
LAG15 = [f"lead{i}" for i in range(7, 0, -1)] + ["day_of"] + [f"lag{i}" for i in range(1, 8)]
XL = np.arange(-7, 8)
# A running cumulative curve answers both questions at once: its SHAPE shows persistence, its
# ENDPOINT is the total. It also makes the pre-event contamination visible — a flat pre-period
# is a clean placebo, a rising one is not, and cold and heat visibly rise before day 0.
axe.axvspan(-7.4, -0.5, color="#F2F2F2", zorder=0)
for h in ORDER:
    cum = np.cumsum([G[h][k]["beta"] for k in LAG15]) * GW
    sc = cum / np.abs(cum).max() * 0.32   # 0.40 let adjacent endpoint labels collide
    axe.plot([-7.4, 7.4], [Y[h], Y[h]], color="#E4E4E4", lw=.4, zorder=1)
    axe.plot(XL, Y[h] + sc, color=CHAN[h], lw=1.15, zorder=3, solid_capstyle="round")
    axe.scatter([7], [Y[h] + sc[-1]], s=7, color=CHAN[h], zorder=4,
                edgecolor="white", linewidth=.35)
    axe.text(8.1, Y[h] + sc[-1], ("%+.2f" % cum[-1]).replace("-", "−"), fontsize=FS_VAL,
             ha="left", va="center", color="black")
    pre = cum[6] / cum[-1]
    axe.text(-7.2, Y[h] + .30, ("%+.0f%%" % (100 * pre)).replace("-", "−"), fontsize=FS_VAL,
             ha="left", va="center", color="#9A9A9A")
axe.axvline(0, color="black", lw=.5, ls=(0, (3, 3)), zorder=2)
axe.axhline((Y["vre_drought"] + Y["ar"]) / 2, color=C_GRID, lw=.7, zorder=1)
axe.set_xlim(-7.8, 12.6); axe.set_xticks([-7, 0, 7])
axe.set_xticklabels(["−7", "0", "+7"], fontsize=FS_TICK)
axe.set_ylim(YLO_L, YHI)
axe.set_yticks([Y[h] for h in ORDER]); axe.set_yticklabels([])
axe.tick_params(axis="y", length=1.8, width=.4)
for sp in ("top", "right"):
    axe.spines[sp].set_visible(False)
axe.spines["left"].set_linewidth(.6)
axe.spines["left"].set_bounds(YLO_L, YHI)
# the bottom spine was hidden while its ticks were kept, so -7 / 0 / +7 floated under nothing
axe.spines["bottom"].set_linewidth(.6)
axe.spines["bottom"].set_bounds(-7.8, 7)
axe.set_xlabel("days from event day", fontsize=FS_AXIS, labelpad=2)
axe.legend(handles=[
    Line2D([], [], color="#666666", lw=1.15, label="running cumulative (GW)"),
    Patch(facecolor="#F2F2F2", label="pre-event window"),
    Line2D([], [], ls="", label="% before day 0")],
    loc="lower left", bbox_to_anchor=(0.022, 0.022), fontsize=FS_TICK, handlelength=1.3,
    handleheight=.85, labelspacing=.26, borderpad=.26, handletextpad=.42, frameon=True,
    facecolor="white", edgecolor="#D8D8D8", framealpha=.94).get_frame().set_linewidth(.3)

save(fig, "fig1_full_ourchain", tight=False, png_dpi=300)
print("page %.0f x %.0f mm  (Nature double-column max = 183 mm, built at final size)" % (W_MM, H_MM))
print("rows 2 and 3 at x = %s mm, each %.1f mm; row 1 at x = %s mm, each %.0f mm"
      % (PX, PW[0], MAP_X, MAP_W))
