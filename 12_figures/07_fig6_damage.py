"""
FIGURE 6. The damage channel: what changes in the storms, what a knot is worth, what it adds to.

  a  HOW THE SEVERITY TAIL MOVES, in all four realizations. At every wind speed, the change in the
     number of county-days per year reaching at least that speed. The grey floor is the historical
     count those percentages sit on, 408 county-days a year at 34 kt falling to 2.7 at 100 kt.
     Nothing else about these storms can move: TGW-future replays the observed synoptic sequence at
     +40 years, so the same storm arrives on the same date along the same track.

  b  WHAT A KNOT IS WORTH, rebuilt on the unified PPML panel. The dose object is the WHOLE-EVENT
     effect of one exposed county-day, impact plus restore plus tail, by wind band. It takes one
     large step of 2.33 log points between the 34-50 kt and 50-64 kt bands. Above that step the
     64-83 kt band is indistinguishable from 50-64 kt (+0.14 log points, z 0.33), while 83 kt and
     above sits a further 1.12 log points higher (z 2.68). The line is the profile made monotone by
     weighted pool-adjacent violators and the points are the unconstrained estimates; on this fit
     the two coincide, because the estimates are already increasing. In the floor, the change in
     county-days per 7 kt bin.

  c  WHAT IT ADDS UP TO. Exposure gained and exposure lost are separate components: 1,412 county-
     days cross above 34 kt and 1,150 fall below it, and an earlier version put the second half
     inside the 34-50 kt bar, which made one symmetric edge process look like two large opposite
     findings. The four band bars therefore hold only county-days a storm reaches in both periods.

LABELS THAT TRAVEL WITH EVERY NUMBER HERE. Thermodynamic intensification only, and fixed
vulnerability: the grid, the fleet and the population are held at their observed values.
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
import json
import numpy as np, pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SR = "/data/scratch_r5"
W_MM, H_MM = 183.0, 70.0
fig = new_fig(W_MM, H_MM)
def R(x, y, w, h): return [x / W_MM, 1.0 - (y + h) / H_MM, w / W_MM, h / H_MM]


PAGE_L, PAGE_R, PAGE_T = 1.5, 5.0, 3.0
GUT_X = 5.0
PAD_L, PAD_B = 11.5, 14.0
ROW_H = 67.0


def cell(c, n):
    w = (W_MM - PAGE_L - PAGE_R - GUT_X * (n - 1)) / n
    x0 = PAGE_L + c * (w + GUT_X)
    return R(x0 + PAD_L, PAGE_T, w - PAD_L, ROW_H - PAD_B)


DZ = json.load(open(f"{SR}/dose_projection_ppml.json"))
BASE = DZ["baseline_share_pct"]
HEAD = "rcp85hotter"                                    # the hottest realization, dT2 = 2.53 K
KT = np.array(DZ["profile"]["kt"])
CO = np.array(DZ["profile"]["coef"])                    # monotone, the line
CR = np.array(DZ["profile"]["coef_raw"])                # unconstrained, the points
SE = np.array(DZ["profile"]["se"])
C34, C50, C64 = "#DFC27D", "#BF812D", "#8C510A"
C_BIN, C_DOSE = "#9A9A9A", "#A6761D"

# ==================== a — how the severity tail moves, in all four ====================
axa = fig.add_axes(cell(0, 3))
TL = pd.read_csv(f"{SR}/tail_by_threshold.csv")
DT = {"rcp45cooler": 1.472, "rcp85cooler": 1.743, "rcp45hotter": 2.084, "rcp85hotter": 2.530}
CW = {"rcp45cooler": "#FDBE85", "rcp85cooler": "#FD8D3C", "rcp45hotter": "#E6550D",
      "rcp85hotter": "#A63603"}
axw = axa.twinx()
axw.fill_between(TL.kt, TL.hist_per_yr, .55, color="#E6E6E6", lw=0, zorder=1)
axw.set_yscale("log"); axw.set_ylim(.55, 1.1e5); axw.set_yticks([])
axw.tick_params(which="both", left=False, right=False, length=0)
axw.set_xlim(32, 102)
for sp in axw.spines.values():
    sp.set_visible(False)
for s in sorted(DT, key=DT.get):
    axa.plot(TL.kt, TL[s], color=CW[s], lw=1.2, zorder=4)
axa.axhline(0, color=C_GRID, lw=.5, zorder=2)
axa.set_xlim(32, 102); axa.set_ylim(-22, 68)
axa.set_xticks([34, 50, 64, 83, 100]); axa.set_yticks([0, 20, 40, 60])
axa.set_xlabel("wind threshold (kt)", fontsize=FS_AXIS, labelpad=2)
axa.set_ylabel("change in county-days per year\nreaching at least that (%)", fontsize=FS_AXIS,
               labelpad=2, linespacing=1.2)
axa.set_zorder(axw.get_zorder() + 1); axa.patch.set_visible(False)
despine(axa)
axa.legend(handles=[Line2D([], [], color=CW[s], lw=1.2, label="+%.2f K" % DT[s])
                    for s in sorted(DT, key=DT.get)]
           + [Patch(facecolor="#E6E6E6", label="historical count")],
           loc="upper left", fontsize=FS_LEG, handlelength=1.3, labelspacing=.28, borderpad=.28,
           frameon=True, facecolor="white", edgecolor="#D8D8D8",
           framealpha=.94).get_frame().set_linewidth(.3)

# ==================== b — what a knot is worth ====================
axb = fig.add_axes(cell(1, 3))
g = np.linspace(34, 145, 400)


def f(x, c=CO):
    return np.interp(x, KT, c, left=c[0], right=c[-1])


solid = g <= KT[-1]
axb.fill_between(g, f(g, CO - SE), f(g, CO + SE), color=C_DOSE, alpha=.16, lw=0, zorder=2)
axb.plot(g[solid], f(g[solid]), color=C_DOSE, lw=1.4, zorder=4)
axb.plot(g[~solid], f(g[~solid]), color=C_DOSE, lw=1.4, ls=(0, (2.5, 2)), zorder=4)
axb.errorbar(KT, CR, yerr=SE, fmt="o", ms=3.2, color=C_DOSE, markeredgecolor="white",
             markeredgewidth=.5, ecolor=C_DOSE, elinewidth=.7, capsize=1.4, capthick=.7, zorder=5)
axb.set_xlim(30, 152); axb.set_ylim(-2.6, 8.4)       # the lower quarter is the histogram's floor
axb.set_yticks([0, 2, 4, 6, 8]); axb.set_xticks([50, 100, 150])
axb.set_ylabel("whole-event outage effect against\nan unexposed county (log points)",
               fontsize=FS_AXIS, labelpad=2, linespacing=1.2)
axb.set_xlabel("county peak wind (kt)", fontsize=FS_AXIS, labelpad=2)
axb.axhline(0, color=C_GRID, lw=.5, zorder=1)
despine(axb)

F = pd.read_parquet(f"{SR}/tc_flags.parquet")
EDG = np.arange(34, 152, 7.0)
hh, _ = np.histogram(F[F.scen == "historical"].wind_kt.values, bins=EDG)
ff, _ = np.histogram(F[F.scen == HEAD].wind_kt.values, bins=EDG)
dd = ff - hh
axz = axb.twinx()
mid = .5 * (EDG[:-1] + EDG[1:])
axz.bar(mid, dd, width=6.2, color=[C64 if v > 0 else C_BIN for v in dd], lw=0, zorder=1)
axz.axhline(0, color=C_GRID, lw=.5, zorder=1)
axz.set_ylim(-98.2, 700.5)
axz.set_yticks([]); axz.set_xlim(30, 152)
for sp in axz.spines.values():
    sp.set_visible(False)
axb.set_zorder(axz.get_zorder() + 1); axb.patch.set_visible(False)
axb.legend(handles=[
    Line2D([], [], color=C_DOSE, lw=1.4, label="monotone profile"),
    Line2D([], [], color=C_DOSE, lw=0, marker="o", ms=3.0, label="band estimate, ±1 s.e."),
    Line2D([], [], color=C_DOSE, lw=1.4, ls=(0, (2.5, 2)), label="flat above the data"),
    Patch(facecolor=C64, label="county-days gained"),
    Patch(facecolor=C_BIN, label="county-days lost")],
    loc="upper left", fontsize=FS_LEG, handlelength=1.3, labelspacing=.28, borderpad=.28,
    frameon=True, facecolor="white", edgecolor="#D8D8D8",
    framealpha=.94).get_frame().set_linewidth(.3)

# ==================== c — what it adds up to ====================
axc = fig.add_axes(cell(2, 3))
TOT = DZ[HEAD]["dose_pp"]
BARS = ["newly exposed", "exposure lost", "34-50 kt", "50-64 kt", "64-83 kt", "83+ kt"]
BC = ["#C6C6C6", "#8C8C8C", C34, C50, C64, "#5A3A08"]
inc = [DZ["bands"][HEAD][b]["pp"] for b in BARS]
run = np.concatenate(([0.0], np.cumsum(inc)))
for i, b in enumerate(BARS):
    axc.bar(i, abs(inc[i]), bottom=min(run[i], run[i + 1]), width=.62, color=BC[i],
            edgecolor="white", linewidth=.4, zorder=3)
    # a label under a bar whose foot sits just above zero lands on the zero line, which is where
    # the "exposure lost" label fell; the white backing keeps the sign legible without moving it
    axc.text(i, (max(run[i], run[i + 1]) + .04) if inc[i] > 0 else (min(run[i], run[i + 1]) - .04),
             "%+.2f" % inc[i], fontsize=FS_VAL, ha="center",
             va="bottom" if inc[i] > 0 else "top", color="black", zorder=6,
             bbox=dict(boxstyle="square,pad=0.06", facecolor="white", edgecolor="none"))
    if i < len(BARS) - 1:
        axc.plot([i + .31, i + 1 - .31], [run[i + 1]] * 2, color="#BDBDBD", lw=.5, zorder=2)
axc.bar(len(BARS), TOT, width=.62, color=C_DOSE, edgecolor="white", linewidth=.4, zorder=3)
axc.plot([len(BARS)] * 2, [DZ[HEAD]["dose_pp_lo"], DZ[HEAD]["dose_pp_hi"]], color="black", lw=.8,
         zorder=5)
axc.text(len(BARS), DZ[HEAD]["dose_pp_hi"] + .04, "%+.2f" % TOT, fontsize=FS_VAL, ha="center",
         va="bottom", color="black")
bmid = DZ[HEAD]["binary_pp"]
axc.bar(len(BARS) + 1, bmid, width=.62, color=C_BIN, edgecolor="white", linewidth=.4, zorder=3)
axc.text(len(BARS) + 1, bmid + .04, "%+.2f" % bmid, fontsize=FS_VAL, ha="center", va="bottom",
         color="black")
axc.axhline(0, color="black", lw=.6, zorder=2)
axc.set_xticks(range(len(BARS) + 2))
axc.set_xticklabels(BARS + ["all", "binary\nestimator"], fontsize=FS_TICK, rotation=34,
                    ha="right", rotation_mode="anchor", linespacing=1.15)
axc.tick_params(axis="x", length=0, pad=1.5)
axc.set_xlim(-.7, len(BARS) + 1.7); axc.set_ylim(-.75, 2.85)
axc.set_ylabel("change in the tropical-cyclone share of\nnational outage customer-hours (pp)",
               fontsize=FS_AXIS, labelpad=2, linespacing=1.2)
despine(axc)
axc.text(3.5, 2.58, "reached in both periods, %.0f%%"
         % DZ["intensification_share_pct"][HEAD], fontsize=FS_VAL, ha="center", va="bottom",
         color="#5A3A08")
axc.plot([1.75, 5.25], [2.52] * 2, color="#5A3A08", lw=.5, zorder=2)
axc.set_xlabel("component of the projected change", fontsize=FS_AXIS, labelpad=2)

save(fig, "fig6_damage", tight=False, png_dpi=300)
print("baseline %.2f%%   headline %s" % (BASE, HEAD))
print("tail change: 34kt %s   64kt %s   100kt %s"
      % tuple(["  ".join("%+.1f%%" % TL[s].iloc[i] for s in sorted(DT, key=DT.get))
               for i in (0, 30, len(TL) - 1)]))
print("dose %+.2f pp [%+.2f, %+.2f] vs binary %+.2f pp   waterfall sum %+.2f"
      % (TOT, DZ[HEAD]["dose_pp_lo"], DZ[HEAD]["dose_pp_hi"], bmid, sum(inc)))
print("bars: " + "  ".join("%s %+.2f" % (b, v) for b, v in zip(BARS, inc)))
print("county-days per 7 kt, change: %s" % dd.tolist())
