"""
FIGURE 4 v2 — who carries the burden, what moves with it, and where the money goes.

Rebuilt on the attribution of Figure 3b, and on tercile contrasts estimated inside the same
Poisson panel rather than in a separate event study.

  a  the five county traits, burden share against population share
  b  the same five traits, the outage gap between the worst and the best third when a severe
     convective storm arrives, with the pre-event window as the test
  c  undergrounding, a property of the distribution system and not of the county. Read as a
     CHRONIC difference, not a storm effect. Measured on the current panel, 79% of the event-day
     gap between the least and the most undergrounded third is already standing 14 to 8 days
     before the storm; the lead block is more significant than the event day, z +2.43 against
     +1.95; the event day alone does not clear even an uncorrected 1.96, let alone this figure's
     Bonferroni bar; the response is not monotone, the middle third sitting below the most
     undergrounded; and the largest block is day +7 to +14, the least storm-like part of the
     window. What the panel shows is that counties with more overhead plant carry a higher
     baseline outage level, visible in the window around convective storms but not caused by
     them. That is the same finding as the burden split in (a): the inequity is exposure plus
     chronic grid quality, not a difference in how a storm is survived.
  d  every investor-owned utility's standing underground share against what it builds
  e  federal discretionary pre-disaster funding against the attributable burden

WHY ONLY ONE HAZARD IS DRAWN IN (b). The tercile interaction is estimated for every hazard that
passes the screen in 04_chk3.py, and the multiplicity bar is counted over all of them. Panel (b)
draws the severe convective column alone, because that is the column the caption names. The
hurricane column fails its own pre-event test for four of the five traits: the worst and best
terciles already differ 14 to 8 days before landfall. Extending the tropical-cyclone tail to day
+35 does not remove it, so the failure is not event overlap. The columns that are estimated but
not drawn are reported in the Supplementary Information and nothing is claimed from them.
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
from scipy import stats as st

AN = "/data/equity_cost/analysis"
AT = "%s/attrib" % AN
W_MM, H_MM = 183.0, 122.0
fig = new_fig(W_MM, H_MM)
def R(x, y, w, h): return [x / W_MM, 1.0 - (y + h) / H_MM, w / W_MM, h / H_MM]

FA = pd.read_csv("%s/fig4a.csv" % AT)
FB = pd.read_csv("%s/fig4b.csv" % AT)
FC = pd.read_csv("%s/fig4c.csv" % AT)
A = pd.read_csv("%s/r4_panelA_utilities.csv" % AN)
B = pd.read_csv("%s/fig4e_states.csv" % AT)
J = json.load(open("%s/r4_ledger.json" % AN))
C_BUR, C_ALL = "#B2182B", "#9A9A9A"
C_SIG, C_NUL, C_PLA = "#5E3C99", "#C4C4C4", "#4D4D4D"
C_LEAST, C_MID, C_MOST = "#8C510A", "#C8C8C8", "#35978F"
# THE BAR IS READ, NOT RESTATED. This was a bare ZB = 2.638 that nothing below ever referenced,
# while 16_fig4data.py decided `clears` with a different number. Two thresholds sat in the tree and
# the figure obeyed the one it did not name. 16_fig4data.py now records the bar it applied, and the
# family it counted, in fig4b.csv, so the drawn colors and the reported bar cannot part.
ZB = float(FB.zb.iloc[0])
N_FAM = int(FB.n_family.iloc[0])
PANEL_HAZ = str(FB.panel_hazard.iloc[0])

# ================================ a — who carries the burden ================================
axa = fig.add_axes(R(20.0, 8.0, 40.0, 40.0))
FA = FA.iloc[::-1].reset_index(drop=True)
yv = np.arange(len(FA))
for i, r in FA.iterrows():
    axa.barh(yv[i] + .16, r.ratio, height=.34, color=C_BUR, alpha=.88, zorder=3)
    axa.text(r.ratio + .07, yv[i] + .16, "%.2f" % r.ratio, fontsize=FS_VAL, va="center",
             ha="left", color="black")
    axa.scatter([r.allcause_ratio], [yv[i] - .22], s=13, facecolor="white", edgecolor=C_ALL,
                linewidth=.8, zorder=4)
    # the same ratio with the two states that dominate the burden removed
    axa.scatter([r.ratio_ex], [yv[i] + .16], s=16, marker="|", color="#3A2606", linewidth=1.0,
                zorder=5)
axa.axvline(1.0, color="black", lw=.6, ls=(0, (3, 3)), zorder=2)
axa.set_yticks(yv); axa.set_yticklabels(FA.trait, fontsize=FS_TICK)
axa.tick_params(axis="y", length=0, pad=2.5)
axa.set_xlim(0, 4.0); axa.set_xticks([0, 1, 2, 3, 4])
axa.set_ylim(-1.75, len(FA) - 0.4)
axa.set_xlabel("share of the burden divided by share of population",
               fontsize=FS_AXIS, labelpad=2)
despine(axa)
axa.legend(handles=[
    Line2D([], [], marker="s", ls="", color=C_BUR, alpha=.88, ms=3.4, label="weather-attributable"),
    Line2D([], [], marker="o", ls="", mfc="white", mec=C_ALL, mew=.8, ms=3.0, label="all-cause"),
    Line2D([], [], marker="|", ls="", color="#3A2606", mew=1.0, ms=4.5,
           label="without Florida and Louisiana")],
    loc="lower left", bbox_to_anchor=(0.012, 0.012), fontsize=FS_VAL, handlelength=1.0,
    labelspacing=.28, borderpad=.26, frameon=True, facecolor="white", edgecolor="#D8D8D8",
    framealpha=.94).get_frame().set_linewidth(.3)

# ============== b — and whether the same storm costs them more ==============
axb = fig.add_axes(R(80.0, 8.0, 40.0, 40.0))
FB = FB.iloc[::-1].reset_index(drop=True)
yv = np.arange(len(FB))
for i, r in FB.iterrows():
    c_ = C_SIG if r.clears else C_NUL
    axb.plot([np.exp(r.gap - 1.96 * r.gap_se), np.exp(r.gap + 1.96 * r.gap_se)],
             [yv[i] + .16] * 2, color=c_, lw=1.4, zorder=3, solid_capstyle="round")
    axb.scatter([np.exp(r.gap)], [yv[i] + .16], s=17, color=c_, edgecolor="white", linewidth=.4,
                zorder=4)
    axb.text(np.exp(r.gap + 1.96 * r.gap_se) * 1.03, yv[i] + .16, "%.2f×" % r.mult,
             fontsize=FS_VAL, va="center", ha="left", color="black" if r.clears else "#8A8A8A")
    axb.scatter([np.exp(r.lead)], [yv[i] - .22], s=11, facecolor="white", edgecolor=C_PLA,
                linewidth=.7, zorder=4)
axb.axvline(1.0, color="black", lw=.6, ls=(0, (3, 3)), zorder=2)
axb.set_yticks(yv); axb.set_yticklabels(FB.trait, fontsize=FS_TICK)
axb.tick_params(axis="y", length=0, pad=2.5)
axb.set_xscale("log"); axb.set_xlim(0.60, 2.45)
axb.set_xticks([0.7, 1, 1.5, 2]); axb.set_xticklabels(["0.7", "1", "1.5", "2"], fontsize=FS_TICK)
axb.xaxis.set_minor_locator(plt.matplotlib.ticker.NullLocator())
axb.set_ylim(-0.9, len(FB) + 0.35)
axb.set_xlabel("worst third against best third, on the day of a\nsevere convective storm",
               fontsize=FS_AXIS, labelpad=2, linespacing=1.25)
despine(axb)
axb.legend(handles=[
    Line2D([], [], marker="o", ls="-", color=C_SIG, ms=3.2, lw=1.4, label="clears the bar"),
    Line2D([], [], marker="o", ls="-", color=C_NUL, ms=3.2, lw=1.4, label="does not"),
    Line2D([], [], marker="o", ls="", mfc="white", mec=C_PLA, mew=.7, ms=2.8,
           label="14 to 8 days before")],
    loc="upper left", bbox_to_anchor=(0.012, 0.988), fontsize=FS_VAL, handlelength=1.2,
    labelspacing=.26, borderpad=.26, frameon=True, facecolor="white", edgecolor="#D8D8D8",
    framealpha=.94).get_frame().set_linewidth(.3)

# ============== c — the grid form, kept apart because it is an association ==============
# This panel used to draw the least-against-most RATIO across the five lag blocks, which says
# whether the gap is there but never how large the burden is. The author asked for the earlier
# picture back: absolute hours per customer, split by where in the event window they accrue. That
# earlier picture came from a retired linear event study, so it is rebuilt here on the panel the
# paper actually uses, att = y (1 - exp(-x'beta)) over each block's convective columns, divided by
# customers and by events. The two agree on the shape, including the middle tercile sitting below
# the most-undergrounded one: 0.286 / 0.076 / 0.130 here against 0.450 / 0.093 / 0.130 there.
FH = pd.read_csv("%s/fig4c_hours.csv" % AT)
axc = fig.add_axes(R(140.0, 8.0, 38.0, 40.0))
SEG = [("impact", "day 0-1", "#DFC27D"), ("restore", "day 2-6", "#BF812D"),
       ("tail", "day 7-14", "#8C510A")]
GT = ["least undergrounded", "middle", "most undergrounded"]
GTL = ["least\nundergrounded", "middle", "most\nundergrounded"]
yg = np.arange(len(GT))[::-1]
axc.axvline(0, color="black", lw=.6, ls=(0, (3, 3)), zorder=1)
P = FH.pivot(index="tercile", columns="block", values="hours_per_customer_per_event")
# the 95% interval on the stacked total, from 400 draws of the convective coefficients
P2 = FH.drop_duplicates("tercile").set_index("tercile")[["total_lo", "total_hi"]]
for i, nm in enumerate(GT):
    left = 0.0
    for key, _lab, c_ in SEG:
        w_ = float(P.loc[nm, key])
        axc.barh(yg[i] + .16, w_, left=left, height=.34, color=c_, zorder=3,
                 edgecolor="white", linewidth=.35)
        left += w_
    lo_, hi_ = float(P2.loc[nm, "total_lo"]), float(P2.loc[nm, "total_hi"])
    axc.plot([lo_, hi_], [yg[i] + .16] * 2, color="#5A3A08", lw=.8, zorder=5,
             solid_capstyle="butt")
    axc.text(hi_ + .012, yg[i] + .16, "%.2f" % left, fontsize=FS_VAL, va="center", ha="left",
             color="black")
    # the placebo, on the same axis and in the same units: the paper's pre-event block
    axc.scatter([float(P.loc[nm, "lead"])], [yg[i] - .20], s=11, facecolor="white",
                edgecolor=C_PLA, linewidth=.7, zorder=5)
axc.set_yticks(yg); axc.set_yticklabels(GTL, fontsize=FS_TICK, linespacing=1.15)
axc.tick_params(axis="y", length=0, pad=2.5)
axc.set_ylim(-1.30, len(GT) - 0.35); axc.set_xlim(-0.055, 0.40)
axc.set_xticks([0, 0.1, 0.2, 0.3])
axc.set_xticklabels(["0", "0.1", "0.2", "0.3"], fontsize=FS_TICK)
axc.set_xlabel("hours per customer, per event\n(a chronic gap, largely present before the storm)",
               fontsize=FS_AXIS, labelpad=2, linespacing=1.2)
despine(axc)
axc.legend(handles=[Patch(facecolor=c_, label=l_) for _k, l_, c_ in SEG]
           + [Line2D([], [], marker="o", ls="", mfc="white", mec=C_PLA, mew=.7, ms=3.0,
                     label="14 to 8 days before, the same gap already standing")],
           loc="lower right", bbox_to_anchor=(1.02, -0.02), ncol=2, fontsize=FS_VAL,
           handlelength=1.1, labelspacing=.26, columnspacing=.8, borderpad=.26, frameon=True,
           facecolor="white", edgecolor="#D8D8D8",
           framealpha=.94).get_frame().set_linewidth(.3)

# ============== d — the grid is being replicated ==============
PY2, PH2 = 62.0, 46.0
axd = fig.add_axes(R(20.0, PY2, 60.0, PH2))
lim = (0.02, 0.80)
axd.plot(lim, lim, color="#4D4D4D", lw=.8, ls=(0, (3.5, 2.5)), zorder=2)
sz = 4.0 + 34.0 * (A.plant / A.plant.max()) ** 0.45
for t, c_, z_ in [("middle", C_MID, 3), ("most", C_MOST, 4), ("least", C_LEAST, 5)]:
    d = A[A.terc == t]
    axd.scatter(d.ug, d.ugadd, s=sz[d.index], facecolor=c_, edgecolor="white", linewidth=.35,
                alpha=.92, zorder=z_)
axd.set_xlim(*lim); axd.set_ylim(*lim)
axd.set_xticks([0, .2, .4, .6, .8]); axd.set_yticks([0, .2, .4, .6, .8])
axd.set_xlabel("underground share of the plant already standing", fontsize=FS_AXIS, labelpad=2)
axd.set_ylabel("underground share of everything\nbuilt, 2014 to 2023", fontsize=FS_AXIS,
               labelpad=2, linespacing=1.25)
despine(axd)
axd.text(0.62, 0.645, "reproduces the existing mix", fontsize=FS_VAL, color="#4D4D4D",
         rotation=45, rotation_mode="anchor", ha="center", va="bottom", zorder=6)
PA = J["panelA"]
axd.text(0.975, 0.055, "median %.3f built, %.3f standing\n%.0f%% above the line;  least third "
         "%.3f / %.3f" % (PA["add_p10_50_90"][1], PA["stock_p10_50_90"][1],
                          100 * PA["frac_rising"], 0.163, 0.199),
         transform=axd.transAxes, fontsize=FS_VAL, ha="right", va="bottom", color="#4D4D4D",
         linespacing=1.35)
axd.legend(handles=[
    Line2D([], [], marker="o", ls="", color=C_LEAST, ms=3.6, label="least undergrounded"),
    Line2D([], [], marker="o", ls="", color=C_MID, ms=3.6, label="middle"),
    Line2D([], [], marker="o", ls="", color=C_MOST, ms=3.6, label="most undergrounded"),
    Line2D([], [], marker="o", ls="", mfc="none", mec="#8A8A8A", mew=.6, ms=5.2,
           label="area: plant")],
    loc="upper left", bbox_to_anchor=(0.022, 0.978), fontsize=FS_VAL, handlelength=1.0,
    labelspacing=.3, borderpad=.28, frameon=True, facecolor="white", edgecolor="#D8D8D8",
    framealpha=.94).get_frame().set_linewidth(.3)

# ============== e — and the money does not follow the burden ==============
axe = fig.add_axes(R(104.0, PY2, 74.0, PH2))
B = B.dropna(subset=["proactive_noncirc_per_cust", "att_rate"])
zero = B.att_rate <= 0
axe.axvline(B.proactive_noncirc_per_cust.median(), color="#BFBFBF", lw=.6, ls=(0, (3, 3)), zorder=1)
axe.axhline(B.att_rate.median(), color="#BFBFBF", lw=.6, ls=(0, (3, 3)), zorder=1)
axe.scatter(B[~zero].proactive_noncirc_per_cust, B[~zero].att_rate, s=13, facecolor=C_SIG,
            edgecolor="white", linewidth=.35, alpha=.9, zorder=3)
axe.scatter(B[zero].proactive_noncirc_per_cust, B[zero].att_rate, s=15, facecolor="none",
            edgecolor=C_SIG, linewidth=.7, alpha=.9, zorder=3)
la = B[B.state == "LA"]
if len(la):
    axe.scatter(la.proactive_noncirc_per_cust, la.att_rate, s=22, facecolor="#B2182B",
                edgecolor="white", linewidth=.5, zorder=4)
    axe.text(float(la.proactive_noncirc_per_cust.iloc[0]) * 0.80, float(la.att_rate.iloc[0]),
             "Louisiana", fontsize=FS_VAL, ha="right", va="center", color="#B2182B",
             clip_on=True)
m = B.state != "LA"
r1 = st.pearsonr(np.log10(B.proactive_noncirc_per_cust), B.att_rate)
r2 = st.pearsonr(np.log10(B[m].proactive_noncirc_per_cust), B[m].att_rate)
s1 = st.spearmanr(B.proactive_noncirc_per_cust, B.att_rate)
s2 = st.spearmanr(B.proactive_noncirc_per_cust, B.obs / B.cust)
FED = float((B.proactive_noncirc_per_cust * B.cust).sum())
UTL = float(A.addtot.sum())
axe.text(0.025, 0.985,
         "Spearman %+.2f (p %.2f) against the attributable burden\n"
         "Spearman %+.2f (p %.2f) against all-cause outage\n"
         "Pearson on log funding %+.2f, %+.2f without Louisiana\n"
         "\\$%.1f bn, %.1f%% of these utilities' own \\$%.0f bn"
         % (s1[0], s1[1], s2[0], s2[1], r1[0], r2[0], FED / 1e9, 100 * FED / UTL, UTL / 1e9),
         transform=axe.transAxes, fontsize=FS_VAL, ha="left", va="top", color="#4D4D4D",
         linespacing=1.4)
axe.set_xscale("log"); axe.set_xlim(0.12, 480); axe.set_ylim(-9, 140)
axe.set_xticks([0.3, 1, 3, 10, 30, 100, 300])
axe.set_xticklabels(["0.3", "1", "3", "10", "30", "100", "300"], fontsize=FS_TICK)
axe.set_xlabel("federal discretionary pre-disaster funding, $ per customer, 2014 to 2023",
               fontsize=FS_AXIS, labelpad=2)
axe.set_ylabel("attributable outage, hours\nper customer, 2015 to 2022", fontsize=FS_AXIS,
               labelpad=2, linespacing=1.25)
despine(axe)
axe.text(0.025, 0.02, "open markers: states with no attributable outage", transform=axe.transAxes,
         fontsize=FS_VAL, ha="left", va="bottom", color="#4D4D4D")

save(fig, "fig4_v2", tight=False, png_dpi=300)
print("page %.0f x %.0f mm" % (W_MM, H_MM))
print("(a) ratios %s" % dict(zip(FA.trait, FA.ratio.round(2))))
print("(a) without FL and LA %s" % dict(zip(FA.trait, FA.ratio_ex.round(2))))
print("(b) hazard drawn %s, Bonferroni family %d tests, bar |z| > %.3f"
      % (PANEL_HAZ, N_FAM, ZB))
print("(b) clears the bar: %s" % list(FB[FB.clears].trait))
print("(c) undergrounding gap by block: %s" % dict(zip(FC.block, np.exp(FC.gap).round(2))))
print("(e) states %d, zero-burden %d, Spearman %+.3f (p %.3f), all-cause %+.3f (p %.3f)"
      % (len(B), int(zero.sum()), s1[0], s1[1], s2[0], s2[1]))
print("    federal $%.2f bn against utility $%.0f bn = %.2f%%" % (FED / 1e9, UTL / 1e9, 100 * FED / UTL))
