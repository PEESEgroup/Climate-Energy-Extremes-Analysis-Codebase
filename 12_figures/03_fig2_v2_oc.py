"""
FIGURE 2 — From hazard to blackout: two distinct outage channels.

Built at Nature double-column final size (183 mm), never rescaled, so type sizes are literal.
No panel letters, no sub-headings, no titles — the author adds those in the layout program.

LAYOUT (restructured 2026-08-05 on the author's instruction)
  row 1   left  half-width  MAP — the ten landfalls, their tracks, and every county coloured by
                            the strongest wind it took across the ten storms
          right half-width  the model's blind spot, in two stacked cells:
                              top    all nine measurable landfalls as observed-drop vs gap
                                     (the population — nearly linear, and nothing lands in the
                                     large-drop-small-gap corner)
                              bottom Irma and Emily as daily series (the case)
  row 2   three cells       how often the grid actually fails | what a hurricane does | dose

Row 1's map is deliberately CROPPED to the Gulf and Atlantic seaboard. Every landfall in the
identification sits on that arc, so the damage channel says nothing about the rest of the
country — a national frame would imply coverage we do not have.

PALETTE. Adequacy keeps Fig 1's generation-side blue: a generation shortfall is that failure.
Damage gets its own dark gold. Row 2's last two cells are the damage channel, so they are gold.

DO NOT REINTRODUCE (verified against the files 2026-08-05)
  - x6.26 / +526% / "84% attributable" — retracted; live headline is PPML x20.53, share 0.951
  - 9-10 day restoration — the contaminated ref -1 reference; use 13 days. The CSV still holds
    `ref-1_zerofill`, so filtering on `spec == 'ref-7_zerofill'` is mandatory
  - any p below 2/1024 = 0.001953 at G = 10
  - "-44.08 is the trough" — that is the day 0..+3 MEAN; the daily trough is -65.52%
  - "~6 days below -20%" — it is 4 days; 6 is the count below -10%
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

EQ = "/data/equity_cost/analysis"; FX = "/data/audit_orphans/fix"
C_ADEQ, C_DAM, C_OBS = C_VRE, "#A6761D", "#1A1A1A"

W_MM, H_MM = 183.0, 119.0
fig = new_fig(W_MM, H_MM)
def R(x, y, w, h): return [x / W_MM, 1.0 - (y + h) / H_MM, w / W_MM, h / H_MM]
def TX(x, y, s, **k):
    k.setdefault("fontsize", 5.4); return fig.text(x / W_MM, 1.0 - y / H_MM, s, **k)

# ======================= row 1 left — the identification footprint =======================
ev = pd.read_csv(f"{EQ}/did/did_events_v2.csv")
ex = pd.read_parquet(f"{EQ}/did/did_exposure_v2.parquet")
gz = pd.read_csv(f"{EQ}/did/2023_Gaz_counties_national.txt", sep="\t", dtype={"GEOID": str})
gz.columns = [c.strip() for c in gz.columns]
gz["fips"] = gz.GEOID.str.zfill(5)
peak = ex.groupby("fips").exposure_kt.max().rename("kt").reset_index()
cty = gz[["fips", "INTPTLAT", "INTPTLONG"]].merge(peak, on="fips", how="inner")

LON0, LON1, LAT0, LAT1 = -100.5, -69.0, 23.5, 40.5
zm = np.load("/data/datasets/grid/subregion_mask.npz", allow_pickle=True)
mask = zm["subregion_mask"]
zc = np.load("/data/datasets/grid/coordinate.npz")
lat, lon = zc["lat"].astype(float), zc["lon"].astype(float)
ri = np.where((lat >= LAT0) & (lat <= LAT1))[0]
ci = np.where((lon >= LON0) & (lon <= LON1))[0]
sub = mask[ri.min():ri.max() + 1, ci.min():ci.max() + 1]
ext = [lon[ci.min()], lon[ci.max()], lat[ri.min()], lat[ri.max()]]
GLON, GLAT = np.meshgrid(lon[ci.min():ci.max() + 1], lat[ri.min():ri.max() + 1])
# nearest county centroid per land cell — county polygons are not on this box, and at 87 mm the
# difference between a Voronoi fill and true boundaries is below one rendered pixel
tree = cKDTree(np.c_[cty.INTPTLONG.values, cty.INTPTLAT.values])
land = sub > 0
_, idx = tree.query(np.c_[GLON[land], GLAT[land]])
kt = np.full(sub.shape, np.nan)
kt[land] = cty.kt.values[idx]
kt[kt < 34] = np.nan                       # below tropical-storm force: never a treated county

ASP = 1.0 / np.cos(np.deg2rad((LAT0 + LAT1) / 2))
MW_, MY_ = 87.0, 4.0
MH_ = MW_ * (ext[3] - ext[2]) * ASP / (ext[1] - ext[0])
axm = fig.add_axes(R(2.0, MY_, MW_, MH_))
axm.imshow(np.where(land, 1.0, np.nan), origin="lower", extent=ext, aspect="auto",
           cmap=plt.matplotlib.colors.ListedColormap(["#EFEFEF"]), vmin=0, vmax=1)
kn = plt.matplotlib.colors.Normalize(34, 115)
kc = plt.get_cmap("YlOrBr")
axm.imshow(kt, origin="lower", extent=ext, aspect="auto", cmap=kc, norm=kn,
           interpolation="nearest")
# Figure 1's 18 subregions as white hairlines — they cut the wind field without adding ink
axm.contour(np.where(sub > 0, sub, np.nan), levels=np.arange(.5, sub.max() + 1),
            colors="white", linewidths=.35, extent=ext, origin="lower", zorder=5)

# ten tracks, from the 1851-2024 HURDAT2 that covers every storm in the design
H2 = open(f"{EQ}/did/hurdat2_latest.txt", errors="ignore").read().splitlines()
want = {(str(r["name"]).upper(), int(r.year)): pd.Timestamp(r.landfall_date)
        for _, r in ev.iterrows()}
i, ntk, TKN = 0, 0, []
while i < len(H2):
    p = [q.strip() for q in H2[i].split(",")]
    if len(p) >= 3 and p[0].startswith("AL") and p[2].isdigit():
        nm, yr, n = p[1].upper(), int(p[0][4:8]), int(p[2])
        if (nm, yr) in want:
            lf = want[(nm, yr)]; tk = []
            for r_ in H2[i + 1:i + 1 + n]:
                q = [z.strip() for z in r_.split(",")]
                try:
                    la = float(q[4][:-1]) * (1 if q[4][-1] == "N" else -1)
                    lo = float(q[5][:-1]) * (-1 if q[5][-1] == "W" else 1)
                except Exception:
                    continue
                tk.append((lo, la))
            if tk:
                tk = np.array(tk); TKN.append((nm, len(tk)))
                axm.plot(tk[:, 0], tk[:, 1], color="#6E6E6E", lw=.45, alpha=.6, zorder=6)
                ntk += 1
        i += n + 1
    else:
        i += 1
axm.scatter(ev.lf_lon, ev.lf_lat, s=20, marker="v", color="#12314F", zorder=8,
            edgecolor="white", linewidth=.6)
axm.set_xlim(ext[0], ext[1]); axm.set_ylim(ext[2], ext[3]); bare(axm)
axm.legend(handles=[
    Line2D([], [], color="#6E6E6E", lw=.6, alpha=.6, label="storm track"),
    Line2D([], [], marker="v", ls="", color="#12314F", mec="white", mew=.5, ms=3.4,
           label="landfall"),
    Line2D([], [], color="#EFEFEF", lw=4, label="below 34 kt")],
    loc="upper left", fontsize=FS_VAL, handlelength=1.5, labelspacing=.35, borderpad=.28,
    frameon=True, facecolor="white", edgecolor="#D8D8D8", framealpha=.92)
mcb = fig.add_axes(R(9.0, MY_ + MH_ + 3.2, 42.0, 2.0))
cb = fig.colorbar(plt.cm.ScalarMappable(norm=kn, cmap=kc), cax=mcb, orientation="horizontal",
                  extend="max")
cb.set_ticks([34, 64, 96]); cb.set_ticklabels(["34", "64", "96"])
cb.ax.tick_params(labelsize=FS_TICK, width=.4, length=1.6, pad=1.2)
cb.set_label("strongest wind across the ten storms (kt)",
             fontsize=FS_VAL, labelpad=1.6)
cb.outline.set_linewidth(.4)

TX(57.0, MY_ + MH_ + 5.2, "1,535 treated / 3,070 control counties\nG = 10 events, 2016–2022",
   color=C_TXT, va="center", linespacing=1.4)

# ===== row 1 right — the generalisation, and then the one case that shows the shape =====
# Nine small multiples stood here, and the scatter beneath them plotted the SAME nine storms: the
# x of every point was the number printed in its own mini panel and the y was the size of its
# shaded gap. One dataset, two panels, 45% of the row. The scatter is the generalisation, so it
# comes first and full size; Irma follows as the single case that shows what the gap looks like
# in time. This is the layout this figure was specified with before the 3 x 3 grew.
D = pd.read_csv(f"{FX}/tc_daily_series_all.csv")
ST = pd.read_csv(f"{FX}/tc_storm_table_v2.csv"); ST = ST[ST.status == "ok"]
from scipy import stats as _st
lr = _st.linregress(ST.obs_pct, ST.div_pts)
se_band = lr.stderr * _st.t.ppf(.975, len(ST) - 2)

GX, GY = 103.0, MY_ + 0.5
SW, SH1, SH2, VG = 78.0, 27.0, 21.0, 11.0
axs = fig.add_axes(R(GX, GY, SW, SH1))
xg = np.linspace(ST.obs_pct.min() - 3, ST.obs_pct.max() + 3, 40)
axs.plot(xg, lr.intercept + lr.slope * xg, color="#8A6A18", lw=.8, ls=(0, (3, 2)), zorder=2)
axs.fill_between(xg, lr.intercept + (lr.slope - se_band) * xg,
                 lr.intercept + (lr.slope + se_band) * xg,
                 color="#8A6A18", alpha=.11, lw=0, zorder=1)
axs.axhline(0, color=C_GRID, lw=.6, zorder=1)
axs.scatter(ST.obs_pct, ST.div_pts, s=np.sqrt(ST.n_ba) * 7 + 4, color=C_DAM, alpha=.85,
            zorder=3, edgecolor="white", linewidth=.35)
DOWN = {"EMILY", "COLIN"}
for _, r_ in ST.iterrows():
    up = (r_.div_pts >= lr.intercept + lr.slope * r_.obs_pct) and r_.storm not in DOWN
    axs.text(r_.obs_pct, r_.div_pts + (1.1 if up else -1.1), str(r_.storm).title(),
             fontsize=FS_VAL, color=C_TXT, ha="center", va="bottom" if up else "top")
axs.set_xlim(-49, 13); axs.set_ylim(-30, 8)
axs.set_xlabel("observed demand drop (%)", fontsize=FS_AXIS, labelpad=2)
axs.set_ylabel("observed \u2212 expected (pts)", fontsize=FS_AXIS, labelpad=2)
despine(axs)
axs.text(0.985, 0.05, "slope %.2f (95%% CI \u00b1%.2f)\n$p$ = %.4f,  n = %d"
         % (lr.slope, se_band, lr.pvalue, len(ST)), transform=axs.transAxes,
         fontsize=FS_VAL, color="#8A6A18", ha="right", va="bottom", linespacing=1.25)

axi = fig.add_axes(R(GX, GY + SH1 + VG, SW, SH2))
d = D[D.storm == "IRMA"].sort_values("day_rel")
axi.fill_between(d.day_rel, d.obs_anom_pct, d.mod_anom_pct, color=C_OBS, alpha=.14, lw=0, zorder=2)
axi.plot(d.day_rel, d.mod_anom_pct, color=C_VRE, lw=1.0, ls=(0, (2.2, 1.4)), zorder=3)
axi.plot(d.day_rel, d.obs_anom_pct, color=C_OBS, lw=1.2, zorder=4)
axi.axvline(0, color="black", lw=.5, ls=(0, (2.4, 2.4)), zorder=1)
axi.axhline(0, color=C_GRID, lw=.5, zorder=1)
axi.set_xlim(-14.5, 10.5); axi.set_ylim(-72, 26)
axi.set_xticks([-14, -7, 0, 7]); axi.set_yticks([-60, -30, 0])
axi.set_xlabel("days from landfall", fontsize=FS_AXIS, labelpad=2)
axi.set_ylabel("demand anomaly (%)", fontsize=FS_AXIS, labelpad=2)
axi.text(-13.5, -66, "Irma, %+.0f%%" % float(ST[ST.storm == "IRMA"].obs_pct.iloc[0]),
         fontsize=FS_VAL, va="bottom", ha="left", color="black")
despine(axi)
axi.legend(handles=[
    Line2D([], [], color=C_OBS, lw=1.2, label="observed"),
    Line2D([], [], color=C_VRE, lw=1.0, ls=(0, (2.2, 1.4)), label="weather-expected"),
    Patch(facecolor=C_OBS, alpha=.14, label="unserved demand")],
    loc="lower right", bbox_to_anchor=(0.978, 0.022), fontsize=FS_VAL, handlelength=1.5,
    labelspacing=.28, borderpad=.28, frameon=True, facecolor="white",
    edgecolor="#D8D8D8").get_frame().set_linewidth(.3)

# ============ row 2, cell 1 — the two channels, on an absolute scale ============
# This was four rate ratios with confidence intervals, and a ratio hides the floor it stands on.
# The floors are not comparable: an ordinary day carries a 0.18% chance of an outage event in the
# adequacy channel and 0.81% in the damage channel, so "13x against 1.4x" is not "nine times
# worse" - the damage channel is already smouldering on a normal day. Both probabilities are on
# the page now and the ratio is the slope between them.
RR = json.load(open("/data/enso/r1_causal/outage_bridge_v3.json"))
PREF = RR
SY, SH = 71.0, 41.0
axa = fig.add_axes(R(13.0, SY, 47.0, SH))
ROWS = [("Adequacy", "adequacy", C_ADEQ, "stress99", "99th pct"),
        ("Adequacy", "adequacy", C_ADEQ, "stress95", "95th pct"),
        ("Damage", "damage", C_DAM, "stress99", "99th pct"),
        ("Damage", "damage", C_DAM, "stress95", "95th pct")]
yy = [3, 2, 1, 0]
TCK = []
for i, (lab, ch, col, st, sub) in enumerate(ROWS):
    v = PREF[st][ch]
    p0 = 100 * v["p_event_given_nostress"]; p1 = 100 * v["p_event_given_stress"]
    axa.plot([p0, p1], [yy[i]] * 2, color=col, lw=1.1, zorder=3, solid_capstyle="round")
    axa.scatter([p0], [yy[i]], s=17, facecolor="white", edgecolor=col, linewidth=.9, zorder=4)
    axa.scatter([p1], [yy[i]], s=19, color=col, edgecolor="white", linewidth=.4, zorder=5)
    axa.text(p1 + .09, yy[i], "%.2f×" % v["rr"], fontsize=FS_VAL, va="center", ha="left",
             color=col)
    TCK.append(sub)
    if i in (0, 2):
        axa.text(p0 - .10, yy[i] + .52, lab, fontsize=FS_VAL, color=col, ha="left", va="center")
axa.set_yticks(yy); axa.set_yticklabels(TCK, fontsize=FS_TICK)
axa.tick_params(axis="y", length=0, pad=2.5)
axa.set_xlim(0, 3.45); axa.set_ylim(-1.25, 3.95)
axa.set_xlabel("county-days with an outage event (%)", fontsize=FS_AXIS, labelpad=2)
despine(axa)
axa.legend(handles=[
    Line2D([], [], marker="o", ls="", mfc="white", mec="#666666", mew=.9, ms=3.4,
           label="ordinary day"),
    Line2D([], [], marker="o", ls="", color="#666666", ms=3.4, label="net-load stress day")],
    loc="lower left", bbox_to_anchor=(0.022, 0.022), fontsize=FS_VAL,
    handlelength=1.2, labelspacing=.30, borderpad=.26, handletextpad=.4,
    frameon=True, facecolor="white", edgecolor="#D8D8D8", framealpha=.92)

# ============== row 2, cell 2 — is the design credible? the leads say so ==============
# The scatter that stood here moved up to row 1, beside the case it generalises. The cell now
# carries the identification, which was nowhere on the page: 20 lead coefficients against a day -7
# reference, then the lags. Day -1 is NOT flat (+0.71, s.e. 0.08) - anticipatory outages
# and pre-landfall de-energisation - which is why the published specification drops it from both
# periods rather than leaving it in the comparison.
ES = pd.read_csv(f"{EQ}/did/eventstudy_coefs_ref7.csv").sort_values("day_rel")
axz = fig.add_axes(R(74.0, SY, 47.0, SH))
pre = ES[ES.day_rel < 0]; post = ES[ES.day_rel >= 0]
axz.axhline(0, color=C_GRID, lw=.6, zorder=1)
axz.axvline(-0.5, color="black", lw=.5, ls=(0, (2.4, 2.4)), zorder=2)
for g_, c_ in [(pre, "#9A9A9A"), (post, C_DAM)]:
    axz.fill_between(g_.day_rel, g_.coef - 1.96 * g_.se, g_.coef + 1.96 * g_.se, color=c_,
                     alpha=.18, lw=0, zorder=3)
    axz.plot(g_.day_rel, g_.coef, color=c_, lw=1.0, zorder=4)
m1 = ES[ES.day_rel == -1]
axz.scatter(m1.day_rel, m1.coef, s=16, color="#8A6A18", edgecolor="white", linewidth=.4, zorder=6)
axz.text(-2.6, float(m1.coef.iloc[0]) + .25, "day −1 already\n+%.2f" % float(m1.coef.iloc[0]),
         fontsize=FS_VAL, ha="right", va="bottom", color="#8A6A18", linespacing=1.25)
axz.set_xlim(-21.5, 21.5); axz.set_ylim(-.85, 7.3); axz.set_xticks([-21, -7, 0, 7, 21]); axz.set_yticks([0, 2, 4, 6])
axz.set_xlabel("days from landfall", fontsize=FS_AXIS, labelpad=2)
axz.set_ylabel("treated − control, log customer-hours", fontsize=FS_AXIS, labelpad=2)
despine(axz)
axz.legend(handles=[
    Line2D([], [], color="#9A9A9A", lw=1.0, label="before landfall (ref −7)"),
    Line2D([], [], color=C_DAM, lw=1.0, label="after"),
    Patch(facecolor="#9A9A9A", alpha=.25, label="95% CI")],
    loc="upper right", bbox_to_anchor=(0.978, 0.978), fontsize=FS_VAL, handlelength=1.4,
    labelspacing=.28, borderpad=.26,
    frameon=True, facecolor="white", edgecolor="#D8D8D8",
    framealpha=.92).get_frame().set_linewidth(.3)

# ================= row 2, cell 3 — it scales with wind, and here is the data =================
# Five estimated coefficients and a fitted curve stood here, which is the same object Figure 6 now
# draws in its applied form. What this panel can show and that one cannot is the DATA. The dose
# regression is two-way fixed-effect (unit = event x county, and event x day), so its identifying
# variation is each county-event's own post-minus-pre change net of the unexposed counties in the
# SAME storm. That per-unit difference-in-differences is computable directly (/data/17_binscatter.py)
# and its bin means land on the published coefficients: 0.990 against 0.995, 1.839 against 1.870,
# 3.717 against 3.609. The top two bins run higher in the cloud than in the regression (5.57 and
# 7.01 against 5.07 and 5.74) because the regression weights county-events differently and those
# bins hold 56 and 13 of them; the published coefficients are what is drawn on top.
DRJ = json.load(open(f"{EQ}/did/did_results_v2.json"))["dose_response"]
BN = DRJ["binned"]
UD = pd.read_csv(f"{EQ}/did/dose_unit_did.csv")
axe = fig.add_axes(R(135.0, SY, 47.0, SH))
axe.add_patch(Rectangle((0, -3.2), 34, 12.4, facecolor="#F2F2F2", edgecolor="none", zorder=1))
axe.scatter(UD.exposure_kt, UD.did, s=1.6, color=C_DAM, alpha=.16, linewidth=0, zorder=2)
axe.axhline(0, color=C_GRID, lw=.6, zorder=1)
xs = [BN[k]["mean_kt"] for k in BN]; ys = [BN[k]["coef"] for k in BN]
for k in BN:
    v = BN[k]
    axe.plot([v["mean_kt"]] * 2, [v["lo"], v["hi"]], color="#4D3208", lw=1.1, zorder=5)
axe.plot(xs, ys, color="#4D3208", lw=1.0, zorder=5)
axe.scatter(xs, ys, s=18, color="#4D3208", zorder=6, edgecolor="white", linewidth=.5)
for k, ha_ in [(list(BN)[0], "center"), (list(BN)[-1], "right")]:
    axe.text(BN[k]["mean_kt"], -2.95, "n = %d" % BN[k]["n_units"], fontsize=FS_VAL,
             ha=ha_, va="bottom", color="#6E6E6E")
axe.set_xlim(0, 108); axe.set_ylim(-3.2, 9.2); axe.set_xticks([0, 34, 60, 85, 105])
axe.set_xlabel("maximum sustained wind (kt)", fontsize=FS_AXIS, labelpad=2)
axe.set_ylabel("county-event effect, log customer-hours", fontsize=FS_AXIS, labelpad=2)
despine(axe)
axe.legend(handles=[
    Line2D([], [], marker="o", ls="", color=C_DAM, alpha=.4, ms=2.0,
           label="county-event (n = %d)" % len(UD)),
    Line2D([], [], marker="o", ls="", color="#4D3208", ms=3.2, label="estimate, 95% CI"),
    Line2D([], [], color="#F2F2F2", lw=4, label="no support")],
    loc="upper left", fontsize=FS_VAL, handlelength=1.5, labelspacing=.30, borderpad=.28,
    frameon=True, facecolor="white", edgecolor="#D8D8D8",
    framealpha=.92).get_frame().set_linewidth(.3)

save(fig, "fig2_v2_ourchain", tight=False)
print("   track points:", TKN)
print("\nmap   tracks drawn %d of 10   counties >=34 kt %d of %d   max %.0f kt"
      % (ntk, int((cty.kt >= 34).sum()), len(cty), cty.kt.max()))
print("scatter slope %.3f +- %.3f p=%.5f over %d storms | binscatter %d county-events"
      % (lr.slope, se_band, lr.pvalue, len(ST), len(UD)))
for st in ["stress95", "stress99"]:
    for ch in ["adequacy", "damage"]:
        v = PREF[st][ch]
        print("%-9s %-9s ordinary %.2f%% -> stress %.2f%%  (%.2fx)"
              % (st, ch, 100 * v["p_event_given_nostress"], 100 * v["p_event_given_stress"],
                 v["rr"]))
