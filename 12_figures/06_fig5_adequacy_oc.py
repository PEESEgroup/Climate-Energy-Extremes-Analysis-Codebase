"""
FIGURE 5 — future adequacy across all 32 realizations.

LAYOUT. Every panel is placed by dividing the page into equal cells and drawing into one, rather
than by hand-picked millimetre boxes. Hand placement is what produced the uneven white margins in
earlier versions: each panel had been sized to its own content instead of to a shared grid.

  row 1  the map | the 24 peak deltas as a grouped matrix
  row 2  year by year | the policy route | the spread of the annual peak
  row 3  every stress event as an object | the hazards themselves | variance decomposition

ENCODING, the same in every panel where the factor is the subject:
  climate realisation -> position (a row in the matrix, a row in the policy panel)
  demand growth (SSP) -> blue / orange
  VRE policy          -> grey / green / purple

THE FINDING THE FIGURE IS BUILT ON. Splitting the four TGW realisations into an RCP axis and a
cooler/hotter axis shows the RCP label is a DEPLOYMENT axis — rcp45 carries 1.98x the renewable
fleet of rcp85 — while cooler against hotter, the only contrast that changes the weather at a
fixed fleet, explains 0–1% of every adequacy metric.

DEFINITION. Everything is on the NO-HYDRO net load, `net = load - (solar + wind + offshore)`,
which is the definition Figure 1 and the whole of R1 are built on. Earlier versions used
`net_hydro_btm`, which additionally subtracts hydro and behind-the-meter rooftop PV and sits about
30 GW lower, so the same historical period appeared 30 GW below Figure 1's level. The delta
results are unchanged by the switch (hydro cancels on both sides); the levels move.

THE HISTORICAL BASELINE. HIST_PK is the mean annual maximum over the reference period that
baseline.py defines, the last ten years of the record, and it is 645.55 GW. It is not the mean over
all forty years. The load product is annually anchored and grows 1.82x across the record, so a
forty-year mean annual maximum is 535.06 GW, a level the modern grid passes routinely, and every
delta on this page would be inflated by 22%. The ten-year figure sits within 1.1% of the 648.4 GW
that cerf_out/R4_NETLOAD_RESULTS.md validates against EIA-930.

UNITS. The net-load arrays are in MW and are divided by 1000 when plotted; HIST_PK is already in
GW. An earlier version divided the reference line twice and drew the historical baseline on zero.
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
import json, glob, os
import sys
import os as _os_rp
for _rp in ("04_demand_model", "07_hazard_calendar", "09_outage_attribution",
            "02_downscale_wind", "12_figures"):
    _ap = _os_rp.path.abspath(_os_rp.path.join(
        _os_rp.path.dirname(_os_rp.path.abspath(__file__)), "..", _rp))
    if _os_rp.path.isdir(_ap) and _ap not in sys.path:
        sys.path.insert(0, _ap)
import hazard_defs as HD
import numpy as np, pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy.spatial import cKDTree

RN = "/data/tell_pred/future/netload_ourchain"
W_MM, H_MM = 183.0, 167.0
fig = new_fig(W_MM, H_MM)
def R(x, y, w, h): return [x / W_MM, 1.0 - (y + h) / H_MM, w / W_MM, h / H_MM]

# Four sizes only. Earlier versions carried eight different hardcoded values between 4.2 and 5.6,
# which reads as sloppiness rather than hierarchy. FS_AXIS/FS_TICK/FS_LABEL come from figstyle.

# ---------------------------------------------------------------- layout grid
PAGE_L, PAGE_R, PAGE_T, PAGE_B = 1.5, 4.0, 3.0, 3.0   # right margin holds the last tick label
GUT_X, GUT_Y = 3.0, 4.5
PAD_L, PAD_B = 12.0, 10.5
ROW_H = [56.0, 48.0, 48.0]


def band(r):
    return PAGE_T + sum(ROW_H[:r]) + GUT_Y * r, ROW_H[r]


def cell(r, c, n, pad=True):
    y0, h = band(r)
    w = (W_MM - PAGE_L - PAGE_R - GUT_X * (n - 1)) / n
    x0 = PAGE_L + c * (w + GUT_X)
    return R(x0, y0, w, h) if not pad else R(x0 + PAD_L, y0, w - PAD_L, h - PAD_B)


S = pd.read_csv(f"{RN}/R4_OURCHAIN_SUMMARY.csv")
DEC = json.load(open(f"{RN}/r5_decomp4_ourchain.json"))
AN = pd.read_csv(f"{RN}/r5_annual_ourchain.csv")
HIST_PK = float(S.hist_meanannmax.iloc[0])
C_SSP = {"ssp3": "#2C7FB8", "ssp5": "#D95F02"}
C_POL = {"NoPolicy": "#9A9A9A", "Ordinances": "#D95F02", "IRA": "#1B7837", "OBBBA": "#762A83"}
VLAB = {"NoPolicy": "no siting policy", "Ordinances": "+ local ordinances",
        "IRA": "+ RPS/CES incentive", "OBBBA": "+ OBBBA capacity cut"}
ORDP = ["NoPolicy", "Ordinances", "IRA", "OBBBA"]
# the keys are the variant names in the data; the page gets these
DISP = {"NoPolicy": "no policy", "Ordinances": "ordinances", "IRA": "IRA",
        "OBBBA": "OBBBA"}
ORDC = ["rcp45cooler", "rcp45hotter", "rcp85cooler", "rcp85hotter"]

# ============================ row 1a — where it lands ============================
# This map used to fill 18 subregions with the change in peak net load, which the matrix beside it,
# the trajectories, the violins and the policy panel all also carry — the same quantity five times.
# The fill is now the one county-level quantity in the figure: how many hours a year a county's own
# load sits above its OWN historical 99.9th percentile, which is 8.8 h/yr in the historical run by
# construction. The subregion peak change it used to carry is now a bar at each subregion's
# centroid, so both live on one map instead of two.
#
# The fill is DEMAND-side only: county-level VRE is not resolved in this pipeline. It is a fair
# climate signal even so — the county load model runs the same fleet and the same socioeconomics on
# both periods, so what moves the hours is the weather. The county threshold is taken on the same
# reference period as the national baseline, the last ten years of the record, because the anchored
# load product carries the real economic growth and a forty-year percentile would sit at a level
# only the last decade ever reached.
# 02_county_stress.py writes here. The copy that used to sit under RN was hand-made on 13 August and
# never moved again, so the map was drawn from a file four rebuilds behind its own producer. Same
# failure as the hand-copied hazard CSV in panel g; the producer's own path is the only one read.
CS = pd.read_csv("/data/cerf_out/r4_netload/county_stress_hours.csv", dtype={"fips": str})
CS["fips"] = CS.fips.str.zfill(5)
gz = pd.read_csv("/data/equity_cost/analysis/did/2023_Gaz_counties_national.txt",
                 sep="\t", dtype={"GEOID": str})
gz.columns = [c.strip() for c in gz.columns]
gz["fips"] = gz.GEOID.str.zfill(5)
cty = gz[["fips", "INTPTLAT", "INTPTLONG"]].merge(CS[["fips", "hrs_mean"]], on="fips", how="inner")

SUB = pd.read_csv(f"{RN}/R4_OURCHAIN_SUBREGION.csv")
# the bars stay the three originally published variants so they are comparable with the text
dp = SUB[SUB.variant.isin(["nopolicy", "policy", "obbba"])].groupby("subregion").d_peak.mean()
zm = np.load("/data/datasets/grid/subregion_mask.npz", allow_pickle=True)
mask = zm["subregion_mask"]
id2 = {int(a_): b_ for a_, b_ in zm["id_to_subregion"]}
zc = np.load("/data/datasets/grid/coordinate.npz")
lat, lon = zc["lat"].astype(float), zc["lon"].astype(float)
rr = np.where((mask > 0).any(1))[0]; cc = np.where((mask > 0).any(0))[0]
sub = mask[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
ext = [lon[cc.min()], lon[cc.max()], lat[rr.min()], lat[rr.max()]]
GLON, GLAT = np.meshgrid(lon[cc.min():cc.max() + 1], lat[rr.min():rr.max() + 1])
land = sub > 0
_, ii = cKDTree(np.c_[cty.INTPTLONG.values, cty.INTPTLAT.values]).query(np.c_[GLON[land], GLAT[land]])
img = np.full(sub.shape, np.nan)
img[land] = cty.hrs_mean.values[ii]

y0, h0 = band(0)
wcell = (W_MM - PAGE_L - PAGE_R - GUT_X) / 2
ASP = 1.0 / np.cos(np.deg2rad(np.mean(ext[2:])))
mh = min(h0 - 9.0, wcell * (ext[3] - ext[2]) * ASP / (ext[1] - ext[0]))
mw = mh * (ext[1] - ext[0]) / ((ext[3] - ext[2]) * ASP)
axa = fig.add_axes(R(PAGE_L + (wcell - mw) / 2, y0, mw, mh))
vlo, vhi = 10.0, float(np.nanpercentile(cty.hrs_mean, 98))
nrm = plt.matplotlib.colors.Normalize(vlo, vhi)
axa.imshow(img, origin="lower", extent=ext, aspect="auto", cmap=plt.get_cmap("YlOrRd"), norm=nrm,
           interpolation="nearest")
axa.contour(np.where(land, sub, np.nan), levels=np.arange(.5, sub.max() + 1),
            colors="white", linewidths=.35, extent=ext, origin="lower", zorder=5)
BMAX = float(dp.abs().max())
BH, BW = 10.5, 0.48                                        # degrees of latitude / longitude
for i, nm in id2.items():
    m_ = sub == i
    if nm not in dp.index or not m_.any():
        continue
    v = float(dp[nm])
    x_, yb = float(GLON[m_].mean()), float(GLAT[m_].mean())
    axa.bar(x_, BH * v / BMAX, bottom=yb, width=2 * BW, color="#3F007D", alpha=.9,
            edgecolor="white", linewidth=.3, zorder=7)
    # a zero line wider than the bar, so up and down are readable against the fill
    axa.plot([x_ - 2.1 * BW, x_ + 2.1 * BW], [yb] * 2, color="white", lw=.9, zorder=8,
             solid_capstyle="butt")
axa.set_xlim(ext[0], ext[1]); axa.set_ylim(ext[2], ext[3]); bare(axa)
mcb = fig.add_axes(R(PAGE_L + (wcell - 44) / 2, y0 + mh + 2.4, 44.0, 1.9))
cb = fig.colorbar(plt.cm.ScalarMappable(norm=nrm, cmap=plt.get_cmap("YlOrRd")), cax=mcb,
                  orientation="horizontal", extend="both")
cb.set_ticks([vlo, (vlo + vhi) / 2, vhi])
cb.set_ticklabels(["%.0f" % vlo, "%.0f" % ((vlo + vhi) / 2), "%.0f" % vhi])
cb.ax.tick_params(labelsize=FS_TICK, width=.4, length=1.6, pad=1.2)
cb.set_label("hours a year above the county's own historical 99.9th pct;  8.8 h in the past",
             fontsize=FS_VAL, labelpad=1.6)
cb.outline.set_linewidth(.4)
# The bars had a legend naming two subregion codes that appear nowhere on the map, and no scale, so
# a height could not be turned into a number. A three-bar scale key does not fit either: at the
# map's own scale a 15 GW reference bar is 18 mm tall and lands on California. So every bar carries
# its own number. In the north-east four subregions sit within a few degrees of each other, so
# their labels alternate above and below the bar rather than all hanging to the right.
CROWD = {"ISONE": (0.0, 1.5, "center"), "NYISO": (2.6, 0.0, "left"),
         "PJM_East": (2.6, -1.4, "left"), "PJM_West": (-2.6, 0.0, "right"),
         "NorthernGrid_East": (0.0, 1.5, "center"), "MISO_Central": (-2.6, 0.0, "right"),
         "SPP_South": (-2.6, 0.0, "right")}
for i, nm in id2.items():
    m_ = sub == i
    if nm not in dp.index or not m_.any():
        continue
    v = float(dp[nm])
    x_, yb = float(GLON[m_].mean()), float(GLAT[m_].mean())
    top = yb + BH * v / BMAX
    dx, dy, ha_ = CROWD.get(nm, (2.4 * BW, 0.0, "left"))
    axa.text(x_ + dx, top + dy, "%+.1f" % v, fontsize=FS_VAL, ha=ha_,
             va="bottom" if v > 0 else "top", color="#3F007D", zorder=9)
axa.text(ext[0] + 0.02 * (ext[1] - ext[0]), ext[2] + 0.045 * (ext[3] - ext[2]),
         "bar: change in peak net load (GW)", fontsize=FS_VAL, ha="left", va="bottom",
         color="#3F007D")

# ============================ row 1b — all 24, grouped ============================
axb = fig.add_axes(cell(0, 1, 2, pad=False))
GAPX, GAPY = 0.42, 0.34
NP = len(ORDP)
xs = [c + (GAPX if c >= NP else 0) for c in range(2 * NP)]
ys = [0, 1, 2 + GAPY, 3 + GAPY]
vals = np.full((4, 2 * NP), np.nan)
for j, ssp in enumerate(["ssp3", "ssp5"]):
    for k_, pol in enumerate(ORDP):
        for i, cl in enumerate(ORDC):
            r = S[(S.climate == cl) & (S.ssp == ssp) & (S.vlab == pol)]
            if len(r):
                vals[i, j * NP + k_] = float(r.d_peak_robust_pct.iloc[0])
vmax = np.nanmax(np.abs(vals))
cmp_ = plt.get_cmap("RdBu_r"); nb = plt.matplotlib.colors.Normalize(-vmax, vmax)
for i in range(4):
    for j in range(2 * NP):
        v = vals[i, j]
        axb.add_patch(Rectangle((xs[j], -ys[i]), 1, -1, facecolor=cmp_(nb(v)),
                                edgecolor="white", linewidth=.5, zorder=3))
        axb.text(xs[j] + .5, -ys[i] - .5, "%+.0f" % v, ha="center", va="center", fontsize=FS_VAL,
                 color="white" if abs(v) > .62 * vmax else "#1A1A1A", zorder=4)
for j, ssp in enumerate(["ssp3", "ssp5"]):
    axb.text(xs[j * NP] + NP / 2.0, .55, ssp, ha="center", va="bottom", fontsize=FS_VAL,
             color="#1A1A1A")
    axb.plot([xs[j * NP], xs[j * NP + NP - 1] + 1], [.30] * 2, color="#8A8A8A", lw=.5,
             clip_on=False)
    for k_, pol in enumerate(ORDP):
        axb.text(xs[j * NP + k_] + .5, -ys[-1] - 1.15, DISP[pol], ha="center", va="top",
                 fontsize=FS_LEG,
                 color="#4D4D4D", rotation=32, rotation_mode="anchor")
for i, cl in enumerate(ORDC):
    axb.text(-0.20, -ys[i] - .5, cl[5:], ha="right", va="center", fontsize=FS_VAL, color="#4D4D4D")
# the source dataset is SSP-forced, so these are SSP2-4.5 and SSP5-8.5. They are drawn as
# the forcing level alone, because the demand axis of this same figure is SSP3 against SSP5
# and repeating "SSP" on both axes would name two different things the same way.
for g, rw in [("4.5 W m$^{-2}$", (0, 1)), ("8.5 W m$^{-2}$", (2, 3))]:
    axb.text(-2.30, -(ys[rw[0]] + ys[rw[1]] + 2) / 2, g, ha="center", va="center", fontsize=FS_VAL,
             color="#1A1A1A", rotation=90)
    axb.plot([-2.02] * 2, [-ys[rw[0]], -ys[rw[1]] - 1], color="#8A8A8A", lw=.5, clip_on=False)
axb.set_xlim(-2.45, xs[-1] + 1.02); axb.set_ylim(-ys[-1] - 2.35, .95)
axb.set_axis_off()
xb0 = PAGE_L + wcell + GUT_X
cbx = fig.add_axes(R(xb0 + 20.0, y0 + h0 - 2.6, 44.0, 1.9))
cbb = fig.colorbar(plt.cm.ScalarMappable(norm=nb, cmap=cmp_), cax=cbx, orientation="horizontal")
cbb.set_ticks([-20, 0, 20]); cbb.ax.tick_params(labelsize=FS_TICK, width=.4, length=1.6, pad=1.2)
cbb.set_label("change in national peak net load (%% of %.0f GW)" % HIST_PK, fontsize=FS_VAL,
              labelpad=1.6)
cbb.outline.set_linewidth(.4)

# ==================== row 2 — the years, the day, the spread ====================
axf = fig.add_axes(cell(1, 0, 3))
for (v_, sc), g in AN.groupby(["vlab", "scenario"]):
    g = g.sort_values("year")
    axf.plot(g.year, g.peak / 1000.0, color=C_SSP[g.ssp.iloc[0]], lw=.4, alpha=.75, zorder=3)
axf.axhline(HIST_PK, color="black", lw=.8, ls=(0, (3, 3)), zorder=4)
PK_LO = 20 * np.floor(AN.peak.min() / 20000.0) - 20
PK_HI = 20 * np.ceil(AN.peak.max() / 20000.0) + 20
axf.set_xlim(2029, 2051); axf.set_xticks([2030, 2040, 2050]); axf.set_ylim(PK_LO, PK_HI)
axf.set_xlabel("year", fontsize=FS_AXIS, labelpad=2)
axf.set_ylabel("annual peak net load (GW)", fontsize=FS_AXIS, labelpad=2)
despine(axf)
NFUT = AN.groupby("ssp").apply(lambda d: d.groupby(["vlab", "scenario"]).ngroups,
                               include_groups=False).to_dict()
axf.legend(handles=[Line2D([], [], color=C_SSP[k], lw=.9,
                           label="%s, %d futures" % (k, NFUT[k]))
                    for k in ("ssp3", "ssp5")]
           + [Line2D([], [], color="black", lw=.8, ls=(0, (3, 3)),
                     label="historical %.0f GW" % HIST_PK)],
           loc="upper left", fontsize=FS_LEG, handlelength=1.3, labelspacing=.26, borderpad=.26,
           frameon=True, facecolor="white", edgecolor="#D8D8D8",
           framealpha=.94).get_frame().set_linewidth(.3)

# The dumbbells that stood here and the violins beside them were both pictures of the peak net load
# by policy variant. This is the question underneath both, which neither answered: the variants
# differ mostly in how much VRE gets built, so the firm requirement against the VRE fleet IS the
# capacity credit — and it is the quantitative form of this figure's own claim, that renewables
# solve the energy problem and not the adequacy one.
FV = pd.read_csv(f"{RN}/firm_vs_vre.csv")
axd = fig.add_axes(cell(1, 1, 3))
X_ = np.c_[np.ones(len(FV)), FV.vre_peak_gw.values, (FV.ssp == "ssp5").astype(float).values]
bt, *_ = np.linalg.lstsq(X_, FV.firm_gw.values, rcond=None)
res = FV.firm_gw.values - X_ @ bt
sl_se = np.sqrt(np.diag(np.linalg.inv(X_.T @ X_) * (res @ res) / (len(FV) - 3)))[1]
for ssp, mk in [("ssp3", "o"), ("ssp5", "s")]:
    g = FV[FV.ssp == ssp]
    xx = np.array([g.vre_peak_gw.min() - 15, g.vre_peak_gw.max() + 15])
    axd.plot(xx, bt[0] + bt[1] * xx + bt[2] * (ssp == "ssp5"), color="#9A9A9A", lw=.7,
             ls=(0, (3, 2)), zorder=2)
    for pol in ORDP:
        h = g[g.vlab == pol]
        axd.scatter(h.vre_peak_gw, h.firm_gw, s=11, marker=mk, color=C_POL[pol],
                    edgecolor="white", linewidth=.35, zorder=4)
axd.axhline(HIST_PK, color="black", lw=.7, ls=(0, (3, 3)), zorder=3)
# Two denominators, and only one of them is a capacity credit. The drawn line regresses the firm
# requirement on the 99.9th percentile of output, which is the x axis. A capacity credit is defined
# per GW of INSTALLED nameplate, so it is estimated separately and labeled as such. The panel used
# to print the output slope under the capacity-credit name, which reads as 18% of nameplate when
# nameplate gives 12%; the fleet supplies 558 GW at the 99.9th percentile against 838 GW installed.
X_C = np.c_[np.ones(len(FV)), FV.vre_cap_gw.values, (FV.ssp == "ssp5").astype(float).values]
bc, *_ = np.linalg.lstsq(X_C, FV.firm_gw.values, rcond=None)
rc = FV.firm_gw.values - X_C @ bc
cc_se = np.sqrt(np.diag(np.linalg.inv(X_C.T @ X_C) * (rc @ rc) / (len(FV) - 3)))[1]
axd.text(0.975, 0.975, "%.2f GW firm per GW of peak output (s.e. %.2f)\n"
                       "capacity credit %.0f%% of nameplate (s.e. %.0f)"
         % (-bt[1], sl_se, -100 * bc[1], 100 * cc_se), transform=axd.transAxes,
         fontsize=FS_VAL, ha="right", va="top", color="black", linespacing=1.35)
# the limits used to be written in; the fleet and the firm requirement both moved when the chain
# was rebuilt, so they now come from the data with a margin, and no point sits off the panel
_xr = FV.vre_peak_gw.max() - FV.vre_peak_gw.min()
_yr = FV.firm_gw.max() - FV.firm_gw.min()
axd.set_xlim(FV.vre_peak_gw.min() - .07 * _xr, FV.vre_peak_gw.max() + .07 * _xr)
axd.set_ylim(FV.firm_gw.min() - .08 * _yr, FV.firm_gw.max() + .30 * _yr)
axd.set_xlabel("VRE fleet, 99.9th percentile of output (GW)", fontsize=FS_AXIS, labelpad=2)
axd.set_ylabel("firm capacity still required, peak\nnet load (GW)", fontsize=FS_AXIS, labelpad=2,
               linespacing=1.2)
despine(axd)
axd.legend(handles=[Line2D([], [], marker="o", ls="", color=C_POL[p], ms=2.8,
                           label=DISP[p]) for p in ORDP]
           + [Line2D([], [], marker="s", ls="", color="#9A9A9A", ms=2.8, label="ssp5 (○ = ssp3)"),
              Line2D([], [], color="black", lw=.7, ls=(0, (3, 3)),
                     label="historical %.0f GW" % HIST_PK)],
           loc="lower left", bbox_to_anchor=(0.022, 0.022), ncol=2, columnspacing=.7,
           fontsize=FS_LEG, handlelength=1.0, labelspacing=.26, borderpad=.26,
           frameon=True, facecolor="white", edgecolor="#D8D8D8",
           framealpha=.94).get_frame().set_linewidth(.3)

axc = fig.add_axes(cell(1, 2, 3))
# The load-duration curve that stood here and the violins that stood in row 3 were two pictures of
# the same thing, the LEVEL of the net-load distribution. The violins keep the four policy variants
# separate, which is the comparison this figure is about, so they are what stays. The duration
# curve's own unique content, the shape of the negative-net-load tail, is a curtailment result and
# does not belong in an adequacy figure.
# Four violins became sixteen: the RCP label is a DEPLOYMENT axis in this ensemble (rcp45 carries
# 1.98x the renewable fleet of rcp85), so collapsing it hid the second-largest source of spread on
# the page. Each violin is still 21 annual peaks x 2 climate realisations.
AN["rcp"] = AN.climate.str[:5]
GRP = [(s_, r_, p_) for s_ in ("ssp3", "ssp5") for r_ in ("rcp45", "rcp85") for p_ in ORDP]
data = [AN[(AN.ssp == a_) & (AN.rcp == r_) & (AN.vlab == b_)].peak.values / 1000.0
        for a_, r_, b_ in GRP]
NB = len(ORDP)
xs_ = [i + .75 * (i // NB) for i in range(len(GRP))]
vp = axc.violinplot(data, positions=xs_, widths=.92, showextrema=False, showmedians=False)
for b_, (a_, r_, p_) in zip(vp["bodies"], GRP):
    b_.set_facecolor(C_POL[p_]); b_.set_edgecolor("white"); b_.set_linewidth(.25); b_.set_alpha(.8)
for x_, d_ in zip(xs_, data):
    axc.plot([x_ - .2, x_ + .2], [np.median(d_)] * 2, color="white", lw=.7, zorder=5)
axc.axhline(HIST_PK, color="black", lw=.8, ls=(0, (3, 3)), zorder=4)
axc.set_xticks([np.mean(xs_[k * NB:(k + 1) * NB]) for k in range(4)])
axc.set_xticklabels(["SSP3\n4.5", "SSP3\n8.5", "SSP5\n4.5", "SSP5\n8.5"],
                    fontsize=FS_TICK, linespacing=1.2)
axc.tick_params(axis="x", length=0, pad=2)
axc.set_xlim(-.8, xs_[-1] + .8); axc.set_ylim(PK_LO, PK_HI)
axc.set_ylabel("annual peak net load (GW)", fontsize=FS_AXIS, labelpad=2)
axc.set_xlabel("21 annual peaks × 2 climates per violin", fontsize=FS_AXIS, labelpad=2)
despine(axc)
axc.legend(handles=[Patch(facecolor=C_POL[p], alpha=.75, label=DISP[p]) for p in ORDP],
           loc="upper left", fontsize=FS_LEG, handlelength=1.0, labelspacing=.26, borderpad=.26,
           frameon=True, facecolor="white", edgecolor="#D8D8D8",
           framealpha=.94).get_frame().set_linewidth(.3)

# ==================== row 3 — the spread and the drivers ====================
axg = fig.add_axes(cell(2, 0, 3))
# Sorting the hours, as a load-duration curve does, destroys the one thing an
# operator needs: whether the hours above the historical peak arrive as isolated 3-hour spikes or
# as a day-long siege. Every contiguous run above the historical baseline printed below is drawn
# here as one point, on the same threshold the rest of the figure uses.
EV = pd.read_csv(f"{RN}/r5_events_ourchain.csv")
# The rate per run-year comes from the file, denominator included. Each row carries the length of
# its own run in `years` and the number of runs in its group in `n_runs`, both written by
# 03_events_oc.py, so a change in the size of the ensemble cannot rescale the rate unnoticed. The
# denominator is every run in the group, not the runs that happen to have an episode: several ssp3
# runs have none, and dividing by the rest would report a rate for a system that spends most of its
# runs below the historical peak.
RUNS = pd.read_csv(f"{RN}/r5_event_runs_ourchain.csv").set_index("grp")
RATE = {g: len(e) / (float(RUNS.years[g]) * float(RUNS.n_runs[g]))
        for g, e in EV.groupby("grp", observed=True)}
if set(RATE) != set(RUNS.index):
    raise SystemExit("groups in the episode table %s do not match the run table %s"
                     % (sorted(RATE), sorted(RUNS.index)))
rng_ = np.random.default_rng(7)
for g, c_, sz, al, zo in [("ssp5", C_SSP["ssp5"], 1.1, .18, 3), ("ssp3", C_SSP["ssp3"], 7, .8, 4),
                          ("historical", "black", 9, .85, 5)]:
    e = EV[EV.grp == g]
    # multiplicative jitter, because the axis is logarithmic: an additive one would smear the
    # 3-hour episodes across a decade and leave the 400-hour ones on top of each other
    jx = e.dur_h.values * np.exp(rng_.uniform(-.055, .055, len(e)))
    if g == "historical":
        axg.scatter(jx, e.depth_gw, s=sz, facecolor="none", edgecolor="black", linewidth=.5,
                    alpha=al, zorder=zo)
    else:
        axg.scatter(jx, e.depth_gw, s=sz, color=c_, alpha=al, linewidth=0, zorder=zo)
axg.set_yscale("log"); axg.set_xscale("log")
# the duration axis used to stop at 23.5 hours and then, once the three-hourly bug was fixed, to
# run to 384 with almost nothing past 48; it now ends where the data end
axg.set_xlim(.8, 1.25 * float(EV.dur_h.max())); axg.set_ylim(.02, 1.6 * float(EV.depth_gw.max()))
TK = [t for t in (1, 2, 4, 8, 16, 32, 64, 128) if t <= 1.25 * float(EV.dur_h.max())]
axg.set_xticks(TK); axg.set_xticklabels([str(t) for t in TK], fontsize=FS_TICK)
axg.xaxis.set_minor_locator(plt.matplotlib.ticker.NullLocator())
# the mean duration of each group, so the centre of each cloud can be read off the axis rather
# than guessed from an overlapping scatter
for g, c_ in [("historical", "black"), ("ssp3", C_SSP["ssp3"]), ("ssp5", C_SSP["ssp5"])]:
    mu = float(EV[EV.grp == g].dur_h.mean())
    axg.axvline(mu, color=c_, lw=.6, ls=(0, (2, 2)), zorder=6, alpha=.85)
    axg.text(mu, 1.35 * float(EV.depth_gw.max()), "%.1f h" % mu, fontsize=FS_VAL, color=c_,
             ha="center", va="top", rotation=90)
axg.set_yticks([.1, 1, 10, 100]); axg.set_yticklabels(["0.1", "1", "10", "100"], fontsize=FS_TICK)
axg.set_xlabel("hours above the historical peak", fontsize=FS_AXIS, labelpad=2)
axg.set_ylabel("GW above the historical peak", fontsize=FS_AXIS, labelpad=2)
despine(axg)
axg.legend(handles=[
    Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor="black",
           markeredgewidth=.5, ms=2.6, label="historical, %.1f/yr" % RATE["historical"]),
    Line2D([], [], marker="o", ls="", color=C_SSP["ssp3"], ms=2.6,
           label="ssp3, %.1f/yr" % RATE["ssp3"]),
    Line2D([], [], marker="o", ls="", color=C_SSP["ssp5"], ms=2.6,
           label="ssp5, %.1f/yr" % RATE["ssp5"])],
    loc="upper left", bbox_to_anchor=(0.022, 0.978), fontsize=FS_TICK, handlelength=1.0, labelspacing=.26, borderpad=.26,
    frameon=True, facecolor="white", edgecolor="#D8D8D8",
    framealpha=.94).get_frame().set_linewidth(.3)

# The one thing this figure never answered: Figure 1 says where each hazard bites historically, and
# nothing here says how often the same hazards happen under warming. Same taxonomy, same thresholds,
# each county's own historical value, aggregated to the 18 subregions.
# Read the canonical stamped product that 05_hazfreq.py writes, not a hand-copied CSV. The copy
# that used to sit under RN was from 14 August and still carried the pre-adoption fire
# threshold, so panel g was drawn at 36.52 fire days a year against the adopted 3.53.
HZ_SRC = "/data/cerf_out/r4_netload/county_hazard_freq.parquet"
HD.require_stamp(HZ_SRC, hazards=["heat", "cold", "fire"])
HZ = pd.read_parquet(HZ_SRC)
HZ["fips"] = HZ["fips"].astype(str).str.zfill(5)
# Figure 1's hazards, and only those. Heavy rain and any-cause 34 kt wind are NOT among them and
# were briefly drawn here by mistake; they are in county_hazard_freq.csv and stay out of the figure.
# The atmospheric river is missing because it needs future IVT, which is not downloaded (task #35).
# VRE drought IS here, on a different basis from the rest and marked as such: the other five are
# county weather against a fixed county threshold, this one is a subregion's wind+solar output
# against its OWN climatology, so the fleet changes between the two periods as well as the weather.
# The ensemble says that does not manufacture the result — within it a LARGER fleet has MORE
# relative drought days (+1.59 d/yr per 100 GW, t = 10.9; refit 2026-08-21), and at a near-fixed fleet the warming
# term alone is -4.1%, negative in all 16 cooler-against-hotter pairs. Details in R3_R4_RESULTS.md.
# EXACTLY Figure 1's six hazards, in Figure 1's own words, so the two panels can be read against
# each other. Humid heat used to sit here and is NOT one of Figure 1's six; heavy rain and any-cause
# 34 kt wind were briefly drawn here by mistake for the same reason. The atmospheric river is
# Figure 1's sixth and cannot be computed forward, because it needs future integrated vapour
# transport, which is not downloaded; it keeps its row and is marked, rather than dropped silently.
HAZN = ["heat", "cold", "fire weather", "VRE drought", "atmospheric river", "tropical cyclone"]
# The label used to read "needs future vapor transport, not available", which is not true: the
# future files carry column moisture (pwat) and wind, but only to 836 m above ground, because they
# were pulled for turbines. What is missing is the free troposphere, so the transport INTEGRAL
# cannot be formed. A boundary-layer proxy was tested and rejected; see RECOMPUTE_2026-08-13.md.
MISSING = {"atmospheric river": "no moisture transport above 836 m"}
# display copy only; the strings above are also column keys in the hazard table
HAZ_DISP = {"VRE drought": "renewable drought"}
VD = pd.read_csv(f"{RN}/vre_drought_future.csv")
SCN = ["rcp45cooler", "rcp85cooler", "rcp45hotter", "rcp85hotter"]
gz2 = gz[["fips", "INTPTLAT", "INTPTLONG"]].merge(HZ, on="fips", how="inner")
# every county to the subregion its centroid falls in
gi = np.clip(np.searchsorted(lat, gz2.INTPTLAT.values), 0, len(lat) - 1)
gj = np.clip(np.searchsorted(lon, gz2.INTPTLONG.values), 0, len(lon) - 1)
gz2["sid"] = mask[gi, gj]
gz2 = gz2[gz2.sid > 0]
axp = fig.add_axes(cell(2, 1, 3))
yh = np.arange(len(HAZN))[::-1]
TICK = []
# per cent, not days: heat moves 27 days a year and the tropical-cyclone count moves 0.01, so a
# linear day axis would show one bar and six lines. The historical base rides on the tick label.
for i, k in enumerate(HAZN):
    if k in MISSING:
        axp.text(-105, yh[i], MISSING[k], fontsize=FS_VAL, va="center", ha="left",
                 color="#8A8A8A", style="italic")
        TICK.append(HAZ_DISP.get(k, k))
        continue
    if k == "VRE drought":
        hn, fn = float(VD.hist_drought.mean()), float(VD.fut_drought.mean())
        sr_pc = 100 * (VD.fut_drought / VD.hist_drought - 1)
        hatch = "///"
    else:
        h_ = gz2["hist_" + k].values
        f_ = np.mean([gz2["%s_%s" % (sc, k)].values for sc in SCN], axis=0)
        hn, fn = float(np.mean(h_)), float(np.mean(f_))
        g_ = pd.DataFrame({"sid": gz2.sid.values, "h": h_, "f": f_}).groupby("sid").mean()
        sr_pc = 100 * (g_.f / g_.h - 1)
        hatch = None
    pc = 100 * (fn / hn - 1)
    c_ = "#B2182B" if pc > 0 else "#2166AC"
    axp.barh(yh[i], pc, height=.56, color=c_, alpha=.85, lw=0, zorder=3, hatch=hatch,
             edgecolor="white")
    axp.scatter(sr_pc, np.full(len(sr_pc), yh[i]), s=4.5, color="#4D4D4D", alpha=.75,
                linewidth=0, zorder=5)
    # every number in one right-aligned column: bars and the subregion cloud both run long, so a
    # label hung off the bar end lands on top of the dots
    axp.text(766, yh[i], "%+.0f%%" % pc, fontsize=FS_VAL, va="center", ha="right", color="black")
    TICK.append("%s, %.1f d" % (HAZ_DISP.get(k, k), hn))
axp.axvline(0, color="black", lw=.6, zorder=4)
axp.set_yticks(yh); axp.set_yticklabels(TICK, fontsize=FS_TICK)
axp.tick_params(axis="y", length=0, pad=2)
axp.set_xlim(-115, 782); axp.set_ylim(-1.35, len(HAZN) - .4)
axp.set_xticks([0, 200, 400])
axp.set_xlabel("change in hazard days per year (%)", fontsize=FS_AXIS, labelpad=2)
despine(axp)
axp.legend(handles=[Patch(facecolor="#B2182B", alpha=.85, label="more frequent"),
                    Patch(facecolor="#2166AC", alpha=.85, label="less frequent"),
                    Patch(facecolor="#2166AC", alpha=.85, hatch="///", edgecolor="white",
                          label="fleet also changes"),
                    Line2D([], [], marker="o", ls="", color="#4D4D4D", ms=2.2,
                           label="subregion")],
           loc="lower right", bbox_to_anchor=(0.978, 0.022), ncol=2, columnspacing=.7,
           fontsize=FS_LEG, handlelength=1.0,
           labelspacing=.26, borderpad=.26, frameon=True, facecolor="white", edgecolor="#D8D8D8",
           framealpha=.94).get_frame().set_linewidth(.3)


axe = fig.add_axes(cell(2, 2, 3))
MET = [("d_peak_robust_pct (published)", "peak net load"),
       ("exceed_histNHp999_pct", "tail hours"), ("vre_ratio_top1", "VRE at top 1%"),
       ("firm_share", "firm share"), ("firm_disp_per_gw_p999", "firm displaced per VRE GW"),
       ("peak_util", "VRE at peak")]
FAC = [("ssp", "demand growth", C_SSP["ssp5"]), ("vlab", "VRE policy", C_POL["OBBBA"]),
       ("rcp", "deployment (RCP)", C_POL["IRA"]),
       ("warm", "climate", C_SSP["ssp3"]), ("resid", "interaction", "#DDDDDD")]
yy = np.arange(len(MET))[::-1]
for i, (key, nm) in enumerate(MET):
    left = 0.0
    for f, _, c_ in FAC:
        w = 100 * DEC[key][f]
        axe.barh(yy[i], w, left=left, height=.62, color=c_, zorder=3, edgecolor="white",
                 linewidth=.3)
        left += w
axe.set_yticks(yy); axe.set_yticklabels([m[1] for m in MET], fontsize=FS_TICK)
axe.tick_params(axis="y", length=0, pad=2.5)
axe.set_xlim(0, 100); axe.set_ylim(-2.75, len(MET) - .35); axe.set_xticks([0, 50, 100])
axe.set_xlabel("share of the variance (%)", fontsize=FS_AXIS, labelpad=2)
despine(axe)
axe.legend(handles=[Patch(facecolor=c_, label=n_) for _, n_, c_ in FAC],
           loc="lower left", bbox_to_anchor=(0.022, 0.022), ncol=2, fontsize=FS_LEG,
           handlelength=1.0, labelspacing=.24, columnspacing=.8, borderpad=.24, frameon=True,
           facecolor="white", edgecolor="#D8D8D8",
           framealpha=.94).get_frame().set_linewidth(.3)

save(fig, "fig5_adequacy", tight=False, png_dpi=300)
print("page %.0f x %.0f mm  rows %s  baseline %.2f GW" % (W_MM, H_MM, ROW_H, HIST_PK))
w_ = AN.groupby(["vlab", "scenario"]).peak.std().mean() / 1000
b_ = AN.groupby(["vlab", "scenario"]).peak.mean().std() / 1000
print("interannual sd %.1f GW vs between-realisation sd %.1f GW (ratio %.2f)" % (w_, b_, w_ / b_))
print("peak delta %.1f..%.1f %%" % (S.d_peak_robust_pct.min(), S.d_peak_robust_pct.max()))
print("firm displaced %.3f GW per GW of p99.9 output (s.e. %.3f, t %.1f); capacity credit "
      "%.3f per GW nameplate (s.e. %.3f)   ssp5 offset %+.0f GW"
      % (-bt[1], sl_se, -bt[1] / sl_se, -bc[1], cc_se, bt[2]))
print("hazard days/yr, historical: %s ; VRE drought %.1f -> %.1f"
      % ({k: round(float(gz2["hist_" + k].mean()), 2) for k in HAZN
          if k != "VRE drought" and k not in MISSING},
         VD.hist_drought.mean(), VD.fut_drought.mean()))
print("map fill top: %s" % dp.sort_values(ascending=False).head(4).round(1).to_dict())
print("episode rate per run-year: %s  (run-years %s)"
      % ({k: round(v, 2) for k, v in RATE.items()},
         {g: int(RUNS.years[g] * RUNS.n_runs[g]) for g in RUNS.index}))
