"""
Rebuild the whole R4 artifact set on the NO-HYDRO net-load definition, to match Figure 1.

Figure 1 / R1 is built on `subregion_netload_1980_2019.npz` whose construction script states
`net = load - (solar + wind)`: no hydro. Every R4 artifact so far was built on `net_hydro_btm`,
which additionally subtracts hydro and behind-the-meter rooftop PV. The two definitions differ by
a nearly constant ~30 GW (hydro mean 31.5 GW, p99 42.9, min 19.5 — it barely stops), so a reader
comparing the historical level in Figure 1 with the historical line in Figure 4 sees the same
quantity for the same period differing by 29 GW with nothing on the page to explain it.

The author's decision: everything on the no-hydro definition.

WHAT CHANGES AND WHAT DOES NOT. Hydro is subtracted from BOTH sides of the future-versus-historical
delta, so it cancels there: the peak DELTA range, the policy decomposition and the variance
decomposition are essentially untouched. What changes is every LEVEL — the historical baseline goes
from 619.17 to 648.14 GW (mean of annual maxima, 3-hourly matched) and every future level rises by
roughly the same ~30 GW.

BTM is dropped with hydro rather than kept, because Figure 1's historical series contains neither;
keeping it would leave a second, smaller inconsistency in place of the one being fixed.

Nothing existing is overwritten. Outputs carry a `_ourchain` tag and the hydro artifacts remain on
disk, because the R4 document's published headline is stated on them.
"""
import json, glob, os
import numpy as np, pandas as pd
import sys as _sys
import os as _os_rp
for _rp in ("04_demand_model", "07_hazard_calendar", "09_outage_attribution",
            "02_downscale_wind", "12_figures"):
    _ap = _os_rp.path.abspath(_os_rp.path.join(
        _os_rp.path.dirname(_os_rp.path.abspath(__file__)), "..", _rp))
    if _os_rp.path.isdir(_ap) and _ap not in _sys.path:
        _sys.path.insert(0, _ap)
import paths as _PATHS   # the one name for the anchored net-load product
import baseline as _BASE  # the one definition of the historical reference period

RN = "/data/tell_pred/future/netload_ourchain"
HP = _PATHS.NETLOAD_NPZ
VLAB = {"nopolicy": "NoPolicy", "ordonly": "Ordinances", "policy": "IRA", "obbba": "OBBBA"}

# ---------------------------------------------------------------- historical, no hydro
zh = np.load(HP, allow_pickle=True)
th = pd.to_datetime(pd.Series(zh["times"].astype(str)), errors="coerce")
ok = th.notna().values
th = th[ok]
hn = zh["net"].sum(0)[ok]                                  # load - solar - wind, no hydro
# The historical series used to be thinned to hours divisible by three, to match a wind product
# that was three-hourly. The chain is hourly on both sides now, and the thinning was one-sided: the
# future series was never thinned, so a historical annual maximum taken over a third of the hours
# was compared with a future one taken over all of them, which biases every delta upward.
sel = np.ones(len(th), dtype=bool)
th3, hn3 = th[sel], hn[sel]
hy = th3.dt.year.values
# The reference period, not the whole record. The anchored product grows 1.82x across the forty
# years, so its forty-year mean annual maximum is 535.06 GW against a modern peak of 642 GW and a
# validated EIA-930 observation of 651.5 GW. baseline.py carries the reasoning.
BY = _BASE.base_years(hy)
bsel = np.isin(hy, BY)
HIST_MAM = float(np.mean([hn3[hy == u].max() for u in BY]))
HIST_P999 = float(np.percentile(hn3[bsel], 99.9))
HIST_MEAN = float(hn3[bsel].mean())
print("historical reference period %d-%d, NO HYDRO: mean %.2f GW  mean-annual-max %.2f GW  "
      "p99.9 %.2f GW" % (BY[0], BY[-1], HIST_MEAN / 1e3, HIST_MAM / 1e3, HIST_P999 / 1e3),
      flush=True)

# ---------------------------------------------------------------- per realisation
rows, ann, prof = [], [], {}
for f in sorted(glob.glob(f"{RN}/netload_*.npz")):
    b = os.path.basename(f)
    if ".bak" in b or b.startswith("netload_hydro") or b.startswith("netload_hedc"):
        continue
    z = np.load(f, allow_pickle=True)
    variant = str(z["variant"]); scenario = str(z["scenario"]); climate = str(z["climate"])
    if variant not in VLAB:
        continue
    t = z["times"].astype(str)
    yr = np.array([int(x[:4]) for x in t])
    # The month and the hour used to be sliced out of the string by position, on the assumption of a
    # YYYYMMDDHH stamp. These files carry ISO timestamps instead, so x[4:6] read "-0" and x[8:10]
    # read the day: every month came out as 0 or -1 and the hour ran 1 to 31. The seasonal masks
    # then selected nothing and all 64 future diurnal profiles were written as NaN. Parsed properly
    # here; the year, which drives every annual maximum in this file, was always correct because it
    # is the first four characters under either convention.
    _ts = pd.to_datetime(pd.Series(t), format="%Y%m%d%H" if len(t[0]) == 10 else None)
    hh = _ts.dt.hour.values
    mo = _ts.dt.month.values
    net = z["net"].sum(0)                               # base: no hydro, no BTM
    load = z["load"].sum(0)
    vre = (z["solar"] + z["wind"] + z["offshore"]).sum(0)
    years = np.unique(yr)
    na, la, vp = [], [], []
    for y in years:
        s = yr == y
        i = int(np.argmax(net[s]))
        na.append(net[s][i]); la.append(load[s].max()); vp.append(vre[s][i])
        ann.append(dict(variant=variant, vlab=VLAB[variant], scenario=scenario,
                        climate=climate, ssp=scenario.split("_")[-1], year=int(y),
                        peak=float(net[s][i]), mean=float(net[s].mean())))
    mam = float(np.mean(na)); lpk = float(np.mean(la))
    vpk = float(np.percentile(vre, 99.9)); vap = float(np.mean(vp))
    thr = np.percentile(net, 99)
    top = net >= thr
    rows.append(dict(
        variant=variant, vlab=VLAB[variant], scenario=scenario, climate=climate,
        ssp=scenario.split("_")[-1], rcp=climate[:5],
        warm="hotter" if "hotter" in climate else "cooler",
        fut_mean=float(net.mean()) / 1e3, fut_meanannmax=mam / 1e3,
        fut_p999=float(np.percentile(net, 99.9)) / 1e3,
        hist_mean=HIST_MEAN / 1e3, hist_meanannmax=HIST_MAM / 1e3,
        d_mean_pct=100 * (net.mean() / HIST_MEAN - 1),
        d_peak_robust_pct=100 * (mam / HIST_MAM - 1),
        exceed_histP999_pct=100 * float((net > HIST_P999).mean()),
        vre_ratio_top1=float(vre[top].mean() / vre.mean()),
        load_peak_gw=lpk / 1e3, vre_peak_gw=vpk / 1e3, vre_at_peak_gw=vap / 1e3,
        firm_share=mam / lpk,
        # firm displacement per GW of the fleet's 99.9th-percentile OUTPUT, not per installed GW.
        # The old name cap_credit invited the installed-capacity reading that the denominator does
        # not support; the nameplate version is attached in 07_vrefut_oc.py where capacity is known.
        firm_disp_per_gw_p999=(lpk - mam) / vpk, peak_util=vap / vpk))
    for seas, mm in [("JJA", (6, 7, 8)), ("DJF", (12, 1, 2))]:
        s = np.isin(mo, mm)
        prof["%s|%s|%s" % (VLAB[variant], scenario, seas)] = [
            float(net[s & (hh == h)].mean()) for h in range(0, 24, 3)]

S = pd.DataFrame(rows).sort_values(["vlab", "scenario"]).reset_index(drop=True)
A = pd.DataFrame(ann)
print("realisations %d   annual rows %d" % (len(S), len(A)), flush=True)
S.to_csv(f"{RN}/R4_OURCHAIN_SUMMARY.csv", index=False)
A.to_csv(f"{RN}/r5_annual_ourchain.csv", index=False)

for seas, mm in [("JJA", (6, 7, 8)), ("DJF", (12, 1, 2))]:
    s = th3.dt.month.isin(mm).values
    prof["HIST||%s" % seas] = [float(hn3[bsel & s & (th3.dt.hour.values == h)].mean())
                               for h in range(0, 24, 3)]
json.dump(prof, open(f"{RN}/r5_diurnal_ourchain.json", "w"), indent=1)

# ---------------------------------------------------------------- policy decomposition
P = S.pivot_table(index="scenario", columns="vlab", values="fut_meanannmax")
D = pd.DataFrame({
    "scenario": P.index,
    "no_peak_mam": P["NoPolicy"].values * 1e3, "ord_peak_mam": P["Ordinances"].values * 1e3,
    "ira_peak_mam": P["IRA"].values * 1e3, "obbba_peak_mam": P["OBBBA"].values * 1e3})
D["dpeak_ord_no_mam"] = (P["Ordinances"] - P["NoPolicy"]).values * 1e3
D["dpeak_ira_ord_mam"] = (P["IRA"] - P["Ordinances"]).values * 1e3
D["dpeak_ira_no_mam"] = (P["IRA"] - P["NoPolicy"]).values * 1e3
D["dpeak_obbba_ira_mam"] = (P["OBBBA"] - P["IRA"]).values * 1e3
for c in D.columns:
    if c != "scenario":
        D[c] = D[c] / 1e3
D.to_csv(f"{RN}/R4_OURCHAIN_ORD_DECOMP.csv", index=False)
print("\nordinance %+.2f..%+.2f (mean %+.2f) | incentive %+.2f..%+.2f (mean %+.2f) | net %+.2f..%+.2f"
      % (D.dpeak_ord_no_mam.min(), D.dpeak_ord_no_mam.max(), D.dpeak_ord_no_mam.mean(),
         D.dpeak_ira_ord_mam.min(), D.dpeak_ira_ord_mam.max(), D.dpeak_ira_ord_mam.mean(),
         D.dpeak_ira_no_mam.min(), D.dpeak_ira_no_mam.max()), flush=True)
print("OBBBA-IRA %+.1f..%+.1f" % (D.dpeak_obbba_ira_mam.min(), D.dpeak_obbba_ira_mam.max()))

# ---------------------------------------------------------------- variance decomposition
FACTORS = ["ssp", "vlab", "rcp", "warm"]


def dec(d, c):
    """One-way sums of squares over a balanced full factorial, plus the interaction remainder.

    This is only a partition because the design is balanced: 2 demand pathways x 4 policy variants
    x 2 forcing pathways x 2 sensitivity members, one realization per cell. Main effects are then
    orthogonal and their shares cannot overlap. The check below enforces that rather than assuming
    it, because the previous version clipped a negative remainder to zero, which would have hidden
    exactly the failure that clipping is a symptom of.

    With one observation per cell there is no replication, so the remainder is the pooled sum of
    squares of every interaction and cannot be separated from error. It is reported as interaction
    on that understanding and must not be read as a residual around a fitted model."""
    # Equal counts among the cells that EXIST does not make a design complete, and an incomplete
    # design leaves the main effects correlated however even the observed cells are. The Cartesian
    # product is therefore checked as well: a three-of-four design passes an equal-count test and
    # still reports a spurious interaction remainder.
    cells = d.groupby(FACTORS).size()
    n_full = 1
    for f in FACTORS:
        n_full *= d[f].nunique()
    if len(cells) != n_full:
        raise ValueError("the scenario design is incomplete: %d of %d factorial cells are present, "
                         "so one-way sums of squares do not partition the variance"
                         % (len(cells), n_full))
    if cells.nunique() != 1:
        raise ValueError("the scenario design is not balanced (cell sizes %s), so one-way sums of "
                         "squares do not partition the variance" % sorted(cells.unique()))
    y = d[c].values.astype(float); tot = ((y - y.mean()) ** 2).sum(); o = {}
    for fct in FACTORS:
        g = d.groupby(fct)[c].transform("mean").values
        o[fct] = ((g - y.mean()) ** 2).sum() / tot
    inter = 1 - sum(o.values())
    if inter < -1e-9:
        raise ValueError("main effects sum to %.4f of the variance for %s, so they are not "
                         "orthogonal and this is not a decomposition" % (sum(o.values()), c))
    o["resid"] = max(0.0, inter)     # numerical only; a real negative raises above
    return o


MET = [("d_peak_robust_pct (published)", "d_peak_robust_pct"),
       ("exceed_histNHp999_pct", "exceed_histP999_pct"),
       ("vre_ratio_top1", "vre_ratio_top1"), ("firm_share", "firm_share"),
       ("firm_disp_per_gw_p999", "firm_disp_per_gw_p999"), ("peak_util", "peak_util")]
DEC = {k: dec(S, c) for k, c in MET}
json.dump(DEC, open(f"{RN}/r5_decomp4_ourchain.json", "w"), indent=1, default=float)
print("\n%-30s %6s %6s %6s %8s" % ("metric", "SSP", "policy", "RCP", "cool/hot"))
for k, c in MET:
    v = DEC[k]
    print("%-30s %5.0f%% %5.0f%% %5.0f%% %7.0f%%"
          % (c, 100 * v["ssp"], 100 * v["vlab"], 100 * v["rcp"], 100 * v["warm"]), flush=True)

# ---------------------------------------------------------------- subregion table
SR = []
zsub = None
for f in sorted(glob.glob(f"{RN}/netload_*.npz")):
    b = os.path.basename(f)
    if ".bak" in b or b.startswith("netload_hydro") or b.startswith("netload_hedc"):
        continue
    z = np.load(f, allow_pickle=True)
    variant = str(z["variant"])
    if variant not in VLAB:
        continue
    subs = [str(x) for x in z["subregions"]]
    t = z["times"].astype(str); yr = np.array([int(x[:4]) for x in t])
    net = z["net"]
    hsub = zh["net"][:, ok][:, sel]
    hsubs = [str(x) for x in zh["subregions"]]
    for i, nm in enumerate(subs):
        j = hsubs.index(nm)
        fm = float(np.mean([net[i][yr == u].max() for u in np.unique(yr)]))
        # the same reference period as the national baseline, so the map and the matrix agree
        hm = float(np.mean([hsub[j][hy == u].max() for u in BY]))
        SR.append(dict(variant=variant, scenario=str(z["scenario"]), subregion=nm,
                       fut_peak=fm / 1e3, hist_peak=hm / 1e3, d_peak=(fm - hm) / 1e3,
                       fut_mean=float(net[i].mean()) / 1e3,
                       hist_mean=float(hsub[j][bsel].mean()) / 1e3,
                       d_mean=float(net[i].mean() - hsub[j][bsel].mean()) / 1e3))
SUB = pd.DataFrame(SR)
SUB.to_csv(f"{RN}/R4_OURCHAIN_SUBREGION.csv", index=False)
n3 = SUB[SUB.variant.isin(["nopolicy", "policy", "obbba"])].groupby("subregion").d_peak.mean()
print("\nsubregion rows %d ; map fill top 5: %s"
      % (len(SUB), n3.sort_values(ascending=False).head(5).round(1).to_dict()), flush=True)
pv = SUB.pivot_table(index=["scenario", "subregion"], columns="variant", values="d_peak")
print("ordinance term top 3: %s"
      % (pv["ordonly"] - pv["nopolicy"]).groupby("subregion").mean()
      .sort_values(key=abs, ascending=False).head(3).round(2).to_dict())
print("\nWROTE R4_OURCHAIN_{SUMMARY,ORD_DECOMP,SUBREGION}.csv, r5_{annual,diurnal,decomp4}_ourchain")
