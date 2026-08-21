"""
R5 TC arm, step 2 - county TC exposure on TGW, frozen calibration, gates, future ratios.

DESIGN (R5_DAMAGE_PLAN 3.4). Inside the HURDAT2 800 km space-time window, a county-day is TC
EXPOSED if the simulated county-maximum 10-m wind reaches w*. w* is fitted ONCE on TGW-historical
1990-2010 so that the TGW county-TC-day count equals the HURDAT2 modified-Rankine count over the
same years, then FROZEN and applied unchanged to all four future runs. Nothing is ever re-fitted
on future data (gate D).

TGW-future is the same synoptic sequence 40 years later (gate A), so the future window is the
historical HURDAT2 window with the date shifted +40 y. Every future/historical comparison is
therefore a PAIRED comparison of the same storm.
"""
import os, json, glob
import numpy as np, pandas as pd
from scipy import stats as st

S = "/data/scratch_r5"
AGG = "%s/county_agg" % S
SCEN = ["rcp45cooler", "rcp45hotter", "rcp85cooler", "rcp85hotter"]
Y0, Y1, OFF = 1990, 2010, 40
NYR = Y1 - Y0 + 1
KT = 0.514444
rng = np.random.default_rng(0)


def load_agg(scen, cols=("fips", "date", "wmax")):
    fs = sorted(glob.glob("%s/agg_%s_s*.parquet" % (AGG, scen)))
    assert fs, "no aggregate for " + scen
    return pd.concat([pd.read_parquet(f, columns=list(cols)) for f in fs], ignore_index=True)


W = pd.read_parquet("%s/tc_window_%d_%d.parquet" % (S, Y0, Y1))
O = pd.read_parquet("%s/tc_obs_%d_%d.parquet" % (S, Y0, Y1))[["fips", "date", "wind_kt"]]

H = load_agg("historical")
H = H[(H.date >= "%d-01-01" % Y0) & (H.date <= "%d-12-31" % Y1)]
print("hist agg rows %s  %s..%s  counties %d"
      % (format(len(H), ","), H.date.min().date(), H.date.max().date(), H.fips.nunique()), flush=True)

P = W.merge(H, on=["fips", "date"], how="left")
cov = P.wmax.notna().mean()
print("window county-days %s   with TGW wind %.4f" % (format(len(P), ","), cov), flush=True)
P = P[P.wmax.notna()].merge(O, on=["fips", "date"], how="left")
P["obs"] = P.wind_kt.notna().astype(int)
NPOS = int(P.obs.sum())
print("observed county-TC-days inside the window: %s of %s window-days"
      % (format(NPOS, ","), format(len(P), ",")), flush=True)

# ---------------------------------------------------------------- future windowed panels (small)
FUT = {}
for sc in SCEN:
    F = load_agg(sc)
    F["date"] = F.date - pd.DateOffset(years=OFF)
    g = W[["fips", "date"]].merge(F, on=["fips", "date"], how="inner")
    FUT[sc] = g
    print("  %-12s windowed county-days with TGW wind %s" % (sc, format(len(g), ",")), flush=True)
    del F

# ---------------------------------------------------------------- FROZEN calibration of w*
WSTAR = float(np.quantile(P.wmax.values, 1.0 - NPOS / len(P)))
P["tgw"] = (P.wmax >= WSTAR).astype(int)
print("\nFROZEN w* = %.4f m/s (%.2f kt)   TGW flags %s   observed %s"
      % (WSTAR, WSTAR / KT, format(int(P.tgw.sum()), ","), format(NPOS, ",")), flush=True)

a, b = P.obs.values, P.tgw.values
po = (a == b).mean()
pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
kappa = (po - pe) / (1 - pe)
tp = int(((a == 1) & (b == 1)).sum()); fp = int(((a == 0) & (b == 1)).sum()); fn = int(((a == 1) & (b == 0)).sum())
pod, far = tp / max(tp + fn, 1), fp / max(tp + fp, 1)
g1 = "PASS(county-level)" if kappa >= .6 else ("DOWNGRADE(storm-aggregate)" if kappa >= .4 else "FAIL(stop TC arm)")
print("GATE B-TC1  kappa %.4f  POD %.4f  FAR %.4f  n %s   -> %s"
      % (kappa, pod, far, format(len(P), ","), g1), flush=True)

cc = pd.concat([P.groupby("fips").obs.sum(), P.groupby("fips").tgw.sum()], axis=1).fillna(0)
rho = float(np.corrcoef(cc.obs, cc.tgw)[0, 1])
rho_s = float(cc.obs.corr(cc.tgw, method="spearman"))
print("GATE B-TC2  county exposure-day count rho = %.4f (spearman %.4f)   %s"
      % (rho, rho_s, "PASS" if rho >= .80 else "FAIL"), flush=True)

src = np.sort(P.loc[P.tgw == 1, "wmax"].values); dst = np.sort(P.loc[P.obs == 1, "wind_kt"].values)
q = np.linspace(0, 1, 501); QX, QY = np.quantile(src, q), np.quantile(dst, q)
qmap = lambda w: np.interp(w, QX, QY, left=QY[0], right=QY[-1])
D_ks = float(st.ks_2samp(qmap(P.loc[P.tgw == 1, "wmax"].values), dst).statistic)
print("GATE B-TC3  KS(mapped TGW wind, HURDAT2 wind) = %.4f   %s"
      % (D_ks, "PASS" if D_ks <= .10 else "FAIL"), flush=True)
print("            TGW raw mean %.2f m/s (%.1f kt) -> mapped %.1f kt ; observed %.1f kt"
      % (src.mean(), src.mean() / KT, qmap(src).mean(), dst.mean()), flush=True)

# ---------------------------------------------------------------- exposure, ratios, sensitivity
hy_all = P.assign(yr=P.date.dt.year)


def counts_by_year(df, thr):
    s = df[df.wmax >= thr]
    return s.groupby(s.date.dt.year).size().reindex(range(Y0, Y1 + 1), fill_value=0)


def boot_R(hy, fy, nb=2000):
    yrs = np.arange(Y0, Y1 + 1)
    bs = np.empty(nb)
    for i in range(nb):
        d = rng.choice(yrs, NYR, replace=True)
        bs[i] = fy.reindex(d).sum() / max(hy.reindex(d).sum(), 1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float(bs.std())


res, FL = {}, [P.loc[P.tgw == 1, ["fips", "date"]].assign(scen="historical", wind_kt=qmap(P.loc[P.tgw == 1, "wmax"].values))]
h_yr = counts_by_year(P, WSTAR)
h_kt = qmap(P.loc[P.tgw == 1, "wmax"].values)
for sc in SCEN:
    G = FUT[sc]
    f_yr = counts_by_year(G, WSTAR)
    R = f_yr.sum() / h_yr.sum()
    lo, hi, sd = boot_R(h_yr, f_yr)
    kt = qmap(G.loc[G.wmax >= WSTAR, "wmax"].values)
    res[sc] = dict(hist_days_per_yr=float(h_yr.sum() / NYR), fut_days_per_yr=float(f_yr.sum() / NYR),
                   R_rawdays=float(R), R_lo=lo, R_hi=hi, R_sd=sd,
                   hist_mean_kt=float(h_kt.mean()), fut_mean_kt=float(kt.mean()),
                   dwind_kt=float(kt.mean() - h_kt.mean()),
                   hist_days50_per_yr=float((h_kt >= 50).sum() / NYR), fut_days50_per_yr=float((kt >= 50).sum() / NYR),
                   hist_days64_per_yr=float((h_kt >= 64).sum() / NYR), fut_days64_per_yr=float((kt >= 64).sum() / NYR),
                   n_counties_ever=int(G.loc[G.wmax >= WSTAR, "fips"].nunique()))
    FL.append(G.loc[G.wmax >= WSTAR, ["fips", "date"]].assign(scen=sc, wind_kt=kt))
    print("\n%-12s exposure-days/yr %.1f -> %.1f   R(raw days) = %.3f [%.3f, %.3f]"
          % (sc, h_yr.sum() / NYR, f_yr.sum() / NYR, R, lo, hi), flush=True)
    print("             mean county wind %.1f -> %.1f kt (%+.2f)   >=50kt/yr %.1f -> %.1f   >=64kt/yr %.1f -> %.1f"
          % (h_kt.mean(), kt.mean(), kt.mean() - h_kt.mean(),
             res[sc]["hist_days50_per_yr"], res[sc]["fut_days50_per_yr"],
             res[sc]["hist_days64_per_yr"], res[sc]["fut_days64_per_yr"]), flush=True)

# threshold sensitivity: is R an artefact of where w* was put?
print("\n--- threshold sensitivity of R (w* moved so the historical count is x0.5 / x1 / x2) ---", flush=True)
SENS = {}
for mult in (0.5, 1.0, 2.0):
    tgt = min(NPOS * mult / len(P), 0.5)
    w = float(np.quantile(P.wmax.values, 1.0 - tgt))
    hy = counts_by_year(P, w)
    row = {sc: float(counts_by_year(FUT[sc], w).sum() / max(hy.sum(), 1)) for sc in SCEN}
    SENS["x%.1f" % mult] = dict(wstar_ms=w, wstar_kt=w / KT, hist_days_per_yr=float(hy.sum() / NYR), R=row)
    print("  count x%.1f  w*=%.2f m/s (%.1f kt)  hist %.1f d/yr   R: %s"
          % (mult, w, w / KT, hy.sum() / NYR, {k: round(v, 3) for k, v in row.items()}), flush=True)

pd.concat(FL, ignore_index=True).to_parquet("%s/tc_flags.parquet" % S, index=False)
PC = pd.DataFrame({"hist": P[P.tgw == 1].groupby("fips").size()})
for sc in SCEN:
    G = FUT[sc]
    PC[sc] = G[G.wmax >= WSTAR].groupby("fips").size()
PC = PC.fillna(0).astype(int)
PC.to_csv("%s/tc_county_exposure.csv" % S)
json.dump(dict(wstar_ms=WSTAR, wstar_kt=WSTAR / KT, n_obs_pos=NPOS, n_window=int(len(P)),
               window_coverage_of_obs=float(cov),
               gate_b_tc1=dict(kappa=kappa, pod=pod, far=far, verdict=g1),
               gate_b_tc2=dict(pearson=rho, spearman=rho_s, verdict="PASS" if rho >= .8 else "FAIL"),
               gate_b_tc3=dict(ks=D_ks, verdict="PASS" if D_ks <= .1 else "FAIL"),
               scenarios=res, threshold_sensitivity=SENS),
          open("%s/tc_results.json" % S, "w"), indent=1, default=float)
print("\nwrote tc_results.json, tc_flags.parquet, tc_county_exposure.csv", flush=True)
