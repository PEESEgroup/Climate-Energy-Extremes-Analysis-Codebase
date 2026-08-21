"""Figure 1 row 1, rebuilt on the published design: a per-subregion composite on the same flags
the joint regression uses, with a Welch test and Benjamini-Hochberg control across subregions.
Only the data changed; the estimator is the one the figure already used."""
import json
import numpy as np, pandas as pd
from scipy import stats as st
import sys as _sys
import os as _os_rp
for _rp in ("04_demand_model", "07_hazard_calendar", "09_outage_attribution",
            "02_downscale_wind", "12_figures"):
    _ap = _os_rp.path.abspath(_os_rp.path.join(
        _os_rp.path.dirname(_os_rp.path.abspath(__file__)), "..", _rp))
    if _os_rp.path.isdir(_ap) and _ap not in _sys.path:
        _sys.path.insert(0, _ap)
import paths as _PATHS   # the one name for each of the three net-load products
import hazard_defs as HD
LO = "/data/tell_pred/future/hist_full40"
PANEL = "/data/enso/r1_causal/panel_v3.parquet"
# Refuse an unstamped or superseded panel before any number is taken from it.
# 15_subregion_flags.py stamps exactly these four hazards. The ar_pub and tc_local columns are
# carried through from panel_v2 and are stamped by no builder, so naming them here would
# raise on a correct file.
PANEL_HAZ = ["cold", "heat", "fire", "vre_drought"]
HD.require_stamp(PANEL, PANEL_HAZ)
P = pd.read_parquet(PANEL)
# The adopted atmospheric-river flag is not a column of the panel. panel_v3.parquet carries the
# retired ar_pub, and 02_fig1_full_oc.py attaches the adopted ivt_p95_cov25 array at draw time. This
# file must attach it the same way, or its AR row describes a different hazard from the figure's.
_Z = np.load("/data/enso/ar_flag_variants.npz", allow_pickle=True)
if "hazard_defs_stamp" not in _Z.files:
    raise ValueError("ar_flag_variants.npz carries no hazard_defs stamp; rerun 06_ar_variants.py")
_arst = json.loads(str(_Z["hazard_defs_stamp"]))
if (_arst.get("definition_hash") or {}).get("ar") != HD.definition_hash("ar"):
    raise ValueError("the atmospheric-river flag is at a superseded definition; rerun 06_ar_variants.py")
_A = pd.DataFrame(_Z["ivt_p95_cov25"].T,
                  index=pd.to_datetime([str(x) for x in _Z["dates"]]),
                  columns=[str(x) for x in _Z["subregions"]]).stack()
_A.index = _A.index.set_names(["date", "subregion"])
P["ar"] = _A.reorder_levels(["subregion", "date"]).reindex(
    pd.MultiIndex.from_arrays([P.subregion.values, pd.to_datetime(P.date).values])
    ).fillna(False).values.astype(float)

# THE ADOPTED ATMOSPHERIC-RIVER FLAG, NOT THE SUPERSEDED ONE. 05_panel_v3.py carries both: ar_pub is
# the retired shapefile flag, firing on 108,700 subregion-days or 41% of the record, and ar is the
# adopted ivt_p95_cov25 flag at 11,998 days. This file wrote 18 ar_pub rows into the figure's own
# input, several of them marked significant, and the resulting day rate comes out as NaN
# against an expected 4.53%. Anyone reading those rows as the atmospheric-river effect got the
# superseded construction.
HAZ = ["cold", "heat", "fire", "vre_drought", "ar", "tc_local"]
# FIXED ECONOMY. Figure 1 is a fixed-fleet counterfactual, so its demand is frozen at 2019 and
# only the weather varies. The anchored product grows 1.82x from 1980 to 2019 and would mix that
# demand trend into every hazard-day contrast below. See paths.NETLOAD_FIXEDECON.
zn = _PATHS.netload_fixedecon()
names = [str(x) for x in zn["subregions"]]
lev = pd.DataFrame(np.asarray(zn["net"], float).T,
                   index=pd.to_datetime([str(x) for x in zn["times"]]),
                   columns=names).resample("D").mean().mean().to_dict()
rows = []
for hz in HAZ:
    for s, d in P.groupby("subregion"):
        m = d[hz].values > 0
        y = d["netload_anom_mean"].values
        yl = d["load_anom_mean"].values; yv = d["vre_anom_mean"].values
        if m.sum() < 10:
            rows.append(dict(hazard=hz, sub=s, net_MW=np.nan, pct=np.nan, p=np.nan,
                             fdr_sig=False, tag_days=int(m.sum()))); continue
        a, b = y[m], y[~m]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        eff = a.mean() - b.mean()          # the effect stays a mean over DAYS, as plotted
        # THE UNIT OF THE TEST IS THE SPELL, NOT THE DAY. Cold requires six consecutive days by
        # definition and heat three, so flagged days arrive in runs and are not independent draws.
        # Measured on this panel, treating them as independent inflates the t statistic by 1.56x
        # for cold, which arrives as 945 days in 133 spells, and by 1.69x for heat, 4,064 days in
        # 930 spells. Fire is barely affected, 2,396 days in 2,069 spells, because it is mostly
        # isolated single days. Collapsing each run to its own mean costs one marginal subregion
        # per hazard, FRCC for cold, SPP_South for heat and NorthernGrid_South for fire, and
        # changes no effect size, no percentage and no map. The effect above is unchanged; only
        # the precision claimed for it is.
        _i = np.where(m)[0]
        _brk = np.r_[0, np.where(np.diff(_i) > 1)[0] + 1, len(_i)]
        _sp = np.array([np.nanmean(y[_i[_brk[k]:_brk[k + 1]]]) for k in range(len(_brk) - 1)])
        _sp = _sp[np.isfinite(_sp)]
        p = float(st.ttest_ind(_sp, b, equal_var=False).pvalue) if len(_sp) >= 3 else np.nan
        rows.append(dict(hazard=hz, sub=s, net_MW=round(eff, 1),
                         load_MW=round(np.nanmean(yl[m]) - np.nanmean(yl[~m]), 1),
                         vre_MW=round(np.nanmean(yv[m]) - np.nanmean(yv[~m]), 1),
                         pct=round(100 * eff / lev[s], 1), p=p, fdr_sig=False,
                         tag_days=int(m.sum()), tag_spells=int(len(_sp))))
R = pd.DataFrame(rows)
for hz in HAZ:                                   # Benjamini-Hochberg across the 18 subregions
    i = R.index[(R.hazard == hz) & R.p.notna()]
    pv = R.loc[i, "p"].values; o = np.argsort(pv); n = len(pv)
    thr = (np.arange(1, n + 1) / n) * 0.05
    keep = pv[o] <= thr
    k = np.where(keep)[0].max() + 1 if keep.any() else 0
    R.loc[i[o[:k]], "fdr_sig"] = True
R.to_csv(f"{LO}/hazard_significance_ourchain.csv", index=False)
print("wrote hazard_significance_ourchain.csv, %d rows" % len(R))
print("\n%-6s %-8s %-24s %-22s %s" % ("hazard", "sig/n", "national GW per hazard day", "pct range", "extremes"))
for hz in ["cold", "heat", "fire"]:
    # THE SAME DENOMINATOR THE FIGURE USES. A subregion is counted only if it was actually tested:
    # at least N_MIN hazard days AND a p value, which needs at least three spells. Without the
    # second condition this line printed "15 of 17" for cold while the figure printed "15 of 16",
    # because FRCC has 12 cold days in 2 spells and carries p = NaN. Two denominators for one
    # quantity is how a caption comes to disagree with the console it was copied from.
    q = R[(R.hazard == hz) & R.net_MW.notna() & (R.tag_days >= 10) & R.p.notna()]
    nat = float((q.net_MW * q.tag_days).sum() / q.tag_days.sum()) / 1e3
    hi = q.loc[q.pct.idxmax()]; big = q.loc[q.net_MW.abs().idxmax()]
    print("  %-6s %2d of %2d   %+7.3f GW           %+.1f to %+.1f%%   steepest %s %+.1f%% = %+.2f GW ; largest %s %+.2f GW at %+.1f%%"
          % (hz, int(q.fdr_sig.sum()), len(q), nat, q.pct.min(), q.pct.max(),
             hi["sub"], hi.pct, hi.net_MW / 1e3, big["sub"], big.net_MW / 1e3, big.pct))
