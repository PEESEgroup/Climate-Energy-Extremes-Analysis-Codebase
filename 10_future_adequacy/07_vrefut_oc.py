"""VRE drought in the futures, on the same Rinaldi rule the historical hazard uses.

`14_vre_drought.py` defines a drought day as subregion daily mean wind+solar below 50% of its own
day-of-year climatology (+-15 d window), and a compound day as a drought day whose net load is also
above the subregion's 90th percentile. The drought half of that rule is RELATIVE to each series' own
climatology, which is what makes it usable here: the future fleet is a different size and in
different places, so an absolute historical threshold would measure the fleet, not the weather.

The high-net-load half is NOT relative. That 90th percentile is computed once on the historical
net load and then held fixed for every future realization. Recomputing it per realization would
grade each future against its own distribution, so a future that is uniformly higher-load would
score the same compound count as today and the growth in load, the reason the compound day matters,
could not show up at all.

Also answers the question the policy panel was asked to make room for: does building more VRE reduce
the firm capacity the system still has to hold?

WHERE THE DEFINITION COMES FROM. The drought rule is one of the seven agreed hazards, so its
constants are read from 07_hazard_calendar/hazard_defs.py and are not restated here: the 50% of
climatology fraction is hazard_defs.VRE_FRACTION, and the window is the shared +/-15 day circular
day-of-year window, hazard_defs.doy_window. hazard_defs.VRE_PERSIST_DAYS is 1 and VRE_MONTHS is
None, so there is no persistence rule and no season gate to apply. The high-net-load percentile of
the compound rule is NOT in the shared table, because the compound day is a power-system condition
rather than a hazard definition; it is the named constant NETLOAD_PCTL below.

A NaN THRESHOLD IS NEVER A ZERO. Every future realization is graded against the historical net-load
threshold of the subregion of the same name. If that threshold is missing or NaN, the comparison is
False on every day and the compound count would read as a true zero. Those subregions are written
as NaN instead, and the two failure modes, a name that does not match and a name that matches a NaN
threshold, are reported separately.
"""
import glob, os
import sys

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


def _nanmean_by(inv, vals, cnt):
    """Group mean over FINITE values only, with the finite count as the denominator."""
    v = np.asarray(vals, float)
    ok = np.isfinite(v)
    num = np.bincount(inv[ok], weights=v[ok], minlength=len(cnt))
    den = np.bincount(inv[ok], minlength=len(cnt)).astype(float)
    out = np.full(len(cnt), np.nan)
    g = den > 0
    out[g] = num[g] / den[g]
    return out

# hazard_defs.py sits in 07_hazard_calendar/ in the repository and beside this script on the
# deployment box, where every script lives flat in /data. Both go on the path, repository first.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, os.pardir, "07_hazard_calendar")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
import hazard_defs as HD

RN = "/data/tell_pred/future/netload_ourchain"
HP = _PATHS.NETLOAD_NPZ
VLAB = {"nopolicy": "NoPolicy", "ordonly": "Ordinances", "policy": "IRA", "obbba": "OBBBA"}
# the compound day's high-net-load percentile. Not a hazard_defs constant; see the docstring.
NETLOAD_PCTL = 90


def drought(vre, net, times, p90=None):
    """days per year: relative VRE drought, and drought coinciding with high net load.

    `p90` is the per-subregion high-net-load threshold, in the row order of this series. Pass the
    HISTORICAL one for every future realization. When it is None it is estimated on this series,
    which is how the historical baseline call sets it; it is returned either way.
    """
    t = pd.DatetimeIndex(pd.to_datetime(times))
    dk = t.strftime("%Y-%m-%d").values
    days, inv = np.unique(dk, return_inverse=True)
    cnt = np.bincount(inv)
    NS = vre.shape[0]
    # Finite-only group means. np.bincount with weights propagates a single NaN into the whole
    # day, and a NaN daily value then compares false against the drought threshold, so a day with
    # one missing hour was silently counted as a non-drought rather than as missing. It also
    # contaminated the day-of-year climatology around it.
    dv = np.vstack([_nanmean_by(inv, vre[s], cnt) for s in range(NS)])
    dn = np.vstack([_nanmean_by(inv, net[s], cnt) for s in range(NS)])
    dd = pd.to_datetime(days)
    doy = HD.clip_doy(days)
    ny = dd.year.nunique()
    dr = np.zeros_like(dv, dtype=bool)
    for s in range(NS):
        cl = np.full(366, np.nan)
        for d_ in range(1, 366):
            # the shared +/-15 day circular window, and the nan-safe mean: one missing hour must
            # not empty a whole day-of-year climatology
            cl[d_] = np.nanmean(dv[s, HD.doy_window(doy, d_)])
        dr[s] = dv[s] < HD.VRE_FRACTION * cl[doy]
    if p90 is None:
        p90 = np.nanpercentile(dn, NETLOAD_PCTL, axis=1)
    p90 = np.asarray(p90, dtype=float)
    comp = (dr & (dn > p90[:, None])).sum(1) / ny
    # a subregion with no usable threshold gets NaN, never 0.0: dn > NaN is False everywhere, so a
    # zero here would be indistinguishable from a subregion that genuinely never compounds
    comp = np.where(np.isfinite(p90), comp, np.nan)
    return dr.sum(1) / ny, comp, p90


zh = np.load(HP, allow_pickle=True)
HN = [str(x) for x in zh["subregions"]]
th = pd.Series(zh["times"].astype(str))
ok = pd.to_datetime(th, errors="coerce").notna().values
hd, hc, HP90 = drought((zh["solar"] + zh["wind"])[:, ok], zh["net"][:, ok],
                       pd.to_datetime(th[ok]))
# the one high-net-load threshold every future realization is graded against, by subregion name
# because the future files do not have to carry the subregions in the historical order
P90 = pd.Series(HP90, index=HN)
if not np.isfinite(HP90).all():
    print("  WARNING historical net load p%d is NaN for %s; their compound counts are written as "
          "NaN, not as zero" % (NETLOAD_PCTL, [n for n, v in zip(HN, HP90) if not np.isfinite(v)]),
          flush=True)
print("historical VRE drought, days/yr: mean %.1f  range %.1f-%.1f ; compound %.1f over %d of %d "
      "subregions" % (np.nanmean(hd), np.nanmin(hd), np.nanmax(hd), np.nanmean(hc),
                      int(np.isfinite(hc).sum()), len(HN)), flush=True)

rows, D_, C_ = [], [], []
for f in sorted(glob.glob(f"{RN}/netload_*.npz")):
    b = os.path.basename(f)
    if ".bak" in b or b.startswith("netload_hydro") or b.startswith("netload_hedc"):
        continue
    z = np.load(f, allow_pickle=True)
    v = str(z["variant"])
    if v not in VLAB:
        continue
    subs = [str(x) for x in z["subregions"]]
    vre = z["solar"] + z["wind"] + z["offshore"]
    ts = z["times"].astype(str)
    q = P90.reindex(subs)
    # two different failures, reported apart: a name this future file carries that the historical
    # file does not, and a name that matches but whose historical threshold is NaN
    unknown = [n for n in subs if n not in P90.index]
    nan_thr = [n for n in q.index[q.isna()] if n not in unknown]
    if unknown:
        print("  WARNING %s: %d subregion names are absent from the historical file: %s"
              % (b, len(unknown), unknown), flush=True)
    if nan_thr:
        print("  WARNING %s: %d subregion names matched but their historical net load p%d is NaN: "
              "%s" % (b, len(nan_thr), NETLOAD_PCTL, nan_thr), flush=True)
    # the futures carry YYYYMMDDHH strings, the historical file carries ISO timestamps
    d_, c_, _ = drought(vre, z["net"],
                        pd.to_datetime(ts, format="%Y%m%d%H") if len(ts[0]) == 10
                        else pd.to_datetime(ts), q.values)
    D_.append(pd.Series(d_, index=subs)); C_.append(pd.Series(c_, index=subs))
    t = z["times"].astype(str)
    yr = np.array([int(x[:4]) for x in t])
    net = z["net"].sum(0); load = z["load"].sum(0); tot = vre.sum(0)
    rows.append(dict(variant=v, vlab=VLAB[v], scenario=str(z["scenario"]),
                     ssp=str(z["scenario"]).split("_")[-1], climate=str(z["climate"]),
                     rcp=str(z["climate"])[:5],
                     firm_gw=float(np.mean([net[yr == u].max() for u in np.unique(yr)])) / 1e3,
                     vre_peak_gw=float(np.percentile(tot, 99.9)) / 1e3,
                     vre_mean_gw=float(tot.mean()) / 1e3,
                     load_peak_gw=float(np.mean([load[yr == u].max() for u in np.unique(yr)])) / 1e3,
                     drought_d_yr=float(np.nanmean(d_)),
                     compound_d_yr=float(np.nanmean(c_)) if np.isfinite(c_).any() else np.nan,
                     n_sub_no_p90=int((~np.isfinite(c_)).sum())))
S = pd.DataFrame(rows)
# INSTALLED NAMEPLATE, carried alongside the p99.9 output. The capacity-credit panel regressed the
# firm requirement on vre_peak_gw, the 99.9th percentile of hourly output, while its annotation
# called the slope a capacity credit per GW of VRE. A reader takes that as installed MW, and the
# two differ by the fleet's availability at the top of the distribution. Both are now available so
# the figure can state which one it uses.
# summary.csv keys the fleet by climate, ssp and arm; S keys it by variant, scenario and climate,
# where variant IS the arm and scenario is climate_ssp. Joining on the wrong names used to fail into
# a printed warning and the column silently never arrived, so the figure kept its output denominator.
# The join is now required: a miss raises, because a capacity credit without nameplate is not one.
_sum = pd.read_csv(f"{RN}/summary.csv")
_cap = (_sum.assign(vre_cap_gw=_sum["wind_GW"].fillna(0) + _sum["solar_GW"].fillna(0)
                    + _sum["offshore_GW"].fillna(0))
        .rename(columns={"arm": "variant"})[["variant", "climate", "ssp", "vre_cap_gw"]])
S = S.merge(_cap, on=["variant", "climate", "ssp"], how="left")
if S.vre_cap_gw.isna().any():
    raise SystemExit("installed capacity did not join for %d of %d rows; keys present in S are %s, "
                     "in summary.csv %s"
                     % (int(S.vre_cap_gw.isna().sum()), len(S),
                        sorted(set(map(tuple, S[["variant", "climate", "ssp"]].values.tolist()))),
                        sorted(set(map(tuple, _cap[["variant", "climate", "ssp"]].values.tolist())))))
DF = pd.concat(D_, axis=1).mean(1)
CF = pd.concat(C_, axis=1).mean(1)
out = pd.DataFrame({"subregion": HN, "hist_drought": hd, "hist_compound": hc})
out["fut_drought"] = out.subregion.map(DF)
out["fut_compound"] = out.subregion.map(CF)
out.to_csv(f"{RN}/vre_drought_future.csv", index=False)
S.to_csv(f"{RN}/firm_vs_vre.csv", index=False)
print("future VRE drought, days/yr: mean %.1f (%.1f to %.1f across realizations)"
      % (S.drought_d_yr.mean(), S.drought_d_yr.min(), S.drought_d_yr.max()))
print("  national change %+.1f%%   compound %.1f -> %.1f d/yr (%+.1f%%)"
      % (100 * (S.drought_d_yr.mean() / np.nanmean(hd) - 1), np.nanmean(hc),
         S.compound_d_yr.mean(), 100 * (S.compound_d_yr.mean() / np.nanmean(hc) - 1)))
if int(S.n_sub_no_p90.max()) > 0:
    print("  %d realization(s) carry a subregion with no usable historical threshold; those "
          "subregions are NaN in both CSVs and are excluded from the means above"
          % int((S.n_sub_no_p90 > 0).sum()))
print("\ndoes more VRE reduce the firm requirement?")
for k in ["vre_peak_gw", "vre_mean_gw"]:
    r = np.corrcoef(S[k], S.firm_gw)[0, 1]
    print("  corr(%s, firm peak net load) = %+.3f" % (k, r))
print(S.groupby(["ssp", "vlab"])[["vre_peak_gw", "vre_mean_gw", "firm_gw", "load_peak_gw",
                                  "drought_d_yr"]].mean().round(1).to_string())
