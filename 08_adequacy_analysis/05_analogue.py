"""Model-internal causal contrast for the hazard arm, by analogue substitution.

The panel regression it replaces put the modeled outcome on the left and a flag derived from the
same weather field on the right. Setting that flag to one is not a well defined intervention, so
its coefficient is a projection rather than a causal effect, even inside the model.

The intervention used here is well defined: replace the weather of a hazard day with the weather
of a comparable non-hazard day. Because the fleet is fixed and the model is a deterministic
function of the weather of that day, the counterfactual output is the output already computed for
the analogue day. Nothing needs to be re-run.

  effect(d) = y(d) - mean over the analogue pool of y(d')

Analogue pool for day d in subregion s: same subregion, day-of-year within 15 days of d, year
within 5 years of d, no hazard of ANY kind flagged, and at least 3 days away from any flagged day.
Matching on day-of-year removes the seasonal cycle and matching on year removes the warming trend,
so no fixed effect is needed and no additivity is assumed.

Uncertainty is a block bootstrap over episodes, since consecutive hazard days are one event.
A placebo repeats the estimator on the same days shifted seven days earlier.
"""
import json, numpy as np, pandas as pd
import sys as _sys
import os as _os_rp
for _rp in ("04_demand_model", "07_hazard_calendar", "09_outage_attribution",
            "02_downscale_wind", "12_figures"):
    _ap = _os_rp.path.abspath(_os_rp.path.join(
        _os_rp.path.dirname(_os_rp.path.abspath(__file__)), "..", _rp))
    if _os_rp.path.isdir(_ap) and _ap not in _sys.path:
        _sys.path.insert(0, _ap)
import hazard_defs as HD

PANEL = "/data/enso/r1_causal/panel_v3.parquet"
# Refuse an unstamped or superseded panel before any number is taken from it.
# 15_subregion_flags.py stamps exactly these four hazards. The ar_pub and tc_local columns are
# carried through from panel_v2 and are stamped by no builder, so naming them here would
# raise on a correct file.
PANEL_HAZ = ["cold", "heat", "fire", "vre_drought"]
HD.require_stamp(PANEL, PANEL_HAZ)
P = pd.read_parquet(PANEL)
P["date"] = pd.to_datetime(P.date)
P["doy"] = P.date.dt.dayofyear.clip(upper=365)
# ADOPTED ATMOSPHERIC-RIVER FLAG. The panel carries an ar_pub column, 108,700 subregion-days of
# the superseded shapefile flag. The adopted flag is ivt_p95_cov25, 11,998 subregion-days, which
# 06_ar_variants.py stamps and 07_ar_adopt_oc.py fits on. It is attached here under the name ar, so the
# analogue estimate and the regression it is compared with describe one construction.
_Z = np.load("/data/enso/ar_flag_variants.npz", allow_pickle=True)
_st = json.loads(str(_Z["hazard_defs_stamp"]))
_want, _got = HD.definition_hash("ar"), (_st.get("definition_hash") or {}).get("ar")
if _got != _want:
    raise ValueError("ar_flag_variants.npz holds the atmospheric-river flag at definition %s, but "
                     "the current definition is %s: rerun 06_ar_variants.py" % (_got, _want))
_A = pd.DataFrame(_Z["ivt_p95_cov25"].T,
                  index=pd.to_datetime([str(x) for x in _Z["dates"]]),
                  columns=[str(x) for x in _Z["subregions"]]).stack()
_A.index = _A.index.set_names(["date", "subregion"])
P["ar"] = _A.reorder_levels(["subregion", "date"]).reindex(
    pd.MultiIndex.from_arrays([P.subregion.values, P.date.values])).fillna(False).values.astype(float)
HAZ = ["heat", "cold", "fire", "vre_drought", "ar", "tc_local"]
OUT = ["netload_anom_mean", "netload_anom_peak", "load_anom_mean", "vre_anom_mean"]
DOY_W, YR_W, BUF, MINPOOL, B = 15, 5, 3, 10, 2000
rng = np.random.default_rng(20260813)
subs = sorted(P.subregion.unique())
print("panel %s   subregions %d   days %d" % (P.shape, len(subs), P.date.nunique()), flush=True)

def episodes(idx_days):
    """contiguous treated days, merged across gaps of one day, are one event"""
    if len(idx_days) == 0:
        return []
    d = np.sort(idx_days); cut = np.where(np.diff(d) > 2)[0]
    return np.split(d, cut + 1)

rows = []
for s in subs:
    Q = P[P.subregion == s].sort_values("date").reset_index(drop=True)
    anyhaz = (Q[HAZ].values > 0).any(1)
    # the leap days dropped upstream arrive here as NaN; one of them in a pool would turn the whole
    # effect into NaN, so they are excluded from the analogue pool and from the treated days
    goodrow = np.isfinite(Q[OUT].values).all(1)
    # a day is a usable analogue only if it and its neighbours are hazard free
    ok = (~anyhaz) & goodrow
    for b in range(1, BUF + 1):
        ok &= ~np.r_[np.zeros(b, bool), anyhaz[:-b]]
        ok &= ~np.r_[anyhaz[b:], np.zeros(b, bool)]
    doy = Q.doy.values; yr = Q.year.values
    dnum = (Q.date - Q.date.min()).dt.days.values
    Y = {o: Q[o].values.astype(float) for o in OUT}
    for hz in HAZ:
        tre = np.where((Q[hz].values > 0) & goodrow)[0]
        if len(tre) < 20:
            rows.append(dict(subregion=s, hazard=hz, n_days=len(tre), status="too few days"))
            continue
        eff = {o: [] for o in OUT}; pool_n = []; kept = []
        pl = {o: [] for o in OUT}
        for i in tre:
            dd = np.abs(((doy - doy[i] + 182) % 365) - 182)
            m = ok & (dd <= DOY_W) & (np.abs(yr - yr[i]) <= YR_W)
            n = int(m.sum()); pool_n.append(n)
            if n < MINPOOL:
                continue
            kept.append(i)
            for o in OUT:
                eff[o].append(Y[o][i] - np.nanmean(Y[o][m]))
            j = i - 7                                     # placebo: same construction, seven days earlier
            if j >= 0 and not anyhaz[j]:
                for o in OUT:
                    pl[o].append(Y[o][j] - np.nanmean(Y[o][m]))
        if len(kept) < 20:
            rows.append(dict(subregion=s, hazard=hz, n_days=len(tre), n_used=len(kept),
                             pool_median=float(np.median(pool_n)) if pool_n else 0.0,
                             status="pool too small"))
            continue
        ep = episodes(np.array(kept)); pos = {v: k for k, v in enumerate(kept)}
        epi = [[pos[x] for x in e] for e in ep]
        rec = dict(subregion=s, hazard=hz, n_days=len(tre), n_used=len(kept), n_episodes=len(ep),
                   pool_median=float(np.median(pool_n)), pool_min=int(np.min(pool_n)),
                   status="ok")
        for o in OUT:
            a = np.array(eff[o]); rec[o] = float(a.mean())
            bs = np.empty(B)
            for b in range(B):
                pick = rng.integers(0, len(epi), len(epi))
                bs[b] = np.concatenate([np.array(epi[k]) for k in pick]).astype(int).size and \
                        a[np.concatenate([np.array(epi[k]) for k in pick]).astype(int)].mean()
            rec[o + "_lo"] = float(np.percentile(bs, 5)); rec[o + "_hi"] = float(np.percentile(bs, 95))
            rec[o + "_p"] = float(2 * min((bs <= 0).mean(), (bs >= 0).mean()))
            rec[o + "_placebo"] = float(np.mean(pl[o])) if pl[o] else np.nan
        rows.append(rec)
    print("  %-10s done" % s, flush=True)

R = pd.DataFrame(rows)
R.to_csv("/data/enso/r1_causal/analogue_effects_ourchain.csv", index=False)
print("\nfeasibility, analogue pool per hazard:")
for hz in HAZ:
    q = R[R.hazard == hz]
    okq = q[q.status == "ok"]
    print("  %-12s subregions usable %2d/%d   median pool %s   days used %s of %s"
          % (hz, len(okq), len(q),
             ("%.0f" % okq.pool_median.median()) if len(okq) else "-",
             int(okq.n_used.sum()) if len(okq) else 0, int(q.n_days.sum())))
print("\nnet-load mean effect, MW, subregions where the estimator is usable:")
for hz in HAZ:
    okq = R[(R.hazard == hz) & (R.status == "ok")]
    if not len(okq):
        continue
    sig = okq[(okq.netload_anom_mean_lo > 0) | (okq.netload_anom_mean_hi < 0)]
    print("  %-12s median %+8.1f   range %+8.1f to %+8.1f   CI excludes zero in %d of %d"
          % (hz, okq.netload_anom_mean.median(), okq.netload_anom_mean.min(),
             okq.netload_anom_mean.max(), len(sig), len(okq)))
print("\nwrote /data/enso/r1_causal/analogue_effects_ourchain.csv")
