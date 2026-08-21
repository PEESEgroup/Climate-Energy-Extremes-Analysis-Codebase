"""
STEP 4 — the equity accounting.

Steps 1-3 were statistical: establish that weather causes outages, and test whether the effect
differs by group. This step is DISTRIBUTIONAL: given the effect, work out who bears it. It needs
no causal assumption of its own, which is the point - inequity is a statement about distribution,
not about mechanism.

What is allocated is the WEATHER-ATTRIBUTABLE burden, not total outage. Total outage is mostly
everyday faults that have nothing to do with climate, and distributing it would answer a
different question.

  attributable_c = (per-event effect) x (number of events at c) x (customers at c)

taken from the event studies: tropical cyclone +4.659 customer-hours per customer per event,
severe convection +0.218. Both are cumulative over day 0..+14 with a flat or near-flat placebo.

DECOMPOSITION. attributable = EXPOSURE x VULNERABILITY, where exposure is how many events a
county actually experienced (pure counting, no estimation) and vulnerability is the per-event
effect. Stages 2-3 found NO demographic gradient in vulnerability under multiplicity correction,
so any inequity in the attributable burden has to come from exposure. That is a testable claim,
not an inference, and it is tested here.

Grid form is reported in a SEPARATE table. Underground share is not a protected characteristic;
its relationship with burden is mechanism, not inequity, and putting the two in one table invites
the reader to treat them symmetrically when they are not.

NO precision weighting is applied and none is needed: because a single common per-event effect is
used rather than county-specific slopes, county attributable burdens carry no county-level
estimation error. (An earlier plan called for shrinkage; that applies only to a design that
estimates per-county effects, which this is not.)
"""
import json, os
import numpy as np, pandas as pd
import os as _os, sys as _sys
_HD = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "07_hazard_calendar"))
if _HD not in _sys.path: _sys.path.insert(0, _HD)
import hazard_defs as HD

OUT = "/data/equity_cost/analysis/step4"; os.makedirs(OUT, exist_ok=True)
T0, T1 = "2015-01-01", "2022-09-30"
BETA = {"tc": 4.65915, "conv": 0.21778}          # cumulative 0..+14, customer-hours per customer

# ------------------------------------------------------------------ events per county
TC = pd.read_parquet("/data/enso/tc_county_ext/county_tc_days.parquet")
TC["date"] = pd.to_datetime(TC.date)
TC = TC[(TC.date >= T0) & (TC.date <= T1)].sort_values(["fips", "date"])

# COUNT THE SAME EVENTS THE EFFECT WAS ESTIMATED ON. The event studies used ISOLATED spells - no
# other event for that county within +/-CLEAN days - because overlapping windows contaminate the
# coefficients. Applying that per-event effect to ALL spells would attribute overlapping events
# separately and double-count: it inflated severe convection from 5.8% to 11.4% and the total
# from 25.4% to 34.0%. Counting isolated spells only UNDERCOUNTS instead, since clustered events
# do cause outages but cannot be separately attributed without double counting. Undercounting is
# the right direction: it keeps the attributable share a lower bound, like every other number
# here.
def isolated_counts(df, clean):
    df = df.sort_values(["fips", "date"])
    f = df.fips.values; d = df.date.values.astype("datetime64[D]").astype(int)
    new = np.r_[True, (f[1:] != f[:-1]) | ((d[1:] - d[:-1]) > 2)]
    ef, ed = f[new], d[new]
    out = {}
    for i in range(len(ef)):
        same = f == ef[i]
        near = same & (np.abs(d - ed[i]) <= clean)
        spell = same & (d >= ed[i]) & (d < ed[i] + 5)
        if near.sum() <= spell.sum():
            out[ef[i]] = out.get(ef[i], 0) + 1
    return pd.Series(out, dtype=float)

n_tc = isolated_counts(TC[["fips", "date"]], 30).rename("n_tc")

HD.require_stamp("/data/enso/county_convective_daily.parquet", hazards=["convection"])
C = pd.read_parquet("/data/enso/county_convective_daily.parquet")
C["date"] = pd.to_datetime(C.date)
CV = C[C.severe][["fips", "date"]]
n_cv = isolated_counts(CV, 21).rename("n_conv")

E = pd.read_csv("/data/equity_cost/analysis/equity_joined_v2.csv", dtype={"fips": str})
E = E[E.denom > 0].copy()
D = pd.read_parquet("/data/equity_cost/analysis/eaglei_county_daily.parquet",
                    columns=["fips", "date", "customer_hours_out"])
D["date"] = pd.to_datetime(D.date); D["fips"] = D.fips.astype(str).str.zfill(5)
span = D.groupby("fips").date.agg(["min", "max"])
keep = span[(span["min"] <= "2015-07-01") & (span["max"] >= "2022-06-30")].index
obs = (D[(D.date >= T0) & (D.date <= T1) & D.fips.isin(keep)]
       .groupby("fips").customer_hours_out.sum().rename("observed_cho"))

M = E[["fips", "state", "denom", "population", "land_km2", "median_income", "poverty_rate",
       "median_age", "minority_pct"]].copy()
M = M[M.fips.isin(keep)]
M = M.join(n_tc, on="fips").join(n_cv, on="fips").join(obs, on="fips")
M[["n_tc", "n_conv", "observed_cho"]] = M[["n_tc", "n_conv", "observed_cho"]].fillna(0)
# TERCILE-SPECIFIC convective effect. Stage 3 showed the per-event effect is NOT common: the
# daily "post" coefficient is +0.0298 / +0.0065 / +0.0086 by underground tercile, and the event
# window is 15 days, so cumulative per event is 15x that. Using a single beta would discard the
# one vulnerability difference the analysis actually established, and would make the attributable
# rate a relabelling of the event count.
GF = pd.read_parquet("/data/equity_cost/gridphys/county_gridform.parquet")[["fips", "ug_share_dom"]]
M = M.merge(GF, on="fips", how="left")
qs = M.ug_share_dom.quantile([1/3, 2/3]).values
BCONV = {"low": 0.029820 * 15, "mid": 0.006529 * 15, "high": 0.008601 * 15}
def bconv(v):
    if not np.isfinite(v):
        return BETA["conv"]                    # no grid form observed -> the pooled effect
    return BCONV["low"] if v < qs[0] else (BCONV["mid"] if v < qs[1] else BCONV["high"])
M["beta_conv"] = M.ug_share_dom.map(bconv)
M["grid_terc"] = np.where(M.ug_share_dom.isna(), "unobserved",
                          np.where(M.ug_share_dom < qs[0], "low",
                                   np.where(M.ug_share_dom < qs[1], "mid", "high")))
print("per-event convective effect by tercile: %s   (pooled %.3f)"
      % ({k: round(v, 3) for k, v in BCONV.items()}, BETA["conv"]))
print("counties by tercile: %s" % M.grid_terc.value_counts().to_dict())
M["att_tc"] = BETA["tc"] * M.n_tc * M.denom
M["att_conv"] = M.beta_conv * M.n_conv * M.denom
M["att_all"] = M.att_tc + M.att_conv
M["att_rate"] = M.att_all / M.denom                      # customer-hours per customer, 7.75 yr
# EXPOSURE is pure counting, valued at the POOLED effect, so it is what the burden would be if
# every county had average vulnerability. attributable - exposure_valued = the vulnerability part.
M["exposure_valued"] = (BETA["tc"] * M.n_tc + BETA["conv"] * M.n_conv) * M.denom
M["vuln_part"] = M.att_all - M.exposure_valued
M["exposure"] = M.n_tc * BETA["tc"] / BETA["conv"] + M.n_conv
M["dens"] = M.population / M.land_km2.replace(0, np.nan)
print("counties %d   observed total %.3e   attributable %.3e (%.1f%%)"
      % (len(M), M.observed_cho.sum(), M.att_all.sum(), 100 * M.att_all.sum() / M.observed_cho.sum()))
print("   of which TC %.1f%%   convection %.1f%%"
      % (100 * M.att_tc.sum() / M.observed_cho.sum(), 100 * M.att_conv.sum() / M.observed_cho.sum()))
print("   counties ever hit: TC %d   convection %d   neither %d"
      % ((M.n_tc > 0).sum(), (M.n_conv > 0).sum(), ((M.n_tc == 0) & (M.n_conv == 0)).sum()))

# ------------------------------------------------------------------ the equity table
VARS = [("median_income", "income", True), ("poverty_rate", "poverty", False),
        ("median_age", "age", False), ("minority_pct", "minority", False),
        ("dens", "rurality", True)]
rows = []
print("\n=== who bears the weather-attributable burden ===")
print("quartiles are of counties; 'disadvantaged' is the poorest / most deprived / most rural end\n")
for col, nm, asc in VARS:
    d = M.dropna(subset=[col]).copy()
    d["q"] = pd.qcut(d[col], 4, labels=[1, 2, 3, 4])
    t = d.groupby("q", observed=True).agg(n=("fips", "size"), pop=("population", "sum"),
                                          cust=("denom", "sum"), att=("att_all", "sum"),
                                          obs=("observed_cho", "sum"), ev=("exposure", "sum"))
    for c in ["pop", "cust", "att", "obs", "ev"]:
        t[c + "_s"] = 100 * t[c] / t[c].sum()
    dis = 1 if asc else 4
    print("   --- %s  (disadvantaged = Q%d)" % (nm, dis))
    print("      %-4s %7s %9s %9s %9s %9s" % ("Q", "pop %", "cust %", "ATTRIB %", "observed %", "events %"))
    for q in [1, 2, 3, 4]:
        r = t.loc[q]
        print("      %-4s %7.1f %9.1f %9.1f %9.1f %9.1f"
              % ("Q%d" % q, r.pop_s, r.cust_s, r.att_s, r.obs_s, r.ev_s))
    r = t.loc[dis]
    print("      -> disadvantaged quartile: %.1f%% of population, %.1f%% of attributable burden"
          % (r.pop_s, r.att_s))
    print("         ratio attributable/population = %.2f ; observed/population = %.2f"
          % (r.att_s / r.pop_s, r.obs_s / r.pop_s))
    rows.append(dict(variable=nm, disadvantaged_q=dis, pop_share=float(r.pop_s),
                     attributable_share=float(r.att_s), observed_share=float(r.obs_s),
                     event_share=float(r.ev_s),
                     ratio_att=float(r.att_s / r.pop_s), ratio_obs=float(r.obs_s / r.pop_s)))

# ------------------------------------------------------------------ exposure vs vulnerability
print("\n=== is any inequity exposure or vulnerability? ===")
print("vulnerability (the per-event effect) is common by construction: Stages 2-3 found no")
print("demographic gradient in it that survives multiplicity. So the whole gradient below is")
print("EXPOSURE - how many events a county gets - and that is what these correlations show.\n")
from scipy import stats as sst
print("   %-10s %14s %14s %14s" % ("trait", "rho exposure", "rho attributable", "rho vulnerab."))
for col, nm, asc in VARS:
    d = M.dropna(subset=[col])
    s = -1 if asc else 1
    r1 = sst.spearmanr(s * d[col], d.exposure)[0]
    r2 = sst.spearmanr(s * d[col], d.att_rate)[0]
    r3 = sst.spearmanr(s * d[col], d.beta_conv)[0]
    print("   %-10s %14.3f %14.3f %14.3f" % (nm, r1, r2, r3))
print("   (sign flipped so positive = the disadvantaged end has MORE)")
tot_a = M.att_all.sum(); tot_e = M.exposure_valued.sum()
print("\n   national split: exposure-valued %.3e (%.1f%%)   vulnerability part %.3e (%+.1f%%)"
      % (tot_e, 100 * tot_e / tot_a, M.vuln_part.sum(), 100 * M.vuln_part.sum() / tot_a))

# ------------------------------------------------------------------ grid form, separate table
print("\n=== grid form — MECHANISM, not inequity (separate by design) ===")
MG = M[M.grid_terc != "unobserved"].copy()
MG["gq"] = pd.Categorical(MG.grid_terc, categories=["low", "mid", "high"], ordered=True)
t = MG.groupby("gq", observed=True).agg(n=("fips", "size"), cust=("denom", "sum"),
                                        att=("att_all", "sum"), ev=("exposure", "sum"))
t["att_per_cust"] = t.att / t.cust
t["ev_per_county"] = t.ev / t.n
print(t[["n", "att_per_cust", "ev_per_county"]].round(3).to_string())
print("\n   share of each demographic quartile's counties that sit in the LOW-underground tercile:")
for col, nm, asc in VARS:
    d = MG.dropna(subset=[col]).copy()
    d["q"] = pd.qcut(d[col], 4, labels=[1, 2, 3, 4])
    sh = d.groupby("q", observed=True).apply(lambda x: 100 * (x.grid_terc == "low").mean())
    dis = 1 if asc else 4
    print("      %-10s Q1 %.0f%%  Q2 %.0f%%  Q3 %.0f%%  Q4 %.0f%%   (disadvantaged Q%d = %.0f%%)"
          % (nm, sh.get(1, 0), sh.get(2, 0), sh.get(3, 0), sh.get(4, 0), dis, sh.get(dis, 0)))
print("   (att_per_cust differences here mix exposure and the tercile effect estimated in Stage 3)")

M.to_parquet(f"{OUT}/county_attributable.parquet", index=False)
json.dump({"beta": BETA, "equity": rows,
           "attributable_total": float(M.att_all.sum()),
           "observed_total": float(M.observed_cho.sum()),
           "pct_attributable": float(100 * M.att_all.sum() / M.observed_cho.sum())},
          open(f"{OUT}/step4_equity.json", "w"), indent=1)
print("\nwrote", OUT)
