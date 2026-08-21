"""The remaining numbers for section 4: the two total-outage ratios in the burden paragraph, and
the funding correlations, both on the attribution that Figure 3b now carries."""
import numpy as np, pandas as pd
import sys
import os as _os_rp
for _rp in ("04_demand_model", "07_hazard_calendar", "09_outage_attribution",
            "02_downscale_wind", "12_figures"):
    _ap = _os_rp.path.abspath(_os_rp.path.join(
        _os_rp.path.dirname(_os_rp.path.abspath(__file__)), "..", _rp))
    if _os_rp.path.isdir(_ap) and _ap not in sys.path:
        sys.path.insert(0, _ap)
import attrib_artifacts as AA          # F3.9: the screen's artifact names live in one place
import hazsets                        # F4.2: the screen's hazard SET lives in one place
from scipy import stats as st
A = "/data/equity_cost/analysis"
M = AA.read_screened("%s/attrib" % A)
hazsets.check_columns(M, where="12_eq4b.py")
E = pd.read_csv("%s/equity_joined_v2.csv" % A, dtype={"fips": str})
GF = pd.read_parquet("/data/equity_cost/gridphys/county_gridform.parquet")[["fips", "ug_share_dom"]]
D = M.merge(E[["fips", "state", "population", "land_km2", "median_income", "poverty_rate",
               "median_age", "minority_pct"]], on="fips", how="left").merge(GF, on="fips", how="left")
D["dens"] = D.population / D.land_km2.replace(0, np.nan)

TR = [("median_age", "median age", False), ("poverty_rate", "poverty", False),
      ("minority_pct", "minority share", False), ("median_income", "income", True),
      ("dens", "rurality", True), ("ug_share_dom", "undergrounding", True)]
print("%-16s %9s %11s %8s %13s" % ("trait", "pop share", "burden sh", "ratio", "all-outage r"))
for col, nm, low_bad in TR:
    v = D[col]; ok = v.notna() & D.population.notna()
    cut = v[ok].quantile(0.2 if low_bad else 0.8)
    sel = ok & ((v <= cut) if low_bad else (v >= cut))
    ps = D.population[sel].sum() / D.population[ok].sum()
    bs = D.att_screened[sel].sum() / D.att_screened[ok].sum()
    os_ = D.observed_cho[sel].sum() / D.observed_cho[ok].sum()
    print("%-16s %8.1f%% %10.1f%% %8.2f %13.2f" % (nm, 100 * ps, 100 * bs, bs / ps, os_ / ps))

print()
print("funding panel, on the new attribution")
# The y axis of Figure 4e is att_rate, built from the screened total, NOT from all five hazards.
print("   its y axis is the SCREENED burden, over %s" % ", ".join(hazsets.screened()))
S = D.groupby("state").agg(att=("att_screened", "sum"), cust=("denom", "sum"),
                           obs=("observed_cho", "sum")).reset_index()
S["att_rate"] = S.att / S.cust
S["obs_rate"] = S.obs / S.cust
B = pd.read_csv("%s/r4_panelB_states.csv" % A)[["state", "proactive_noncirc_per_cust", "zero_cov"]]
J = S.merge(B, on="state", how="inner").dropna(subset=["proactive_noncirc_per_cust"])
print("states %d, of which zero attributable %d" % (len(J), int((J.att_rate == 0).sum())))
r, p = st.spearmanr(J.proactive_noncirc_per_cust, J.att_rate)
print("   Spearman funding vs attributable per customer: %+.3f (p = %.3f)" % (r, p))
r2, p2 = st.pearsonr(np.log10(J.proactive_noncirc_per_cust.clip(lower=1e-3)), J.att_rate)
print("   Pearson on log funding:                        %+.3f (p = %.3f)" % (r2, p2))
K = J[J.state != "LA"]
r3, p3 = st.pearsonr(np.log10(K.proactive_noncirc_per_cust.clip(lower=1e-3)), K.att_rate)
print("   the same without Louisiana:                    %+.3f (p = %.3f)" % (r3, p3))
r4, p4 = st.spearmanr(J.proactive_noncirc_per_cust, J.obs_rate)
print("   Spearman funding vs ALL-cause outage:          %+.3f (p = %.5f)" % (r4, p4))
west = J[(J.att_rate == 0)]
print("   states at zero attributable: %s" % ", ".join(sorted(west.state)))
J.to_csv("%s/attrib/fig4e_states.csv" % A, index=False)
print("wrote fig4e_states.csv")
