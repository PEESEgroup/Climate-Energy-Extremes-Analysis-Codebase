"""
R4, reduced to the two facts that carry the argument.

R3 identified exactly one modifiable county characteristic that changes how hard a storm lands:
the form of the distribution grid. R4 asks who is moving it, and the answer is nobody. Two
quantities, both from filed data, no modelling:

  A. For each investor-owned utility, the underground share of the distribution plant it ALREADY
     has against the underground share of everything it BUILT in 2014-2023. A utility on the 45
     degree line is replicating its own grid: it is spending, and its underground share will never
     change. This is the same variable Stage 3 used, so the panel attaches directly to R3.

  B. Federal discretionary pre-disaster money per customer against the weather-attributable outage
     burden per customer that R3 measured. The y axis is our own Stage-1 output, not a proxy.

Nothing here is a projection and nothing is causal - these are accounting facts about filed
capital and appropriated dollars.
"""
import json
import numpy as np, pandas as pd
from scipy import stats as st

GP = "/data/equity_cost/gridphys"
AN = "/data/equity_cost/analysis"
OUT = "/data/equity_cost/analysis/r4_ledger.json"
YRS = (2014, 2023)
ACC = ["364", "365", "366", "367"]      # overhead conductors, poles, underground conduit, cable

# ------------------------------------------------------------------ panel A
f = pd.read_parquet(f"{GP}/out_ferc1__yearly_plant_in_service_sched204.parquet")
f = f[(f.utility_type == "electric") & f.report_year.between(*YRS)].copy()
f["acc"] = f.ferc_account.astype(str)
f = f[f.acc.isin(ACC)]

# stock: the underground share of what stands, averaged over the decade so one odd filing year
# cannot move a utility across the 45 degree line
stk = f.pivot_table(index=["utility_id_pudl", "report_year"], columns="acc",
                    values="ending_balance", aggfunc="sum").dropna()
stk["ug"] = (stk["366"] + stk["367"]) / stk[ACC].sum(1)
stk["tot"] = stk[ACC].sum(1)
stk = stk[(stk.ug > 0) & (stk.ug < 1)].reset_index()

# flow: the underground share of the DECADE's additions, summed first and divided once, so a year
# with tiny additions cannot produce a wild ratio
add = f.pivot_table(index="utility_id_pudl", columns="acc", values="additions", aggfunc="sum")
add = add.dropna()
add = add[(add[ACC] >= 0).all(1)]                       # negative additions are restatements
add["addtot"] = add[ACC].sum(1)
add["ugadd"] = (add["366"] + add["367"]) / add["addtot"]
add = add[add.addtot > 0][["ugadd", "addtot"]]

ny = stk.groupby("utility_id_pudl").report_year.nunique()
keep = ny[ny >= 8].index                                # a utility must be present most of the decade
A = (stk[stk.utility_id_pudl.isin(keep)]
     .groupby("utility_id_pudl").agg(ug=("ug", "mean"), plant=("tot", "mean"))
     .join(add, how="inner").reset_index())
A["z"] = (A.ug - A.ug.mean()) / A.ug.std()              # within-source z, as Stage 3 built it
qz = A.z.quantile([1 / 3, 2 / 3]).values
A["terc"] = np.where(A.z < qz[0], "least", np.where(A.z < qz[1], "middle", "most"))
A["drift"] = A.ugadd - A.ug                             # >0 the share rises, <0 it falls
nm = f[["utility_id_pudl", "utility_name_ferc1"]].drop_duplicates("utility_id_pudl")
A = A.merge(nm, on="utility_id_pudl", how="left")
# California is the one state where a hazard forced a decade of unprecedented distribution capital,
# so it is the test case for whether hazard pressure changes the MIX. Its three IOUs are labelled.
CA_PAT = "pacific gas|southern california edison|san diego gas"
A["ca"] = A.utility_name_ferc1.str.lower().str.contains(CA_PAT, na=False)
A["short"] = A.utility_name_ferc1.str.replace(
    "PACIFIC GAS AND ELECTRIC COMPANY", "PG&E", regex=False).str.replace(
    "Southern California Edison Company", "SCE", regex=False).str.replace(
    "San Diego Gas & Electric Company", "SDG&E", regex=False)

print("panel A: %d utilities, %d-%d" % (len(A), *YRS))
print("   stock ug share      p10 %.3f  p50 %.3f  p90 %.3f" %
      tuple(A.ug.quantile([.1, .5, .9])))
print("   additions ug share  p10 %.3f  p50 %.3f  p90 %.3f" %
      tuple(A.ugadd.quantile([.1, .5, .9])))
print("   drift (add - stock) p10 %+.3f  p50 %+.3f  p90 %+.3f  mean %+.3f" %
      (*A.drift.quantile([.1, .5, .9]), A.drift.mean()))
print("   share of utilities building a HIGHER underground share than they hold: %.1f%%"
      % (100 * (A.drift > 0).mean()))
tt = st.ttest_1samp(A.drift, 0.0)
print("   is the mean drift different from zero?  t %+.2f  p %.3f" % (tt.statistic, tt.pvalue))
for t in ["least", "middle", "most"]:
    d = A[A.terc == t]
    print("   %-6s tercile n %3d   stock %.3f   additions %.3f   drift %+.4f"
          % (t, len(d), d.ug.median(), d.ugadd.median(), d.drift.median()))

# how long to move the least-undergrounded tercile up one tercile, at the observed drift?
gap_z = qz[0] - A.z.min()                                # not used; the honest gap is tercile-to-tercile
need_z = qz[0] - A[A.terc == "least"].z.median()
need_share = need_z * A.ug.std()
lo = A[A.terc == "least"]
# annual movement of the stock share implied by building at `ugadd` when you hold `ug`:
# d(ug)/dt ~ (ugadd - ug) * additions / plant
lo_rate = ((lo.ugadd - lo.ug) * lo.addtot / 10.0 / lo.plant)
print("   least tercile must gain %.3f share points to reach the middle tercile" % need_share)
print("   its implied annual drift: median %+.5f /yr  p90 %+.5f /yr"
      % (lo_rate.median(), lo_rate.quantile(.9)))
yrs_med = need_share / lo_rate.median() if lo_rate.median() > 0 else np.inf
yrs_p90 = need_share / lo_rate.quantile(.9) if lo_rate.quantile(.9) > 0 else np.inf
print("   years to close it: median utility %.0f   fastest decile %.0f" % (yrs_med, yrs_p90))

# ------------------------------------------------------------------ panel B
S = pd.read_csv(f"{AN}/state_risk_investment_v3.csv")
S = S[S.panel == "50_states_plus_DC"]
M = pd.read_parquet(f"{AN}/step4/county_attributable.parquet")   # already carries `state`
M["fips"] = M.fips.astype(str).str.zfill(5)
M = M.dropna(subset=["state"])
B = (M.groupby("state")[["att_all", "att_tc", "att_conv", "denom"]].sum()
     .assign(att_per_cust=lambda d: d.att_all / d.denom,
             tc_per_cust=lambda d: d.att_tc / d.denom).reset_index())
B = B.merge(S[["state", "proactive_noncirc_per_cust", "cho_per_cust", "total_customers"]],
            on="state", how="inner")
B = B.dropna(subset=["proactive_noncirc_per_cust", "att_per_cust"])
# THE COVERAGE HOLE, stated rather than hidden. The attributable measure contains tropical cyclones
# and severe convection only; the fire-weather arm did not replicate and was withdrawn. Nine states
# therefore have an attributable burden of exactly zero - a floor set by what the hazard set covers,
# not a measurement that their grids are safe. They are flagged so the figure can show it.
B["zero_cov"] = B.att_per_cust < 0.05
r_att = st.pearsonr(B.proactive_noncirc_per_cust, B.att_per_cust)
s_att = st.spearmanr(B.proactive_noncirc_per_cust, B.att_per_cust)
r_all = st.pearsonr(B.proactive_noncirc_per_cust, B.cho_per_cust)
print("\npanel B: %d states + DC" % len(B))
print("   discretionary pre-disaster $/customer over 10 yrs: median %.2f  p90 %.2f"
      % (B.proactive_noncirc_per_cust.median(), B.proactive_noncirc_per_cust.quantile(.9)))
print("   weather-attributable outage, customer-hours per customer: median %.1f  p90 %.1f"
      % (B.att_per_cust.median(), B.att_per_cust.quantile(.9)))
print("   money vs WEATHER-ATTRIBUTABLE burden: r %+.3f (p %.3f)   rho %+.3f (p %.3f)"
      % (r_att[0], r_att[1], s_att[0], s_att[1]))
print("   money vs ALL outage burden:           r %+.3f (p %.3f)" % (r_all[0], r_all[1]))
z = B[B.zero_cov]
print("   COVERAGE FLOOR: %d states have zero attributable burden: %s"
      % (len(z), " ".join(sorted(z.state))))
print("      they hold %.1f%% of customers, take %.1f%% of the discretionary money,"
      % (100 * z.total_customers.sum() / B.total_customers.sum(),
         100 * (z.proactive_noncirc_per_cust * z.total_customers).sum()
         / (B.proactive_noncirc_per_cust * B.total_customers).sum()))
print("      and their median TOTAL outage burden is %.1f vs %.1f for the rest"
      % (z.cho_per_cust.median(), B[~B.zero_cov].cho_per_cust.median()))
ca = A[A.ca]
print("\n   California IOUs (the hazard-pressure test case):")
for _, r_ in ca.iterrows():
    print("      %-6s stock %.3f  additions %.3f  drift %+.4f  = pctile %.0f of %d"
          % (r_.short, r_.ug, r_.ugadd, r_.drift, 100 * (A.drift < r_.drift).mean(), len(A)))

A.to_csv("/data/equity_cost/analysis/r4_panelA_utilities.csv", index=False)
B.to_csv("/data/equity_cost/analysis/r4_panelB_states.csv", index=False)
json.dump({"panelA": {"n_utilities": int(len(A)), "years": list(YRS),
                      "stock_p10_50_90": A.ug.quantile([.1, .5, .9]).tolist(),
                      "add_p10_50_90": A.ugadd.quantile([.1, .5, .9]).tolist(),
                      "drift_p10_50_90": A.drift.quantile([.1, .5, .9]).tolist(),
                      "drift_mean": float(A.drift.mean()),
                      "drift_t": float(tt.statistic), "drift_p": float(tt.pvalue),
                      "frac_rising": float((A.drift > 0).mean()),
                      "tercile_cuts_z": qz.tolist(),
                      "need_share_points": float(need_share),
                      "years_to_close_median": float(yrs_med),
                      "years_to_close_p90": float(yrs_p90)},
           "coverage_floor_states": sorted(B[B.zero_cov].state.tolist()),
           "california": {r_.short: dict(stock=float(r_.ug), additions=float(r_.ugadd),
                                         drift=float(r_.drift),
                                         pctile=float(100 * (A.drift < r_.drift).mean()))
                          for _, r_ in A[A.ca].iterrows()},
           "panelB": {"n_states": int(len(B)),
                      "money_median_per_cust_10yr": float(B.proactive_noncirc_per_cust.median()),
                      "att_median_per_cust": float(B.att_per_cust.median()),
                      "r_att": list(r_att), "rho_att": [float(s_att[0]), float(s_att[1])],
                      "r_all": list(r_all)}},
          open(OUT, "w"), indent=1, default=float)
print("\nwrote %s" % OUT)
