"""Assemble every number Figure 4 needs, and check each one before it is drawn.

Panels (a) and (b) carry the five county traits. Undergrounding is a property of the distribution
system that a utility chooses, not a characteristic of the people in the county, so it is moved to
its own panel (c) and labelled as an association.
"""
import json
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
A = "/data/equity_cost/analysis"
AT = "%s/attrib" % A
M = AA.read_screened(AT)
hazsets.check_columns(M, where="16_fig4data.py")
print("screened hazards: %s" % ", ".join(hazsets.screened()))
E = pd.read_csv("%s/equity_joined_v2.csv" % A, dtype={"fips": str})
GF = pd.read_parquet("/data/equity_cost/gridphys/county_gridform.parquet")[["fips", "ug_share_dom"]]
D = M.merge(E[["fips", "state", "population", "land_km2", "median_income", "poverty_rate",
               "median_age", "minority_pct"]], on="fips", how="left").merge(GF, on="fips", how="left")
D["dens"] = D.population / D.land_km2.replace(0, np.nan)

TR = [("median_age", "median age", False), ("poverty_rate", "poverty", False),
      ("minority_pct", "minority share", False), ("median_income", "income", True),
      ("dens", "rurality", True)]
T = json.load(open("%s/tercile_gaps_v2.json" % AT))
# EVERY INPUT IS CHECKED BEFORE ANY OUTPUT IS WRITTEN. This check used to sit below the panel (a)
# block, so a gaps file left over from an earlier hazard set rewrote fig4a.csv and then stopped,
# leaving the three CSVs describing two different screens.
FAM = sorted({(nm, k.split("|")[0]) for nm, d in T.items()
              for k in d["results"] if k.split("|")[1] == "impact"})
HZ_T = sorted({h for _, h in FAM})
if HZ_T != sorted(hazsets.screened()):
    raise SystemExit("tercile_gaps_v2.json carries %s but the screen kept %s; rerun 07_terc.py"
                     % (HZ_T, sorted(hazsets.screened())))
# THE BONFERRONI BAR IS COUNTED, NOT TYPED. It was the literal 2.865, the bar for six traits by two
# hazards. The screen no longer keeps two hazards, so a typed constant is the bar for a family that
# no longer exists. It is now counted from the gaps file: every (trait, hazard) event-day contrast
# 07_terc.py actually estimated.
ZB = hazsets.bonferroni_z(len(FAM))
# Panels (b) and (c) draw ONE hazard, the one the caption names. Checked, never assumed.
PANEL_HAZ = "convective"
if PANEL_HAZ not in HZ_T:
    raise SystemExit("panels (b) and (c) draw %s, which the screen no longer keeps" % PANEL_HAZ)

print("PANEL A, five county traits, disadvantaged fifth")
print("%-16s %7s %10s %11s %8s %13s %10s"
      % ("trait", "n", "pop share", "burden sh", "ratio", "all-outage r", "no FL+LA"))
rows = []
for col, nm, low_bad in TR:
    v = D[col]; ok = v.notna() & D.population.notna()
    cut = v[ok].quantile(0.2 if low_bad else 0.8)
    sel = ok & ((v <= cut) if low_bad else (v >= cut))
    ps = D.population[sel].sum() / D.population[ok].sum()
    bs = D.att_screened[sel].sum() / D.att_screened[ok].sum()
    os_ = D.observed_cho[sel].sum() / D.observed_cho[ok].sum()
    # The burden is concentrated enough that a national ratio can be one state. The same ratio is
    # therefore computed with Florida and Louisiana removed, and both are reported.
    d2 = D[~D.state.isin(["FL", "LA"])]
    v2 = d2[col]; ok2 = v2.notna() & d2.population.notna()
    cut2 = v2[ok2].quantile(0.2 if low_bad else 0.8)
    sel2 = ok2 & ((v2 <= cut2) if low_bad else (v2 >= cut2))
    r2 = (d2.att_screened[sel2].sum() / d2.att_screened[ok2].sum()) / (d2.population[sel2].sum() / d2.population[ok2].sum())
    rows.append(dict(trait=nm, n=int(sel.sum()), pop_share=ps, burden_share=bs,
                     ratio=bs / ps, allcause_ratio=os_ / ps, ratio_ex=r2))
    print("%-16s %7d %9.1f%% %10.1f%% %8.2f %13.2f %10.2f"
          % (nm, sel.sum(), 100 * ps, 100 * bs, bs / ps, os_ / ps, r2))
pd.DataFrame(rows).to_csv("%s/fig4a.csv" % AT, index=False)

NM5 = ["median age", "poverty", "minority share", "income", "rurality"]
print()
print("PANEL B, %s only. The other screened hazards are estimated, and counted into the bar,\n"
      "         but are reported in the Supplementary Information rather than drawn here."
      % hazsets.label(PANEL_HAZ))
print("%-16s %10s %8s %10s %8s %9s %s" % ("trait", "lead", "z", "impact", "z", "multiple", "verdict"))
rowsb = []
for nm in NM5:
    r = T[nm]["results"]
    L, I = r["%s|lead" % PANEL_HAZ], r["%s|impact" % PANEL_HAZ]
    ok = abs(L["z"]) < 1.96
    surv = ok and abs(I["z"]) > ZB
    print("%-16s %+10.3f %+8.2f %+10.3f %+8.2f %9.2f %s"
          % (nm, L["gap"], L["z"], I["gap"], I["z"], np.exp(I["gap"]),
             "clears" if surv else ("placebo fails" if not ok else "below the bar")))
    rowsb.append(dict(trait=nm, lead=L["gap"], lead_z=L["z"], gap=I["gap"], gap_se=I["gap_se"],
                      z=I["z"], mult=float(np.exp(I["gap"])), clears=bool(surv),
                      zb=ZB, n_family=len(FAM), panel_hazard=PANEL_HAZ))
pd.DataFrame(rowsb).to_csv("%s/fig4b.csv" % AT, index=False)
# The bar travels with the data it gated, so the figure cannot draw one bar and print another.
print("   Bonferroni bar over %d traits x %d hazards (%s) = %d tests: |z| > %.3f"
      % (len(NM5) + 1, len(HZ_T), ", ".join(HZ_T), len(FAM), ZB))

print()
print("PANEL C, undergrounding, %s, by block, against the middle tercile"
      % hazsets.label(PANEL_HAZ))
r = T["undergrounding"]["results"]
print("%-12s %12s %12s %12s %8s" % ("block", "least ug", "most ug", "gap", "z"))
rowsc = []
for b, lab in (("lead", "-14 to -8"), ("antic", "-7 to -1"), ("impact", "0 to +1"),
               ("restore", "+2 to +6"), ("tail", "+7 to +14")):
    k = "%s|%s" % (PANEL_HAZ, b)
    if k not in r:
        continue
    d = r[k]
    print("%-12s %12.3f %12.3f %+12.3f %+8.2f" % (lab, d["worst"], d["best"], d["gap"], d["z"]))
    rowsc.append(dict(block=lab, least=d["worst"], most=d["best"], gap=d["gap"],
                      gap_se=d["gap_se"], z=d["z"]))
pd.DataFrame(rowsc).to_csv("%s/fig4c.csv" % AT, index=False)
print("   counties: least undergrounded %d, most %d"
      % (T["undergrounding"]["n_worst"], T["undergrounding"]["n_best"]))
print("   this panel is an association: undergrounding is chosen by the utility and moves with")
print("   density, income and urban form, so the tercile contrast is not the effect of burying line")
