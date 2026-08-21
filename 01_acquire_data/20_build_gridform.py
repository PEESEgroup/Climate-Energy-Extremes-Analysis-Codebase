"""
County-level grid FORM, for Stage 3 (#66). Physical, pre-determined characteristics only - never
SAIDI, which is an accumulation of past weather and would be a post-treatment control.

The two federal filings use different units, so the harmonizable quantity is the UNDERGROUND
SHARE, which is dimensionless and means the same thing in both:
   FERC-1 (IOUs)  : (366 underground conduit + 367 underground conductors)
                    / (365 overhead conductors + 366 + 367)     -- dollars of plant in service
   RUS-7 (co-ops) : underground miles / (overhead + underground miles)
RUS additionally yields miles per customer, which FERC-1 cannot give.

Utility -> county through the EIA-861 service territory. Two allocations are produced because
neither is obviously right: the DOMINANT utility (largest by state customers among those serving
the county) and a customer-weighted average over all serving utilities. If Stage 3 moves between
them, that is reported rather than hidden.

Municipal, political-subdivision and state utilities file neither form: ~19% of US customers have
no observable grid form at all. That hole is written into the output as a coverage column so any
downstream regression can test whether it is correlated with the demographics of interest.
"""
import difflib, json, re
import numpy as np, pandas as pd

GP = "/data/equity_cost/gridphys"
OUT = "/data/equity_cost/gridphys/county_gridform.parquet"
YRS = (2015, 2022)

# ---------------------------------------------------------------- FERC-1 underground share
f1 = pd.read_parquet(f"{GP}/out_ferc1__yearly_plant_in_service_sched204.parquet")
f1 = f1[(f1.utility_type == "electric") & f1.report_year.between(*YRS)]
acc = f1[f1.ferc_account.astype(str).isin(["365", "366", "367"])]
w = acc.pivot_table(index=["utility_id_pudl", "report_year"], columns="ferc_account",
                    values="ending_balance", aggfunc="sum")
w.columns = [str(c) for c in w.columns]
w = w.dropna(subset=["365"])
w["ug"] = w.get("366", 0).fillna(0) + w.get("367", 0).fillna(0)
w["tot"] = w["365"].fillna(0) + w["ug"]
w = w[w.tot > 0]
w["ug_share"] = w["ug"] / w["tot"]
ferc = w.groupby("utility_id_pudl").ug_share.mean().rename("ug_share_ferc").reset_index()
assn = pd.read_parquet(f"{GP}/core_pudl__assn_eia_pudl_utilities.parquet")
ferc = ferc.merge(assn[["utility_id_pudl", "utility_id_eia"]].dropna().drop_duplicates(),
                  on="utility_id_pudl", how="inner")
print("FERC-1 utilities with an underground share and an EIA id: %d" % ferc.utility_id_eia.nunique())
print("   ug_share: median %.3f  IQR %.3f-%.3f"
      % (ferc.ug_share_ferc.median(), ferc.ug_share_ferc.quantile(.25),
         ferc.ug_share_ferc.quantile(.75)))

# ---------------------------------------------------------------- RUS-7 underground share
mil = pd.read_parquet(f"{GP}/core_rus7__yearly_transmission_and_distribution_mileage.parquet")
mil["year"] = pd.to_datetime(mil.report_date).dt.year
mil = mil[mil.year.between(2015, 2021)]
pv = mil.pivot_table(index=["borrower_id_rus", "year"], columns="line_type", values="miles",
                     aggfunc="sum")
pv = pv.dropna(subset=["distribution_overhead"])
pv["ug"] = pv.get("distribution_underground", 0).fillna(0)
pv["tot"] = pv["distribution_overhead"].fillna(0) + pv["ug"]
pv = pv[pv.tot > 0]
pv["ug_share"] = pv["ug"] / pv["tot"]
cus = pd.read_parquet(f"{GP}/core_rus7__yearly_power_requirements_electric_customers.parquet")
cus["year"] = pd.to_datetime(cus.report_date).dt.year
cus = cus[(cus.observation_period == "avg")].groupby(["borrower_id_rus", "year"]).customers_num.sum()
pv = pv.join(cus.rename("cust"))
pv["mi_per_cust"] = pv["tot"] / pv["cust"].replace(0, np.nan)
rus = pv.groupby("borrower_id_rus")[["ug_share", "mi_per_cust"]].mean().reset_index()
rus.columns = ["borrower_id_rus", "ug_share_rus", "mi_per_cust_rus"]
bor = pd.read_parquet(f"{GP}/core_rus7__entity_borrowers.parquet")
rus = rus.merge(bor, on="borrower_id_rus", how="left")
print("\nRUS-7 borrowers with a share: %d   ug_share median %.3f   miles/customer median %.4f"
      % (len(rus), rus.ug_share_rus.median(), rus.mi_per_cust_rus.median()))

# fuzzy name+state match to EIA
sal = pd.read_parquet(f"{GP}/core_eia861__yearly_sales.parquet")
eia = (sal.groupby(["utility_id_eia", "utility_name_eia", "state"], observed=True)
          .customers.max().reset_index())
STOP = ["electric", "cooperative", "coop", "co op", "corporation", "corp", "inc", "company",
        "co", "association", "assn", "power", "membership", "the", "utilities", "utility",
        "of", "and", "rural", "system", "systems", "energy", "emc", "rea", "public", "service"]


def norm(s):
    s = re.sub(r"[^a-z0-9 ]", " ", str(s).lower())
    for x in STOP:
        s = re.sub(r"\b%s\b" % x, " ", s)
    return re.sub(r"\s+", " ", s).strip()


eia["key"] = eia.utility_name_eia.map(norm)
rus["key"] = rus.borrower_name_rus.map(norm)
by_state = {s: g for s, g in eia.groupby("state")}
hits = []
for _, r in rus.iterrows():
    g = by_state.get(r.state)
    if g is None or not r.key:
        continue
    exact = g[g.key == r.key]
    if len(exact):
        hits.append((r.borrower_id_rus, int(exact.iloc[0].utility_id_eia), 1.0)); continue
    cand = difflib.get_close_matches(r.key, g.key.tolist(), n=1, cutoff=0.86)
    if cand:
        m = g[g.key == cand[0]].iloc[0]
        hits.append((r.borrower_id_rus, int(m.utility_id_eia),
                     difflib.SequenceMatcher(None, r.key, cand[0]).ratio()))
X = pd.DataFrame(hits, columns=["borrower_id_rus", "utility_id_eia", "score"]).drop_duplicates(
    "borrower_id_rus")
print("   RUS -> EIA matched: %d of %d (%.1f%%)   exact %d"
      % (len(X), len(rus), 100 * len(X) / len(rus), int((X.score == 1).sum())))
rus = rus.merge(X, on="borrower_id_rus", how="inner")

# ---------------------------------------------------------------- combine and allocate
U = pd.concat([
    ferc[["utility_id_eia", "ug_share_ferc"]].rename(columns={"ug_share_ferc": "ug_share"})
        .assign(src="ferc1", mi_per_cust=np.nan),
    rus[["utility_id_eia", "ug_share_rus", "mi_per_cust_rus"]]
        .rename(columns={"ug_share_rus": "ug_share", "mi_per_cust_rus": "mi_per_cust"})
        .assign(src="rus7")], ignore_index=True)
U = U.dropna(subset=["utility_id_eia"]).groupby("utility_id_eia").agg(
    ug_share=("ug_share", "mean"), mi_per_cust=("mi_per_cust", "mean"),
    src=("src", "first")).reset_index()

# THE SHARES ARE NOT ON ONE SCALE. FERC-1 is dollars of plant, RUS-7 is miles, and undergrounding
# costs roughly 8x per mile - which is exactly what the medians imply (0.111 miles <-> 0.514
# dollars => k ~ 8.2). Each is internally consistent; pooled, the variable would partly encode
# WHICH FORM A UTILITY FILES rather than how its grid is built, manufacturing a spurious
# IOU-vs-cooperative contrast. So standardise WITHIN source: the variable becomes "more
# underground than peers of the same type", which is the comparison Stage 3 actually wants.
U["ug_z"] = U.groupby("src").ug_share.transform(lambda v: (v - v.mean()) / v.std())
print("\nutilities with an observable grid form: %d" % len(U))
print("   ug_share by source:")
print(U.groupby("src").ug_share.describe()[["count", "50%", "std"]].to_string())
print("   -> pooling raw shares would encode the filing form; using the within-source z-score")

st = pd.read_parquet(f"{GP}/core_eia861__yearly_service_territory.parquet")
st["year"] = pd.to_datetime(st.report_date).dt.year
st = st[st.year.between(*YRS)][["utility_id_eia", "county_id_fips", "state"]].dropna()
st = st.drop_duplicates()
sz = eia.groupby(["utility_id_eia", "state"]).customers.max().rename("cust").reset_index()
st = st.merge(sz, on=["utility_id_eia", "state"], how="left")
st["cust"] = st.cust.fillna(0)
st = st.merge(U, on="utility_id_eia", how="left")
st["has"] = st.ug_z.notna()

g = st.groupby("county_id_fips")
cov = g.apply(lambda d: d.loc[d.has, "cust"].sum() / max(d.cust.sum(), 1)).rename("cust_covered")
wavg = g.apply(lambda d: np.average(d.loc[d.has, "ug_z"],
               weights=d.loc[d.has, "cust"] + 1) if d.has.any() else np.nan).rename("ug_share_wavg")
dom = g.apply(lambda d: d.loc[d.has].sort_values("cust").iloc[-1].ug_z
              if d.has.any() else np.nan).rename("ug_share_dom")
raw = g.apply(lambda d: d.loc[d.has].sort_values("cust").iloc[-1].ug_share
              if d.has.any() else np.nan).rename("ug_share_raw_dom")
src = g.apply(lambda d: d.loc[d.has].sort_values("cust").iloc[-1].src
              if d.has.any() else None).rename("src_dom")
mipc = g.apply(lambda d: np.average(d.loc[d.has & d.mi_per_cust.notna(), "mi_per_cust"],
               weights=d.loc[d.has & d.mi_per_cust.notna(), "cust"] + 1)
               if (d.has & d.mi_per_cust.notna()).any() else np.nan).rename("mi_per_cust")
nut = g.size().rename("n_utilities")
C = pd.concat([cov, wavg, dom, raw, src, mipc, nut], axis=1).reset_index()
C = C.rename(columns={"county_id_fips": "fips"})
C.to_parquet(OUT, index=False)
print("\ncounties with any grid form: %d of %d   median customer coverage %.2f"
      % (C.ug_share_dom.notna().sum(), len(C), C.cust_covered.median()))
print("   ug z-score (within source)  median %.3f  IQR %.3f-%.3f"
      % (C.ug_share_dom.median(), C.ug_share_dom.quantile(.25), C.ug_share_dom.quantile(.75)))
print("   raw share by dominant source:")
print(C.groupby("src_dom").ug_share_raw_dom.agg(["size", "median"]).to_string())
print("   wavg vs dom correlation: %.3f" % C[["ug_share_wavg", "ug_share_dom"]].corr().iloc[0, 1])
print("WROTE", OUT)

# is the unobserved hole correlated with what Stage 3 cares about?
E = pd.read_csv("/data/equity_cost/analysis/equity_joined_v2.csv", dtype={"fips": str})
M = C.merge(E[["fips", "median_income", "poverty_rate", "median_age", "minority_pct",
               "population", "land_km2"]], on="fips", how="inner")
M["dens"] = M.population / M.land_km2.replace(0, np.nan)
M["observed"] = M.ug_share_dom.notna().astype(float)
print("\nIS THE COVERAGE HOLE SELECTIVE? (observed = 1 if a grid form exists)")
from scipy import stats as sst
for v in ["median_income", "poverty_rate", "median_age", "minority_pct", "dens"]:
    a = M.loc[M.observed == 1, v].dropna(); b = M.loc[M.observed == 0, v].dropna()
    if len(b) > 5:
        t, p = sst.ttest_ind(a, b, equal_var=False)
        print("   %-14s observed %10.2f   unobserved %10.2f   p %.4g"
              % (v, a.median(), b.median(), p))
json.dump({"n_counties_with_form": int(C.ug_share_dom.notna().sum()),
           "n_counties": int(len(C)), "median_cust_coverage": float(C.cust_covered.median()),
           "n_utilities_with_form": int(len(U))},
          open("/data/equity_cost/gridphys/gridform_build.json", "w"), indent=1)
