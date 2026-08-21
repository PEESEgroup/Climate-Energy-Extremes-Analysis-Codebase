#!/usr/bin/env python
"""Distributional outage-equity analysis. Writes only under /data/equity_cost/analysis/."""
import numpy as np, pandas as pd, json
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/data/equity_cost/analysis"
np.random.seed(0)

# ---------- (1) JOIN ----------
b = pd.read_csv(f"{BASE}/eaglei_county_total_v2.csv", dtype={"fips": str})
a = pd.read_csv(f"{BASE}/acs_county.csv", dtype={"fips": str})
b["fips"] = b["fips"].str.zfill(5)
a["fips"] = a["fips"].str.zfill(5)
df = b.merge(a, on="fips", how="inner")
print(f"[join] burden rows {len(b)}, acs rows {len(a)}, merged {len(df)}")

# clean sentinel / missing
df.loc[df["median_income"] < 0, "median_income"] = np.nan
# analysis subset: valid per-customer burden + demographics
D0 = df.dropna(subset=["customer_hours_per_customer", "median_income", "minority_pct",
                       "poverty_rate", "median_age"]).copy()
D0 = D0[D0["customer_hours_per_customer"] > 0].copy()
# PHYSICAL-VALIDITY FILTER: peak customers-out can't exceed served customers (MCC denom).
# Counties failing this have a broken MCC denominator that inflates per-customer burden
# by orders of magnitude (e.g. 37089: mcc=24 but peak_out=131,460). Drop them for the
# per-customer disparity analysis (they stay in national/event totals, which use raw hours).
valid = D0["mcc_customers"] >= D0["peak_customers_out"]
n_bad = int((~valid).sum())
bad_examples = D0[~valid].sort_values("customer_hours_per_customer", ascending=False).head(5)[
    ["fips", "state", "mcc_customers", "peak_customers_out", "customer_hours_per_customer"]]
print(f"[validity filter] dropped {n_bad} counties where peak_out > mcc_customers "
      f"(broken denom). Examples:\n{bad_examples.to_string(index=False)}")
D = D0[valid].copy()
print(f"[analysis subset] {len(D)} valid counties (of {len(D0)} w/ demographics); "
      f"pop covered {D['population'].sum():,.0f}")
D["burden"] = D["customer_hours_per_customer"]
D["logburden"] = np.log(D["burden"])

natl_ch = df["total_customer_hours_out"].sum()
print(f"[national] total customer-hours-out {natl_ch:,.0f}")

# ---------- (2) QUINTILE DISPARITY ----------
def quintile_table(data, col, ascending=True, label=None):
    d = data.dropna(subset=[col]).copy()
    d["q"] = pd.qcut(d[col].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    # population-weighted AND unweighted mean burden per quintile
    g = d.groupby("q", observed=True).apply(
        lambda x: pd.Series({
            "n": len(x),
            "mean_" + col: x[col].mean(),
            "mean_burden": x["burden"].mean(),
            "median_burden": x["burden"].median(),
            "popwt_burden": np.average(x["burden"], weights=x["population"]),
            "pop": x["population"].sum(),
        }), include_groups=False)
    return g

results = {}
for col, asc in [("median_income", True), ("poverty_rate", True),
                 ("minority_pct", True), ("median_age", True)]:
    g = quintile_table(D, col)
    results[col] = g
    print(f"\n===== quintiles by {col} =====")
    print(g.to_string())

# key ratios
def ratio(data, col, hi_is_group5=True):
    d = data.dropna(subset=[col]).copy()
    d["q"] = pd.qcut(d[col].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    m = d.groupby("q", observed=True)["burden"].mean()
    mp = d.groupby("q", observed=True).apply(
        lambda x: np.average(x["burden"], weights=x["population"]), include_groups=False)
    return m, mp

def med_by_q(data, col):
    d = data.dropna(subset=[col]).copy()
    d["q"] = pd.qcut(d[col].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    return d.groupby("q", observed=True)["burden"].median()

inc_m, inc_mp = ratio(D, "median_income")
pov_m, pov_mp = ratio(D, "poverty_rate")
min_m, min_mp = ratio(D, "minority_pct")
age_m, age_mp = ratio(D, "median_age")
inc_md = med_by_q(D, "median_income"); pov_md = med_by_q(D, "poverty_rate")
min_md = med_by_q(D, "minority_pct"); age_md = med_by_q(D, "median_age")

ratios = {
    "poorest_vs_richest_income_unw": inc_m[1] / inc_m[5],
    "poorest_vs_richest_income_popwt": inc_mp[1] / inc_mp[5],
    "highest_vs_lowest_poverty_unw": pov_m[5] / pov_m[1],
    "highest_vs_lowest_poverty_popwt": pov_mp[5] / pov_mp[1],
    "highest_vs_lowest_minority_unw": min_m[5] / min_m[1],
    "highest_vs_lowest_minority_popwt": min_mp[5] / min_mp[1],
    "oldest_vs_youngest_age_unw": age_m[5] / age_m[1],
    "oldest_vs_youngest_age_popwt": age_mp[5] / age_mp[1],
    # ROBUST (median-based) ratios — primary, insensitive to residual denom outliers
    "poorest_vs_richest_income_MEDIAN": inc_md[1] / inc_md[5],
    "highest_vs_lowest_poverty_MEDIAN": pov_md[5] / pov_md[1],
    "highest_vs_lowest_minority_MEDIAN": min_md[5] / min_md[1],
    "oldest_vs_youngest_age_MEDIAN": age_md[5] / age_md[1],
}
print("\n===== KEY RATIOS =====")
for k, v in ratios.items():
    print(f"  {k}: {v:.3f}")

# ---------- (3) CORRELATION + REGRESSION ----------
print("\n===== SPEARMAN (burden vs demographics) =====")
spear = {}
for col in ["median_income", "poverty_rate", "minority_pct", "median_age"]:
    rho, p = stats.spearmanr(D[col], D["burden"])
    spear[col] = (rho, p)
    print(f"  {col:15s} rho={rho:+.3f}  p={p:.2e}")

# Multivariate OLS: logburden ~ z(income)+z(minority)+z(age)+z(poverty) + STATE FE (exposure control)
def zscore(x):
    return (x - x.mean()) / x.std(ddof=0)

R = D.copy()
for c in ["median_income", "minority_pct", "median_age", "poverty_rate"]:
    R["z_" + c] = zscore(R[c])

def ols_hc1(y, X, names):
    """OLS with HC1 robust SE. X includes intercept col."""
    XtX = X.T @ X
    XtXinv = np.linalg.inv(XtX)
    beta = XtXinv @ (X.T @ y)
    resid = y - X @ beta
    n, k = X.shape
    # HC1
    S = (X * resid[:, None])
    meat = S.T @ S
    cov = XtXinv @ meat @ XtXinv * (n / (n - k))
    se = np.sqrt(np.diag(cov))
    t = beta / se
    from scipy.stats import t as tdist
    pvals = 2 * tdist.sf(np.abs(t), n - k)
    ss_res = (resid ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    return dict(beta=beta, se=se, t=t, p=pvals, names=names, r2=r2, n=n, k=k)

# Model A: no exposure control (demographics only)
yv = R["logburden"].values
predsA = ["z_median_income", "z_minority_pct", "z_median_age"]
XA = np.column_stack([np.ones(len(R))] + [R[c].values for c in predsA])
mA = ols_hc1(yv, XA, ["const"] + predsA)

# Model B: + STATE fixed effects (controls for WHERE hazards hit)
st = pd.get_dummies(R["state"], prefix="st", drop_first=True).astype(float)
XB = np.column_stack([np.ones(len(R))] + [R[c].values for c in predsA] + [st[c].values for c in st.columns])
namesB = ["const"] + predsA + list(st.columns)
mB = ols_hc1(yv, XB, namesB)

# Model C: also add poverty in place-of/alongside income (report separately)
predsC = ["z_poverty_rate", "z_minority_pct", "z_median_age"]
XC = np.column_stack([np.ones(len(R))] + [R[c].values for c in predsC] + [st[c].values for c in st.columns])
mC = ols_hc1(yv, XC, ["const"] + predsC + list(st.columns))

def show(m, title):
    print(f"\n----- {title}  (n={m['n']}, k={m['k']}, R2={m['r2']:.3f}) -----")
    for nm, be, se, pv in zip(m["names"], m["beta"], m["se"], m["p"]):
        if nm.startswith("st_"):
            continue
        star = "***" if pv < .001 else "**" if pv < .01 else "*" if pv < .05 else ""
        print(f"  {nm:20s} beta={be:+.4f}  se={se:.4f}  p={pv:.2e} {star}")

show(mA, "Model A: demographics only (no exposure control)")
show(mB, "Model B: demographics + STATE FE (exposure-controlled) income")
show(mC, "Model C: demographics + STATE FE, poverty instead of income")

# ---------- (4) CONCENTRATION BY GROUP ----------
print("\n===== CONCENTRATION: burden share vs pop share (bottom/top quartile) =====")
conc = {}
total_burden_ch = D["total_customer_hours_out"].sum()
total_pop = D["population"].sum()
def concentration(col, take_bottom):
    d = D.dropna(subset=[col]).copy()
    d["q4"] = pd.qcut(d[col].rank(method="first"), 4, labels=[1, 2, 3, 4])
    grp = 1 if take_bottom else 4
    sub = d[d["q4"] == grp]
    bshare = sub["total_customer_hours_out"].sum() / d["total_customer_hours_out"].sum()
    pshare = sub["population"].sum() / d["population"].sum()
    return bshare, pshare, bshare / pshare
for col, tb, name in [("median_income", True, "bottom-income Q"),
                      ("poverty_rate", False, "highest-poverty Q"),
                      ("minority_pct", False, "highest-minority Q")]:
    bs, ps, rr = concentration(col, tb)
    conc[name] = dict(burden_share=bs, pop_share=ps, ratio=rr)
    print(f"  {name:20s}: burden {bs*100:5.1f}%  pop {ps*100:5.1f}%  ratio {rr:.3f}")

# ---------- (5) NEWLY-RECOVERED EVENTS ----------
print("\n===== EVENT MAGNITUDES (newly recovered) =====")
d22 = pd.read_parquet(f"{BASE}/_daily_v2_2022.parquet")
d24 = pd.read_parquet(f"{BASE}/_daily_v2_2024.parquet")
for d in (d22, d24):
    d["date"] = pd.to_datetime(d["date"])
    d["ch"] = d["ch_sum"] * 0.25  # customer-hours

events = {}
def event_mag(daily, start, end, name, fips_prefixes=None):
    m = (daily["date"] >= start) & (daily["date"] <= end)
    sub = daily[m].copy()
    if fips_prefixes:
        sub = sub[sub["fips"].str[:2].isin(fips_prefixes)]
    ch = sub["ch"].sum()
    peak = sub.groupby("date")["ch_max"].sum().max()  # national concurrent peak (approx = sum county peaks/day)
    ncnty = sub[sub["ch"] > 0]["fips"].nunique()
    events[name] = dict(customer_hours=float(ch), peak_day_customers=float(peak),
                        n_counties=int(ncnty), window=f"{start}..{end}")
    print(f"  {name:28s} {ch:14,.0f} cust-hrs  peakday~{peak:11,.0f}  counties={ncnty}  [{start}..{end}]")
    return ch

# Winter Storm Elliott: Dec 21-26 2022 (national)
event_mag(d22, "2022-12-21", "2022-12-26", "Winter Storm Elliott (natl)")
# Hurricane Helene: Sep 24 - Oct 3 2024 (SE US: FL,GA,SC,NC,TN,VA)
event_mag(d24, "2024-09-24", "2024-10-03", "Hurricane Helene (SE US)",
          fips_prefixes=["12", "13", "45", "37", "47", "51"])
event_mag(d24, "2024-09-24", "2024-10-03", "Hurricane Helene (natl window)")
# Hurricane Milton: Oct 9-14 2024 (FL)
event_mag(d24, "2024-10-09", "2024-10-14", "Hurricane Milton (FL)", fips_prefixes=["12"])
event_mag(d24, "2024-10-09", "2024-10-14", "Hurricane Milton (natl window)")

# whole-year context
print(f"  [context] 2022 full-year cust-hrs {d22['ch'].sum():,.0f}; "
      f"2024 full-year {d24['ch'].sum():,.0f}")

# ---------- SAVE numeric outputs ----------
out = {
    "n_counties_analysis": int(len(D)),
    "n_dropped_broken_denom": n_bad,
    "n_counties_with_demographics": int(len(D0)),
    "pop_covered": float(D["population"].sum()),
    "national_customer_hours_out": float(natl_ch),
    "ratios": {k: float(v) for k, v in ratios.items()},
    "spearman": {k: {"rho": float(v[0]), "p": float(v[1])} for k, v in spear.items()},
    "ols_modelA": {n: {"beta": float(b), "p": float(p)} for n, b, p in zip(mA["names"], mA["beta"], mA["p"]) if not n.startswith("st_")},
    "ols_modelB_stateFE": {n: {"beta": float(b), "p": float(p)} for n, b, p in zip(mB["names"], mB["beta"], mB["p"]) if not n.startswith("st_")},
    "ols_modelC_poverty_stateFE": {n: {"beta": float(b), "p": float(p)} for n, b, p in zip(mC["names"], mC["beta"], mC["p"]) if not n.startswith("st_")},
    "ols_r2": {"A": float(mA["r2"]), "B": float(mB["r2"]), "C": float(mC["r2"])},
    "concentration": conc,
    "events": events,
}
with open(f"{BASE}/equity_numeric_outputs.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\n[saved] {BASE}/equity_numeric_outputs.json")

# stash quintile tables
for col, g in results.items():
    g.to_csv(f"{BASE}/equity_quintiles_{col}.csv")
D[["fips", "state", "burden", "total_customer_hours_out", "population", "median_income",
   "poverty_rate", "minority_pct", "median_age"]].to_csv(f"{BASE}/equity_joined.csv", index=False)
print("[saved] quintile tables + joined csv")

# ---------- FIGURES ----------
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

# FIG 1: quintile bar panels
fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
panels = [("median_income", "Median income quintile\n(1=poorest → 5=richest)", inc_md, inc_mp),
          ("poverty_rate", "Poverty-rate quintile\n(1=lowest → 5=highest)", pov_md, pov_mp),
          ("minority_pct", "Minority-share quintile\n(1=lowest → 5=highest)", min_md, min_mp),
          ("median_age", "Median-age quintile\n(1=youngest → 5=oldest)", age_md, age_mp)]
for ax, (col, lab, m, mp) in zip(axes, panels):
    x = np.arange(1, 6)
    ax.bar(x, m.values, 0.6, color="#4C72B0")
    ax.set_xticks(x); ax.set_xlabel(lab)
    ax.set_title(col)
axes[0].set_ylabel("MEDIAN customer-hours-out per customer\n(2014–2025, valid-denom counties)")
fig.suptitle("Median per-customer outage burden across demographic quintiles (US counties, EAGLE-I 2014–2025)", fontweight="bold")
fig.tight_layout()
fig.savefig(f"{BASE}/equity_fig1_quintiles.png", dpi=130)
plt.close(fig)

# FIG 2: concentration (Lorenz-ish bars) + scatter
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
names = list(conc.keys())
bshares = [conc[n]["burden_share"] * 100 for n in names]
pshares = [conc[n]["pop_share"] * 100 for n in names]
xx = np.arange(len(names))
ax1.bar(xx - 0.2, pshares, 0.4, label="population share", color="#55A868")
ax1.bar(xx + 0.2, bshares, 0.4, label="outage-burden share", color="#C44E52")
ax1.set_xticks(xx); ax1.set_xticklabels([n.replace(" Q", "\nquartile") for n in names], fontsize=8)
ax1.set_ylabel("% of national total (analysis counties)")
ax1.set_title("Concentration: burden share vs population share")
ax1.legend()
for i, (b, p) in enumerate(zip(bshares, pshares)):
    ax1.text(i + 0.2, b + 0.5, f"{b:.0f}%", ha="center", fontsize=8)
    ax1.text(i - 0.2, p + 0.5, f"{p:.0f}%", ha="center", fontsize=8)
# scatter minority vs burden (log)
sc = ax2.scatter(D["minority_pct"] * 100, D["burden"], s=6, alpha=0.35,
                 c=D["median_income"], cmap="viridis")
ax2.set_yscale("log")
ax2.set_xlabel("County minority share (%)")
ax2.set_ylabel("Customer-hours-out per customer (log)")
ax2.set_title(f"Burden vs minority share (Spearman ρ={spear['minority_pct'][0]:+.2f})")
plt.colorbar(sc, ax=ax2, label="median income ($)")
fig.tight_layout()
fig.savefig(f"{BASE}/equity_fig2_concentration.png", dpi=130)
plt.close(fig)

# FIG 3: event magnitudes + exposure-vs-vulnerability coefficients
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ev_names = ["Winter Storm Elliott (natl)", "Hurricane Helene (SE US)", "Hurricane Milton (FL)"]
ev_vals = [events[n]["customer_hours"] / 1e6 for n in ev_names]
ax1.barh(range(len(ev_names)), ev_vals, color=["#4C72B0", "#C44E52", "#DD8452"])
ax1.set_yticks(range(len(ev_names)))
ax1.set_yticklabels([n.replace(" (", "\n(") for n in ev_names], fontsize=9)
ax1.set_xlabel("Customer-hours-out (millions)")
ax1.set_title("Newly-recovered events (2022-full + 2024)")
for i, v in enumerate(ev_vals):
    ax1.text(v, i, f" {v:,.0f}M", va="center", fontsize=9)
# coefficient comparison A vs B
coefs = ["z_median_income", "z_minority_pct", "z_median_age"]
bA = [mA["beta"][mA["names"].index(c)] for c in coefs]
bB = [mB["beta"][mB["names"].index(c)] for c in coefs]
seB = [mB["se"][mB["names"].index(c)] for c in coefs]
xx = np.arange(len(coefs))
ax2.bar(xx - 0.2, bA, 0.4, label="Model A (no exposure ctrl)", color="#8172B3")
ax2.bar(xx + 0.2, bB, 0.4, yerr=[1.96 * s for s in seB], label="Model B (+state FE)", color="#937860", capsize=3)
ax2.axhline(0, color="k", lw=0.8)
ax2.set_xticks(xx); ax2.set_xticklabels(["income", "minority", "age"])
ax2.set_ylabel("Coefficient on log per-customer burden")
ax2.set_title("Exposure vs vulnerability:\ndoes demographic signal survive state FE?")
ax2.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{BASE}/equity_fig3_events_regression.png", dpi=130)
plt.close(fig)
print("[saved] 3 figures")
print("DONE")
