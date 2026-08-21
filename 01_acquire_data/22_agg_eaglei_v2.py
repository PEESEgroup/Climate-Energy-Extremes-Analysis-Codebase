"""STEP 3: re-aggregate EAGLE-I county totals 2014-2025 with the BEST files.
 - 2014-2021 + 2023: reuse validated legacy county-day intermediates (_daily_{y}.parquet,
   same-provenance as source CSVs in 2014_2022/ and 2023/).
 - 2022: REGENERATE from figshare full-year file (old _daily_2022 was truncated 2022-11-12).
 - 2024, 2025: process from figshare files.
EAGLE-I rows are 15-min snapshots of customers_out; customer_hours = sum(customers_out)*0.25.
Writes /data/equity_cost/analysis/eaglei_county_total_v2.csv, joined to MCC per-customer denom.
Memory-safe chunked reads (chunksize 2e6)."""
import os, gc, time
import pandas as pd, numpy as np

BASE = "/data/equity_cost"
OUTDIR = f"{BASE}/analysis"
FIG = f"{BASE}/eaglei/figshare_v4"
CHUNK = 2_000_000
MCC = f"{FIG}/MCC.csv"

# legacy county-day intermediates to reuse (correct full-year span, same provenance)
LEGACY_YEARS = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2023]
# (year -> figshare csv) to process fresh; value column is 'customers_out'
FIG_FILES = {
    2022: f"{FIG}/eaglei_outages_2022.csv",
    2024: f"{FIG}/eaglei_outages_2024.csv",
    2025: f"{FIG}/eaglei_outages_2025.csv",
}

ST = {
"01":"AL","02":"AK","04":"AZ","05":"AR","06":"CA","08":"CO","09":"CT","10":"DE","11":"DC",
"12":"FL","13":"GA","15":"HI","16":"ID","17":"IL","18":"IN","19":"IA","20":"KS","21":"KY",
"22":"LA","23":"ME","24":"MD","25":"MA","26":"MI","27":"MN","28":"MS","29":"MO","30":"MT",
"31":"NE","32":"NV","33":"NH","34":"NJ","35":"NM","36":"NY","37":"NC","38":"ND","39":"OH",
"40":"OK","41":"OR","42":"PA","44":"RI","45":"SC","46":"SD","47":"TN","48":"TX","49":"UT",
"50":"VT","51":"VA","53":"WA","54":"WV","55":"WI","56":"WY","60":"AS","66":"GU","69":"MP",
"72":"PR","78":"VI"}
def st_abbr(f): return ST.get(f[:2], "??")

def process_fig_year(y, path):
    inter = f"{OUTDIR}/_daily_v2_{y}.parquet"
    if os.path.exists(inter):
        print(f"[{y}] v2 intermediate exists -> reuse", flush=True)
        return pd.read_parquet(inter)
    t0 = time.time(); parts = []; nrows = 0; nbad = 0
    for ci, ch in enumerate(pd.read_csv(path, chunksize=CHUNK,
            usecols=["fips_code", "customers_out", "run_start_time"],
            dtype={"fips_code": str, "customers_out": "float64"})):
        nrows += len(ch)
        good = ch["fips_code"].notna()
        nbad += int((~good).sum())
        ch = ch[good]
        ch["fips"] = ch["fips_code"].str.zfill(5)
        ch["date"] = ch["run_start_time"].str.slice(0, 10)
        v = ch["customers_out"].fillna(0.0).to_numpy()
        ch["cust"] = v
        ch["pos"] = (v > 0).astype("int64")
        a = ch.groupby(["fips", "date"], sort=False).agg(
            ch_sum=("cust", "sum"), ch_max=("cust", "max"), ch_npos=("pos", "sum")).reset_index()
        parts.append(a); del ch
        if ci % 10 == 0:
            gc.collect(); print(f"[{y}] chunk {ci} rows~{nrows:,} t={time.time()-t0:.0f}s", flush=True)
    yr = pd.concat(parts, ignore_index=True); del parts; gc.collect()
    yr = yr.groupby(["fips", "date"], sort=False).agg(
        ch_sum=("ch_sum", "sum"), ch_max=("ch_max", "max"), ch_npos=("ch_npos", "sum")).reset_index()
    yr.to_parquet(inter, index=False)
    print(f"[{y}] DONE rows_in={nrows:,} bad_fips={nbad:,} county_days={len(yr):,} t={time.time()-t0:.0f}s", flush=True)
    return yr

frames = []
for y in LEGACY_YEARS:
    d = pd.read_parquet(f"{OUTDIR}/_daily_{y}.parquet")
    assert list(d.columns) == ["fips", "date", "ch_sum", "ch_max", "ch_npos"], (y, d.columns)
    print(f"[{y}] legacy reuse: {len(d):,} county-days {d['date'].min()}..{d['date'].max()}", flush=True)
    frames.append(d)
for y, p in FIG_FILES.items():
    frames.append(process_fig_year(y, p))

allyr = pd.concat(frames, ignore_index=True); del frames; gc.collect()
allyr = allyr.groupby(["fips", "date"], sort=False).agg(
    ch_sum=("ch_sum", "sum"), ch_max=("ch_max", "max"), ch_npos=("ch_npos", "sum")).reset_index()

daily = pd.DataFrame({
    "fips": allyr["fips"], "date": allyr["date"],
    "customer_hours_out": allyr["ch_sum"] * 0.25,
    "peak_customers_out": allyr["ch_max"],
    "npos": allyr["ch_npos"],
})
# outage days = county-days with at least one positive-outage snapshot
od = daily[daily["npos"] > 0].copy()
od["year"] = od["date"].str.slice(0, 4).astype(int)
tot = od.groupby("fips").agg(
    total_customer_hours_out=("customer_hours_out", "sum"),
    peak_customers_out=("peak_customers_out", "max"),
    n_outage_days=("date", "nunique"),
    years_present=("year", "nunique"),
    first_date=("date", "min"),
    last_date=("date", "max")).reset_index()
tot["state"] = tot["fips"].map(st_abbr)
tot["peak_customers_out"] = tot["peak_customers_out"].round().astype("int64")

# join MCC per-county customer denominator (proper per-customer burden)
mcc = pd.read_csv(MCC, encoding="utf-8-sig")
mcc.columns = [c.strip() for c in mcc.columns]
mcc = mcc[pd.to_numeric(mcc["County_FIPS"], errors="coerce").notna()].copy()   # drop 'Grand Total' row
mcc["fips"] = pd.to_numeric(mcc["County_FIPS"]).astype("int64").astype(str).str.zfill(5)
mcc = mcc.rename(columns={"Customers": "mcc_customers"})[["fips", "mcc_customers"]]
tot = tot.merge(mcc, on="fips", how="left")
tot["customer_hours_per_customer"] = np.where(
    tot["mcc_customers"] > 0, tot["total_customer_hours_out"] / tot["mcc_customers"], np.nan)

tot = tot[["fips", "state", "total_customer_hours_out", "peak_customers_out",
           "n_outage_days", "years_present", "first_date", "last_date",
           "mcc_customers", "customer_hours_per_customer"]].sort_values("fips")
out = f"{OUTDIR}/eaglei_county_total_v2.csv"
tot.to_csv(out, index=False)

print("\n=== AUDIT ===", flush=True)
print(f"counties               : {len(tot):,}")
print(f"national cust-hours-out : {daily['customer_hours_out'].sum():,.0f}")
print(f"date span              : {daily['date'].min()} .. {daily['date'].max()}")
print(f"years covered          : {sorted(od['year'].unique())}")
print(f"counties matched to MCC : {tot['mcc_customers'].notna().sum():,} / {len(tot):,}")
print(f"WROTE {out}")
print("\nTop-10 counties by customer_hours_per_customer (>=5000 customers):")
top = tot[tot['mcc_customers'] >= 5000].sort_values('customer_hours_per_customer', ascending=False).head(10)
print(top[['fips','state','customer_hours_per_customer','total_customer_hours_out','mcc_customers','years_present']].to_string(index=False))
