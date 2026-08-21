"""State customer denominator from EAGLE-I coverage files -> state_customers.csv (state, total_customers).
Uses the MOST RECENT year available per state across 2014-2022 (coverage_history.csv) and
2023 (State_Coverage_by_Year.csv). Keeps year_source + max_pct_covered for auditing."""
import pandas as pd, numpy as np
BASE = "/data/equity_cost/eaglei"
OUT = "/data/equity_cost/analysis/state_customers.csv"

def parse_year(s):
    s = str(s).strip().strip('"')
    for fmt in ("%m/%d/%y", "%m/%d/%Y"):
        try:
            return pd.to_datetime(s, format=fmt).year
        except Exception:
            pass
    return int(pd.to_datetime(s).year)

frames = []
for p in [f"{BASE}/2014_2022/eaglei_outages/coverage_history.csv",
          f"{BASE}/2023/State_Coverage_by_Year.csv"]:
    d = pd.read_csv(p)
    d.columns = [c.strip().strip('"') for c in d.columns]
    d["state"] = d["state"].astype(str).str.strip().str.strip('"')
    d["yr"] = d["year"].map(parse_year)
    d["total_customers"] = pd.to_numeric(d["total_customers"], errors="coerce")
    frames.append(d[["yr", "state", "total_customers", "max_pct_covered"]])
cov = pd.concat(frames, ignore_index=True)
print("year range in coverage:", int(cov.yr.min()), "..", int(cov.yr.max()))
print("years present:", sorted(cov.yr.unique()))
print("n state-year rows:", len(cov), "n states:", cov.state.nunique())
# most recent year per state
idx = cov.sort_values(["state", "yr"]).groupby("state")["yr"].idxmax()
latest = cov.loc[idx].rename(columns={"yr": "year_source"}).sort_values("state")
latest = latest[["state", "total_customers", "year_source", "max_pct_covered"]]
latest["total_customers"] = latest["total_customers"].round().astype("Int64")
latest.to_csv(OUT, index=False)
print("WROTE", OUT)
print("n states:", len(latest), "US total_customers (sum of latest):", int(latest.total_customers.sum()))
print(latest.head(6).to_string(index=False))
