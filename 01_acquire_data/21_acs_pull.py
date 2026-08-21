"""STEP 4: pull ACS 5-year 2022 county demographics nationwide from the Census API.
Key read from .census_key (never printed). Loops states; builds acs_county.csv.
Sentinel negatives (e.g. -666666666) -> NaN."""
import os, urllib.request, json, time
import pandas as pd, numpy as np

KEY = open(os.environ.get("CENSUS_KEY_FILE", ".census_key")).read().strip()   # your own Census API key
BASE = "https://api.census.gov/data/2022/acs/acs5"
VARS = ["B19013_001E",  # median household income
        "B01003_001E",  # total population
        "B01002_001E",  # median age
        "B03002_001E",  # race/ethnicity universe (total)
        "B03002_003E",  # white alone, not hispanic
        "B17001_001E",  # poverty universe
        "B17001_002E"]  # income below poverty
STATES = ["01","02","04","05","06","08","09","10","11","12","13","15","16","17","18","19",
          "20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35",
          "36","37","38","39","40","41","42","44","45","46","47","48","49","50","51","53",
          "54","55","56","72"]

rows = []
for st in STATES:
    url = f"{BASE}?get=NAME,{','.join(VARS)}&for=county:*&in=state:{st}&key={KEY}"
    try:
        with urllib.request.urlopen(url, timeout=90) as r:
            data = json.load(r)
    except Exception as e:
        print(f"SKIP {st}: {type(e).__name__} {getattr(e,'code','')}", flush=True)
        continue
    hdr = data[0]
    for rec in data[1:]:
        rows.append(dict(zip(hdr, rec)))
    print(f"OK {st}: {len(data)-1} counties", flush=True)
    time.sleep(0.1)

df = pd.DataFrame(rows)
df["fips"] = df["state"].str.zfill(2) + df["county"].str.zfill(3)

def num(col):
    v = pd.to_numeric(df[col], errors="coerce")
    return v.where(v > -1e8)   # census sentinel negatives -> NaN

inc  = num("B19013_001E")
pop  = num("B01003_001E")
age  = num("B01002_001E")
tot  = num("B03002_001E")
wnh  = num("B03002_003E")
pu   = num("B17001_001E")
below= num("B17001_002E")

minority = 1.0 - (wnh / tot)
poverty  = below / pu

out = pd.DataFrame({
    "fips": df["fips"],
    "median_income": inc,
    "population": pop,
    "median_age": age,
    "minority_pct": minority.round(6),
    "poverty_rate": poverty.round(6),
}).sort_values("fips").reset_index(drop=True)

path = "/data/equity_cost/analysis/acs_county.csv"
out.to_csv(path, index=False)
print(f"\nWROTE {path}")
print(f"counties               : {len(out):,}")
print(f"median_income non-null : {out['median_income'].notna().sum():,}")
print(f"population sum         : {out['population'].sum():,.0f}")
print(f"minority_pct mean      : {out['minority_pct'].mean():.3f}")
print(f"poverty_rate mean      : {out['poverty_rate'].mean():.3f}")
print("\nsample rows:")
print(out.head(3).to_string(index=False))
print(out[out['fips'].isin(['36061','06037','48201','17031'])].to_string(index=False))
