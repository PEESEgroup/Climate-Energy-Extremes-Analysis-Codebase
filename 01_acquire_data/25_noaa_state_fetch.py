#!/usr/bin/env python3
"""Fetch NOAA NCEI Billion-Dollar Disaster per-STATE time series, parse to tidy + rollup.
Costs are CPI-adjusted $millions, given as ranges -> take midpoint. '0-0' -> 0.
Writes only under /data/equity_cost/analysis/.
"""
import subprocess, os, sys
import pandas as pd
import numpy as np

OUT = '/data/equity_cost/analysis'
RAW = os.path.join(OUT, 'noaa_state_raw')
os.makedirs(RAW, exist_ok=True)

# 50 states + DC (2-letter codes)
STATES = ['AL','AK','AZ','AR','CA','CO','CT','DE','DC','FL','GA','HI','ID','IL',
          'IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE',
          'NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD',
          'TN','TX','UT','VT','VA','WA','WV','WI','WY']
assert len(STATES) == 51, len(STATES)

HAZ = ['Drought','Flooding','Freeze','Severe Storm','Tropical Cyclone',
       'Wildfire','Winter Storm']
SHORT = {'Drought':'drought','Flooding':'flooding','Freeze':'freeze',
         'Severe Storm':'severe','Tropical Cyclone':'tc','Wildfire':'wildfire',
         'Winter Storm':'winter'}

def midpoint(rng):
    s = str(rng).strip().strip('"')
    if s in ('', '0-0', 'nan', 'None'):
        return 0.0
    if '-' in s:
        a, b = s.split('-', 1)
        try:
            return (float(a) + float(b)) / 2.0
        except ValueError:
            return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan

rows = []
missing = []
allcheck = []  # per (state,year) All Disasters midpoint for cross-check
for st in STATES:
    url = f'https://www.ncei.noaa.gov/access/billions/time-series/{st}.csv'
    dst = os.path.join(RAW, f'{st}.csv')
    r = subprocess.run(['curl', '-s', '--max-time', '60', '-w', '%{http_code}',
                        '-o', dst, url], capture_output=True, text=True)
    code = r.stdout.strip()[-3:]
    if code != '200' or not os.path.exists(dst) or os.path.getsize(dst) < 100:
        missing.append((st, code))
        continue
    df = pd.read_csv(dst, comment='#')
    df.columns = [c.strip() for c in df.columns]
    for _, row in df.iterrows():
        try:
            yr = int(row['Year'])
        except (ValueError, TypeError):
            continue
        if yr < 1980 or yr > 2024:
            continue
        for h in HAZ:
            cnt = row.get(f'{h} Count')
            cost = midpoint(row.get(f'{h} Cost Range'))
            cnt = int(cnt) if pd.notna(cnt) else 0
            rows.append((st, yr, SHORT[h], cnt, round(cost, 3)))
        allcheck.append((st, yr,
                         int(row['All Disasters Count']) if pd.notna(row.get('All Disasters Count')) else 0,
                         midpoint(row.get('All Disasters Cost Range'))))

tidy = pd.DataFrame(rows, columns=['state','year','hazard_type','count','cost_musd'])
tidy = tidy.sort_values(['state','year','hazard_type']).reset_index(drop=True)
tidy.to_csv(os.path.join(OUT, 'noaa_state_hazard.csv'), index=False)

allc = pd.DataFrame(allcheck, columns=['state','year','all_count','all_cost_musd'])

# ---- Full-period rollup (1980-2024) ----
def build_rollup(tdf, adf):
    piv_cost = tdf.pivot_table(index='state', columns='hazard_type',
                               values='cost_musd', aggfunc='sum', fill_value=0.0)
    piv_cnt = tdf.pivot_table(index='state', columns='hazard_type',
                              values='count', aggfunc='sum', fill_value=0)
    out = pd.DataFrame(index=piv_cost.index)
    for short in SHORT.values():
        out[f'{short}_cost'] = piv_cost.get(short, 0.0)
        out[f'{short}_count'] = piv_cnt.get(short, 0)
    out['total_cost_musd'] = out[[f'{s}_cost' for s in SHORT.values()]].sum(axis=1)
    out['total_count'] = out[[f'{s}_count' for s in SHORT.values()]].sum(axis=1)
    # NCEI's own "All Disasters" totals (cross-check; not additive with type midpoints)
    ad = adf.groupby('state').agg(all_disasters_cost_musd=('all_cost_musd','sum'),
                                  all_disasters_count=('all_count','sum'))
    out = out.join(ad)
    # order columns
    cols = (['total_cost_musd','total_count'] +
            [f'{s}_cost' for s in SHORT.values()] +
            [f'{s}_count' for s in SHORT.values()] +
            ['all_disasters_cost_musd','all_disasters_count'])
    out = out[cols].reset_index().rename(columns={'index':'state'})
    out = out.sort_values('total_cost_musd', ascending=False).reset_index(drop=True)
    return out

roll = build_rollup(tidy, allc)
roll.to_csv(os.path.join(OUT, 'noaa_state_total.csv'), index=False)

# ---- 2014-2023 window (matches EAGLE-I outage window) ----
tidy_w = tidy[(tidy.year >= 2014) & (tidy.year <= 2023)].reset_index(drop=True)
tidy_w.to_csv(os.path.join(OUT, 'noaa_state_hazard_2014_2023.csv'), index=False)
allc_w = allc[(allc.year >= 2014) & (allc.year <= 2023)]
roll_w = build_rollup(tidy_w, allc_w)
roll_w.to_csv(os.path.join(OUT, 'noaa_state_total_2014_2023.csv'), index=False)

# ---- Sanity / audit ----
n_states = tidy.state.nunique()
natl_recon_types = tidy.cost_musd.sum()           # sum of per-state type midpoints
natl_recon_all = allc.all_cost_musd.sum()          # sum of per-state All Disasters midpoints
yr_min, yr_max = int(tidy.year.min()), int(tidy.year.max())
top5 = roll.head(5)[['state','total_cost_musd','total_count','tc_cost','winter_cost','severe_cost']]

print('=== SANITY ===')
print(f'n_states_parsed = {n_states} (expected 51)')
print(f'missing/404 = {missing}')
print(f'year range = {yr_min}-{yr_max}')
print(f'tidy rows = {len(tidy)}  (51 states x 45 yr x 7 haz = {51*45*7})')
print(f'SUM of per-state per-type midpoints  = ${natl_recon_types/1e6:.3f} T  (${natl_recon_types:,.0f} M)')
print(f'SUM of per-state All-Disasters midpts = ${natl_recon_all/1e6:.3f} T  (${natl_recon_all:,.0f} M)')
print('  (NOTE: multi-state events attributed per-state; sum >= national ~$2.9T)')
print('=== TOP 5 STATES BY TOTAL COST ===')
for _, r in top5.iterrows():
    print(f"  {r['state']}: total=${r['total_cost_musd']/1e6:.3f}T  "
          f"n={int(r['total_count'])}  tc=${r['tc_cost']/1e3:.1f}B  "
          f"winter=${r['winter_cost']/1e3:.1f}B  severe=${r['severe_cost']/1e3:.1f}B")
print('=== 2014-2023 window ===')
print(f'window tidy rows = {len(tidy_w)}  window type-sum = ${tidy_w.cost_musd.sum()/1e6:.3f}T')
print('=== FILES ===')
for f in ['noaa_state_hazard.csv','noaa_state_total.csv',
          'noaa_state_hazard_2014_2023.csv','noaa_state_total_2014_2023.csv']:
    p = os.path.join(OUT, f)
    print(f'  {p}  ({os.path.getsize(p)} bytes)')
