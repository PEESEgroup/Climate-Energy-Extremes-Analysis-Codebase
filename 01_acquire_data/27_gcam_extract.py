"""Extract state-level annual electricity demand from a GCAM-USA BaseX db (query 'Total electricity delivered'),
convert EJ->TWh, save TELL-format electricity_demand_{scen}.csv + state growth factors (value/value@2020).
Validate rcp85hotter_ssp5 against IM3's sample electricity_demand CSV (growth-factor comparison)."""
import sys, gcamreader, pandas as pd, numpy as np, os
MI="/data/tellenv/lib/python3.10/site-packages/gcamreader/ModelInterface"
scen=sys.argv[1]; DBROOT=sys.argv[2]              # e.g. rcp85hotter_ssp5, /data/gcam_usa/db_extracted
OUTDIR="/data/gcam_usa/processed"; os.makedirs(OUTDIR,exist_ok=True)
EJ2TWh=1e18/3.6e15                                # 1 EJ = 277.778 TWh
STATES=set("AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC".split())
dbfile=f"database_{scen}"
conn=gcamreader.LocalDBConn(DBROOT, dbfile, suppress_gabble=True)
q=[x for x in gcamreader.parse_batch_query(MI+"/Main_queries.xml") if x.title=="Total electricity delivered"][0]
df=conn.runQuery(q)
us=df[df.region.isin(STATES)].copy()
us=us.groupby(["region","Year"],as_index=False)["value"].sum()   # ensure one per state-year
us["TWh"]=us["value"]*EJ2TWh
# TELL electricity_demand_{scen}.csv format
out=pd.DataFrame({"scenario":scen,"region":"USA","subRegion":us.region,"param":"elecFinalBySecTWh",
    "xLabel":"Year","x":us.Year,"vintage":"Vint_"+us.Year.astype(str),
    "units":"Final Electricity by Sector (TWh)","value":us.TWh})
out.to_csv(f"{OUTDIR}/electricity_demand_{scen}.csv",index=False)
# growth factors vs 2020
piv=us.pivot(index="region",columns="Year",values="TWh")
gf=piv.div(piv[2020],axis=0)
gf.to_csv(f"{OUTDIR}/growth_factor_{scen}.csv")
print(f"[{scen}] states={us.region.nunique()} years={sorted(us.Year.unique())[:3]}..{sorted(us.Year.unique())[-2:]}")
print(f"  US total TWh: 2020={piv[2020].sum():.0f}  2050={piv[2050].sum():.0f}  growth2050/2020={piv[2050].sum()/piv[2020].sum():.3f}")

# ---- validate vs IM3 sample (only for rcp85hotter_ssp5) ----
SAMP="/data/tell_data/sample_forcing_data/sample_gcam_usa_data/electricity_demand_rcp85hotter_ssp5.csv"
if scen=="rcp85hotter_ssp5" and os.path.exists(SAMP):
    s=pd.read_csv(SAMP)
    s=s[s.param=="elecFinalBySecTWh"].groupby(["subRegion","x"],as_index=False)["value"].sum()
    sp=s.pivot(index="subRegion",columns="x",values="value")
    common_y=[y for y in [2020,2030,2040,2050] if y in sp.columns and y in piv.columns]
    sg=sp.div(sp[2020],axis=0); og=gf
    print("\n=== validate vs IM3 sample (rcp85hotter_ssp5) ===")
    print(f"{'year':>6} {'ourGrowth':>10} {'IM3Growth':>10}  (US-agg value/2020)")
    for y in common_y:
        ov=piv[y].sum()/piv[2020].sum(); sv=sp[y].sum()/sp[2020].sum()
        print(f"{y:>6} {ov:>10.3f} {sv:>10.3f}")
    # per-state growth correlation at 2050
    st=set(og.index)&set(sg.index)
    a=np.array([og.loc[k,2050] for k in st]); b=np.array([sg.loc[k,2050] for k in st])
    print(f"per-state 2050 growth corr (ours vs IM3): {np.corrcoef(a,b)[0,1]:.3f}  (n={len(st)})")
    print(f"abs level ratio ours/IM3 at 2020 (US): {piv[2020].sum()/sp[2020].sum():.3f}")
