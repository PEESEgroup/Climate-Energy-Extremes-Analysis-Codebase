"""Stage 2 (full): our native-12km TGW -> pop-weighted BA hourly weather, 2016-2019.
Writes TELL forcing CSVs: {BA}_WRF_Hourly_Mean_Meteorology_{year}.csv  cols Time_UTC,T2,Q2,SWDOWN,GLW,WSPD.
Validated (proto): reproduces TELL compiled_historical_data (T2/Q2/GLW/SWDOWN corr~1.0; scalar-wind WSPD corr~0.99)."""
import numpy as np, pandas as pd, glob, os, scipy.sparse as sp, sys, time

TGW="/data/tgw_hist"; QS="/data/tell_qs/tell_quickstarter_data/outputs"
POP="/data/tell_data/sample_forcing_data/sample_population_projections/ssp5_county_population.csv"
OUT="/data/tell_forcing/historic"; os.makedirs(OUT, exist_ok=True)
YEARS=[2016,2017,2018,2019]
t0=time.time(); log=lambda s: print(f"[{time.time()-t0:6.1f}s] {s}", flush=True)

# ---- county mask, county-mean matrix M (nC x nCell) ----
m=np.load("/data/loads_measured/county_mask_tgw.npz", allow_pickle=True)
fips=np.array([str(x).zfill(5) for x in m["fips"]]); nC=len(fips)
H=int(m["H"]); W=int(m["W"]); nCell=H*W
pf=m["pair_fips"].astype(int); pc=m["pair_cell"].astype(int)
counts=np.bincount(pf,minlength=nC).astype(float); counts[counts==0]=1
M=sp.csr_matrix((1.0/counts[pf],(pf,pc)),shape=(nC,nCell))
fips2row={f:i for i,f in enumerate(fips)}

# ---- county pop (2020 base) + county->BA (equal-split multi-BA), pop-weighted BA matrix Wba ----
pop=pd.read_csv(POP,dtype={"FIPS":str}); pop["FIPS"]=pop["FIPS"].str.zfill(5)
popmap=dict(zip(pop["FIPS"],pop["2020"].astype(float)))
st=pd.read_csv(QS+"/ba_service_territory/ba_service_territory_2019.csv")
st["FIPS"]=st["County_FIPS"].astype(int).astype(str).str.zfill(5)
nBAof=st.groupby("FIPS")["BA_Code"].nunique().to_dict()
bas=sorted(st["BA_Code"].unique()); ba2col={b:i for i,b in enumerate(bas)}; nBA=len(bas)
rows=[];cols=[];vals=[]
for _,r in st.iterrows():
    f=r["FIPS"]; b=r["BA_Code"]
    if f in fips2row and f in popmap:
        rows.append(ba2col[b]); cols.append(fips2row[f]); vals.append(popmap[f]/nBAof[f])
Wba=sp.csr_matrix((vals,(rows,cols)),shape=(nBA,nC))
rs=np.asarray(Wba.sum(1)).ravel(); rs[rs==0]=1
Wba=sp.diags(1.0/rs)@Wba
log(f"nC={nC} nBA={nBA} nCell={nCell}")

# ---- global hourly index 2016-2019 ----
idx=pd.date_range("2016-01-01 00:00","2019-12-31 23:00",freq="h")
stamp2i={t.strftime("%Y%m%d%H"):k for k,t in enumerate(idx)}
NH=len(idx)
# per-BA accumulators
ACC={v:np.full((nBA,NH),np.nan,np.float32) for v in ["T2","Q2","SWDOWN","GLW","WSPD"]}

# ---- stream files overlapping 2016-2019 ----
files=sorted(glob.glob(TGW+"/tgw_historical_*_20[12]*-*.npz"))
def yr(fn): return fn.split("_hourly_")[1][:4]
files=[f for f in files if yr(f) in {"2015","2016","2017","2018","2019","2020"}]
log(f"{len(files)} candidate files")
used=0
for fi,fn in enumerate(files):
    d=np.load(fn); ts=d["times"];
    keep=[(ti,stamp2i[str(s)]) for ti,s in enumerate(ts) if str(s) in stamp2i]
    if not keep: continue
    dat=d["data"].astype(np.float32); T=dat.shape[0]
    spd=np.sqrt(dat[:,0]**2+dat[:,1]**2).reshape(T,nCell)
    ba={}
    for v,ix in {"Q2":2,"T2":4,"GLW":5,"SWDOWN":6}.items():
        cm=(M@dat[:,ix].reshape(T,nCell).T).T; ba[v]=(Wba@cm.T).T
    ba["WSPD"]=(Wba@(M@spd.T)).T
    ba["Q2"]=np.clip(ba["Q2"],0,None)
    for ti,gi in keep:
        for v in ACC: ACC[v][:,gi]=ba[v][ti]
    used+=1
    if fi%20==0: log(f"file {fi}/{len(files)} used={used}")
log(f"aggregation done, files used={used}")

# ---- coverage + write per BA per year ----
cov=np.isfinite(ACC["T2"]).all(0).mean()
log(f"hourly coverage (all-BA finite): {cov*100:.2f}%")
modeled=set(__import__('tell').get_balancing_authority_to_model_dict().keys())
yv=idx.year.values
written=0
for b,col in ba2col.items():
    for Y in YEARS:
        sel=np.where(yv==Y)[0]
        t2=ACC["T2"][col,sel]
        if not np.isfinite(t2).any(): continue
        df=pd.DataFrame({
            "Time_UTC":[idx[k].strftime("%Y-%m-%d %H:%M:%S") for k in sel],
            "T2":ACC["T2"][col,sel],"Q2":ACC["Q2"][col,sel],
            "SWDOWN":ACC["SWDOWN"][col,sel],"GLW":ACC["GLW"][col,sel],"WSPD":ACC["WSPD"][col,sel]})
        # fill any residual NaN by forward/back fill (TGW is continuous so should be none)
        nnan=int(df[["T2","Q2","SWDOWN","GLW","WSPD"]].isna().sum().sum())
        if nnan: df=df.ffill().bfill()
        df.to_csv(f"{OUT}/{b}_WRF_Hourly_Mean_Meteorology_{Y}.csv",index=False)
        written+=1
        if b in modeled and Y==2019 and nnan: log(f"WARN {b} {Y}: {nnan} NaN filled")
log(f"wrote {written} BA-year forcing files to {OUT}  ({len(modeled&set(ba2col))} modeled BAs covered)")
