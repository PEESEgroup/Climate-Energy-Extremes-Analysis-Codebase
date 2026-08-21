"""Future climate-driven load pipeline (Tier 1: no GCAM growth; socioeconomics held at present).
For one scenario: aggregate future TGW 2030-2050 -> BA weather -> TELL MLP predict -> per-BA/month bias
correction (QDM-mean) -> county disaggregation (Wba.T). Saves BA + county hourly load + summary.
Usage: 09_future_pipeline.py <scenario>   e.g. rcp85hotter"""
import sys, tell, pandas as pd, numpy as np, os, glob, scipy.sparse as sp, time, warnings
warnings.filterwarnings("ignore")
SCEN="hist_full40"
SRC="/data/tgw_hist"; FDIR=f"/data/tell_forcing/{SCEN}"; os.makedirs(FDIR,exist_ok=True)
OUT=f"/data/tell_pred/future/{SCEN}"; os.makedirs(OUT,exist_ok=True)
QS="/data/tell_qs/tell_quickstarter_data/outputs"
POP="/data/tell_data/sample_forcing_data/sample_population_projections/ssp5_county_population.csv"
YEARS=list(range(1980,2020)); t0=time.time(); log=lambda s: print(f"[{SCEN} {time.time()-t0:6.0f}s] {s}",flush=True)

# ---- county-mean matrix M + pop-weighted BA matrix Wba (same as historical, validated) ----
m=np.load("/data/loads_measured/county_mask_tgw.npz",allow_pickle=True)
fips=np.array([str(x).zfill(5) for x in m["fips"]]); nC=len(fips); fips2row={f:i for i,f in enumerate(fips)}
H=int(m["H"]);W=int(m["W"]);nCell=H*W
pf=m["pair_fips"].astype(int);pc=m["pair_cell"].astype(int); counts=np.bincount(pf,minlength=nC).astype(float);counts[counts==0]=1
M=sp.csr_matrix((1.0/counts[pf],(pf,pc)),shape=(nC,nCell))
pop=pd.read_csv(POP,dtype={"FIPS":str}); pop["FIPS"]=pop["FIPS"].str.zfill(5); popmap=dict(zip(pop["FIPS"],pop["2020"].astype(float)))
st=pd.read_csv(QS+"/ba_service_territory/ba_service_territory_2019.csv"); st["FIPS"]=st["County_FIPS"].astype(int).astype(str).str.zfill(5)
nBAof=st.groupby("FIPS")["BA_Code"].nunique().to_dict(); bas=sorted(st["BA_Code"].unique()); ba2col={b:i for i,b in enumerate(bas)}; nBA=len(bas)
rows=[];cols=[];vals=[]
for _,r in st.iterrows():
    f=r["FIPS"];b=r["BA_Code"]
    if f in fips2row and f in popmap: rows.append(ba2col[b]);cols.append(fips2row[f]);vals.append(popmap[f]/nBAof[f])
Wba=sp.csr_matrix((vals,(rows,cols)),shape=(nBA,nC)); rs=np.asarray(Wba.sum(1)).ravel();rs[rs==0]=1; Wba=sp.diags(1.0/rs)@Wba
modeled=sorted(tell.get_balancing_authority_to_model_dict().keys())

# ---- Stage 2: aggregate future TGW -> BA weather -> forcing CSVs ----
idx=pd.date_range(f"{YEARS[0]}-01-01 00:00",f"{YEARS[-1]}-12-31 23:00",freq="h"); NH=len(idx)
stamp2i={t.strftime("%Y%m%d%H"):k for k,t in enumerate(idx)}
ACC={v:np.full((nBA,NH),np.nan,np.float32) for v in ["T2","Q2","SWDOWN","GLW","WSPD"]}
files=sorted(glob.glob(SRC+"/tgw_historical_*hourly*.npz")); log(f"{len(files)} future npz")
used=0
for fi,fn in enumerate(files):
    d=np.load(fn); ts=d["times"]; keep=[(ti,stamp2i[str(s)]) for ti,s in enumerate(ts) if str(s) in stamp2i]
    if not keep: continue
    dat=d["data"].astype(np.float32); T=dat.shape[0]
    spd=np.sqrt(dat[:,0]**2+dat[:,1]**2).reshape(T,nCell)
    ba={}
    for v,ix in {"Q2":2,"T2":4,"GLW":5,"SWDOWN":6}.items():
        ba[v]=(Wba@(M@dat[:,ix].reshape(T,nCell).T)).T
    ba["WSPD"]=(Wba@(M@spd.T)).T; ba["Q2"]=np.clip(ba["Q2"],0,None)
    for ti,gi in keep:
        for v in ACC: ACC[v][:,gi]=ba[v][ti]
    used+=1
    if fi%100==0: log(f"agg {fi}/{len(files)}")
log(f"agg done used={used} cov={np.isfinite(ACC['T2']).all(0).mean()*100:.1f}%")
yv=idx.year.values
for b in modeled:
    if b not in ba2col: continue
    col=ba2col[b]
    for Y in YEARS:
        sel=np.where(yv==Y)[0]
        df=pd.DataFrame({"Time_UTC":[idx[k].strftime("%Y-%m-%d %H:%M:%S") for k in sel],
            "T2":ACC["T2"][col,sel],"Q2":ACC["Q2"][col,sel],"SWDOWN":ACC["SWDOWN"][col,sel],
            "GLW":ACC["GLW"][col,sel],"WSPD":ACC["WSPD"][col,sel]})
        if df[["T2","Q2","SWDOWN","GLW","WSPD"]].isna().any().any(): df=df.ffill().bfill()
        df.to_csv(f"{FDIR}/{b}_WRF_Hourly_Mean_Meteorology_{Y}.csv",index=False)
log("forcing CSVs written")

# ---- Stage 3+5: predict + per-BA/month bias correction ----
# The validated calibration, not the undocumented per-month scalar. See 04_demand_model/loadcal.py.
import loadcal as _LC
QM = _LC.load_maps()
print("  quantile maps loaded for %d balancing authorities; the rest take no transform"
      % len(QM), flush=True)
BA=np.zeros((nBA,NH),np.float32)
mo_of=idx.month.values
for b in modeled:
    if b not in ba2col: continue
    ser=[]
    for Y in YEARS:
        f=f"{FDIR}/{b}_WRF_Hourly_Mean_Meteorology_{Y}.csv"
        if not os.path.exists(f): continue
        pr=tell.predict(b,Y,FDIR); pr.index=pd.to_datetime(pr["Time_UTC"]); ser.append(pr["Load"])
    if not ser: continue
    s=pd.concat(ser).reindex(idx).values.astype(np.float32)
    s=_LC.apply_ba(s, mo_of, b, QM)      # the selected quantile transform, or none
    BA[ba2col[b]]=s
log("predict+bias done")

# ---- Stage 4: county disaggregation ----
county=(Wba.T@BA).astype(np.float32)     # (nC, NH) MW
np.save(f"{OUT}/ba_load_hourly.npy",BA)
np.savez(f"{OUT}/meta.npz",bas=np.array(bas),fips=fips,years=np.array(YEARS),
         t0=idx[0].strftime("%Y-%m-%d %H:%M"),NH=NH)
# county hourly is big; save as fp32 memmap-friendly .npy
np.save(f"{OUT}/county_load_hourly.npy",county)
# summary per year: annual TWh + peak GW (US total)
usa=BA.sum(0)   # US-total hourly MW (sum over BAs = sum over counties, conserved)
ann=pd.Series(usa,index=idx).groupby(idx.year).agg(["sum","max","mean"])
ann["TWh"]=ann["sum"]/1e6; ann["peakGW"]=ann["max"]/1e3
ann[["TWh","peakGW"]].to_csv(f"{OUT}/annual_summary.csv")
log(f"DONE. US mean {ann['TWh'].mean():.0f} TWh/yr, peak {ann['peakGW'].max():.0f} GW. saved -> {OUT}")
