"""Stage 5 FINAL: adaptive per-BA calibration. Candidate configs {none, month-EQM, season-EQM};
choose per BA by CV-RMSE (fit 2016-17, select on 2018). Final: refit choice on 2016-18, HELD-OUT test 2019.
Also refit on all 2016-19 -> production transfer funcs (for future QDM). Save metrics, transfers, plots."""
import tell, pandas as pd, numpy as np, os, warnings, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
FORCING="/data/tell_forcing/historic"; COMP="/data/tell_qs/tell_quickstarter_data/outputs/compiled_historical_data"
OUT="/data/tell_pred"; os.makedirs(OUT+"/figs",exist_ok=True)
YEARS=[2016,2017,2018,2019]; ba54=sorted(tell.get_balancing_authority_to_model_dict().keys())
CANDS=[("none",None,0),("month","mo",40),("season","seas",25)]

def get_pred(b):
    ps=[]
    for Y in YEARS:
        if os.path.exists(f"{FORCING}/{b}_WRF_Hourly_Mean_Meteorology_{Y}.csv"):
            pr=tell.predict(b,Y,FORCING); pr.index=pd.to_datetime(pr["Time_UTC"]); ps.append(pr["Load"])
    return pd.concat(ps) if ps else None
def get_obs(b):
    d=pd.read_csv(f"{COMP}/{b}_historical_data.csv")
    d["dt"]=pd.to_datetime(dict(year=d.Year,month=d.Month,day=d.Day,hour=d.Hour))
    return d.set_index("dt")["Adjusted_Demand_MWh"]
def cache(b):
    pr=get_pred(b); ob=get_obs(b)
    if pr is None: return None
    idx=pr.index.intersection(ob.index)
    df=pd.DataFrame({"pred":pr.reindex(idx),"obs":ob.reindex(idx)}).dropna(); df=df[df.obs>0]
    df["mo"]=df.index.month; df["seas"]=(df.index.month%12//3); df["yr"]=df.index.year
    return df
def fit_tf(tr,sc,nq):
    qs=np.linspace(0.5,99.5,nq)/100.0; tf={}
    for s in tr[sc].unique():
        trm=tr[tr[sc]==s]
        if len(trm)>=50: tf[s]=(np.quantile(trm["pred"],qs),np.quantile(trm["obs"],qs))
    return tf
def apply_tf(pred,strat,tf):
    out=pred.copy().astype(float)
    for s,(pp,oo) in tf.items():
        m=(strat==s).values if hasattr(strat,"values") else (strat==s)
        out[m]=np.interp(pred[m],pp,oo)
    return out
def mets(p,o):
    m=np.isfinite(p)&np.isfinite(o)&(o>0); p=p[m];o=o[m]
    rmse=np.sqrt(((p-o)**2).mean())
    return dict(n=len(o),r2=1-((p-o)**2).sum()/((o-o.mean())**2).sum(),mape=np.mean(np.abs(p-o)/o)*100,
        cvrmse=rmse/o.mean()*100,rmse=rmse,nmbe=(p-o).sum()/o.sum()*100,
        p999=(np.percentile(p,99.9)-np.percentile(o,99.9))/np.percentile(o,99.9)*100)

rows=[]; CHOICE={}; PROD={}
for b in ba54:
    df=cache(b)
    if df is None or (df.yr.isin([2016,2017,2018])).sum()<3000 or (df.yr==2019).sum()<3000: continue
    # --- model selection by CV-RMSE: fit 2016-17, score 2018 ---
    cvtr=df[df.yr.isin([2016,2017])]; cvte=df[df.yr==2018]
    best=("none",None,0); bestr=None
    for name,sc,nq in CANDS:
        if sc is None: cal=cvte["pred"].values.astype(float)
        else: cal=apply_tf(cvte["pred"].values.astype(float),cvte[sc],fit_tf(cvtr,sc,nq))
        rm=np.sqrt(((cal-cvte["obs"].values)**2).mean())
        rm_eff=rm*(0.999 if name=="none" else 1.0)   # tiny bias toward raw on ties
        if bestr is None or rm_eff<bestr: bestr=rm_eff; best=(name,sc,nq)
    CHOICE[b]=best[0]
    # --- final held-out: refit choice on 2016-18, apply 2019 ---
    tr=df[df.yr.isin([2016,2017,2018])]; te=df[df.yr==2019]
    name,sc,nq=best
    raw=te["pred"].values.astype(float); obs=te["obs"].values.astype(float)
    cal=raw if sc is None else apply_tf(raw,te[sc],fit_tf(tr,sc,nq))
    mr=mets(raw,obs); mc=mets(cal,obs)
    rows.append(dict(BA=b,choice=name,**{f"raw_{k}":v for k,v in mr.items()},**{f"cal_{k}":v for k,v in mc.items()}))
    # --- production transfer: refit on ALL 2016-19 ---
    if sc is not None: PROD[b]=(sc,nq,fit_tf(df,sc,nq))

R=pd.DataFrame(rows); R.to_csv(OUT+"/calibration_2019_adaptive.csv",index=False)
flat={}
for b in PROD:
    sc,nq,tf=PROD[b]
    for s in tf:
        flat[f"{b}|{sc}|{nq}|{s}|pred"]=tf[s][0]; flat[f"{b}|{sc}|{nq}|{s}|obs"]=tf[s][1]
np.savez(OUT+"/qm_transfer_prod.npz",**flat)

def med(c): return R[c].median()
print("choice counts:", R.choice.value_counts().to_dict())
print("\n=== held-out 2019 median (adaptive) BEFORE -> AFTER ===")
for k in ["r2","mape","cvrmse","nmbe","p999"]:
    print("  %-7s raw=%8.3f  cal=%8.3f" % (k,med("raw_"+k),med("cal_"+k)))
hurt=(R.cal_r2<R.raw_r2-0.02).sum(); help_=(R.cal_r2>R.raw_r2+0.02).sum()
print("  R2: helped(>0.02)=%d  hurt(>0.02)=%d ;  |P99.9|err raw=%.2f cal=%.2f ;  |nMBE|<2%%: %d->%d" % (
    help_,hurt,R.raw_p999.abs().median(),R.cal_p999.abs().median(),(R.raw_nmbe.abs()<2).sum(),(R.cal_nmbe.abs()<2).sum()))
print("\n=== big BAs ===")
big=["PJM","MISO","CISO","ERCO","SOCO","ISNE","NYIS","DUK","FPL","TVA","SWPP","AZPS"]
print(R.set_index("BA").reindex([b for b in big if b in R.BA.values])[
    ["choice","raw_r2","cal_r2","raw_mape","cal_mape","raw_nmbe","cal_nmbe","raw_p999","cal_p999"]].round(2).to_string())

# plots
for b in ["PJM","ERCO","CISO","MISO"]:
    df=cache(b); tr=df[df.yr.isin([2016,2017,2018])]; te=df[df.yr==2019].copy()
    name=CHOICE.get(b,"none"); sc={"month":"mo","season":"seas","none":None}[name]; nq={"month":40,"season":25,"none":0}[name]
    te["cal"]=te["pred"].values if sc is None else apply_tf(te["pred"].values.astype(float),te[sc],fit_tf(tr,sc,nq))
    fig,ax=plt.subplots(1,2,figsize=(13,4.2))
    for s,c,l in [("obs","k","measured"),("pred","C3","raw MLP"),("cal","C0","calibrated")]:
        ax[0].plot(np.sort(te[s])[::-1]/1000,c,lw=1.4,label=l)
    ax[0].set_title(f"{b} 2019 load-duration ({name})"); ax[0].set_xlabel("hours ranked"); ax[0].set_ylabel("GW"); ax[0].legend()
    pk=te["obs"].idxmax(); wk=te.loc[pk-pd.Timedelta("3.5D"):pk+pd.Timedelta("3.5D")]
    ax[1].plot(wk.index,wk.obs/1000,"k",label="measured"); ax[1].plot(wk.index,wk.pred/1000,"C3",alpha=.6,label="raw")
    ax[1].plot(wk.index,wk.cal/1000,"C0",alpha=.8,label="calibrated")
    ax[1].set_title(f"{b} 2019 peak week"); ax[1].set_ylabel("GW"); ax[1].legend(); fig.autofmt_xdate()
    fig.tight_layout(); fig.savefig(OUT+f"/figs/load_{b}.png",dpi=110); plt.close(fig)
print("\nplots -> /data/tell_pred/figs/")
