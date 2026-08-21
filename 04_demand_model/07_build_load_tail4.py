"""Part A (v4, FINAL): base 1980-2019 = hist_full40_seds AS-IS (validated model/EIA930 basis). Tail 2020-24
= OEDI-8562 real COUNTY hourly (county-level real -> correct multi-BA splits, unlike EIA930 capacity-split),
scaled per-county by t_c = mean_base/mean_8562 (2016-19) onto the model basis so the 2019/2020 seam is clean.
2024 = 8562-2023 calendar-matched x per-state SEDS(2024/2023) growth."""
import numpy as np, pandas as pd, os
HF="/data/tell_pred/future/hist_full40"; B="/data/tell_pred/future/hist_full40_seds"
OUT="/data/tell_pred/future/hist_full45_seds"; os.makedirs(OUT,exist_ok=True)
meta=np.load(f"{HF}/meta.npz",allow_pickle=True); fips=np.array([str(f).zfill(5) for f in meta["fips"]]); nC=len(fips)
base_idx=pd.date_range("1980-01-01",periods=int(meta["NH"]),freq="h")
tail_idx=pd.date_range("2020-01-01","2024-12-31 23:00",freq="h")
full_idx=pd.date_range("1980-01-01","2024-12-31 23:00",freq="h")
NHb=len(base_idx); NHt=len(tail_idx); NH=len(full_idx)
FA={"01":"AL","02":"AK","04":"AZ","05":"AR","06":"CA","08":"CO","09":"CT","10":"DE","11":"DC","12":"FL","13":"GA",
"16":"ID","17":"IL","18":"IN","19":"IA","20":"KS","21":"KY","22":"LA","23":"ME","24":"MD","25":"MA","26":"MI","27":"MN",
"28":"MS","29":"MO","30":"MT","31":"NE","32":"NV","33":"NH","34":"NJ","35":"NM","36":"NY","37":"NC","38":"ND","39":"OH",
"40":"OK","41":"OR","42":"PA","44":"RI","45":"SC","46":"SD","47":"TN","48":"TX","49":"UT","50":"VT","51":"VA","53":"WA",
"54":"WV","55":"WI","56":"WY"}
st_of=np.array([FA.get(f[:2],"??") for f in fips])
fm=pd.read_csv("/tmp/fips_to_subregion_mapping.csv"); fm["FIPS"]=fm["FIPS"].astype(str).str.zfill(5)
f2sub=dict(zip(fm["FIPS"],fm["Subregion_Code"])); c2n=fm.drop_duplicates("Subregion_Code").set_index("Subregion_Code")["Subregion"].to_dict()
subcode=np.array([f2sub.get(f,0) for f in fips])
oe=pd.read_hdf("/data/loads_measured/historic_load_hourly_2016_2023_county.h5"); oe.index=pd.to_datetime(oe.index)
if oe.index.tz is not None: oe.index=oe.index.tz_convert("UTC").tz_localize(None)
oe=oe.rename(columns={c:str(c)[1:].zfill(5) for c in oe.columns}).reindex(columns=fips)
oeM=np.nan_to_num(oe.values.astype("float32")); oe_time=oe.index; oe_pos={t:i for i,t in enumerate(oe_time)}
S=pd.read_csv("/data/loads_measured/seds_use_all_phy.csv"); es=S[S.MSN=="ESTCP"].set_index("State")
g2423={s:(float(es.loc[s,"2024"])/float(es.loc[s,"2023"]) if s in es.index and float(es.loc[s,"2023"])>0 else 1.0) for s in set(st_of) if s!="??"}
gcty=np.array([g2423.get(st_of[c],1.0) for c in range(nC)],"float32")
# per-county scale to model basis
base_cty=np.load(f"{B}/county_load_hourly_realdist.npy",mmap_mode="r")
mb=np.where((base_idx.year>=2016)&(base_idx.year<=2019))[0]; oe1619=(oe_time.year>=2016)&(oe_time.year<=2019)
mean_oe=oeM[oe1619].mean(0); mean_bs=np.asarray(base_cty[:,mb]).mean(1)
nat=float(np.nansum(mean_bs)/np.nansum(mean_oe))
t_c=np.where(mean_oe>0, mean_bs/mean_oe, nat); t_c=np.clip(np.nan_to_num(t_c,nan=nat),0.5,2.0).astype("float32")
print(f"tail->model scale t_c: national {nat:.3f} median {np.median(t_c):.3f}",flush=True)
cty_tail=np.zeros((nC,NHt),"float32")
for j,t in enumerate(tail_idx):
    if t.year<=2023:
        if t in oe_pos: cty_tail[:,j]=oeM[oe_pos[t]]*t_c
    else:
        src=pd.Timestamp(2023,t.month,28 if (t.month==2 and t.day==29) else t.day,t.hour)
        if src in oe_pos: cty_tail[:,j]=oeM[oe_pos[src]]*gcty*t_c
    if j%8760==0: print(f"  tail {t.year}",flush=True)
cty_out=np.lib.format.open_memmap(f"{OUT}/county_load_hourly.npy",mode="w+",dtype="float32",shape=(nC,NH))
for i0 in range(0,NHb,50000):
    e=min(i0+50000,NHb); cty_out[:,i0:e]=np.asarray(base_cty[:,i0:e])
cty_out[:,NHb:]=cty_tail; cty_out.flush()
sub_full=np.zeros((18,NH),"float32")
for sc in range(1,19):
    ids=np.where(subcode==sc)[0]
    if ids.size:
        for i0 in range(0,NH,50000): sub_full[sc-1,i0:i0+50000]=cty_out[ids,i0:i0+50000].sum(0)
np.save(f"{OUT}/subregion_load_hourly.npy",sub_full)
np.savez(f"{OUT}/meta.npz",fips=meta["fips"],t0="1980-01-01",NH=NH,subcode=subcode,subnames=np.array([c2n[c] for c in range(1,19)]))
us=sub_full.sum(0)
rows=[(y,round(float(us[full_idx.year==y].sum())/1e6,1),"real-8562->modelbasis" if 2020<=y<=2023 else ("8562x SEDS24" if y==2024 else "model/EIA930-basis")) for y in range(1980,2025)]
pd.DataFrame(rows,columns=["year","US_TWh","basis"]).to_csv(f"{OUT}/annual_summary.csv",index=False)
print("US last8:",[(y,v) for y,v,_ in rows[-8:]])
a19=sub_full[:,full_idx.year==2019].sum(1)/1e6; a20=sub_full[:,full_idx.year==2020].sum(1)/1e6
jump=100*(a20/a19-1); print(f"SEAM US {a19.sum():.0f}->{a20.sum():.0f} ({100*(a20.sum()/a19.sum()-1):+.1f}%); per-sub max|{np.abs(jump).max():.1f}%| med|{np.median(np.abs(jump)):.1f}%|")
print(f"WROTE {OUT}/ (v4: base as-is + 8562 tail on model basis)")
