"""Part B step 2: subregion x sector x hourly load, 1980-2024. Splits the real subregion total into
res/com/ind/trans by: annual sector fraction f[sub,s,year] from REAL SEDS (state->subregion weighted, every
year) x dsgrid diurnal/seasonal SHAPE Sh[s,mo,hod]. sector_hourly = total x normalize(f*Sh).
Result preserves the real subregion total (sum over sectors = total) and real annual sector mix per year."""
import numpy as np, pandas as pd
O="/data/tell_pred/future/hist_full45_seds"
m=np.load(f"{O}/meta.npz",allow_pickle=True); NH=int(m["NH"]); fips=np.array([str(f).zfill(5) for f in m["fips"]])
subcode=m["subcode"]; names=list(m["subnames"]); idx=pd.date_range("1980-01-01",periods=NH,freq="h")
sub_tot=np.load(f"{O}/subregion_load_hourly.npy")                     # (18,NH)
FA={"01":"AL","02":"AK","04":"AZ","05":"AR","06":"CA","08":"CO","09":"CT","10":"DE","11":"DC","12":"FL","13":"GA",
"16":"ID","17":"IL","18":"IN","19":"IA","20":"KS","21":"KY","22":"LA","23":"ME","24":"MD","25":"MA","26":"MI","27":"MN",
"28":"MS","29":"MO","30":"MT","31":"NE","32":"NV","33":"NH","34":"NJ","35":"NM","36":"NY","37":"NC","38":"ND","39":"OH",
"40":"OK","41":"OR","42":"PA","44":"RI","45":"SC","46":"SD","47":"TN","48":"TX","49":"UT","50":"VT","51":"VA","53":"WA",
"54":"WV","55":"WI","56":"WY"}
st_of=np.array([FA.get(f[:2],"??") for f in fips])
SEC=["res","com","ind","trans"]; MSN=["ESRCP","ESCCP","ESICP","ESACP"]

# state weight within each subregion (county load 2016-19 by state)
cty=np.load(f"{O}/county_load_hourly.npy",mmap_mode="r"); m1619=np.where((idx.year>=2016)&(idx.year<=2019))[0]
cmean=np.asarray(cty[:,m1619]).mean(1)                               # (nC,) county mean load
w=np.zeros((18,60)); states=sorted(set(st_of)-{"??"}); si={s:i for i,s in enumerate(states)}
for c in range(len(fips)):
    if subcode[c]>0 and st_of[c]!="??": w[subcode[c]-1, si[st_of[c]]]+=cmean[c]
w=w[:, :len(states)]; w=w/np.maximum(w.sum(1,keepdims=True),1e-9)    # (18, nstate) load-share

# SEDS state x sector fraction per year
S=pd.read_csv("/data/loads_measured/seds_use_all_phy.csv")
def seds_frac_year(y):
    yc=str(y) if str(y) in S.columns else "2024"
    F=np.zeros((len(states),4))
    for k,code in enumerate(MSN):
        r=S[S.MSN==code].set_index("State")[yc]
        F[:,k]=[float(r.get(s,0.0)) for s in states]
    F=F/np.maximum(F.sum(1,keepdims=True),1e-9); return F            # (nstate,4)
# subregion sector fraction per year f[sub,4,year]
years=list(range(1980,2025)); fSY=np.zeros((18,4,len(years)))
for yi,y in enumerate(years):
    Fst=seds_frac_year(y); fSY[:,:,yi]=w@Fst                         # (18,4)

Sh=np.load("/data/analysis/national_sector_shape.npz")
SHAPE=np.stack([Sh["res"],Sh["com"],Sh["ind"],np.ones((12,24))],0)   # (4,12,24), trans flat

out=np.zeros((18,4,NH),"float32")
moA=idx.month.values; hodA=idx.hour.values; yrA=idx.year.values
for yi,y in enumerate(years):
    hsel=np.where(yrA==y)[0]; fY=fSY[:,:,yi]                          # (18,4)
    # shareY[sub,4,mo,hod] = fY*SHAPE normalized over sector
    num=fY[:,:,None,None]*SHAPE[None,:,:,:]                           # (18,4,12,24)
    shareY=num/np.maximum(num.sum(1,keepdims=True),1e-12)
    mo_h=moA[hsel]-1; hod_h=hodA[hsel]
    sh=shareY[:,:,mo_h,hod_h]                                         # (18,4,len(hsel))
    out[:,:,hsel]=(sub_tot[:,None,hsel]*sh).astype("float32")
    if y%10==0: print(f"  sector hourly {y}",flush=True)
np.save(f"{O}/subregion_sector_hourly.npy",out)

# --- validate ---
tot_err=np.abs(out.sum(1)-sub_tot).sum()/np.abs(sub_tot).sum()
print(f"\nsum-over-sector == total: rel err {tot_err:.2e} (should ~0)")
us=out.sum(0)                                                        # (4,NH)
for y in [1990,2019,2024]:
    a=us[:,yrA==y].sum(1)/1e6; f=a/a.sum()
    seF=seds_frac_year(y); seUS=(w.sum(0)@seF)/(w.sum(0)@seF).sum()  # approx US SEDS frac (load-weighted)
    print(f"  {y}: US sector TWh res {a[0]:.0f} com {a[1]:.0f} ind {a[2]:.0f} trans {a[3]:.1f} | frac res {f[0]:.0%} com {f[1]:.0%} ind {f[2]:.0%}")
np.savez(f"{O}/sector_meta.npz",sectors=np.array(SEC),subnames=np.array(names),years=np.array(years),fSY=fSY)
print(f"WROTE {O}/subregion_sector_hourly.npy (18x4x{NH}) + sector_meta.npz")
