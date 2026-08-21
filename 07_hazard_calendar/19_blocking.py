"""STEP B: Tibaldi-Molteni sector blocking + storm-track index from NCEP/NCAR Z500 daily 1997-2019.
Daily per-longitude TM blocking; sector fractions (Pacific 140-220E, Atlantic 280-360E); DAILY binary flags.
Storm-track = DJF 2-6d band-pass Z500 std over N-America sector, per winter.
Saves /data/enso/blocking_daily.csv + /data/enso/stormtrack_winter.csv."""
import numpy as np, pandas as pd, xarray as xr, glob
from scipy.signal import butter, filtfilt
fs=sorted(glob.glob("/data/ncep/hgt.*.nc"))
ds=xr.open_mfdataset(fs,combine="by_coords").sel(level=500)
z=ds["hgt"]; lat=ds.lat.values; lon=ds.lon.values
time=pd.to_datetime(ds.time.values)
print(f"Z500 {z.shape} lat {lat.max()}..{lat.min()} lon {lon.min()}..{lon.max()} {time.min().date()}..{time.max().date()}")
Z=z.values  # (T, nlat, nlon)  meters
def zlat(phi): return Z[:,int(np.argmin(np.abs(lat-phi))),:]  # (T,nlon)
# Tibaldi-Molteni: blocked at lon if for some delta in {-5,0,5}: GHGS>0 and GHGN<-10 m/deg
blk=np.zeros(Z.shape[::2]+ (Z.shape[2],) if False else (Z.shape[0],Z.shape[2]),bool)
blk=np.zeros((Z.shape[0],Z.shape[2]),bool)
for d in (-5.,0.,5.):
    ZN=zlat(80+d); Z0=zlat(60+d); ZS=zlat(40+d)
    GHGN=(ZN-Z0)/((80+d)-(60+d)); GHGS=(Z0-ZS)/((60+d)-(40+d))
    blk |= (GHGS>0)&(GHGN<-10)
def sector_frac(lo,hi):
    m=(lon>=lo)&(lon<=hi); return blk[:,m].mean(1)
pac=sector_frac(140,220); atl=sector_frac(280,360)
df=pd.DataFrame({"date":time,"pac_frac":pac,"atl_frac":atl})
# binary "sector blocked day": >=15% of sector longitudes blocked (a coherent block, not a single lon)
df["pac_block"]=(df.pac_frac>=0.15).astype(int); df["atl_block"]=(df.atl_frac>=0.15).astype(int)
df.to_csv("/data/enso/blocking_daily.csv",index=False)
djf=df[df.date.dt.month.isin([12,1,2])]
print(f"DJF blocked-day freq: Pacific {100*djf.pac_block.mean():.1f}% | Atlantic {100*djf.atl_block.mean():.1f}%  (typical 5-20%)")
# storm-track: 2-6 day band-pass Z500 std over N-America sector (30-60N, 230-300E), per winter DJF
la=(lat>=30)&(lat<=60); loo=(lon>=230)&(lon<=300)
Zna=Z[:,la,:][:,:,loo].reshape(Z.shape[0],-1)  # (T, ncell)
b,a=butter(4,[1/6,1/2],btype="band")  # daily data: 2-6 day = freq 1/6..1/2 cyc/day
Zbp=filtfilt(b,a,Zna,axis=0)
st=pd.DataFrame({"date":time,"stvar":Zbp.std(1)})
st["wy"]=np.where(st.date.dt.month==12,st.date.dt.year+1,st.date.dt.year)
stw=st[st.date.dt.month.isin([12,1,2])].groupby("wy")["stvar"].mean()
stw.to_csv("/data/enso/stormtrack_winter.csv")
print("storm-track winter index built (DJF 2-6d bandpass Z500 std, N-Am sector)")
# annual blocking climatology check by month
print("Pacific block freq by month:", {int(m):round(100*df[df.date.dt.month==m].pac_block.mean(),1) for m in [1,4,7,10]})
print("saved blocking_daily.csv + stormtrack_winter.csv")
