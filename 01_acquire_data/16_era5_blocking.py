"""ERA5 'downoven' STEP 1 — Tibaldi-Molteni blocking on ERA5 z500 (1.0deg, 6-hourly->daily), 1980-2019.
Replaces the coarse NCEP 2.5deg blocking. Same TM index + >=5-day persistence as blocking2.py, PLUS a new
NORTH-AMERICAN sector (230-290E) that covers the western-US/Rockies ridge sitting in the GAP between the
existing Pacific(140-220) and Atlantic(280-360) sectors -> the sector actually relevant to western VRE-drought.
Out: /data/enso/blocking_daily_era5.csv  (+ prints ERA5-vs-NCEP DJF/JJA blocked-day frequency)."""
import numpy as np, pandas as pd, xarray as xr, glob
G="/data/era5"; EN="/data/enso"
files=sorted(glob.glob(f"{G}/z500_*.nc"))
print(f"{len(files)} z500 files")
def persist(binary,minrun=5):
    out=np.zeros_like(binary); n=len(binary); i=0
    while i<n:
        if binary[i]:
            j=i
            while j<n and binary[j]: j+=1
            if j-i>=minrun: out[i:j]=True
            i=j
        else: i+=1
    return out
blk_parts=[]; date_parts=[]; lon360=None
for f in files:
    ds=xr.open_dataset(f)
    z=ds["z"]
    for dim in ["pressure_level","number","expver"]:
        if dim in z.dims: z=z.isel({dim:0})
    z=z/9.80665                                   # geopotential -> geopotential height (m)
    zd=z.resample(valid_time="1D").mean()
    lat=ds.latitude.values; lon=ds.longitude.values
    if lon360 is None: lon360=(lon+360.0)%360.0
    Z=zd.values                                   # (ndays, nlat, nlon)
    dates=pd.to_datetime(zd.valid_time.values)
    def zl(phi): return Z[:, int(np.argmin(np.abs(lat-phi))), :]
    blk=np.zeros((Z.shape[0], Z.shape[2]), bool)
    for d in (-5.,0.,5.):
        GHGN=(zl(80+d)-zl(60+d))/20.; GHGS=(zl(60+d)-zl(40+d))/20.
        blk |= (GHGS>0)&(GHGN<-10)
    blk_parts.append(blk); date_parts.append(dates); ds.close()
    print(f"  {f.split('/')[-1]}  days {len(dates)}", flush=True)
BLK=np.concatenate(blk_parts,0); dates=pd.to_datetime(np.concatenate([d.values for d in date_parts]))
o=np.argsort(dates.values); dates=dates[o]; BLK=BLK[o]
def sector(lo,hi,thr=0.20):
    m=(lon360>=lo)&(lon360<=hi); frac=BLK[:,m].mean(1); return frac, persist(frac>=thr,5)
pf,pe=sector(140,220); af,ae=sector(280,360); nf,ne=sector(230,290)
df=pd.DataFrame({"date":dates,"pac_frac":pf,"atl_frac":af,"nam_frac":nf,
                 "pac_block":pe.astype(int),"atl_block":ae.astype(int),"nam_block":ne.astype(int)})
df.to_csv(f"{EN}/blocking_daily_era5.csv",index=False)
mo=df.date.dt.month; djf=df[mo.isin([12,1,2])]; jja=df[mo.isin([6,7,8])]
print(f"\nERA5 (1.0deg) DJF blocked-day freq: Pac {100*djf.pac_block.mean():.1f}%  Atl {100*djf.atl_block.mean():.1f}%  NAm {100*djf.nam_block.mean():.1f}%")
print(f"ERA5 (1.0deg) JJA blocked-day freq: Pac {100*jja.pac_block.mean():.1f}%  Atl {100*jja.atl_block.mean():.1f}%  NAm {100*jja.nam_block.mean():.1f}%")
try:
    nc=pd.read_csv(f"{EN}/blocking_daily.csv"); nc["date"]=pd.to_datetime(nc.date); ncm=nc.date.dt.month
    ncdjf=nc[ncm.isin([12,1,2])]
    print(f"NCEP (2.5deg) DJF blocked-day freq: Pac {100*ncdjf.pac_block.mean():.1f}%  Atl {100*ncdjf.atl_block.mean():.1f}%  (no NAm sector)")
except Exception as e: print("NCEP compare skipped:",e)
print(f"\nspan {dates.min().date()} -> {dates.max().date()}  ndays {len(df)}")
print("saved /data/enso/blocking_daily_era5.csv  [ERA5_BLOCK_DONE]")
