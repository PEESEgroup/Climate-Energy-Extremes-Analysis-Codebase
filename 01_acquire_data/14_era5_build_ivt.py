"""ERA5 'downoven' STEP 2a — build daily IVT (0.5deg) from ERA5 viwve/viwvn, 1980-2019, keys matching the
NCEP ivt npz so 09_hazard_ar.py logic drops in. Out: /data/ncep/ivt_daily_era5_1980_2019.npz"""
import numpy as np, pandas as pd, xarray as xr, glob
G="/data/era5"
files=sorted(glob.glob(f"{G}/ivt_*.nc")); print(f"{len(files)} ivt files")
ivt=[];u=[];v=[];dts=[]; lat=None; lon360=None
for f in files:
    ds=xr.open_dataset(f)
    eu=ds["viwve"]; ev=ds["viwvn"]
    for dim in ["number","expver","pressure_level"]:
        if dim in eu.dims: eu=eu.isel({dim:0}); ev=ev.isel({dim:0})
    ud=eu.resample(valid_time="1D").mean(); vd=ev.resample(valid_time="1D").mean()
    U=ud.values.astype("f4"); V=vd.values.astype("f4")
    ivt.append(np.sqrt(U**2+V**2)); u.append(U); v.append(V)
    dts.append(pd.to_datetime(ud.valid_time.values))
    if lat is None: lat=ds.latitude.values.astype("f4"); lon360=((ds.longitude.values+360.0)%360.0).astype("f4")
    ds.close(); print(f"  {f.split('/')[-1]}",flush=True)
IVT=np.concatenate(ivt,0); IVTU=np.concatenate(u,0); IVTV=np.concatenate(v,0)
dates=pd.to_datetime(np.concatenate([d.values for d in dts]))
o=np.argsort(dates.values); dates=dates[o]; IVT=IVT[o]; IVTU=IVTU[o]; IVTV=IVTV[o]
np.savez(f"/data/ncep/ivt_daily_era5_1980_2019.npz", ivt=IVT, ivtu=IVTU, ivtv=IVTV,
         lat=lat, lon=lon360, dates=np.array([str(d.date()) for d in dates]))
print(f"shape {IVT.shape}  lat {lat.min():.1f}..{lat.max():.1f} ({len(lat)})  lon360 {lon360.min():.1f}..{lon360.max():.1f} ({len(lon360)})")
print(f"span {dates.min().date()}..{dates.max().date()}  ndays {len(dates)}  IVT mean {IVT.mean():.0f} max {IVT.max():.0f}")
print("[ERA5_IVT_BUILD_DONE]")
