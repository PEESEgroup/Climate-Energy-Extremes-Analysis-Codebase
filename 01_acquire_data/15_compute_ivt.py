"""Compute daily IVT (vertically integrated vapor transport) from NCEP/NCAR R1, 1980-2019.
IVT = 1/g * integral(q * V dp) over 1000..300 hPa (8 levels where shum is available).
Save a North-America window (lat 15-60N, lon 180-300E) for AR detection + AR footprint labels."""
import numpy as np, xarray as xr, glob
NCEP="/data/ncep"; g=9.80665
LEV=[1000.,925.,850.,700.,600.,500.,400.,300.]        # hPa, shum-available
p=np.array(LEV)*100.0                                  # Pa, descending
# trapezoidal weights over descending p
def ivt_from(qu):                                      # qu: (...,nlev) of q*wind at LEV
    acc=0.0
    for k in range(len(p)-1):
        acc=acc+0.5*(qu[...,k]+qu[...,k+1])*(p[k]-p[k+1])
    return acc/g
IU=[];IV=[];IMAG=[];DATES=[]
la=lo=None
for y in range(1980,2020):
    ds_q=xr.open_dataset(f"{NCEP}/shum.{y}.nc"); ds_u=xr.open_dataset(f"{NCEP}/uwnd.{y}.nc"); ds_v=xr.open_dataset(f"{NCEP}/vwnd.{y}.nc")
    q=ds_q["shum"].sel(level=LEV); u=ds_u["uwnd"].sel(level=LEV); v=ds_v["vwnd"].sel(level=LEV)
    lat=q["lat"].values; lon=q["lon"].values
    mla=(lat>=15)&(lat<=60); mlo=(lon>=180)&(lon<=300)
    q=q.values[:,:,mla][:,:,:,mlo]; u=u.values[:,:,mla][:,:,:,mlo]; v=v.values[:,:,mla][:,:,:,mlo]  # (t,lev,nla,nlo)
    q=np.moveaxis(q,1,-1); u=np.moveaxis(u,1,-1); v=np.moveaxis(v,1,-1)   # (t,nla,nlo,lev)
    iu=ivt_from(q*u); iv=ivt_from(q*v); im=np.sqrt(iu**2+iv**2)
    IU.append(iu.astype("f4")); IV.append(iv.astype("f4")); IMAG.append(im.astype("f4"))
    DATES.append(ds_q["time"].values)
    if la is None: la=lat[mla]; lo=lon[mlo]
    ds_q.close(); ds_u.close(); ds_v.close()
    print("IVT",y,"days",im.shape[0],flush=True)
IU=np.concatenate(IU); IV=np.concatenate(IV); IMAG=np.concatenate(IMAG)
dates=np.concatenate(DATES).astype("datetime64[D]").astype(str)
np.savez_compressed(f"{NCEP}/ivt_daily_1980_2019.npz",ivt=IMAG,ivtu=IU,ivtv=IV,lat=la,lon=lo,dates=dates)
print("SAVED ivt_daily_1980_2019.npz shape",IMAG.shape,"lat",la.min(),la.max(),"lon",lo.min(),lo.max())
print("IVT stats kg/m/s: mean",round(float(np.nanmean(IMAG)),1),"p99",round(float(np.nanpercentile(IMAG,99)),1),"max",round(float(np.nanmax(IMAG)),1))
