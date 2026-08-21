"""ERA5 upgrade download (1980-2019) via CDS API. Replaces NCEP 2.5deg with:
- Z500 (blocking): geopotential@500hPa, NH 20-90N, 1.0deg, 00/06/12/18 -> daily-mean at analysis time.
- IVT (AR): vertical integral of eastward/northward water-vapour flux, NA 15-60N/180-300E, 0.5deg.
Year-by-year, idempotent (skip existing). Runs in background; slow (CDS queue)."""
import cdsapi, os, sys
c=cdsapi.Client()
OUT="/data/era5"; os.makedirs(OUT,exist_ok=True)
MONTHS=[f"{m:02d}" for m in range(1,13)]; DAYS=[f"{d:02d}" for d in range(1,32)]; TIMES=["00:00","06:00","12:00","18:00"]
for y in range(1980,2020):
    zf=f"{OUT}/z500_{y}.nc"
    if not os.path.exists(zf) or os.path.getsize(zf)<10000:
        try:
            c.retrieve("reanalysis-era5-pressure-levels",{"product_type":"reanalysis","variable":"geopotential",
                "pressure_level":"500","year":str(y),"month":MONTHS,"day":DAYS,"time":TIMES,
                "area":[90,-180,20,180],"grid":[1.0,1.0],"data_format":"netcdf"},zf)
            print(f"Z500 {y} OK {os.path.getsize(zf)/1e6:.0f}MB",flush=True)
        except Exception as e: print(f"Z500 {y} ERR {repr(e)[:200]}",flush=True)
    vf=f"{OUT}/ivt_{y}.nc"
    if not os.path.exists(vf) or os.path.getsize(vf)<10000:
        try:
            c.retrieve("reanalysis-era5-single-levels",{"product_type":"reanalysis",
                "variable":["vertical_integral_of_eastward_water_vapour_flux","vertical_integral_of_northward_water_vapour_flux"],
                "year":str(y),"month":MONTHS,"day":DAYS,"time":TIMES,
                "area":[60,-180,15,-60],"grid":[0.5,0.5],"data_format":"netcdf"},vf)
            print(f"IVT  {y} OK {os.path.getsize(vf)/1e6:.0f}MB",flush=True)
        except Exception as e: print(f"IVT  {y} ERR {repr(e)[:200]}",flush=True)
print("ERA5_DL_DONE")
