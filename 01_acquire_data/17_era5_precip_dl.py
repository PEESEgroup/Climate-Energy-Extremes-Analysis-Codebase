"""ERA5 monthly total precipitation, western US, 1980-2019 — the direct weather driver of hydro (accumulated
precip -> snowpack -> melt -> runoff). Small monthly-means product (fast)."""
import cdsapi
c=cdsapi.Client()
c.retrieve("reanalysis-era5-single-levels-monthly-means",{
  "product_type":"monthly_averaged_reanalysis",
  "variable":["total_precipitation","2m_temperature","snowmelt","snow_depth_water_equivalent"],
  "year":[str(y) for y in range(1980,2020)],
  "month":[f"{m:02d}" for m in range(1,13)],
  "time":"00:00","area":[55,-130,25,-100],"grid":[0.5,0.5],"data_format":"netcdf"},
  "/data/era5/precip_monthly_west_1980_2019.nc")
print("PRECIP_DL_DONE")
