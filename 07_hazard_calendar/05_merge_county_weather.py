"""
Merge the county-weather shards, recombining boundary-split days EXACTLY (same rule as
merge_subweather.py): tmax = max, tmin = min, everything else = hour-count-weighted mean.

Then validate two ways:
  1. every county x date present exactly once, 24 h/day except at UTC file edges
  2. re-aggregate counties to the 18 subregions (weighted by each county's TGW cell count) and
     compare against /data/enso/subregion_weather_daily.csv. These cannot match to machine
     precision - counties do not tile a subregion (coastal/offshore cells belong to no county) -
     but a large discrepancy would mean the county mapping is wrong.
"""
import glob
import numpy as np, pandas as pd

SH = sorted(glob.glob("/data/enso/county_weather/county_weather_shard*.parquet"))
print("shards:", len(SH))
d = pd.concat([pd.read_parquet(f) for f in SH], ignore_index=True)
print("partial rows %s   memory %.1f GB" % (format(len(d), ","), d.memory_usage(deep=True).sum() / 1e9))

d["fips"] = d.fips.astype("category")
d["date"] = pd.to_datetime(d.date)
SUMS = ["tmean", "q", "ps", "sw", "wspd"]
g = d.groupby(["fips", "date"], observed=True, sort=False)
out = g.agg(tmax=("tmax", "max"), tmin=("tmin", "min"), n=("n", "sum"),
            **{c: (c, "sum") for c in SUMS}).reset_index()
del d
for c in SUMS:
    out[c] = (out[c] / out["n"]).astype("f4")
print("clean rows %s   counties %d   dates %d"
      % (format(len(out), ","), out.fips.nunique(), out.date.nunique()))
assert not out.duplicated(["fips", "date"]).any(), "duplicate county-date"
bad = int((out.n != 24).sum())
print("days with != 24 h: %d (%.3f%%)   n range %d..%d"
      % (bad, 100 * bad / len(out), out.n.min(), out.n.max()))
print("dates %s .. %s" % (out.date.min().date(), out.date.max().date()))

out["fips"] = out.fips.astype(str)
out = out.sort_values(["fips", "date"])
out.drop(columns=["n"]).to_parquet("/data/enso/county_weather_daily.parquet", index=False)
print("WROTE /data/enso/county_weather_daily.parquet")

# ------------------------------------------------------------------ validation vs subregions
cm = np.load("/data/loads_measured/county_mask_tgw.npz", allow_pickle=True)
fips = np.array([str(f).zfill(5) for f in cm["fips"]])
pc, pf = cm["pair_cell"].astype(np.int64), cm["pair_fips"].astype(np.int64)
sm = np.load("/data/datasets/grid/subregion_mask.npz", allow_pickle=True)
smask = sm["subregion_mask"]
id2sub = {int(a): str(b) for a, b in sm["id_to_subregion"]}
zc = np.load("/data/datasets/grid/coordinate.npz")
glat, glon = zc["lat"].astype(float), zc["lon"].astype(float)
g_ = np.load("/data/tgw_hist/tgw_grid.npz")
XLAT, XLON = g_["XLAT"].ravel(), g_["XLONG"].ravel()
ila = np.clip(np.searchsorted(glat, XLAT[pc]), 0, len(glat) - 1)
ilo = np.clip(np.searchsorted(glon, XLON[pc]), 0, len(glon) - 1)
cell_sub = smask[ila, ilo]
tab = pd.DataFrame({"fips": fips[pf], "sub": cell_sub})
tab = tab[tab["sub"] > 0]
cw = tab.groupby(["fips", "sub"]).size().rename("cells").reset_index()
dom = cw.sort_values("cells").drop_duplicates("fips", keep="last")
print("\ncounties assigned to a subregion: %d" % len(dom))

S = pd.read_csv("/data/enso/subregion_weather_daily.csv")
S["date"] = pd.to_datetime(S.date)
S["subname"] = S["sub"].map(id2sub) if S["sub"].dtype != object else S["sub"]
chk = out.merge(dom[["fips", "cells", "sub"]], on="fips")
for c in ["tmean", "wspd"]:
    chk[c + "_w"] = chk[c] * chk.cells
agg = chk.groupby(["sub", "date"]).agg(cells=("cells", "sum"),
                                       **{c + "_w": (c + "_w", "sum") for c in ["tmean", "wspd"]})
for c in ["tmean", "wspd"]:
    agg[c] = agg[c + "_w"] / agg.cells
agg = agg.reset_index()
agg["subname"] = agg["sub"].map(id2sub)
m = agg.merge(S, on=["subname", "date"], suffixes=("_cty", "_sub"))
print("matched subregion-days: %s" % format(len(m), ","))
for c in ["tmean", "wspd"]:
    r = np.corrcoef(m[c + "_cty"], m[c + "_sub"])[0, 1]
    bias = (m[c + "_cty"] - m[c + "_sub"]).mean()
    rmse = np.sqrt(((m[c + "_cty"] - m[c + "_sub"]) ** 2).mean())
    print("   %-6s r = %.5f   bias %+.4f   rmse %.4f" % (c, r, bias, rmse))
print("\n(a bias is expected: counties omit offshore/unassigned cells that the subregion mean includes)")
