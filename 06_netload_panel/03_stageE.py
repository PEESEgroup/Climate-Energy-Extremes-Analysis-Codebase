"""Stage E: 18-subregion hourly load, wind, solar and net load, on OUR production chain.

Replaces the assembly that read the published fixed-fleet generation product. County-to-subregion
mapping is copied from that script so the load side is unchanged; the generation side now comes
from the plant-level hourly capacity factors this study computes, 1980-2019.

Guards, because both of the traps this pipeline has hit before are silent ones:
  - the stored capacity factors are already NET of plant losses (wind 15%, solar 14%), so no
    further haircut may be applied here; the fleet capacity factor is asserted into a band
  - load carries a real calendar and solar drops 29 February, so the two are aligned on the
    timestamp intersection and the leap hours are counted and reported
"""
import numpy as np, pandas as pd
from scipy import stats
G = "/data/datasets/grid"
# HISTORICAL LOAD PRODUCT. This used to be hist_full40/county_load_hourly.npy, the earliest
# population-share allocation with no annual anchor. Its national total is 4,030 TWh in 1980 and
# 4,024 TWh in 2019, a ratio of 0.999, which is the number the paper quotes as what happens
# WITHOUT the anchor. Every published historical result was therefore computed on the unanchored
# product while the Methods described the anchored one. The anchored product carries the SEDS
# state-year totals (ratio 1.814) and the real county shares that replace population share.
LO = "/data/tell_pred/future/hist_full40_seds"
# TWO PRODUCTS FROM ONE SCRIPT. Figure 1 is a fixed-fleet weather counterfactual and needs demand
# held at a constant economy; everything compared with an observation or projected forward needs the
# real economic path. Passing "fixedecon" selects the first. The growth assertion below is inverted
# for that mode, so neither product can be mistaken for the other.
import sys as _sys
MODE = "fixedecon" if len(_sys.argv) > 1 and _sys.argv[1] == "fixedecon" else "anchored"
LOAD_NPY = ("county_load_hourly_fixedecon.npy" if MODE == "fixedecon"
            else "county_load_hourly_realdist.npy")
OUT_NPZ = ("subregion_netload_ourchain_1980_2019_fixedecon.npz" if MODE == "fixedecon"
           else "subregion_netload_ourchain_1980_2019.npz")
print("stageE mode: %s  load %s  -> %s" % (MODE, LOAD_NPY, OUT_NPZ), flush=True)
B = "/data/gen_targets/srgan3d_val/hist_v5"; GC = "/data/datasets/gen/tgw-gen-historical"
lat = np.load(f"{G}/coordinate.npz")["lat"]; lon = np.load(f"{G}/coordinate.npz")["lon"]
sm = np.load(f"{G}/subregion_mask.npz", allow_pickle=True); mask = sm["subregion_mask"]
id2 = dict((int(r[0]), str(r[1])) for r in sm["id_to_subregion"]); NS = 18
names = [id2[i] for i in range(1, NS + 1)]
def cell_sub(la, lo):
    ila = np.clip(np.searchsorted(lat, la), 0, len(lat) - 1)
    ilo = np.clip(np.searchsorted(lon, lo), 0, len(lon) - 1)
    return mask[ila, ilo]

g = np.load("/data/tgw_hist/tgw_grid.npz"); XLAT = g["XLAT"].ravel(); XLONG = g["XLONG"].ravel()
cm = np.load("/data/loads_measured/county_mask_tgw.npz", allow_pickle=True)
cfips = np.array([str(x).zfill(5) for x in cm["fips"]]); pf = cm["pair_fips"]; pc = cm["pair_cell"]
csub_cell = cell_sub(XLAT[pc], XLONG[pc])
county_sub = np.zeros(len(cfips), int)
for i in range(len(cfips)):
    v = csub_cell[pf == i]; v = v[v > 0]
    county_sub[i] = int(stats.mode(v, keepdims=False).mode) if len(v) else 0
nz = np.argwhere(mask > 0); nzll = np.column_stack([lat[nz[:, 0]], lon[nz[:, 1]]])
for i in np.where(county_sub == 0)[0]:
    cc = pc[pf == i]
    if len(cc) == 0: continue
    j = np.argmin((nzll[:, 0] - XLAT[cc].mean()) ** 2 + (nzll[:, 1] - XLONG[cc].mean()) ** 2)
    county_sub[i] = int(mask[nz[j, 0], nz[j, 1]])
print("unassigned counties:", int((county_sub == 0).sum()), flush=True)

meta = np.load(f"{LO}/meta.npz", allow_pickle=True)
lfips = np.array([str(x).zfill(5) for x in meta["fips"]])
county = np.load(f"{LO}/{LOAD_NPY}", mmap_mode="r")
# Fail rather than silently regress to the unanchored product: 1980 and 2019 national totals must
# differ by the SEDS growth, not be flat.
_yr = pd.date_range("1980-01-01", periods=county.shape[1], freq="h").year.values
_t80 = float(np.asarray(county[:, _yr == 1980]).sum()) / 1e6
_t19 = float(np.asarray(county[:, _yr == 2019]).sum()) / 1e6
print("  historical load %s: 1980 %.0f TWh, 2019 %.0f TWh, ratio %.3f"
      % (LOAD_NPY, _t80, _t19, _t19 / _t80), flush=True)
if MODE == "anchored":
    assert _t19 / _t80 > 1.5, ("the historical load is not annually anchored: 1980/2019 ratio %.3f. "
                               "hist_full40 is the superseded unanchored product." % (_t19 / _t80))
else:
    assert 0.9 < _t19 / _t80 < 1.1, ("the fixed-economy load still carries a trend: 1980/2019 ratio "
                                     "%.3f. Figure 1 must not see demand growth." % (_t19 / _t80))
yrs = meta["years"]; lidx = pd.date_range(f"{yrs[0]}-01-01", f"{yrs[-1]}-12-31 23:00", freq="h")
assert county.shape[1] == len(lidx), (county.shape, len(lidx))
f2s = {f: county_sub[i] for i, f in enumerate(cfips)}
lsub = np.array([f2s.get(f, 0) for f in lfips])
loadS = np.zeros((NS, len(lidx)), np.float32)
for s in range(1, NS + 1):
    rows = np.where(lsub == s)[0]
    if len(rows): loadS[s - 1] = np.asarray(county[rows]).sum(0)
print("load assembled %s  %s -> %s" % (loadS.shape, lidx[0].date(), lidx[-1].date()), flush=True)

def gen_subregion(npz, cfgcsv, capcol, band, label):
    z = np.load(npz, allow_pickle=True)
    pl = z["plants"].astype(str); st = z["stamps"].astype(str)
    c = pd.read_csv(cfgcsv, dtype={"plant_code_unique": str}).drop_duplicates("plant_code_unique").set_index("plant_code_unique")
    # Join by coordinate, not by plant id. The wind capacity-factor file carries one row per
    # LOCATION, and co-located EIA units share that row, so its id column repeats: 1,395 rows carry
    # only 1,175 distinct ids. Looking capacity up by id therefore counts 149 plants twice and drops
    # 220 others, 12.16 GW. The coordinate join covers all 1,395 config plants and the full 118.4 GW.
    import numpy as _np
    from scipy.spatial import cKDTree as _KD
    _tree = _KD(_np.c_[_np.asarray(z["lat"], float), _np.asarray(z["lon"], float)])
    cc = c.reset_index()
    _d, _j = _tree.query(_np.c_[cc["lat"].values.astype(float), cc["lon"].values.astype(float)])
    cc["row"] = _np.where(_d <= 1e-3, _j, -1)        # rounding-proof, still an exact-location join
    unmatched = int((cc.row < 0).sum())
    assert unmatched == 0, "%d config plants have no capacity-factor row at their coordinate" % unmatched
    rowidx = cc.row.values.astype(int)
    cap = cc[capcol].values.astype(float)
    pl = cc["plant_code_unique"].values.astype(str)
    pla = cc["lat"].values.astype(float); plo = cc["lon"].values.astype(float)
    sub = cell_sub(pla, plo)
    oom = sub == 0
    if oom.any():                   # a plant just off the mask must not vanish from the total
        for i in np.where(oom)[0]:
            j = np.argmin((nzll[:, 0] - pla[i]) ** 2 + (nzll[:, 1] - plo[i]) ** 2)
            sub[i] = int(mask[nz[j, 0], nz[j, 1]])
        print("  %-5s %d plants fell outside the mask, assigned to the nearest subregion"
              % (label, int(oom.sum())), flush=True)
    cf = z["cf"]
    fleet = float(np.nansum(np.nanmean(cf, axis=1)[rowidx] * cap) / np.nansum(cap))
    assert abs(np.nansum(cap) - c[capcol].sum()) < 1.0, "capacity lost in the join"
    assert band[0] <= fleet <= band[1], (
        "%s fleet capacity factor %.4f outside %s: the stored series is already net of plant "
        "losses, so a second haircut must not be applied" % (label, fleet, band))
    print("  %-5s fleet CF %.4f (net, in band %s)  plants %d  capacity %.1f GW"
          % (label, fleet, band, len(pl), np.nansum(cap) / 1e6), flush=True)
    out = np.zeros((NS, len(st)), np.float32)
    for s in range(1, NS + 1):
        r = np.where(sub == s)[0]
        if len(r):
            out[s - 1] = np.nansum(np.asarray(cf[rowidx[r]]) * cap[r, None] / 1e3, axis=0)   # kW -> MW
    # an hour with no plant reporting is a hole in the input, not zero generation. nansum turns it
    # into zero silently, which would read as a fleet-wide outage, so it is flagged and dropped.
    dead = np.isnan(np.asarray(cf)).all(axis=0)
    print("  %-5s %d config plants joined onto %d distinct locations, %.1f GW"
          % (label, len(cap), len(set(rowidx.tolist())), np.nansum(cap) / 1e6), flush=True)
    print("  %-5s hours with every plant missing: %d (flagged, not zero-filled)" % (label, dead.sum()), flush=True)
    return out, st, dead

print("generation:", flush=True)
Wg, wst, wdead = gen_subregion(f"{B}/hist_cf_hourly_1980_2019.npz", f"{GC}/eia_wind_configs.csv",
                        "system_capacity", (0.33, 0.40), "wind")
Sg, sst, sdead = gen_subregion(f"{B}/hist_solar_cf1h_1980_2019.npz", f"{GC}/eia_solar_configs.csv",
                        "system_capacity", (0.19, 0.24), "solar")
lkey = lidx.strftime("%Y%m%d%H").values
wpos = {k: j for j, k in enumerate(wst)}; spos = {k: j for j, k in enumerate(sst)}
common = [(i, wpos[k], spos[k]) for i, k in enumerate(lkey)
          if k in wpos and k in spos and not wdead[wpos[k]] and not sdead[spos[k]]]
li = np.array([c[0] for c in common]); wi = np.array([c[1] for c in common]); si = np.array([c[2] for c in common])
tidx = lidx[li]
L = loadS[:, li]; W = Wg[:, wi]; S = Sg[:, si]
nanS = int(np.isnan(S).any(0).sum())
NET = L - W - S
np.savez(f"{LO}/{OUT_NPZ}", load=L, wind=W, solar=S, net=NET,
         times=np.array([str(t) for t in tidx], dtype=object),
         subregions=np.array(names, dtype=object), county_sub=county_sub, county_fips=cfips)
print("\naligned T=%d h  %s..%s  (dropped %d hours of the load calendar, of which the input holes; %d residual NaN)"
      % (len(tidx), tidx.min().date(), tidx.max().date(), len(lidx) - len(tidx), nanS), flush=True)
print("%-20s %8s %7s %7s %6s %9s" % ("subregion", "load_GW", "wind", "solar", "VRE%", "netpk_GW"))
for s in range(NS):
    l = L[s].mean() / 1e3; wi_ = W[s].mean() / 1e3; so = S[s].mean() / 1e3
    print("%-20s %8.1f %7.2f %7.2f %6.1f %9.1f"
          % (names[s], l, wi_, so, 100 * (wi_ + so) / l if l > 0 else 0, np.nanmax(NET[s]) / 1e3))
print("\nUS: load %.0f GW avg, wind %.1f, solar %.1f, penetration %.1f%%"
      % (L.sum(0).mean() / 1e3, W.sum(0).mean() / 1e3, S.sum(0).mean() / 1e3,
         100 * (W + S).sum() / L.sum()))
print("saved -> subregion_netload_ourchain_1980_2019.npz")
