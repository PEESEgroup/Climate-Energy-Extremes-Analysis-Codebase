"""County-level stress hours: the one thing in Figure 5 that is neither a peak nor a delta of one.

The map in that figure fills 18 subregions with the change in peak net load, which the matrix, the
trajectories, the violins and the policy panel all also carry. This computes something none of them
can: for every county, how many hours a year its own load sits above its OWN historical 99.9th
percentile. The threshold is per county, so a small rural county and Los Angeles are on the same
scale, and the historical value is 8.76 h/yr by construction — the future number is a multiplier on
that, directly comparable everywhere.

Load only. County-level VRE is not resolved to counties in this pipeline, so this is a demand-side
stress map and is labelled as one; the net-load statements stay at subregion level where the
generation actually is.
"""
import numpy as np, pandas as pd

# WHICH HISTORICAL PRODUCT, AND WHY NOT THE ANCHORED ONE. This reads hist_full40, the flat-economy
# county load, and it must keep reading it until the future county arrays are rebuilt. Both sides of
# this comparison have to carry the same county allocation and the same economy, and today only
# hist_full40 does: county by county its mean load and the futures' agree to r = 0.9999 and a ratio
# of 1.00 to 1.03, while against the SEDS-anchored product the same ratio runs 0.45 to 1.33. Its
# annual total is flat as well, 4,029 TWh in 1980 against 4,033 in 2019, which matches the futures'
# 4,017 to 4,107 TWh; the anchored product instead grows from 2,192 to 3,999 TWh. Substituting it
# makes the median county read 0.65 stress hours a year and puts 21 counties at 8,766, every hour of
# the year, which measures the mismatch in the county shares rather than the weather.
#
# The panel is therefore a FIXED-ECONOMY, CLIMATE-ONLY map and the figure says so. It is not the
# Figure 1 fixed-economy product (paths.COUNTY_FIXEDECON); it is this older flat product, which the
# future arm happens to share. Moving it under the X1 rename needs the future county arrays first.
BASE = "/data/tell_pred/future"
SC = ["rcp45cooler", "rcp85cooler", "rcp45hotter", "rcp85hotter"]
M = np.load(f"{BASE}/hist_full40/meta.npz", allow_pickle=True)
FIPS = M["fips"].astype(str)
NH_H = int(M["NH"]); YH = len(M["years"])
H = np.load(f"{BASE}/hist_full40/county_load_hourly.npy", mmap_mode="r")
FU = {s: np.load(f"{BASE}/{s}/county_load_hourly.npy", mmap_mode="r") for s in SC}
YF = {s: len(np.load(f"{BASE}/{s}/meta.npz", allow_pickle=True)["years"]) for s in SC}
print("counties %d  historical %d h over %d y  future %s" % (len(FIPS), NH_H, YH, YF), flush=True)

CH = 300
thr = np.empty(len(FIPS)); hrs = {s: np.empty(len(FIPS)) for s in SC}
hmean = np.empty(len(FIPS))
for i in range(0, len(FIPS), CH):
    h = np.asarray(H[i:i + CH], dtype=np.float32)
    t = np.percentile(h, 99.9, axis=1)
    thr[i:i + CH] = t
    hmean[i:i + CH] = h.mean(1)
    del h
    for s in SC:
        f = np.asarray(FU[s][i:i + CH], dtype=np.float32)
        hrs[s][i:i + CH] = (f > t[:, None]).sum(1) / YF[s]
        del f
    print("  %d/%d" % (min(i + CH, len(FIPS)), len(FIPS)), flush=True)

D = pd.DataFrame({"fips": FIPS, "hist_p999_mw": thr, "hist_mean_mw": hmean})
for s in SC:
    D["hrs_%s" % s] = hrs[s]
D["hrs_mean"] = D[["hrs_%s" % s for s in SC]].mean(1)
D["ratio"] = D.hrs_mean / (0.001 * NH_H / YH)          # historical is 8.76 h/yr by construction
D.to_csv("/data/cerf_out/r4_netload/county_stress_hours.csv", index=False)
print("\nhistorical by construction: %.2f h/yr" % (0.001 * NH_H / YH))
print(D[["hrs_%s" % s for s in SC] + ["hrs_mean", "ratio"]].describe().round(2).to_string())
print("\ntop 10 counties by ratio:")
print(D.nlargest(10, "ratio")[["fips", "hist_mean_mw", "hrs_mean", "ratio"]].round(2)
      .to_string(index=False))
print("\ncounties with ratio > 5: %d ; < 1: %d" % ((D.ratio > 5).sum(), (D.ratio < 1).sum()))
print("wrote county_stress_hours.csv")
