"""
The two things every version of Figure 4 so far has thrown away: the 21 years, and the day.

Everything published from this ensemble is a 21-year aggregate - the mean of annual maxima, the
share of hours above a threshold - so the simulation's own time axis is invisible, and so is the
diurnal shape that the VRE fleet is actually changing. This script extracts both, once, for all 24
realisations plus the historical baseline:

  per year   national annual peak net load and annual mean, 2030-2050 (21 values per realisation)
  per hour   the mean diurnal profile of national net load, on the 3-hourly grid the futures live
             on, computed separately for summer (JJA) and winter (DJF) because the two seasons have
             opposite VRE signatures and averaging them hides both

The historical side uses `net_hydro` subsampled to the same 3-hourly phase, which is the series the
619.08 GW baseline is defined on.
"""
import glob, os, json
import numpy as np, pandas as pd

RN = "/data/cerf_out/r4_netload"
HP = "/data/tell_pred/future/hist_full40/subregion_netload_hydro_1980_2019.npz"
rows, prof = [], {}

for f in sorted(glob.glob(f"{RN}/netload_hydro_*.npz")):
    if ".bak" in f:
        continue
    z = np.load(f, allow_pickle=True)
    base = os.path.basename(f).replace("netload_hydro_", "").replace(".npz", "")
    zp = np.load(f"{RN}/netload_{base}.npz", allow_pickle=True)
    t = zp["times"].astype(str)
    yr = np.array([int(x[:4]) for x in t]); hh = np.array([int(x[8:10]) for x in t])
    mo = np.array([int(x[4:6]) for x in t])
    net = z["net_hydro_btm"].sum(0)
    variant = str(z["variant"]); scenario = str(z["scenario"])
    vlab = {"nopolicy": "NoPolicy", "ordonly": "Ordinances",
            "policy": "IRA", "obbba": "OBBBA"}[variant]
    for y in np.unique(yr):
        s = yr == y
        rows.append(dict(variant=variant, vlab=vlab, scenario=scenario,
                         climate=str(z["climate"]), ssp=scenario.split("_")[-1],
                         year=int(y), peak=float(net[s].max()), mean=float(net[s].mean())))
    for seas, mm in [("JJA", (6, 7, 8)), ("DJF", (12, 1, 2))]:
        s = np.isin(mo, mm)
        prof["%s|%s|%s" % (vlab, scenario, seas)] = [
            float(net[s & (hh == h)].mean()) for h in range(0, 24, 3)]

D = pd.DataFrame(rows)
D.to_csv(f"{RN}/r5_annual.csv", index=False)
print("annual rows %d  (%d realisations x %d years)"
      % (len(D), D.scenario.nunique() * D.vlab.nunique(), D.year.nunique()), flush=True)
print(D.groupby(["ssp", "vlab"]).peak.agg(["mean", "std", "min", "max"]).round(1).to_string())

zh = np.load(HP, allow_pickle=True)
th = pd.to_datetime(pd.Series(zh["times"].astype(str)), errors="coerce")
hn = zh["net_hydro"].sum(0)
ok = th.notna().values
th, hn = th[ok], hn[ok]
sel = (th.dt.hour.values % 3) == 0                     # same 3-hourly phase as the futures
th, hn = th[sel], hn[sel]
for seas, mm in [("JJA", (6, 7, 8)), ("DJF", (12, 1, 2))]:
    s = th.dt.month.isin(mm).values
    prof["HIST||%s" % seas] = [float(hn[s & (th.dt.hour.values == h)].mean())
                               for h in range(0, 24, 3)]
    print("historical %s diurnal (GW): %s"
          % (seas, [round(x / 1000, 1) for x in prof["HIST||%s" % seas]]), flush=True)

json.dump(prof, open(f"{RN}/r5_diurnal.json", "w"), indent=1)
print("wrote %s/r5_annual.csv and r5_diurnal.json" % RN, flush=True)

# how much does the annual peak move BETWEEN years, against how much it moves between scenarios?
w = D.groupby(["vlab", "scenario"]).peak.std().mean()
b = D.groupby(["vlab", "scenario"]).peak.mean().std()
print("\ninterannual sd of the annual peak, averaged over realisations: %.1f GW" % w)
print("between-realisation sd of the mean annual peak:                 %.1f GW" % b)
print("ratio interannual / between-scenario: %.2f" % (w / b))
