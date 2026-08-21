"""Every individual exceedance event, as an object with a duration and a depth.

The load-duration curve sorts the hours and therefore destroys the one thing an operator needs:
whether the hours above the historical peak arrive as isolated 3-hour spikes or as a day-long
siege. This extracts each contiguous run above the historical mean annual maximum (645.55 GW,
no hydro) and records how long it lasted and how far above the line it went.
"""
import glob, os
import numpy as np, pandas as pd
import sys as _sys
import os as _os_rp
for _rp in ("04_demand_model", "07_hazard_calendar", "09_outage_attribution",
            "02_downscale_wind", "12_figures"):
    _ap = _os_rp.path.abspath(_os_rp.path.join(
        _os_rp.path.dirname(_os_rp.path.abspath(__file__)), "..", _rp))
    if _os_rp.path.isdir(_ap) and _ap not in _sys.path:
        _sys.path.insert(0, _ap)
import paths as _PATHS   # the one name for the anchored net-load product
import baseline as _BASE  # the one definition of the historical reference period

RN = "/data/tell_pred/future/netload_ourchain"
HP = _PATHS.NETLOAD_NPZ
VLAB = {"nopolicy": "NoPolicy", "ordonly": "Ordinances", "policy": "IRA", "obbba": "OBBBA"}

zh = np.load(HP, allow_pickle=True)
th = pd.to_datetime(pd.Series(zh["times"].astype(str)), errors="coerce")
ok = th.notna().values
th = th[ok]; hn = zh["net"].sum(0)[ok]
# Same one-sided thinning as in 01_r4_ourchain.py, and here it was worse: `events` multiplied every
# run length by three, which is right for a three-hourly sample and wrong for the hourly future
# series, so every future duration was three times too long. Both arms are hourly now.
sel = np.ones(len(th), dtype=bool)
th3, hn3 = th[sel], hn[sel]
hy = th3.dt.year.values
# The threshold and the window it is counted over are the same reference period 01_r4_ourchain.py
# uses, so panel g and the rest of Figure 5 stand on one number. Taking the mean annual maximum
# over the whole forty years instead puts the line at 535.06 GW, 110 GW under the modern peak, and
# the historical arm then records 33 episodes a year: 0 in the 1980s and 810 in the 2010s, which
# measures the economy rather than the weather. baseline.py carries the reasoning.
BY = _BASE.base_years(hy)
bsel = np.isin(hy, BY)
THR = float(np.mean([hn3[hy == u].max() for u in BY]))
NY_HIST = len(BY)


def events(x, thr):
    m = x > thr
    d = np.diff(np.concatenate(([0], m.astype(np.int8), [0])))
    s = np.flatnonzero(d > 0); e = np.flatnonzero(d < 0)
    return [(int(b - a), float(x[a:b].max() - thr) / 1e3) for a, b in zip(s, e)]


# Every run in the group carries the denominator with it, including the runs that never exceed the
# line. The figure used to hold its own hardcoded run count, so a change in the ensemble would have
# rescaled the rate silently.
rows = []
NRUN = {"historical": 1}
for dur, dep in events(hn3[bsel], THR):
    rows.append(dict(grp="historical", ssp="hist", vlab="hist", scenario="historical",
                     dur_h=dur, depth_gw=dep, years=NY_HIST, n_runs=1))
for f in sorted(glob.glob(f"{RN}/netload_*.npz")):
    b = os.path.basename(f)
    if ".bak" in b or b.startswith("netload_hydro") or b.startswith("netload_hedc"):
        continue
    z = np.load(f, allow_pickle=True)
    v = str(z["variant"])
    if v not in VLAB:
        continue
    sc = str(z["scenario"])
    g_ = sc.split("_")[-1]
    NRUN[g_] = NRUN.get(g_, 0) + 1
    ny = len(np.unique([int(x[:4]) for x in z["times"].astype(str)]))
    for dur, dep in events(z["net"].sum(0), THR):
        rows.append(dict(grp=g_, ssp=g_, vlab=VLAB[v],
                         scenario=sc, dur_h=dur, depth_gw=dep, years=ny, n_runs=0))
E = pd.DataFrame(rows)
E["n_runs"] = E.grp.map(NRUN)
# a group whose every run stays under the line would vanish from the table and take its denominator
# with it, so the counts are written out separately as well
pd.DataFrame([dict(grp=g, n_runs=n,
                   years=int(E[E.grp == g].years.iloc[0]) if (E.grp == g).any() else 0)
              for g, n in NRUN.items()]).to_csv(f"{RN}/r5_event_runs_ourchain.csv", index=False)
E.to_csv(f"{RN}/r5_events_ourchain.csv", index=False)
print("threshold %.2f GW (mean annual max, %d-%d) ; events: %s"
      % (THR / 1e3, BY[0], BY[-1], E.grp.value_counts().to_dict()))
print("runs per group %s ; rate per run-year %s"
      % (NRUN, {g: round(len(e) / (float(e.years.iloc[0]) * float(e.n_runs.iloc[0])), 2)
                for g, e in E.groupby("grp")}))
print(E.groupby("grp")[["dur_h", "depth_gw"]].describe().round(2).to_string())
print("\nper realisation, events per year and the longest one:")
q = (E[E.grp != "historical"].groupby(["ssp", "vlab", "scenario"])
     .agg(n=("dur_h", "size"), med=("dur_h", "median"), mx=("dur_h", "max"),
          dmx=("depth_gw", "max")))
q["per_yr"] = q.n / 21
print(q.groupby(["ssp", "vlab"])[["per_yr", "med", "mx", "dmx"]].mean().round(1).to_string())
print("\nhistorical: %.2f/yr, median %.0f h, max %.0f h, deepest %.1f GW"
      % (len(E[E.grp == "historical"]) / NY_HIST,
         E[E.grp == "historical"].dur_h.median(), E[E.grp == "historical"].dur_h.max(),
         E[E.grp == "historical"].depth_gw.max()))
