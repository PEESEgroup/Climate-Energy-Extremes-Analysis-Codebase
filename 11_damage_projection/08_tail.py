"""How the SEVERITY tail moves, as a continuum in the threshold, in all four realizations.

The earlier panel reported the change at three hand-picked thresholds (34, 50, 64 kt). The choice
of threshold is doing work there, so it is drawn as a curve in the threshold instead: at every wind
speed, the change in the number of county-days per year that reach at least that speed.
"""
import numpy as np, pandas as pd

F = pd.read_parquet("/data/scratch_r5/tc_flags.parquet")
F["date"] = pd.to_datetime(F.date)
NY = F[F.scen == "historical"].date.dt.year
NY = NY.max() - NY.min() + 1
SC = ["rcp45cooler", "rcp85cooler", "rcp45hotter", "rcp85hotter"]
DT = {"rcp45cooler": 1.472, "rcp85cooler": 1.743, "rcp45hotter": 2.084, "rcp85hotter": 2.530}
TH = np.arange(34, 101, 1.0)
h = F[F.scen == "historical"].wind_kt.values
H = np.array([(h >= t).sum() for t in TH]) / NY
print("years %d ; historical county-days per year: 34kt %.1f  50kt %.1f  64kt %.1f  83kt %.2f  "
      "100kt %.2f" % (NY, H[0], H[16], H[30], H[49], H[-1]))
out = {}
for s in SC:
    f = F[F.scen == s].wind_kt.values
    Fc = np.array([(f >= t).sum() for t in TH]) / NY
    out[s] = 100 * (Fc / H - 1)
    print("%-12s dT2 %.2f : %+6.1f%% at 34, %+6.1f%% at 50, %+6.1f%% at 64, %+6.1f%% at 83, "
          "%+6.1f%% at 100" % (s, DT[s], out[s][0], out[s][16], out[s][30], out[s][49], out[s][-1]))
D = pd.DataFrame(out, index=TH)
D.index.name = "kt"
D["hist_per_yr"] = H
D.to_csv("/data/scratch_r5/tail_by_threshold.csv")
print("\nmonotone in the threshold?  " + "  ".join(
    "%s %s" % (s, "yes" if np.all(np.diff(np.convolve(out[s], np.ones(9) / 9, "valid")) > -1e-9)
               else "no") for s in SC))
print("wrote tail_by_threshold.csv")
