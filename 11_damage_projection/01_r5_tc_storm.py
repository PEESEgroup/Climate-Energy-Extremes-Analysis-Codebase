"""
R5 TC arm, step 3 - STORM-AGGREGATE statistics.

Gate B-TC1 returned kappa = 0.465 and gate B-TC2 rho = 0.781, both short of the county-level bar,
so the pre-registered fallback applies: county-level maps are WITHDRAWN and the projection is
reported at storm aggregate and above. This script produces exactly that granularity - per storm,
how many counties it reaches, and at what wind - for the historical run and each future run, on
the same storms (the +40 y replay).
"""
import json
import numpy as np, pandas as pd

S = "/data/scratch_r5"
SCEN = ["rcp45cooler", "rcp45hotter", "rcp85cooler", "rcp85hotter"]
Y0, Y1 = 1990, 2010
NYR = Y1 - Y0 + 1
rng = np.random.default_rng(3)

W = pd.read_parquet("%s/tc_window_%d_%d.parquet" % (S, Y0, Y1))[["fips", "date", "sid"]]
F = pd.read_parquet("%s/tc_flags.parquet" % S)
F["date"] = pd.to_datetime(F.date)
F = F.merge(W, on=["fips", "date"], how="left")
print("flag rows %s   with storm id %.4f" % (format(len(F), ","), F.sid.notna().mean()), flush=True)


def storm_stats(g):
    """per storm: counties ever exposed, county-days, county-days >=50/64 kt, max county wind"""
    s = g.groupby("sid").agg(counties=("fips", "nunique"), county_days=("fips", "size"),
                             maxkt=("wind_kt", "max"), meankt=("wind_kt", "mean"))
    s["cd50"] = g[g.wind_kt >= 50].groupby("sid").size().reindex(s.index).fillna(0)
    s["cd64"] = g[g.wind_kt >= 64].groupby("sid").size().reindex(s.index).fillna(0)
    return s


H = storm_stats(F[F.scen == "historical"])
print("\nTGW-historical %d-%d: %d storms reach at least one county, %.1f storms/yr"
      % (Y0, Y1, len(H), len(H) / NYR), flush=True)
print("  per storm: counties %.1f   county-days %.1f   >=50kt county-days %.1f   >=64kt %.1f"
      % (H.counties.mean(), H.county_days.mean(), H.cd50.mean(), H.cd64.mean()), flush=True)

OUT = dict(hist=dict(n_storms=int(len(H)), storms_per_yr=float(len(H) / NYR),
                     counties_per_storm=float(H.counties.mean()),
                     county_days_per_storm=float(H.county_days.mean()),
                     cd50_per_storm=float(H.cd50.mean()), cd64_per_storm=float(H.cd64.mean()),
                     max_kt=float(H.maxkt.max())), scenarios={})
for sc in SCEN:
    G = storm_stats(F[F.scen == sc])
    common = H.index.intersection(G.index)
    # paired, same storms - bootstrap over storms
    d_c = (G.loc[common, "counties"] - H.loc[common, "counties"])
    d_64 = (G.loc[common, "cd64"] - H.loc[common, "cd64"])
    d_k = (G.loc[common, "maxkt"] - H.loc[common, "maxkt"])
    bs = np.array([d_c.sample(len(common), replace=True, random_state=int(r)).mean()
                   for r in rng.integers(0, 1 << 31, 1000)])
    OUT["scenarios"][sc] = dict(
        n_storms=int(len(G)), n_common=int(len(common)),
        counties_per_storm=float(G.counties.mean()),
        d_counties_per_storm=float(d_c.mean()),
        d_counties_lo=float(np.percentile(bs, 2.5)), d_counties_hi=float(np.percentile(bs, 97.5)),
        frac_storms_larger=float((d_c > 0).mean()), frac_storms_smaller=float((d_c < 0).mean()),
        d_cd64_per_storm=float(d_64.mean()),
        d_maxkt_per_storm=float(d_k.mean()),
        cd50_per_storm=float(G.cd50.mean()), cd64_per_storm=float(G.cd64.mean()),
        max_kt=float(G.maxkt.max()))
    o = OUT["scenarios"][sc]
    print("\n%-12s storms %d (common %d)   counties/storm %.1f -> %.1f  (%+.2f [%+.2f, %+.2f])"
          % (sc, len(G), len(common), H.loc[common, "counties"].mean(), G.loc[common, "counties"].mean(),
             o["d_counties_per_storm"], o["d_counties_lo"], o["d_counties_hi"]), flush=True)
    print("             storms that grow %.0f%% / shrink %.0f%% / unchanged %.0f%%   "
          "hurricane-force county-days per storm %.2f -> %.2f   peak county wind %.0f -> %.0f kt"
          % (100 * o["frac_storms_larger"], 100 * o["frac_storms_smaller"],
             100 * (1 - o["frac_storms_larger"] - o["frac_storms_smaller"]),
             H.cd64.mean(), G.cd64.mean(), H.maxkt.max(), G.maxkt.max()), flush=True)

json.dump(OUT, open("%s/tc_storm.json" % S, "w"), indent=1, default=float)
H.to_csv("%s/tc_storm_hist.csv" % S)
print("\nwrote %s/tc_storm.json" % S, flush=True)
