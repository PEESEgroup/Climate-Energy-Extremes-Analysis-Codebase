"""
County-resolved tropical-cyclone wind exposure, 1980-2022, from HURDAT2.

A county is TC-exposed on day d if its centroid falls inside the 34-kt wind field of any
6-hourly track point on that day; the stored intensity is the maximum implied wind at the
centroid across the day's track points. Track points whose maximum wind is the -999 missing code
are dropped, and their count is printed; the subregion build (10_hazard_tc.py) drops them too.

CONSTANTS. The 34 kt threshold and the -999 missing-wind code are read from
07_hazard_calendar/hazard_defs.py (TC_WIND_KT, TC_MISSING_WIND). This file holds no private copy
of either. The subregion radius TC_RADIUS_KM is not used here: it is the subregion-scale rule, and
county exposure is the four-quadrant wind-radii rule instead.

STAMP. The output parquet is written through hazard_defs.write_flags, so it carries the name of
this script and the SHA-256 of the tropical-cyclone constants it used. A consumer that reads it
through hazard_defs.require_stamp will refuse a file written by a superseded builder. The stamp
also records the county-day denominator, which is every county in the domain on every day of
Y0 to Y1, not only the days a storm was present.

THE 2004 BREAK, STATED UP FRONT. HURDAT2 carries the four-quadrant 34/50/64-kt wind radii only
from 2004 onward. For 1980-2003 there are none. Rather than importing a literature formula, the
quadrant radii are predicted from a relation FIT ON OUR OWN 2004-2022 records:

    log R34_q  ~  a + b log(vmax - V0) + c |lat| + quadrant effects,   V0 = TC_WIND_KT - 1

and applied backwards. Everything downstream is reported separately for 2004-2022 (measured
radii) and 1980-2003 (predicted radii) so a reader can see whether the answer moves across the
break. If it does, only the post-2004 arm is usable.

Wind at the centroid is a modified-Rankine decay between the radius of maximum wind and R34,
which is the standard shape; no attempt is made to model asymmetry beyond the four quadrants,
and no terrain or surface-roughness correction is applied. This is a WIND-FIELD ENVELOPE, not a
damage model.
"""
import json, os, re
import sys as _sys
for _p in (os.path.dirname(os.path.abspath(__file__)),
           os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "07_hazard_calendar")):
    if os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
import hazard_defs as HD
import numpy as np, pandas as pd

OUT = "/data/enso/tc_county_ext"; os.makedirs(OUT, exist_ok=True)
H2 = "/data/equity_cost/analysis/did/hurdat2_latest.txt"
Y0, Y1 = 1980, 2022
V0 = HD.TC_WIND_KT - 1.0                         # the offset in log(vmax - V0); vmax = TC_WIND_KT
                                                 # gives log(1) = 0, so the fit is anchored at the
                                                 # threshold itself

# ------------------------------------------------------------------ parse HURDAT2
rows, hdr = [], None
for ln in open(H2, errors="ignore"):
    p = [x.strip() for x in ln.split(",")]
    if len(p) >= 3 and re.match(r"^AL\d{6}$", p[0]) and p[2].isdigit():
        hdr = (p[0], p[1], int(p[0][4:8]))
        continue
    if hdr is None or len(p) < 8:
        continue
    try:
        dt = pd.to_datetime(p[0] + " " + p[1][:2], format="%Y%m%d %H")
        la = float(p[4][:-1]) * (1 if p[4][-1] == "N" else -1)
        lo = float(p[5][:-1]) * (-1 if p[5][-1] == "W" else 1)
        vmax = float(p[6])
    except Exception:
        continue
    r34 = [np.nan] * 4
    if len(p) >= 12:
        try:
            v = [float(p[8 + i]) for i in range(4)]
            r34 = [np.nan if x < 0 else x for x in v]
        except Exception:
            pass
    rows.append((hdr[0], hdr[1], hdr[2], dt, la, lo, vmax, *r34))
T = pd.DataFrame(rows, columns=["sid", "name", "year", "dt", "lat", "lon", "vmax",
                                "r_ne", "r_se", "r_sw", "r_nw"])
miss = (T.vmax == HD.TC_MISSING_WIND) | (T.vmax < 0)   # HURDAT2 writes -999 for a missing wind
print("track points with missing wind (%d), dropped: %s"
      % (HD.TC_MISSING_WIND, format(int(miss.sum()), ",")), flush=True)
T = T[~miss]
T = T[(T.year >= Y0) & (T.year <= Y1) & (T.vmax >= HD.TC_WIND_KT)]
T = T[(T.lat > 10) & (T.lat < 55) & (T.lon > -110) & (T.lon < -50)]
print("track points >= %.0f kt in domain: %s   storms %d   years %d..%d"
      % (HD.TC_WIND_KT, format(len(T), ","), T.sid.nunique(), T.year.min(), T.year.max()), flush=True)
has = T[["r_ne", "r_se", "r_sw", "r_nw"]].notna().all(1)
print("   with measured %.0f-kt radii: %s (%.1f%%)   first year with radii: %d"
      % (HD.TC_WIND_KT, format(int(has.sum()), ","), 100 * has.mean(),
         int(T[has].year.min())), flush=True)

# ------------------------------------------------------------------ fit R34 on the measured era
FIT = T[has & (T.year >= 2004)].copy()
QN = ["r_ne", "r_se", "r_sw", "r_nw"]
long = FIT.melt(id_vars=["vmax", "lat"], value_vars=QN, var_name="q", value_name="r")
long = long[long.r > 0]
X = np.column_stack([np.ones(len(long)), np.log(long.vmax.values - V0),
                     np.abs(long.lat.values)] +
                    [(long.q.values == q).astype(float) for q in QN[1:]])
y = np.log(long.r.values)
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
pred = X @ beta
r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
print("   R34 model fit on %s quadrant-records 2004-%d:  R2 = %.3f   rmse(log) = %.3f"
      % (format(len(long), ","), Y1, r2, float(np.sqrt(((y - pred) ** 2).mean()))), flush=True)

def predict_r34(vmax, lat):
    n = len(vmax)
    out = np.empty((n, 4))
    for j, q in enumerate(QN):
        Xp = np.column_stack([np.ones(n), np.log(np.maximum(vmax - V0, 1.0)), np.abs(lat)] +
                             [np.full(n, 1.0 if q == qq else 0.0) for qq in QN[1:]])
        out[:, j] = np.exp(Xp @ beta)
    return out

R = T[QN].values.copy()
need = ~has.values
R[need] = predict_r34(T.vmax.values[need], T.lat.values[need])
T["radii_source"] = np.where(has.values, "measured", "predicted")

# ------------------------------------------------------------------ county centroids
gz = pd.read_csv("/data/equity_cost/analysis/did/2023_Gaz_counties_national.txt",
                 sep="\t", dtype={"GEOID": str})
gz.columns = [c.strip() for c in gz.columns]
gz["fips"] = gz.GEOID.str.zfill(5)
gz = gz[(gz.INTPTLONG > -110) & (gz.INTPTLONG < -50) & (gz.INTPTLAT > 20) & (gz.INTPTLAT < 55)]
clat, clon, cf = gz.INTPTLAT.values, gz.INTPTLONG.values, gz.fips.values
print("counties in the Atlantic/Gulf domain: %d" % len(cf), flush=True)

NM_PER_DEG = 60.0
RMW_NM = 25.0                                    # nominal radius of maximum wind

def haversine_nm(la1, lo1, la2, lo2):
    r = 3440.065
    p = np.pi / 180
    a = (np.sin((la2 - la1) * p / 2) ** 2 +
         np.cos(la1 * p) * np.cos(la2 * p) * np.sin((lo2 - lo1) * p / 2) ** 2)
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

recs = []
tl, to, tv = T.lat.values, T.lon.values, T.vmax.values
tdt = T.dt.values
tsid, tsrc = T.sid.values, T.radii_source.values
for i in range(len(T)):
    d = haversine_nm(tl[i], to[i], clat, clon)
    r34q = R[i]
    if not np.isfinite(r34q).any():
        continue
    rmax = np.nanmax(r34q)
    near = d <= rmax
    if not near.any():
        continue
    dy = clat[near] - tl[i]
    dx = (clon[near] - to[i]) * np.cos(np.deg2rad(tl[i]))
    qi = np.where(dy >= 0, np.where(dx >= 0, 0, 3), np.where(dx >= 0, 1, 2))   # NE SE SW NW
    rq = r34q[qi]
    dd = d[near]
    inside = dd <= rq
    if not inside.any():
        continue
    # modified-Rankine decay from vmax at RMW down to TC_WIND_KT at the quadrant radius
    ddi, rqi = dd[inside], rq[inside]
    w = np.where(ddi <= RMW_NM, tv[i],
                 HD.TC_WIND_KT + (tv[i] - HD.TC_WIND_KT)
                 * np.clip((rqi - ddi) / np.maximum(rqi - RMW_NM, 1e-6), 0, 1) ** 0.6)
    idx = np.where(near)[0][inside]
    recs.append(pd.DataFrame({"fips": cf[idx], "dt": tdt[i], "sid": tsid[i],
                              "wind_kt": w, "radii_source": tsrc[i]}))
    if i % 2000 == 0:
        print("   track point %d/%d" % (i, len(T)), flush=True)

X_ = pd.concat(recs, ignore_index=True)
X_["date"] = pd.to_datetime(X_.dt).dt.normalize()
G = (X_.groupby(["fips", "date"])
       .agg(wind_kt=("wind_kt", "max"), sid=("sid", "first"),
            radii_source=("radii_source", "first")).reset_index())
G["tc"] = 1

# The stamp travels inside the parquet, so a consumer can refuse a file from a superseded builder.
# The denominator is every county in the domain on every day of Y0 to Y1, not only the county-days
# a storm produced, because that is the universe the recorded county.tc day rate is a share of.
ND_SPAN = len(pd.date_range("%d-01-01" % Y0, "%d-12-31" % Y1))
HD.write_flags(G, f"{OUT}/county_tc_days.parquet", script=__file__,
               n_units=len(cf), n_dates=ND_SPAN, hazards=["tc"], hazard_col=None,
               extra={"wind_kt": HD.TC_WIND_KT, "rmw_nm": RMW_NM,
                      "geometry": "county centroid inside the four-quadrant 34 kt wind radii",
                      "years": [Y0, Y1], "counties_in_domain": int(len(cf)),
                      "radii_measured_from_year": int(T[has].year.min())})
_rate = len(G) / float(len(cf) * ND_SPAN)
_ok, _e, _t, _b = HD.day_rate_ok("county", "tc", _rate)
print("\ncounty TC-days (>= %.0f kt): %s   counties %d   storms %d   dates %d"
      % (HD.TC_WIND_KT, format(len(G), ","), G.fips.nunique(), G.sid.nunique(), G.date.nunique()))
print("   day rate %.4f of %d county x %d day cells; recorded band %.4f +/- %.4f [%s] -> %s"
      % (_rate, len(cf), ND_SPAN, _e, _t, _b.split(":")[0], "in band" if _ok else "OUT OF BAND"))
for era, sel in [("%d-2003 predicted radii" % Y0, G.date.dt.year < 2004),
                 ("2004-%d measured radii" % Y1, G.date.dt.year >= 2004)]:
    g = G[sel]
    print("   %-28s county-days %8s   counties %4d   mean wind %.1f kt"
          % (era, format(len(g), ","), g.fips.nunique(), g.wind_kt.mean()))

json.dump({"r34_model_r2": float(r2), "r34_beta": beta.tolist(),
           "n_county_tc_days": int(len(G)), "n_counties": int(G.fips.nunique()),
           "n_storms": int(G.sid.nunique()),
           "measured_radii_from_year": int(T[has].year.min())},
          open(f"{OUT}/county_tc_build.json", "w"), indent=1)
print("wrote", OUT)
