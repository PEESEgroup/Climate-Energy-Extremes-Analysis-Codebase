#!/usr/bin/env python
"""Build the HEDC demand-transformation increment on the TELL 3-hourly UTC timeline.

incr[sub,t] = magnitude-controlled Cambium (HighDemandGrowth - MidCase) enduse_load
increment, carrying the SHAPE (spatial concentration + winter heat-pump seasonality +
EV/DC diurnal) but with national annual magnitude scaled to a published-anchored target
to avoid any electrification-magnitude double-count with the ssp5 GCAM base.

Column: enduse_load (end-use electricity demand). High-Mid isolates the incremental
electrification+DC demand and excludes scenario-dependent storage charging / T&D losses.

TIMEZONE (audit R4-3 fix): the Cambium GEA hourly files are indexed by each region's
LOCAL STANDARD time -- row 0 = local Jan-1 00:00. Verified directly from each file's own
columns: `timestamp` is a single national Eastern-Time clock, `timestamp_local` is the
region's local clock (CAISO timestamp = local+3h=PT->ET; ERCOT = local+1h=CT->ET; ISONE
= local=ET), and file rows advance by local time. To land each region's diurnal shape at
the correct absolute UTC hour we therefore roll by THAT region's own UTC offset
(ET+5, CT+6, MT+7, PT+8), read authoritatively from each file's `tz` metadata field --
NOT a uniform +5 (which was correct only for Eastern regions and misaligned the 13
non-Eastern subregions by 1-3h). TELL netload timeline is UTC.
Cambium is a non-leap 8760h typical year; Feb-29 mapped to Feb-28.
"""
import numpy as np, pandas as pd, json

GEA = "/tmp/camb_peek/hourly_gea"
OUT = "/data/cerf_out/r4_netload"
YEARS_C = [2030, 2035, 2040, 2045, 2050]
SLOTS = [0, 3, 6, 9, 12, 15, 18, 21]          # 3-hourly UTC phase-0 slots
DEMCOL = "enduse_load"
# National added-demand target (net-new above ssp5), TWh/yr. The +750 by-2050 value is an
# AUTHOR-SYNTHESIZED central estimate within the published +300-900 TWh/yr range, NOT a single
# cited figure (audit F2): EIA AEO2026 data-center ~+440 TWh by 2050; LBNL 2024 data-center
# 325-580; EPRI 2024 data-center 380-790; plus NREL EFS electrification (net-new heat-pump/EV).
# Composite = DC plateau (~+400-500) + net-new electrification shape (~+250-350); 2030 is
# DC-dominated (~+150). Full-doc relabel is Phase 3; this comment is the code-level honesty fix.
TARGET = {2030: 150., 2035: 350., 2040: 500., 2045: 650., 2050: 750.}

# ---- subregion order + TELL timeline from a base netload file ----
z = np.load(f"{OUT}/netload_nopolicy_rcp45cooler_ssp5.npz", allow_pickle=True)
SUBS = list(z["subregions"]); TIMES = z["times"]
tt = pd.to_datetime([str(x) for x in TIMES], format="%Y%m%d%H")
NT = len(TIMES)
Tyear = tt.year.values; Tmon = tt.month.values; Tday = tt.day.values; Tslot = (tt.hour.values // 3).astype(int)
# calendar map (month,day)->non-leap doy(0..364); Feb-29 -> Feb-28
cal = pd.date_range("2001-01-01", "2001-12-31")
md = {(d.month, d.day): i for i, d in enumerate(cal)}
Tdoy = np.array([md.get((m, dd), md[(2, 28)]) for m, dd in zip(Tmon, Tday)])
NS = len(SUBS)

# ---- per-subregion UTC offset from each file's authoritative `tz` metadata (audit R4-3) ----
# roll = hours to ADD to the Cambium local-standard hourly index to reach UTC.
TZ2OFF = {"ET": 5, "CT": 6, "MT": 7, "PT": 8}
def read_tz(reg):
    with open(f"{GEA}/Cambium24_HighDemandGrowth_hourly_{reg}_2050.csv") as f:
        keys = f.readline().lstrip("﻿").rstrip("\n").split(",")
        vals = f.readline().rstrip("\n").split(",")
    return dict(zip(keys, vals))["tz"]
TZ = {reg: read_tz(reg) for reg in SUBS}
OFFSET = {reg: TZ2OFF[TZ[reg]] for reg in SUBS}
print("per-subregion tz -> UTC roll offset (audit R4-3 timezone fix):")
for reg in SUBS:
    print("  %-20s %-3s +%d" % (reg, TZ[reg], OFFSET[reg]))

def read_incr(reg, yr):
    h = pd.read_csv(f"{GEA}/Cambium24_HighDemandGrowth_hourly_{reg}_{yr}.csv", skiprows=5)
    m = pd.read_csv(f"{GEA}/Cambium24_MidCase_hourly_{reg}_{yr}.csv", skiprows=5)
    return (h[DEMCOL].values - m[DEMCOL].values).astype(np.float64)   # MWh/h LOCAL STD, len 8760

# ---- build Cambium nodes: (NS,365,8) UTC 3-hourly increment per Cambium year ----
nodes = {}; raw_nat_full = {}; node_nat_3h = {}
for yr in YEARS_C:
    C = np.zeros((NS, 365, 8), np.float64); natfull = 0.0
    for si, reg in enumerate(SUBS):
        inc = read_incr(reg, yr)                     # local-standard hourly
        natfull += inc.sum()                         # MWh full-year
        u = np.roll(inc, OFFSET[reg]).reshape(365, 24)  # local-standard->UTC (+per-region tz offset)
        C[si] = u[:, SLOTS]
    nodes[yr] = C
    raw_nat_full[yr] = natfull / 1e6                 # TWh (full 8760)
    node_nat_3h[yr] = C.sum() * 3.0 / 1e6            # TWh implied by 3-hourly samples

# ---- magnitude control: scale each node so added 3-hourly energy == TARGET ----
scale = {yr: TARGET[yr] / node_nat_3h[yr] for yr in YEARS_C}
nodes_s = {yr: nodes[yr] * scale[yr] for yr in YEARS_C}

# ---- interpolate scaled increment across Cambium 5-yr nodes onto TELL timeline ----
incr = np.zeros((NS, NT), np.float64)
for y in range(2030, 2051):
    mask = (Tyear == y)
    if not mask.any():
        continue
    lo = max(c for c in YEARS_C if c <= y); hi = min(c for c in YEARS_C if c >= y)
    w = 0.0 if lo == hi else (y - lo) / (hi - lo)
    vlo = nodes_s[lo][:, Tdoy[mask], Tslot[mask]]
    vhi = nodes_s[hi][:, Tdoy[mask], Tslot[mask]]
    incr[:, mask] = (1 - w) * vlo + w * vhi

neg_frac = float((incr < 0).mean())
# scaled national added TWh/yr on TELL timeline (mean-power*8760 basis)
scaled_nat = {}
for y in YEARS_C:
    m = (Tyear == y)
    scaled_nat[y] = float(incr[:, m].sum() * 3.0 / 1e6)

# ---- spatial + seasonal validation (RAW shape; scaling is uniform national) ----
raw2050 = nodes[2050]                                  # (NS,365,8) unscaled 3h
sub_twh = raw2050.sum(axis=(1, 2)) * 3.0 / 1e6         # per-sub 2050 raw TWh
sub_share = 100 * sub_twh / sub_twh.sum()
# seasonal: DJF vs JJA mean increment per sub (2050 raw)
doy_mon = np.array([cal[d].month for d in range(365)])
djf = np.isin(doy_mon, [12, 1, 2]); jja = np.isin(doy_mon, [6, 7, 8])
sub_djf = raw2050[:, djf, :].mean(axis=(1, 2))
sub_jja = raw2050[:, jja, :].mean(axis=(1, 2))
winter_ratio = sub_djf / np.where(sub_jja == 0, np.nan, sub_jja)
val = pd.DataFrame(dict(subregion=SUBS, raw2050_TWh=sub_twh, share_pct=sub_share,
                        DJF_MW=sub_djf, JJA_MW=sub_jja, winter_ratio=winter_ratio))
val = val.sort_values("share_pct", ascending=False)
val.to_csv(f"{OUT}/_hedc_spatial_seasonal.csv", index=False, float_format="%.3f")

np.savez(f"{OUT}/future_subregion_hedc_load.npz",
         incr=incr.astype(np.float32), times=TIMES, subregions=np.array(SUBS, object),
         target_twh=json.dumps(TARGET), raw_nat_full_twh=json.dumps(raw_nat_full),
         scaled_nat_twh=json.dumps(scaled_nat), scale=json.dumps(scale), demcol=DEMCOL,
         node_years=np.array(YEARS_C),
         tz=json.dumps(TZ), utc_offset=json.dumps(OFFSET))

print("=== HEDC increment build ===")
print("subs:", NS, "NT:", NT, "neg_frac: %.4f" % neg_frac)
print("RAW national added TWh/yr (full 8760):", {y: round(raw_nat_full[y], 1) for y in YEARS_C})
print("TARGET (published-anchored) TWh/yr   :", TARGET)
print("SCALED national added TWh/yr (TELL)  :", {y: round(scaled_nat[y], 1) for y in YEARS_C})
print("scale factors:", {y: round(scale[y], 3) for y in YEARS_C})
print("\nSPATIAL (top-8 share of 2050 increment):")
print(val.head(8)[["subregion", "raw2050_TWh", "share_pct", "winter_ratio"]].to_string(index=False))
print("\nSEASONAL winter_ratio (DJF/JJA) cold subs:")
cold = ["ISONE", "NYISO", "MISO_North", "NorthernGrid_East", "NorthernGrid_West"]
print(val[val.subregion.isin(cold)][["subregion", "winter_ratio", "DJF_MW", "JJA_MW"]].to_string(index=False))
print("DONE increment")
