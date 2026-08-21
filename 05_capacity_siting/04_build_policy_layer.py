"""
Build the STATE/COUNTY renewable-policy layer on the GRIDCERF Albers-1km grid.

Outputs (all under /data/policy_data/):
  county_fips_albers.tif   : (H,W) int32 county GEOID (5-digit FIPS) rasterized to the grid
  policy_arrays.npz        : wind_derate (H,W float32; 0..1 fraction of buildable wind cells
                             removed by local ordinance; 1.0 = full ban/moratorium),
                             state_fips (H,W int16), rps_fraction (H,W float32 per-state RPS pull)
  wind_policy_rules.csv    : per-county FIPS rule (state,county,derate,reason,binding_setback_m)
  rps_state_table.csv      : hand-coded state RPS/CES effective final fraction (documented)

Mechanism (documented in the report):
  - 'Prohibitions' ordinance -> county wind EXCLUSION, classified by classify_prohibition():
      * genuine whole-county ban       -> derate 1.0 (full exclusion)
      * zone-/buffer-/parcel-/use-      -> derate 0.5 (partial)   [only these are "partial"]
        specific or single-municipality
      * clearly-temporary moratorium    -> derate 0.5 ('temporary-moratorium'), NOT a permanent
        (fixed term / interim / emergency)  1.0 exclusion, because such moratoria expire well
                                            before the 2050 siting horizon (see NOTE below).
  - MUNICIPAL / township prohibitions (home-rule states: NY towns, MI/MN townships, ME towns):
    the Municipal-Level sheet is INGESTED, aggregated to county FIPS as a partial county derate
    (author-calibrated per-jurisdiction removal, capped), instead of being silently dropped.
    Municipal *setback* rows are NOT ingested (disclosed + counted in the run log).
  - Property-line / structure setbacks -> buildable-fraction derate, keyed to a
    property-line-equivalent sterilizing distance. The bin edges/values are AUTHOR-CALIBRATED
    heuristics, only qualitatively informed by Lopez et al. 2023 (NREL) which reports the most
    restrictive setback ordinances remove ~50-90% of developable wind land (the paper does NOT
    supply these specific thresholds).
  - State-level setback laws applied as a baseline to every county in that state.
  - RPS/CES: per-state effective final clean/renewable fraction -> NLC bonus at siting time.

NOTE (temporary-moratorium assumption, disclosed for rigor): moratoria with no explicit expiry
that read as durable (e.g. long-standing multi-year moratoria) are kept as full 1.0 bans; only
moratoria with clearly temporary/interim/emergency/fixed-term language are down-weighted to 0.5.
Treating any temporary moratorium as a permanent 2050 exclusion would OVERSTATE restriction; the
0.5 value is an author-calibrated midpoint retaining near-term restriction + renewal uncertainty.
"""
import os, numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.transform import Affine
from rasterio.features import rasterize
from rasterio.crs import CRS
import openpyxl

PD="/data/policy_data"
XLSX=f"{PD}/wind_ordinances_2025.xlsx"
CTY_SHP=f"{PD}/tiger_county/tl_2023_us_county.shp"

# ---- GRIDCERF Albers-1km grid (identical to 06_cerf_full.py) ----
ALBERS=("+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=37.5 +lon_0=-96 +x_0=0 +y_0=0 "
        "+datum=NAD83 +units=m +no_defs")
DST_CRS=CRS.from_proj4(ALBERS); RES=1000.0
LEFT,TOP=-2831615.2280,1690434.1707; W,H=5460,3229
DST_T=Affine(RES,0,LEFT,0,-RES,TOP)

# ------------------------------------------------------------------
# 1. setback -> meters helper
# Reference turbine used to convert height-multiplier setbacks to metres. These are an
# AUTHOR-CHOSEN representative modern utility turbine (not a per-project value): 115 m hub,
# 170 m rotor -> 200 m tip.
TIP=200.0; ROTOR=170.0; HUB=115.0
def to_m(val, units):
    try: v=float(val)
    except (TypeError,ValueError): return None
    if units is None: return None
    u=str(units).strip().lower()
    if u=="tip-height-multiplier": return v*TIP
    if u=="rotor-diameter-multiplier": return v*ROTOR
    if u=="hub-height-multiplier": return v*HUB
    if u=="feet": return v*0.3048
    if u=="meters": return v
    if u=="miles": return v*1609.34
    return None

# property-line-equivalent sterilizing distance -> buildable-fraction derate.
# AUTHOR-CALIBRATED heuristic step function (NOT literature values). The mapping is only
# qualitatively anchored to Lopez et al. 2023 (NREL), which finds the most restrictive setback
# ordinances remove ~50-90% of developable wind land; the specific bin edges (800/500/300/180/60 m)
# and derate values (0.70/0.50/0.32/0.18/0.08) are the author's monotonic interpolation of that
# range and are not taken from the paper.
def setback_derate(deff_m):
    if deff_m is None: return 0.0
    if deff_m>=800: return 0.70
    if deff_m>=500: return 0.50
    if deff_m>=300: return 0.32
    if deff_m>=180: return 0.18
    if deff_m>=60:  return 0.08
    return 0.0

# ------------------------------------------------------------------
# classify a 'Prohibitions' summary into full-ban / partial / temporary-moratorium.
# Guiding PRINCIPLE (fixes the old bug where every unmatched ban defaulted to 0.5 'partial',
# wrongly halving genuine whole-county bans): a Prohibition sterilizes the WHOLE developable
# county for utility-scale wind UNLESS it is explicitly (a) a clearly-temporary moratorium,
# (b) scoped to a single municipality, or (c) limited to specific zones / buffers / parcels /
# uses while leaving wind permitted elsewhere. Anything that is an unqualified prohibition of
# commercial/utility wind is therefore a FULL ban (derate 1.0), not a 0.5 partial.
import re as _re
def _norm(x):
    if x is None: return ""
    s=str(x).lower()
    for a,b in (("’","'"),("‘","'"),("“",'"'),("”",'"'),
                ("–","-"),("—","-"),("\xa0"," ")):
        s=s.replace(a,b)
    return _re.sub(r"\s+"," ",s).strip()

# (a) clearly-temporary moratorium: a legal SUSPENSION with fixed term / interim / emergency
def _is_temporary(s):
    if ("moratorium" in s) or ("interim ordinance" in s) or ("temporary prohibit" in s):
        for q in ("temporary","emergency","interim"," until ","one year","one-year",
                  "six month","6 month","12 month","12-month","pending","proposed",
                  "considering","period of","suspend"):
            if q in s: return True
    return False

# (b) whole-jurisdiction ban language (county-wide / entire unincorporated area / blanket use ban)
_COUNTYWIDE=("all zoning district","all zone district","in all zones","in all districts",
             "prohibited in all zoning","not permitted in any zoning","not permitted in any district",
             "entire county","throughout the county","unincorporated area of","unincorporated areas of",
             "unincorporated portion of","no property shall be used","specifically prohibited as a use",
             "not allow commercial wecs","ban on commercial wind","prohibited use within all zoning",
             "prohibited use in all zoning","designating all")
def _countywide_ban(s):
    if any(k in s for k in _COUNTYWIDE): return True
    # blanket "no person/entity shall construct ..." unless scoped to a single named city/town
    if ("no entity or person shall construct" in s) or ("no person shall construct" in s):
        return not any(m in s for m in ("city of","town of","village of","borough of"))
    return False

# (c1) single-municipality scope (only that town/township, NOT the county) -> partial
def _muni_scope_only(s):
    if not any(m in s for m in ("township","city of","town of","towns of","village of","borough of")):
        return False
    if ("unincorporated" in s) and ("county" in s): return False   # ...also bans the whole county
    return True

# ------------------------------------------------------------------
# REG-FIX (2026-07-23): the broad substrings in _COUNTYWIDE ("in all zoning district(s)",
# "in all districts") short-circuited to full=1.0 BEFORE any permissive / carve-out / municipal
# test ran, wrongly full-excluding: (a) PERMISSIVE ordinances that PERMIT wind "in all zoning
# districts except <one overlay>" (SIGN INVERSION, e.g. 55021 Columbia WI); (b) carve-out bans
# "prohibited in all districts EXCEPT <a permitted zone / special use>" (e.g. 17197 Will IL,
# 18085 Kosciusko IN); (c) single-township bans "prohibited in all zoning districts within X
# Township" (e.g. 26039 Crawford MI).  These three guards run BEFORE _countywide_ban below.

# (g1) PERMISSIVE: wind AFFIRMATIVELY permitted/allowed as a PRINCIPAL use (a sign-inverted
# 'Prohibitions' row where wind is actually allowed county-wide, e.g. 55021 Columbia WI).  We
# require "principal" (NOT merely "accessory"): permitting only *accessory* (small on-site) systems
# while banning the utility/principal use is NOT permissive for utility-scale siting (e.g. Garrett
# MD, Orchard Park NY ban commercial solar but permit accessory rooftop).  Guard negations too
# (e.g. Wabaunsee KS: wind may NOT be "permitted as a Conditional Use").
_PERMIT_PRIN=_re.compile(r"(?:permitted|allowed)\s+(?:use\s+)?as\s+(?:a\s+|an\s+)?principal")
def _is_permissive(s):
    m=_PERMIT_PRIN.search(s)
    if not m: return False
    pre=s[max(0,m.start()-30):m.start()]
    if _re.search(r"\bnot\b|\bno\b|prohibit|denied|shall not|may not|cannot", pre): return False
    return True

# (g2) CARVE-OUT ban: a prohibition that explicitly leaves a PERMITTED zone / special-use /
# special-exception path -> the whole county is NOT sterilized -> PARTIAL, not full.
def _is_carveout_ban(s):
    if not any(b in s for b in ("prohibit","not permitted","not allow","shall not")):
        return False
    if ("special use" in s) or ("special exception" in s): return True
    if _re.search(r"permitted\s+only\s+(?:after|in|with|as|upon)", s): return True  # "permitted only after special exception"
    if ("however" in s) and (("permitted" in s) or ("allowed" in s)): return True
    if ("in certain district" in s) and (("permitted" in s) or ("allowed" in s)): return True
    # bare "except <zone>" carve-out, but NOT "except as noted (no exceptions)" style non-carve-outs
    if ("except " in s) and not any(n in s for n in ("no exception","without exception","except as noted")):
        return True
    return False

# (c2) zone / buffer / cap / parcel / use-specific partial signals
_PARTIAL_KEYS=("zoning district","zone district"," zones"," zone,","district)","overlay","floodplain",
               "floodway","flood plain","shoreland","shore land","coastal zone","scenic","williamson act",
               "farmland security","efu zone","residential zone","residential district","recreational",
               "conservation","special use","special exception","conditional use","permitted only in",
               "permitted as","other districts allow","capped at","land use categories","exclusion zone",
               "wildlife","historic","not a permitted use","type 3","tier 3",
               "siting approval","prior siting")  # IL PA 102-1123 permitting regime = permitted w/ approval
_BUFFER=_re.compile(r"within\s+[\d,\.]+\s*(?:feet|foot|ft|mile|miles|meter|meters)\b")
_CAP=_re.compile(r"(?:exceed|more than|no more than|cap(?:ped)?(?:\s+at)?)\s+[\d,\.]+\s*"
                 r"(?:megawatt|mw|turbine|wecs|wind)")

def classify_prohibition(summary, scope="county"):
    """Return (label, derate): 'full'/1.0, 'partial'/0.5, 'temporary'/0.5, or 'none'/0.0.
    scope='county' applies the single-municipality-scope downgrade (a county-level row scoped to
    one township should NOT sterilize the whole county); scope='municipal' skips it because a
    Municipal-Level row is ALREADY one municipality -> 'full' is its intended per-town calibration."""
    s=_norm(summary)
    if not s: return ("partial",0.5)                 # no text -> conservative partial (logged)
    if _is_temporary(s):     return ("temporary",0.5)# time-limited; expires before 2050 horizon
    # --- REG-FIX guards: MUST precede _countywide_ban so broad "in all districts" substrings do
    #     not sign-invert permissive ordinances nor promote carve-out / single-township bans to full.
    if _is_permissive(s):    return ("none",0.0)     # wind PERMITTED as a principal use in the county
                                                     # (a trailing "except <one overlay>" is a minor
                                                     # zone carve-out, NOT a county-wide ban) -> ~0
    if _is_carveout_ban(s):  return ("partial",0.5)  # prohibited EXCEPT a permitted zone/special-use
    if scope=="county":
        # county-level row scoped to a single town/township must NOT sterilize the whole county:
        if _muni_scope_only(s): return ("partial",0.5)
        if _countywide_ban(s):  return ("full",1.0)  # explicit whole-county ban wins
    else:
        # municipal row: preserve ORIG order (countywide before muni-scope) so a genuine full town
        # ban keeps its per-town 'full' calibration; only permissive/carve-out (above) are corrected.
        if _countywide_ban(s):  return ("full",1.0)
        if _muni_scope_only(s): return ("partial",0.5)
    if any(k in s for k in _PARTIAL_KEYS) or _BUFFER.search(s) or _CAP.search(s):
        return ("partial",0.5)                       # zone / buffer / cap / parcel / use-specific
    return ("full",1.0)                              # unqualified commercial/utility-wind ban

# ------------------------------------------------------------------
# 2. parse ordinance workbook
wb=openpyxl.load_workbook(XLSX, read_only=True, data_only=True)

def sheet_rows(name, header_row_is_first=True):
    ws=wb[name]; it=ws.iter_rows(values_only=True)
    hdr=next(it)
    if hdr[0] is not None and "generative AI" in str(hdr[0]):  # disclaimer row -> real header next
        hdr=next(it)
    idx={h:i for i,h in enumerate(hdr)}
    return idx, list(it)

# --- state-level baseline (setbacks apply to all counties in the state) ---
sidx, srows = sheet_rows('State-Level')
state_pl={}; state_st={}   # state -> best property-line / structure setback (m)
for r in srows:
    st=str(r[sidx['State']]).strip() if r[sidx['State']] else None
    if not st: continue
    f=r[sidx['Feature']]; d=to_m(r[sidx['Value']], r[sidx['Units']])
    if d is None: continue
    if f=='Property Line (Non-Participating)': state_pl[st]=max(state_pl.get(st,0),d)
    if f=='Structures (Non-Participating)':    state_st[st]=max(state_st.get(st,0),d)

# --- county-level ordinances ---
cidx, crows = sheet_rows('County-Level')
FIPSCOL='County Subdivision FIPS Code'
county={}   # fips -> dict(state,county,pl,st,prohib,reasons)
for r in crows:
    fips=r[cidx[FIPSCOL]]
    if fips is None: continue
    try: fips=int(fips)
    except (TypeError,ValueError): continue
    if fips<1000: continue
    d=county.setdefault(fips, dict(state=str(r[cidx['State']]).strip() if r[cidx['State']] else '',
                                   county=str(r[cidx['County']]).strip() if r[cidx['County']] else '',
                                   pl=0.0, st=0.0, prohib=0.0, reasons=set()))
    f=r[cidx['Feature']]; m=to_m(r[cidx['Value']], r[cidx['Units']])
    if f=='Property Line (Non-Participating)' and m: d['pl']=max(d['pl'],m); d['reasons'].add('propline-setback')
    elif f=='Structures (Non-Participating)' and m: d['st']=max(d['st'],m); d['reasons'].add('structure-setback')
    elif f=='Prohibitions':
        lab,pr=classify_prohibition(r[cidx['Summary']]); d['prohib']=max(d['prohib'],pr)
        d['reasons'].add({'full':'ban','partial':'partial-ban','temporary':'temporary-moratorium',
                          'none':'permitted-use'}[lab])

# ------------------------------------------------------------------
# 2b. MUNICIPAL / township ordinances (previously dropped silently).  Home-rule states let
# towns/townships zone independently, so dropping them under-restricts NY/MI/MN/ME etc.  We
# INGEST municipal *prohibitions*, classify each with the same rule, and aggregate to the county
# FIPS (first 5 digits of the 10-digit County-Subdivision FIPS) as a partial county-level derate.
# A single municipality covers only part of a county, so we do NOT promote it to a full county
# ban; instead each banning municipality removes an AUTHOR-CALIBRATED slice of county wind land.
# LIMITATION (disclosed): a precise treatment needs county-subdivision (cousub) geometry to
# rasterize each municipality; that geometry is not loaded here, so this is a county-aggregate
# approximation.  Municipal SETBACK rows are NOT ingested (counted + disclosed below).
MUNI_UNIT_FULL=0.10   # AUTHOR-CALIBRATED: each municipality with a full wind ban removes ~10% of
                      #   county buildable wind land (rural wind land mostly lies outside towns).
MUNI_UNIT_PARTIAL=0.04# AUTHOR-CALIBRATED: a partial/temporary municipal restriction removes ~4%.
MUNI_CAP=0.60         # AUTHOR-CALIBRATED: cap county-aggregate municipal removal at 60% (many
                      #   townships banning still leaves some unincorporated / permitted land).
muni_derate={}        # county_fips -> aggregated partial derate from municipal prohibitions
muni_stats=dict(rows_total=0, proh_rows=0, proh_full=0, proh_partial=0, proh_temp=0,
                counties=0, unmapped=0, setback_rows_dropped=0, dropped_total_nonproh=0)
try:
    midx, mrows = sheet_rows('Municipal-Level')
    _muni_full={}; _muni_part={}
    for r in mrows:
        if not any(x is not None for x in r): continue
        muni_stats['rows_total']+=1
        feat=r[midx.get('Feature')] if 'Feature' in midx else None
        fp=r[midx.get(FIPSCOL)] if FIPSCOL in midx else None
        try: cty=int(str(int(fp)).zfill(10)[:5])
        except (TypeError,ValueError): cty=None
        if feat=='Prohibitions':
            muni_stats['proh_rows']+=1
            if cty is None: muni_stats['unmapped']+=1; continue
            lab,_=classify_prohibition(r[midx['Summary']], scope="municipal")
            if lab=='none':   continue                       # permissive muni ordinance -> no derate
            if lab=='full':   muni_stats['proh_full']+=1;   _muni_full[cty]=_muni_full.get(cty,0)+1
            elif lab=='temporary': muni_stats['proh_temp']+=1; _muni_part[cty]=_muni_part.get(cty,0)+1
            else:             muni_stats['proh_partial']+=1;_muni_part[cty]=_muni_part.get(cty,0)+1
        elif feat in ('Property Line (Non-Participating)','Structures (Non-Participating)'):
            muni_stats['setback_rows_dropped']+=1
        else:
            muni_stats['dropped_total_nonproh']+=1
    for cty in set(list(_muni_full)+list(_muni_part)):
        dr=MUNI_UNIT_FULL*_muni_full.get(cty,0)+MUNI_UNIT_PARTIAL*_muni_part.get(cty,0)
        muni_derate[cty]=min(MUNI_CAP, dr)
    muni_stats['counties']=len(muni_derate)
except KeyError:
    print("[muni] Municipal-Level sheet not found; skipping municipal ingestion")
print(f"[muni] Municipal-Level rows scanned: {muni_stats['rows_total']:,} | "
      f"prohibition rows: {muni_stats['proh_rows']} "
      f"(full {muni_stats['proh_full']}, partial {muni_stats['proh_partial']}, "
      f"temporary {muni_stats['proh_temp']}, unmapped {muni_stats['unmapped']})")
print(f"[muni] INGESTED municipal bans into {muni_stats['counties']} counties (county-aggregate "
      f"partial derate). NOT ingested/disclosed: {muni_stats['setback_rows_dropped']} municipal "
      f"setback rows + {muni_stats['dropped_total_nonproh']:,} other municipal-feature rows.")

def _apply_muni(cty_fips, base_derate):
    """Combine a county's base derate with its municipal-aggregate derate as independent removals."""
    md=muni_derate.get(int(cty_fips),0.0)
    if md<=0: return base_derate, False
    return min(1.0, 1.0-(1.0-base_derate)*(1.0-md)), True

# fold state baseline into every county of that state (so all counties see state law)
# build state -> list of its county fips from the TIGER file later; here store state setbacks
# ------------------------------------------------------------------
# 3. rasterize TIGER counties to the Albers grid
print("[rasterize] reading TIGER counties ...")
gdf=gpd.read_file(CTY_SHP)[['GEOID','STATEFP','NAME','geometry']]
gdf=gdf[gdf['STATEFP'].astype(int)<=56]            # CONUS+ (drop territories >56)
gdf=gdf[~gdf['STATEFP'].isin(['02','15'])]         # drop AK, HI (outside grid)
gdf['fips']=gdf['GEOID'].astype(int)
gdf=gdf.to_crs(DST_CRS)
fips_raster=rasterize([(g,f) for g,f in zip(gdf.geometry,gdf['fips'])],
                      out_shape=(H,W), transform=DST_T, fill=0, dtype='int32', all_touched=False)
print("[rasterize] county cells:", int((fips_raster>0).sum()))

# state name -> STATEFP map for applying state baselines
STATE_NAME_FP={  # full state name -> 2-digit fp
 'Alabama':1,'Arizona':4,'Arkansas':5,'California':6,'Colorado':8,'Connecticut':9,'Delaware':10,
 'Florida':12,'Georgia':13,'Idaho':16,'Illinois':17,'Indiana':18,'Iowa':19,'Kansas':20,'Kentucky':21,
 'Louisiana':22,'Maine':23,'Maryland':24,'Massachusetts':25,'Michigan':26,'Minnesota':27,'Mississippi':28,
 'Missouri':29,'Montana':30,'Nebraska':31,'Nevada':32,'New Hampshire':33,'New Jersey':34,'New Mexico':35,
 'New York':36,'North Carolina':37,'North Dakota':38,'Ohio':39,'Oklahoma':40,'Oregon':41,'Pennsylvania':42,
 'Rhode Island':44,'South Carolina':45,'South Dakota':46,'Tennessee':47,'Texas':48,'Utah':49,'Vermont':50,
 'Virginia':51,'Washington':53,'West Virginia':54,'Wisconsin':55,'Wyoming':56}
FP_STATE={v:k for k,v in STATE_NAME_FP.items()}

# ------------------------------------------------------------------
# 4. build per-county derate table (state baseline + county rules + municipal aggregate)
# deff = property-line-equivalent binding distance. The 0.6 weight on structure/dwelling
# setbacks is an AUTHOR-CALIBRATED heuristic (not a literature value): a dwelling/structure
# setback sterilizes ~0.6 as much land as an equal property-line setback because dwellings are
# sparser than parcel boundaries, so the same distance reaches fewer buildable cells.
STRUCT_WEIGHT=0.6
rows=[]
muni_used=set()
for f, d in county.items():
    stname=d['state']
    pl=max(d['pl'], state_pl.get(stname,0.0))
    st=max(d['st'], state_st.get(stname,0.0))
    deff=max(pl, STRUCT_WEIGHT*st)
    sd=setback_derate(deff)
    base=min(1.0, max(d['prohib'], sd))
    derate, used = _apply_muni(f, base)
    if used: d['reasons'].add('muni-ban'); muni_used.add(int(f))
    reason='|'.join(sorted(d['reasons'])) or 'none'
    rows.append(dict(fips=f, state=stname, county=d['county'], derate=round(derate,3),
                     prohib=d['prohib'], setback_derate=round(sd,3),
                     binding_setback_m=round(deff,1), reason=reason))

# also add state-baseline-only counties (states with a state law but county not individually listed)
listed=set(county.keys())
statefp_with_law=set()
for stname in set(list(state_pl)+list(state_st)):
    fp=STATE_NAME_FP.get(stname)
    if fp: statefp_with_law.add(fp)
for _,g in gdf.iterrows():
    f=int(g['fips']); fp=int(g['STATEFP'])
    if f in listed: continue
    if fp in statefp_with_law:
        stname=FP_STATE.get(fp,'')
        pl=state_pl.get(stname,0.0); st=state_st.get(stname,0.0)
        deff=max(pl,STRUCT_WEIGHT*st); sd=setback_derate(deff)
        derate, used = _apply_muni(f, sd)
        if (sd>0) or used:
            reason='state-law-setback' + ('|muni-ban' if used else '')
            if used: muni_used.add(int(f))
            rows.append(dict(fips=f, state=stname, county=g['NAME'], derate=round(derate,3),
                             prohib=0.0, setback_derate=round(sd,3),
                             binding_setback_m=round(deff,1), reason=reason))

# municipal-only counties: have municipal bans but no county-level rule and no state-law setback
fp_to_name={int(g['fips']):(FP_STATE.get(int(g['STATEFP']),''), g['NAME']) for _,g in gdf.iterrows()}
for cty, md in muni_derate.items():
    if md<=0 or int(cty) in muni_used or int(cty) in listed: continue
    stname, cname = fp_to_name.get(int(cty), ('',''))
    rows.append(dict(fips=int(cty), state=stname, county=cname, derate=round(min(1.0,md),3),
                     prohib=0.0, setback_derate=0.0, binding_setback_m=0.0, reason='muni-ban'))

rules=pd.DataFrame(rows).sort_values('fips').reset_index(drop=True)
rules.to_csv(f"{PD}/wind_policy_rules.csv", index=False)

# ------------------------------------------------------------------
# 5. rasterize derate to grid + state raster
derate_map={int(r.fips): float(r.derate) for r in rules.itertuples()}
wind_derate=np.zeros((H,W), 'f4')
uniq=np.unique(fips_raster)
for f in uniq:
    if f<=0: continue
    dr=derate_map.get(int(f))
    if dr and dr>0: wind_derate[fips_raster==f]=dr
state_fips=(fips_raster//1000).astype('int16')     # 5-digit county fips -> 2-digit state

# ------------------------------------------------------------------
# 6. RPS/CES hand-coded table (effective final clean/renewable fraction near 2050)
#    Sources consulted: LBNL "U.S. State Renewables Portfolio & Clean Electricity Standards 2024
#    Status Update" + DSIRE + NCSL. The per-state fractions below are AUTHOR-COMPILED from those
#    trackers (the author's reading of each state's enforceable final target as a fraction of
#    retail sales; CES total where a 100%-clean law exists) rather than a single published table.
RPS={ 'California':1.00,'New York':1.00,'Washington':1.00,'Oregon':1.00,'New Mexico':1.00,
 'Colorado':0.80,'Virginia':1.00,'Nevada':0.50,'Illinois':1.00,'New Jersey':1.00,'Maryland':0.50,
 'Massachusetts':0.80,'Connecticut':1.00,'Rhode Island':1.00,'Michigan':1.00,'Minnesota':1.00,
 'Wisconsin':0.10,'Maine':1.00,'Vermont':0.75,'Pennsylvania':0.18,'North Carolina':0.60,
 'Delaware':0.40,'Arizona':0.15,'Montana':0.15,'Missouri':0.15,'Ohio':0.085,'New Hampshire':0.252,
 'Texas':0.0,'Iowa':0.0,'Kansas':0.0 }
rps_state_fp={STATE_NAME_FP[k]:v for k,v in RPS.items() if k in STATE_NAME_FP}
rps_fraction=np.zeros((H,W),'f4')
for fp,val in rps_state_fp.items():
    if val>0: rps_fraction[state_fips==fp]=val
pd.DataFrame([dict(state=k, rps_ces_final_fraction=v, statefp=STATE_NAME_FP.get(k)) for k,v in RPS.items()]
            ).to_csv(f"{PD}/rps_state_table.csv", index=False)

# ------------------------------------------------------------------
# 7. save
with rasterio.open(f"{PD}/county_fips_albers.tif","w",driver="GTiff",height=H,width=W,count=1,
                   dtype="int32",crs=DST_CRS,transform=DST_T,nodata=0,compress="lzw") as ds:
    ds.write(fips_raster,1)
np.savez_compressed(f"{PD}/policy_arrays.npz",
                    wind_derate=wind_derate, state_fips=state_fips, rps_fraction=rps_fraction)

# ------------------------------------------------------------------
# 8. report
n_ban=(rules.derate>=1.0).sum(); n_partial=((rules.derate>0)&(rules.derate<1.0)).sum()
n_temp=rules.reason.str.contains('temporary-moratorium').sum()
n_muni=rules.reason.str.contains('muni-ban').sum()
print("\n==== POLICY LAYER BUILT ====")
print(f"counties with any wind rule : {len(rules)}")
print(f"  full ban/exclude (derate=1.0): {n_ban}")
print(f"  partial derate (0<d<1)       : {n_partial}")
print(f"  incl. temporary-moratorium rows (down-weighted from permanent 1.0): {n_temp}")
print(f"  incl. counties with ingested municipal bans                       : {n_muni}")
print(f"grid cells fully excluded (wind): {(wind_derate>=1.0).sum():,}")
print(f"grid cells derated (0<d<1)      : {((wind_derate>0)&(wind_derate<1)).sum():,}")
print(f"RPS states with pull>0          : {(np.array(list(RPS.values()))>0).sum()}")
# audit VERIFY: the 5 named genuine full county bans must now be fully excluded (derate 1.0)
_chk=rules[rules.fips.isin([18059,18099,20139,20173,20197])][['fips','state','county','derate','reason']]
print("audit-named full bans (expect derate=1.0):")
print(_chk.to_string(index=False))
print("top-15 most-restrictive counties:")
print(rules.sort_values(['derate','binding_setback_m'],ascending=False)
          [['fips','state','county','derate','binding_setback_m','reason']].head(15).to_string(index=False))
print("\nDONE")
