"""
Build the SOLAR policy layer, symmetric to the wind layer, and merge into policy_arrays.npz.

Adds three solar mechanisms:
  (1) SOLAR ORDINANCE exclusion/derate : county solar bans/moratoria + property-line/structure
      setbacks -> buildable-fraction derate (same logic as wind).  Source: NREL 2025 Solar
      Ordinance database (OpenEI 8602).
      Prohibitions are classified by classify_prohibition() (full 1.0 / partial 0.5 /
      temporary-moratorium 0.5), IDENTICAL to the wind builder, so genuine whole-county solar
      bans get full exclusion and only zone/buffer/parcel/use/single-municipality restrictions
      are partial.  Municipal-Level solar prohibitions are INGESTED (county-aggregate), and a
      STATE-baseline setback step is applied to unlisted counties (symmetric to the wind layer,
      fixing the prior solar<->wind asymmetry).

  AUTHOR-CALIBRATED heuristics in this file (labeled inline, NOT literature values): the
  setback_derate() bin edges/values, the 0.6 structure-setback weight, and the AG_STRONG/AG_MOD
  farmland-protection strengths.  The NCSL/DSIRE trackers identify WHICH states protect farmland
  but do not supply a percentage derate; the strengths below are the author's calibration.
  (2) PRIME-FARMLAND / AG-PROTECTION derate : in states with documented utility-scale-solar
      farmland-protection policies (NCSL Farmland & Solar tracker / DSIRE), prime-farmland cells
      (USDA gSSURGO prime-farmland layer from GRIDCERF) are derated for solar.  GRIDCERF's solar
      composite does NOT include prime farmland, so this is fully incremental (no double-count).
  (3) SOLAR RPS CARVE-OUT incentive : states with a solar-specific RPS carve-out (DSIRE) get an
      ADDITIONAL solar NLC bonus on top of the general RPS pull.

Outputs merged into /data/policy_data/policy_arrays.npz:
  solar_derate (H,W f4)  solar_rps_fraction (H,W f4)   (plus the existing wind arrays)
And solar_policy_rules.csv, solar_ag_states.csv, solar_carveout_table.csv.
"""
import os, numpy as np, pandas as pd, rasterio, openpyxl

PD="/data/policy_data"
XLSX=f"{PD}/solar_ordinances_2025.xlsx"
FIPS_TIF=f"{PD}/county_fips_albers.tif"
FARM_TIF=f"{PD}/gridcerf_usda_nrsc_prime_farmland_classification.tif"

fips=rasterio.open(FIPS_TIF).read(1)
prime=(rasterio.open(FARM_TIF).read(1)==1)          # True = USDA prime farmland
H,W=fips.shape
state_fips=(fips//1000).astype("int16")

# ---- solar turbine-agnostic setback -> meters (panels: use feet/meters/miles; multipliers rare) ----
def to_m(val,units):
    try: v=float(val)
    except (TypeError,ValueError): return None
    if units is None: return None
    u=str(units).strip().lower()
    if u=="feet": return v*0.3048
    if u=="meters": return v
    if u=="miles": return v*1609.34
    # AUTHOR-CHOSEN reference: height-multiplier setbacks are rare for solar; 30 m is a nominal
    # panel/inverter reference height so such rows still map to a finite distance.
    if u in ("tip-height-multiplier","hub-height-multiplier"): return v*30.0
    return None
# AUTHOR-CALIBRATED heuristic step function (NOT literature values): solar-specific bin edges
# (300/150/75/30 m) and derate values (0.55/0.35/0.20/0.08) chosen for utility-scale PV, which
# is more land-flexible than wind so the same distance sterilizes a smaller land fraction.
def setback_derate(d):
    if d is None: return 0.0
    if d>=300: return 0.55
    if d>=150: return 0.35
    if d>=75:  return 0.20
    if d>=30:  return 0.08
    return 0.0

# ------------------------------------------------------------------
# Prohibition classifier -- IDENTICAL logic to the wind builder (04_build_policy_layer.py).
# PRINCIPLE: a Prohibition sterilizes the WHOLE developable county for utility-scale solar UNLESS
# it is explicitly (a) a clearly-temporary moratorium, (b) scoped to a single municipality, or
# (c) limited to specific zones/buffers/parcels/uses.  This replaces the old keyword-whitelist
# that defaulted every unmatched ban to 0.5 'partial' (which wrongly halved genuine full bans).
import re as _re
def _norm(x):
    if x is None: return ""
    s=str(x).lower()
    for a,b in (("’","'"),("‘","'"),("“",'"'),("”",'"'),("–","-"),("—","-"),("\xa0"," ")):
        s=s.replace(a,b)
    return _re.sub(r"\s+"," ",s).strip()
def _is_temporary(s):
    if ("moratorium" in s) or ("interim ordinance" in s) or ("temporary prohibit" in s):
        for q in ("temporary","emergency","interim"," until ","one year","one-year",
                  "six month","6 month","12 month","12-month","pending","proposed",
                  "considering","period of","suspend"):
            if q in s: return True
    return False
_COUNTYWIDE=("all zoning district","all zone district","in all zones","in all districts",
             "prohibited in all zoning","not permitted in any zoning","not permitted in any district",
             "entire county","throughout the county","unincorporated area of","unincorporated areas of",
             "unincorporated portion of","no property shall be used","specifically prohibited as a use",
             "not allow commercial","ban on commercial","prohibited use within all zoning",
             "prohibited use in all zoning","designating all")
def _countywide_ban(s):
    if any(k in s for k in _COUNTYWIDE): return True
    if ("no entity or person shall construct" in s) or ("no person shall construct" in s):
        return not any(m in s for m in ("city of","town of","village of","borough of"))
    return False
def _muni_scope_only(s):
    if not any(m in s for m in ("township","city of","town of","towns of","village of","borough of")):
        return False
    if ("unincorporated" in s) and ("county" in s): return False
    return True
# --- REG-FIX (2026-07-23): same guards as the wind builder.  The broad _COUNTYWIDE substrings
# ("in all zoning district(s)","in all districts") short-circuited to full=1.0 BEFORE any
# permissive / carve-out / municipal test, wrongly full-excluding permissive ordinances
# (sign-inversion), carve-out bans, and single-township bans.  These guards run BEFORE
# _countywide_ban in classify_prohibition() below.
# PERMISSIVE requires "principal" (NOT merely "accessory"): permitting only accessory/rooftop solar
# while banning the commercial/utility use is NOT permissive for utility-scale siting (e.g. Garrett MD,
# Columbiana OH, Orchard Park NY ban commercial/ground-mounted solar but permit accessory rooftop).
_PERMIT_PRIN=_re.compile(r"(?:permitted|allowed)\s+(?:use\s+)?as\s+(?:a\s+|an\s+)?principal")
def _is_permissive(s):
    m=_PERMIT_PRIN.search(s)
    if not m: return False
    pre=s[max(0,m.start()-30):m.start()]
    if _re.search(r"\bnot\b|\bno\b|prohibit|denied|shall not|may not|cannot", pre): return False
    return True
def _is_carveout_ban(s):
    if not any(b in s for b in ("prohibit","not permitted","not allow","shall not")):
        return False
    if ("special use" in s) or ("special exception" in s): return True
    if _re.search(r"permitted\s+only\s+(?:after|in|with|as|upon)", s): return True
    if ("however" in s) and (("permitted" in s) or ("allowed" in s)): return True
    if ("in certain district" in s) and (("permitted" in s) or ("allowed" in s)): return True
    if ("except " in s) and not any(n in s for n in ("no exception","without exception","except as noted")):
        return True
    return False
_PARTIAL_KEYS=("zoning district","zone district"," zones"," zone,","district)","overlay","floodplain",
               "floodway","flood plain","shoreland","shore land","coastal zone","scenic","williamson act",
               "farmland security","efu zone","residential zone","residential district","recreational",
               "conservation","special use","special exception","conditional use","permitted only in",
               "permitted as","other districts allow","capped at","land use categories","exclusion zone",
               "wildlife","historic","not a permitted use","ground-mounted","roof-mounted","rooftop",
               "type 3","tier 3","siting approval","prior siting")
_BUFFER=_re.compile(r"within\s+[\d,\.]+\s*(?:feet|foot|ft|mile|miles|meter|meters)\b")
_CAP=_re.compile(r"(?:exceed|more than|no more than|cap(?:ped)?(?:\s+at)?)\s+[\d,\.]+\s*"
                 r"(?:megawatt|mw|acre|acres)")
def classify_prohibition(summary, scope="county"):
    """Return (label, derate): 'full'/1.0, 'partial'/0.5, 'temporary'/0.5, or 'none'/0.0.
    scope='municipal' skips the single-municipality downgrade (a Municipal-Level row is already
    one municipality -> 'full' is its intended per-town calibration)."""
    s=_norm(summary)
    if not s: return ("partial",0.5)
    if _is_temporary(s):    return ("temporary",0.5)
    # REG-FIX guards MUST precede _countywide_ban (see note above):
    if _is_permissive(s):   return ("none",0.0)      # solar PERMITTED as a principal use -> ~0
    if _is_carveout_ban(s): return ("partial",0.5)
    if scope=="county":
        if _muni_scope_only(s): return ("partial",0.5)   # county row scoped to a single town/township
        if _countywide_ban(s):  return ("full",1.0)
    else:
        if _countywide_ban(s):  return ("full",1.0)      # preserve ORIG per-town calibration
        if _muni_scope_only(s): return ("partial",0.5)
    if any(k in s for k in _PARTIAL_KEYS) or _BUFFER.search(s) or _CAP.search(s):
        return ("partial",0.5)
    return ("full",1.0)

wb=openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
def sheet(name):
    ws=wb[name]; it=ws.iter_rows(values_only=True); h=next(it)
    if h[0] and 'generative AI' in str(h[0]): h=next(it)
    return {c:i for i,c in enumerate(h)}, list(it)

STATE_NAME_FP={'Alabama':1,'Arizona':4,'Arkansas':5,'California':6,'Colorado':8,'Connecticut':9,
 'Delaware':10,'District of Columbia':11,'Florida':12,'Georgia':13,'Idaho':16,'Illinois':17,'Indiana':18,
 'Iowa':19,'Kansas':20,'Kentucky':21,'Louisiana':22,'Maine':23,'Maryland':24,'Massachusetts':25,
 'Michigan':26,'Minnesota':27,'Mississippi':28,'Missouri':29,'Montana':30,'Nebraska':31,'Nevada':32,
 'New Hampshire':33,'New Jersey':34,'New Mexico':35,'New York':36,'North Carolina':37,'North Dakota':38,
 'Ohio':39,'Oklahoma':40,'Oregon':41,'Pennsylvania':42,'Rhode Island':44,'South Carolina':45,
 'South Dakota':46,'Tennessee':47,'Texas':48,'Utah':49,'Vermont':50,'Virginia':51,'Washington':53,
 'West Virginia':54,'Wisconsin':55,'Wyoming':56}

# ---- state-level solar setback baseline ----
sidx,srows=sheet('State-Level'); state_pl={}; state_st={}
for r in srows:
    st=str(r[sidx['State']]).strip() if r[sidx['State']] else None
    if not st: continue
    f=r[sidx['Feature']]; d=to_m(r[sidx['Value']],r[sidx['Units']])
    if d is None: continue
    if f=='Property Line (Non-Participating)': state_pl[st]=max(state_pl.get(st,0),d)
    if f=='Structures (Non-Participating)':    state_st[st]=max(state_st.get(st,0),d)

# ---- county-level solar ordinances ----
cidx,crows=sheet('County-Level'); FC='County Subdivision FIPS Code'
county={}
for r in crows:
    fp=r[cidx[FC]]
    try: fp=int(fp)
    except (TypeError,ValueError): continue
    if fp<1000: continue
    d=county.setdefault(fp,dict(state=str(r[cidx['State']]).strip() if r[cidx['State']] else '',
                                county=str(r[cidx['County']]).strip() if r[cidx['County']] else '',
                                pl=0.0,st=0.0,prohib=0.0,reasons=set()))
    f=r[cidx['Feature']]; m=to_m(r[cidx['Value']],r[cidx['Units']])
    if f=='Property Line (Non-Participating)' and m: d['pl']=max(d['pl'],m); d['reasons'].add('propline-setback')
    elif f=='Structures (Non-Participating)' and m: d['st']=max(d['st'],m); d['reasons'].add('structure-setback')
    elif f=='Prohibitions':
        lab,pr=classify_prohibition(r[cidx['Summary']]); d['prohib']=max(d['prohib'],pr)
        d['reasons'].add({'full':'ban','partial':'partial-ban','temporary':'temporary-moratorium',
                          'none':'permitted-use'}[lab])

# ---- MUNICIPAL / township solar prohibitions (previously dropped silently) ----
# Ingested + aggregated to county FIPS as a partial county derate (same approach + AUTHOR-CALIBRATED
# constants as the wind builder). Single-municipality => partial county removal, not a full ban.
# LIMITATION (disclosed): precise treatment needs cousub geometry; this is a county aggregate.
MUNI_UNIT_FULL=0.10; MUNI_UNIT_PARTIAL=0.04; MUNI_CAP=0.60   # AUTHOR-CALIBRATED (see wind builder)
muni_derate={}
mstat=dict(rows_total=0,proh=0,full=0,part=0,temp=0,unmapped=0,setback_dropped=0,other_dropped=0)
try:
    midx,mrows=sheet('Municipal-Level'); _mf={}; _mp={}
    for r in mrows:
        if not any(x is not None for x in r): continue
        mstat['rows_total']+=1
        feat=r[midx['Feature']] if 'Feature' in midx else None
        fpv=r[midx[FC]] if FC in midx else None
        try: cty=int(str(int(fpv)).zfill(10)[:5])
        except (TypeError,ValueError): cty=None
        if feat=='Prohibitions':
            mstat['proh']+=1
            if cty is None: mstat['unmapped']+=1; continue
            lab,_=classify_prohibition(r[midx['Summary']], scope="municipal")
            if lab=='none': continue                         # permissive muni ordinance -> no derate
            if lab=='full': mstat['full']+=1; _mf[cty]=_mf.get(cty,0)+1
            elif lab=='temporary': mstat['temp']+=1; _mp[cty]=_mp.get(cty,0)+1
            else: mstat['part']+=1; _mp[cty]=_mp.get(cty,0)+1
        elif feat in ('Property Line (Non-Participating)','Structures (Non-Participating)'):
            mstat['setback_dropped']+=1
        else: mstat['other_dropped']+=1
    for cty in set(list(_mf)+list(_mp)):
        muni_derate[cty]=min(MUNI_CAP, MUNI_UNIT_FULL*_mf.get(cty,0)+MUNI_UNIT_PARTIAL*_mp.get(cty,0))
except KeyError:
    print("[muni] Municipal-Level sheet not found; skipping")
print(f"[muni] solar municipal rows scanned: {mstat['rows_total']:,} | prohibition rows {mstat['proh']} "
      f"(full {mstat['full']}, partial {mstat['part']}, temporary {mstat['temp']}, unmapped {mstat['unmapped']})")
print(f"[muni] INGESTED solar municipal bans into {len(muni_derate)} counties; NOT ingested/disclosed: "
      f"{mstat['setback_dropped']} municipal setback rows + {mstat['other_dropped']:,} other rows.")
def _apply_muni(cty_fips, base):
    md=muni_derate.get(int(cty_fips),0.0)
    if md<=0: return base, False
    return min(1.0, 1.0-(1.0-base)*(1.0-md)), True

# all county FIPS present on the grid, grouped by state (for the state-baseline step below)
_all_grid=[int(x) for x in np.unique(fips) if x>0]
_state_counties={}
for cfp in _all_grid: _state_counties.setdefault(cfp//1000, []).append(cfp)
FP_STATE={v:k for k,v in STATE_NAME_FP.items()}

# ---- per-county derate: county rules + municipal aggregate ----
# deff: 0.6 weight on structure setbacks is an AUTHOR-CALIBRATED heuristic (see wind builder):
# dwelling/structure setbacks reach fewer buildable cells than an equal property-line setback.
STRUCT_WEIGHT=0.6
rows=[]; muni_used=set(); listed=set(county.keys())
for fp,d in county.items():
    pl=max(d['pl'],state_pl.get(d['state'],0.0)); st=max(d['st'],state_st.get(d['state'],0.0))
    deff=max(pl,STRUCT_WEIGHT*st); sd=setback_derate(deff)
    base=min(1.0,max(d['prohib'],sd))
    der,used=_apply_muni(fp,base)
    if used: d['reasons'].add('muni-ban'); muni_used.add(int(fp))
    rows.append(dict(fips=fp,state=d['state'],county=d['county'],derate=round(der,3),
                     prohib=d['prohib'],binding_setback_m=round(deff,1),
                     reason='|'.join(sorted(d['reasons'])) or 'none'))

# ---- FINDING 4 fix: STATE-baseline setback for counties in a state with a state solar law but
#      not individually listed (symmetric to the wind builder's 'state-law-setback' step) ----
statefp_with_law={STATE_NAME_FP[s] for s in set(list(state_pl)+list(state_st)) if s in STATE_NAME_FP}
for stfp in statefp_with_law:
    stname=FP_STATE.get(stfp,'')
    pl=state_pl.get(stname,0.0); st=state_st.get(stname,0.0)
    deff=max(pl,STRUCT_WEIGHT*st); sd=setback_derate(deff)
    for cfp in _state_counties.get(stfp,[]):
        if cfp in listed: continue
        der,used=_apply_muni(cfp,sd)
        if sd>0 or used:
            muni_used.add(int(cfp))
            rows.append(dict(fips=cfp,state=stname,county='',derate=round(der,3),
                             prohib=0.0,binding_setback_m=round(deff,1),
                             reason='state-law-setback'+('|muni-ban' if used else '')))

# ---- municipal-only counties: municipal bans but no county rule and no state-law setback ----
for cty,md in muni_derate.items():
    if md<=0 or int(cty) in muni_used or int(cty) in listed: continue
    rows.append(dict(fips=int(cty),state=FP_STATE.get(cty//1000,''),county='',
                     derate=round(min(1.0,md),3),prohib=0.0,binding_setback_m=0.0,reason='muni-ban'))

rules=pd.DataFrame(rows).sort_values('fips')
rules.to_csv(f"{PD}/solar_policy_rules.csv",index=False)

# ---- ordinance derate raster ----
der_map={int(r.fips):float(r.derate) for r in rules.itertuples()}
ord_derate=np.zeros((H,W),"f4")
for f in np.unique(fips):
    if f<=0: continue
    dr=der_map.get(int(f))
    if dr: ord_derate[fips==f]=dr

# ---- ag-protection derate (prime-farmland cells in ag-protection states) ----
# State SELECTION is sourced (NCSL "State Farmland and Solar" tracker + DSIRE identify which
# states protect utility-scale-solar farmland). The derate STRENGTHS below are AUTHOR-CALIBRATED
# heuristics (NOT literature values): 0.55 'strong' vs 0.40 'moderate' express the author's
# judgment of how strongly each state's policy discourages prime-farmland solar; the trackers do
# not supply a percentage. (Sensitivity is exposed downstream via the SCALE knob.)
AG_STRONG=0.55; AG_MOD=0.40
ag_strength={'Oregon':AG_STRONG,'New York':AG_STRONG,'Ohio':AG_STRONG,'Vermont':AG_STRONG,'California':AG_STRONG,
 'Illinois':AG_MOD,'Indiana':AG_MOD,'Iowa':AG_MOD,'Wisconsin':AG_MOD,'Minnesota':AG_MOD,'Michigan':AG_MOD,
 'Virginia':AG_MOD,'North Carolina':AG_MOD,'Pennsylvania':AG_MOD,'Maryland':AG_MOD,'Massachusetts':AG_MOD,
 'Washington':AG_MOD,'Kansas':AG_MOD,'Nebraska':AG_MOD,'Missouri':AG_MOD}
ag_derate=np.zeros((H,W),"f4")
for stn,strg in ag_strength.items():
    fp=STATE_NAME_FP.get(stn)
    if fp is None: continue
    m=(state_fips==fp)&prime
    ag_derate[m]=strg
pd.DataFrame([dict(state=k,ag_derate=v,statefp=STATE_NAME_FP.get(k)) for k,v in ag_strength.items()]
            ).to_csv(f"{PD}/solar_ag_states.csv",index=False)

# ---- combined solar derate = independent removals: 1-(1-ord)*(1-ag) ----
solar_derate=1.0-(1.0-ord_derate)*(1.0-ag_derate)
solar_derate=np.clip(solar_derate,0.0,1.0).astype("f4")

# ---- solar RPS carve-out (DSIRE) -> per-state additional solar fraction ----
solar_carveout={'New Jersey':0.051,'Maryland':0.145,'Massachusetts':0.04,'Delaware':0.10,
 'District of Columbia':0.10,'Pennsylvania':0.005,'North Carolina':0.002,'New Mexico':0.06,
 'Missouri':0.003,'Illinois':0.02,'Arizona':0.045,'Colorado':0.03,'New Hampshire':0.007,'Nevada':0.0,'Ohio':0.0}
solar_rps_fraction=np.zeros((H,W),"f4")
for stn,v in solar_carveout.items():
    fp=STATE_NAME_FP.get(stn)
    if fp and v>0: solar_rps_fraction[state_fips==fp]=v
pd.DataFrame([dict(state=k,solar_carveout_fraction=v,statefp=STATE_NAME_FP.get(k)) for k,v in solar_carveout.items()]
            ).to_csv(f"{PD}/solar_carveout_table.csv",index=False)

# ---- merge into policy_arrays.npz (keep existing wind arrays) ----
old=dict(np.load(f"{PD}/policy_arrays.npz"))
old['solar_derate']=solar_derate
old['solar_rps_fraction']=solar_rps_fraction
np.savez_compressed(f"{PD}/policy_arrays.npz",**old)

# ---- report ----
inv=(fips>0)
nban=(rules.derate>=1.0).sum(); npart=((rules.derate>0)&(rules.derate<1)).sum()
ntemp=rules.reason.str.contains('temporary-moratorium').sum()
nmuni=rules.reason.str.contains('muni-ban').sum()
nstate=rules.reason.str.contains('state-law-setback').sum()
print("==== SOLAR POLICY LAYER BUILT ====")
print(f"solar counties with any ordinance rule: {len(rules)}  (full ban {nban}, partial {npart})")
print(f"  temporary-moratorium rows (down-weighted from 1.0): {ntemp}")
print(f"  counties with ingested municipal bans             : {nmuni}")
print(f"  state-baseline-only counties (FINDING-4 symmetry) : {nstate}")
print(f"prime-farmland cells total: {int(prime.sum()):,}")
print(f"ag-protection prime-farmland cells (derated): {int((ag_derate>0).sum()):,}")
print(f"solar cells hard-banned (derate>=1): {int((solar_derate>=0.999).sum()):,}")
print(f"solar cells derated 0<d<1: {int(((solar_derate>0)&(solar_derate<0.999)).sum()):,}")
print(f"solar RPS carve-out states: {sum(1 for v in solar_carveout.values() if v>0)}")
print("top solar-restrictive counties:")
print(rules.sort_values(['derate','binding_setback_m'],ascending=False)
          [['fips','state','county','derate','binding_setback_m','reason']].head(10).to_string(index=False))
print("DONE")
