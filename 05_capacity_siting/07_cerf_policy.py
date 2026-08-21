"""
POLICY-AWARE CERF siting: identical to 06_cerf_full.py but with a STATE/COUNTY wind-siting
policy layer coupled into the Net-Locational-Cost siting.

Two policy mechanisms (built by 04_build_policy_layer.py -> /data/policy_data/policy_arrays.npz):
  (1) WIND EXCLUSION / DERATE  : counties with wind bans or moratoria are removed from the
      wind suitability mask; counties with large ordinance setbacks have their buildable wind
      fraction reduced (a fixed spatial share of suitable cells is sterilised).  This is UNIONED
      into GRIDCERF's suitability -> it only ADDS the incremental local-ordinance effect.
  (2) RPS / CES INCENTIVE       : states with a binding renewable / clean-energy standard get an
      NLC bonus (subtracted from Net-Locational-Cost) proportional to their final target fraction,
      pulling siting toward RPS states within each subregion.  Applied to wind AND solar.

Everything else (grid, resource, GCAM capacity, NOV, seed fleet, per-period expansion) is
IDENTICAL to 06_cerf_full.py so the two fleets are directly comparable.

Env knobs for sensitivity:  DERATE_SCALE (default 1.0)  RPS_ALPHA (default 0.08)
Outputs: /data/cerf_out/fleet_policy_{scen}.csv (+ policy_diag_{scen}.csv, map_policy_{scen}.png,
         SUMMARY_policy_fleets.csv)
"""
import os, sys, time, numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.transform import Affine
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
from pyproj import Transformer
from scipy.spatial import cKDTree
from scipy.ndimage import binary_dilation
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from cerf.nov import NetOperationalValue
from cerf.process_region import process_region

T0=time.time()
RESDIR="/data/gen_targets/srgan3d_val/future_resource"
GCAM_CSV="/data/gcam_usa/gcam_capacity_by_subregion.csv"
SHP="/data/cerf_smoke/inputs/subregion_regions.shp"
EIA={"wind":"/data/datasets/gen/tgw-gen-historical/eia_wind_configs.csv",
     "solar":"/data/datasets/gen/tgw-gen-historical/eia_solar_configs.csv"}
GCDIR="/data/gridcerf/gridcerf/compiled/compiled_technology_layers"
GC_FN={"wind":"wind/gridcerf_wind_onshore_hubheight100_no-cooling_10cf.tif",
       "solar":"solar/gridcerf_solar_pv_centralized_no-cooling_10cf.tif"}
OUT="/data/cerf_out"; os.makedirs(OUT,exist_ok=True); REG_TIF=f"{OUT}/region_raster.tif"
sys.path.insert(0,OUT); from cerf_lmp_zones import build_lmp2d   # LMP-aware economics (Option B)
LMP2D=build_lmp2d().astype("f4")                                 # (H,W) real annual-mean zonal LMP $/MWh
PD="/data/policy_data"

CLIMATES=["rcp45cooler","rcp45hotter","rcp85cooler","rcp85hotter"]
SSPS=["ssp3","ssp5"]; YEARS=[2025,2030,2035,2040,2045,2050]; SEED_YEAR=2020

ALBERS=("+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=37.5 +lon_0=-96 +x_0=0 +y_0=0 "
        "+datum=NAD83 +units=m +no_defs")
DST_CRS=CRS.from_proj4(ALBERS); RES=1000.0
LEFT,TOP=-2831615.2280,1690434.1707; W,H=5460,3229
DST_T=Affine(RES,0,LEFT,0,-RES,TOP)
COMMON=dict(discount_rate=0.05,lifetime_yrs=25,operational_life_yrs=25,
  variable_om_esc_rate_fraction=0.0,fuel_price_esc_rate_fraction=0.0,carbon_tax_esc_rate_fraction=0.0,
  heat_rate_btu_per_kWh=0.0,fuel_price_usd_per_mmbtu=0.0,carbon_tax_usd_per_ton=0.0,
  carbon_capture_rate_fraction=0.0,fuel_co2_content_tons_per_btu=0.0)
TECH={1:dict(tech_name="wind_onshore",unit_size_mw=300,variable_om_usd_per_mwh=1.0,
        buffer_in_km=5,capacity_factor_fraction=0.40,**COMMON),
      2:dict(tech_name="solar_pv_non_dist",unit_size_mw=200,variable_om_usd_per_mwh=0.5,
        buffer_in_km=3,capacity_factor_fraction=0.25,**COMMON)}
TECH_ORDER=[1,2]; TECH_KIND={1:"wind",2:"solar"}; KIND_TID={"wind":1,"solar":2}
BUF_CELLS={1:5,2:3}; LMP_UNIFORM=40.0; IC_RATE=90.0

# ============ POLICY LAYER (wind + solar, symmetric) ============
POL=np.load(f"{PD}/policy_arrays.npz")
WIND_DERATE=POL["wind_derate"].astype("f4")          # (H,W) 0..1 county wind ordinance derate
SOLAR_DERATE=POL["solar_derate"].astype("f4")        # (H,W) 0..1 county solar ordinance + ag-land derate
RPS_FRAC=POL["rps_fraction"].astype("f4")            # (H,W) per-state final RPS/CES fraction (both techs)
SOLAR_RPS_FRAC=POL["solar_rps_fraction"].astype("f4")# (H,W) per-state solar RPS carve-out (solar only)
FIPS_RAS=rasterio.open(f"{PD}/county_fips_albers.tif").read(1)
DERATE_SCALE=float(os.environ.get("DERATE_SCALE","1.0"))
RPS_ALPHA=float(os.environ.get("RPS_ALPHA","0.08"))  # AUTHOR-CALIBRATED heuristic weight
# (NOT a literature value): converts a per-state RPS/CES pull into an NLC siting bonus. 0.08 is the
# authors' combined federal+state clean-siting incentive strength under the IRA scenario; the OBBBA
# variant (08_cerf_obbba.py) uses 0.04 = the state-only half. Exposed for sensitivity via RPS_ALPHA env.
# separate fixed land fields for wind vs solar so removals are not spatially identical
THIN_W=np.random.default_rng(12345).random((H,W)).astype("f4")
THIN_S=np.random.default_rng(999).random((H,W)).astype("f4")
WIND_DERATE_EFF=np.clip(WIND_DERATE*DERATE_SCALE,0.0,1.0)
SOLAR_DERATE_EFF=np.clip(SOLAR_DERATE*DERATE_SCALE,0.0,1.0)
WIND_BAN=WIND_DERATE_EFF>=0.999
SOLAR_BAN=SOLAR_DERATE_EFF>=0.999
WIND_POLICY_EXCL =WIND_BAN |((WIND_DERATE_EFF>0)&(WIND_DERATE_EFF<0.999)&(THIN_W<WIND_DERATE_EFF))
SOLAR_POLICY_EXCL=SOLAR_BAN|((SOLAR_DERATE_EFF>0)&(SOLAR_DERATE_EFF<0.999)&(THIN_S<SOLAR_DERATE_EFF))
POLICY_EXCL={1:WIND_POLICY_EXCL, 2:SOLAR_POLICY_EXCL}
print(f"[policy] DERATE_SCALE={DERATE_SCALE} RPS_ALPHA={RPS_ALPHA}")
print(f"[policy] WIND  hard-banned={int(WIND_BAN.sum()):,}  excl(total)={int(WIND_POLICY_EXCL.sum()):,}")
print(f"[policy] SOLAR hard-banned={int(SOLAR_BAN.sum()):,}  excl(total)={int(SOLAR_POLICY_EXCL.sum()):,}")

# ---- region raster (shared) ----
gdf=gpd.read_file(SHP).to_crs(DST_CRS).reset_index(drop=True); gdf["rid"]=gdf.index+1
REGIONS={r.subregion.lower():int(r.rid) for r in gdf.itertuples()}
RID2NAME={int(r.rid):r.subregion for r in gdf.itertuples()}
LC2NAME={k:RID2NAME[v] for k,v in REGIONS.items()}
region_raster=rasterize([(g,r) for g,r in zip(gdf.geometry,gdf.rid)],out_shape=(H,W),
  transform=DST_T,fill=0,dtype="int32",all_touched=False)
cols=np.arange(W); rows=np.arange(H)
XC=np.tile(LEFT+(cols+0.5)*RES,(H,1)).astype("f8")
YC=np.tile((TOP-(rows+0.5)*RES)[:,None],(1,W)).astype("f8")
IDX2D=np.arange(H*W).reshape(H,W)

def cf_to_albers(cf,lat,lon):
    dx=float((lon[-1]-lon[0])/(len(lon)-1)); dy=float((lat[-1]-lat[0])/(len(lat)-1))
    cf_ns=cf[::-1,:].copy(); src_top=lat[-1]+dy/2.0; src_left=lon[0]-dx/2.0
    src_t=Affine(dx,0,src_left,0,-dy,src_top); dst=np.full((H,W),np.nan,"f4")
    reproject(cf_ns,dst,src_transform=src_t,src_crs=CRS.from_epsg(4326),dst_transform=DST_T,
      dst_crs=DST_CRS,resampling=Resampling.bilinear,src_nodata=np.nan,dst_nodata=np.nan)
    return dst

_CFC={}
def get_cf(climate):
    if climate in _CFC: return _CFC[climate]
    o={}
    for kind in ("wind","solar"):
        d=np.load(f"{RESDIR}/future_{kind}_resource_{climate}.npz")
        o[kind]=cf_to_albers(d["mean_cf"].astype("f4"),d["lat"].astype("f8"),d["lon"].astype("f8"))
    _CFC[climate]=o; return o

_NOVC={}
def get_nov(climate):
    if climate in _NOVC: return _NOVC[climate]
    cf_alb=get_cf(climate); lmp=np.repeat(LMP2D[None,:,:],2,axis=0).copy()  # LMP-aware zonal $/MWh (was flat LMP_UNIFORM)
    gen=np.zeros((2,H,W),"f4"); op=np.zeros((2,H,W),"f4"); nov=np.zeros((2,H,W),"f4")
    for ix,tid in enumerate(TECH_ORDER):
        kind=TECH_KIND[tid]; td=TECH[tid]; cf2d=np.nan_to_num(cf_alb[kind],nan=0.0).astype("f8")
        econ=NetOperationalValue(discount_rate=td["discount_rate"],lifetime_yrs=td["lifetime_yrs"],
          unit_size_mw=td["unit_size_mw"],capacity_factor_fraction=cf2d,
          variable_om_esc_rate_fraction=0.0,fuel_price_esc_rate_fraction=0.0,carbon_tax_esc_rate_fraction=0.0,
          variable_om_usd_per_mwh=td["variable_om_usd_per_mwh"],heat_rate_btu_per_kWh=0.0,
          fuel_price_usd_per_mmbtu=0.0,carbon_tax_usd_per_ton=0.0,carbon_capture_rate_fraction=0.0,
          fuel_co2_content_tons_per_btu=0.0,lmp_arr=lmp[ix],target_year=2050)
        gg,oo,nn=econ.calc_nov(); gen[ix]=gg.astype("f4"); op[ix]=float(oo); nov[ix]=nn.astype("f4")
    # ---- RPS bonus: per-tech reference NOV * alpha * state RPS pull ----
    # wind  gets the general RPS/CES fraction; solar gets general RPS/CES + its solar carve-out.
    rps_bonus=np.zeros((2,H,W),"f4")
    for ix in (0,1):
        m=(region_raster>0)&np.isfinite(nov[ix])&(nov[ix]>0)
        nref=float(np.nanmean(nov[ix][m])) if m.any() else 0.0
        pull=RPS_FRAC if ix==0 else (RPS_FRAC+SOLAR_RPS_FRAC)
        rps_bonus[ix]=(RPS_ALPHA*pull*nref).astype("f4")
        print(f"[policy] {climate} {TECH_KIND[TECH_ORDER[ix]]}: NOV_ref={nref:.3e}  "
              f"max RPS bonus={float(rps_bonus[ix].max()):.3e}")
    _NOVC[climate]=(lmp,gen,op,nov,rps_bonus); return _NOVC[climate]

_GCC={}
def get_comp(ssp,year,kind):
    k=(ssp,year,kind)
    if k not in _GCC: _GCC[k]=rasterio.open(f"{GCDIR}/{ssp}/{year}/{GC_FN[kind]}").read(1)
    return _GCC[k]

TR_FWD=Transformer.from_crs(4326,DST_CRS,always_xy=True)
TR_BACK=Transformer.from_crs(DST_CRS,4326,always_xy=True)

def build_seed(cf_alb):
    rows=[]; occ={1:np.zeros((H,W),bool),2:np.zeros((H,W),bool)}; xy={1:[],2:[]}
    for kind in ("wind","solar"):
        tid=KIND_TID[kind]; d=pd.read_csv(EIA[kind]).dropna(subset=["lat","lon","system_capacity"])
        lon=d["lon"].to_numpy(float); lat=d["lat"].to_numpy(float); cap=d["system_capacity"].to_numpy(float)/1000.0
        px,py=TR_FWD.transform(lon,lat); px=np.asarray(px); py=np.asarray(py)
        col=np.floor((px-LEFT)/RES).astype(int); row=np.floor((TOP-py)/RES).astype(int)
        ok=np.isfinite(px)&np.isfinite(py)&(row>=0)&(row<H)&(col>=0)&(col<W)
        for i in np.where(ok)[0]:
            r,c=int(row[i]),int(col[i]); rid=int(region_raster[r,c])
            if rid==0: continue
            cf=float(cf_alb[kind][r,c]); occ[tid][r,c]=True; xy[tid].append((px[i],py[i]))
            rows.append(dict(tech=kind,capacity_mw=cap[i],lon=lon[i],lat=lat[i],xcoord=px[i],ycoord=py[i],
                sited_year=SEED_YEAR,retirement_year="",subregion=RID2NAME[rid],
                cell_cf_4km=cf if np.isfinite(cf) else np.nan,
                net_operational_value_usd_per_year=np.nan,interconnection_cost_usd_per_year=np.nan,
                net_locational_cost_usd_per_year=np.nan,generation_mwh_per_year=np.nan,cell_lmp=np.nan))
    return pd.DataFrame(rows), occ, xy

def build_ic(xy_list,unit,suit_mask):
    tree=cKDTree(np.asarray(xy_list)); ic=np.zeros((H,W),"f4")
    rr,cc=np.where(suit_mask)
    if len(rr)==0: return ic
    cx=LEFT+(cc+0.5)*RES; cy=TOP-(rr+0.5)*RES
    dd,_=tree.query(np.c_[cx,cy],k=1,workers=-1); ic[rr,cc]=(dd/1000.0)*unit*IC_RATE; return ic

def reconcile_fleet_to_gcam(fl,piv):
    """FLEET-FIX: force the realized 2050 fleet to EQUAL the GCAM-2050 capacity target per
    (subregion,tech). Over-built subregions (EIA-2020 seed overhang and/or a GCAM projected
    decline) are capped to GCAM-2050 by economic retirement: retire the lowest-CF operating cells
    first (partial-retire the marginal cell), so real HIGH-CF plant LOCATIONS are retained. A
    sub-unit rounding shortfall is absorbed exactly by bumping the best operating added cell; a
    shortfall of >= one unit is genuine unmet demand (here, wind/solar capacity STRANDED by the
    policy exclusion mask) and is left untouched. Retirement is recorded via retirement_year=2050
    -> excluded from the 2050 fleet, matching fut_gen.load_fleet's `retirement_year>2050` filter."""
    tol=1e-6
    for name_lc,rid in REGIONS.items():
        name=RID2NAME[rid]
        for kind in ("wind","solar"):
            tid=KIND_TID[kind]; unit=float(TECH[tid]["unit_size_mw"])
            try: tgt=float(piv.loc[(name,kind),2050])*1000.0
            except KeyError: tgt=0.0
            op=fl.index[(fl.subregion==name)&(fl.tech==kind)&(fl.retirement_year=="")]
            if len(op)==0: continue
            realized=float(fl.loc[op,"capacity_mw"].sum()); diff=realized-tgt
            if diff>tol:                       # over-built -> retire lowest-CF operating cells first
                order=fl.loc[op].sort_values("cell_cf_4km",na_position="first").index
                rem=diff
                for i in order:
                    if rem<=tol: break
                    cap=float(fl.at[i,"capacity_mw"])
                    if cap<=rem+tol: fl.at[i,"retirement_year"]="2050"; rem-=cap
                    else: fl.at[i,"capacity_mw"]=cap-rem; rem=0.0
            elif diff<-tol and (-diff)<unit:   # sub-unit rounding shortfall -> exact top-up
                pool=fl.loc[op]; added=pool[pool.sited_year>SEED_YEAR]
                pool=added if len(added) else pool
                cfmax=pool["cell_cf_4km"].max()
                j=pool["cell_cf_4km"].idxmax() if pd.notna(cfmax) else pool.index[0]
                fl.at[j,"capacity_mw"]=float(fl.at[j,"capacity_mw"])+(-diff)
    return fl

def run_scenario(climate,ssp):
    scen=f"{climate}_{ssp}"; t=time.time()
    cf_alb=get_cf(climate); lmp,gen,op,nov,rps_bonus=get_nov(climate)
    seed_df,occ,xy=build_seed(cf_alb)
    fleet=[seed_df]
    g=pd.read_csv(GCAM_CSV); g=g[(g.scenario==scen)&(g.tech.isin(["wind","solar"]))]
    piv=g.pivot_table(index=["subregion","tech"],columns="year",values="capacity_GW").fillna(0.0)
    # --- FLEET-FIX: track OPERATING capacity per (subregion,tech), seeded from the real EIA-2020
    #     fleet, so additions are driven by the ABSOLUTE GCAM path (reconciled to the seed), not by
    #     gross period-to-period diffs that ignore GCAM declines. ---
    realized={(nl,tid):0.0 for nl in REGIONS for tid in TECH_ORDER}
    for rr_ in seed_df.itertuples():
        realized[(rr_.subregion.lower(),KIND_TID[rr_.tech])]+=float(rr_.capacity_mw)
    log=[]
    for Y in YEARS:
        Y0=Y-5
        suit=np.ones((2,H,W),"f4")
        for ix,tid in enumerate(TECH_ORDER):
            kind=TECH_KIND[tid]; comp=get_comp(ssp,Y,kind)
            valid=np.isfinite(cf_alb[kind])&(cf_alb[kind]>0.02)&(region_raster>0)&(comp==0)
            valid&=~POLICY_EXCL[tid]                     # <<< POLICY: wind & solar exclusion / derate
            if occ[tid].any():
                valid&=~binary_dilation(occ[tid],iterations=BUF_CELLS[tid])
            suit[ix]=np.where(valid,0.0,1.0)
        ic=np.zeros((2,H,W),"f4")
        for ix,tid in enumerate(TECH_ORDER):
            ic[ix]=build_ic(xy[tid],TECH[tid]["unit_size_mw"],suit[ix]==0)
        nlc=ic-nov-rps_bonus                            # <<< POLICY: RPS/CES NLC bonus
        # FLEET-FIX: net additions to raise OPERATING capacity toward the ABSOLUTE GCAM path.
        # GCAM declines are NOT sited here (the cumulative fleet ratchets up to the running max);
        # the net trajectory down to the EXACT GCAM-2050 target is reconciled by retirement in
        # reconcile_fleet_to_gcam() after the loop.
        expansion={}; req={1:0,2:0}
        for name_lc,rid in REGIONS.items():
            name=RID2NAME[rid]; expansion[name_lc]={}
            for tid in TECH_ORDER:
                kind=TECH_KIND[tid]
                try: tgt_mw=float(piv.loc[(name,kind),Y])*1000.0
                except KeyError: tgt_mw=0.0
                deficit=tgt_mw-realized[(name_lc,tid)]
                n=int(round(deficit/TECH[tid]["unit_size_mw"])) if deficit>0 else 0
                req[tid]+=n; realized[(name_lc,tid)]+=n*TECH[tid]["unit_size_mw"]
                expansion[name_lc][tid]={"n_sites":n,"tech_name":TECH[tid]["tech_name"]}
        settings={"run_year":Y,"region_raster_file":REG_TIF}
        pf=[]
        for name_lc in REGIONS:
            if sum(expansion[name_lc][t]["n_sites"] for t in TECH_ORDER)<=0: continue
            try:
                res=process_region(target_region_name=name_lc,settings_dict=settings,technology_dict=TECH,
                  technology_order=TECH_ORDER,expansion_dict={name_lc:{t:dict(expansion[name_lc][t]) for t in TECH_ORDER}},
                  regions_dict=REGIONS,suitability_arr=suit,lmp_arr=lmp,generation_arr=gen,operating_cost_arr=op,
                  nov_arr=nov,ic_arr=ic,nlc_arr=nlc,zones_arr=region_raster.astype("f8"),
                  xcoords=XC,ycoords=YC,indices_2d=IDX2D,randomize=False,seed_value=42,verbose=False,write_output=False)
            except Exception as e:
                print(f"    [WARN] {scen} {Y} {name_lc}: {e}"); continue
            if res is not None and len(res.run_data.sited_df): pf.append(res.run_data.sited_df)
        got={1:0,2:0}
        if pf:
            sp=pd.concat(pf,ignore_index=True)
            idx=sp["index"].astype(int).values; rr=idx//W; cc=idx%W
            lonp,latp=TR_BACK.transform(sp["xcoord"].values,sp["ycoord"].values)
            out=pd.DataFrame(dict(
                tech=sp["tech_id"].map(TECH_KIND), capacity_mw=sp["unit_size_mw"].astype(float),
                lon=lonp, lat=latp, sited_year=Y, retirement_year="",
                subregion=sp["region_name"].map(LC2NAME),
                cell_cf_4km=[float(cf_alb[TECH_KIND[t]][r,c]) for t,r,c in zip(sp["tech_id"],rr,cc)],
                xcoord=sp["xcoord"].values, ycoord=sp["ycoord"].values,
                net_operational_value_usd_per_year=sp["net_operational_value_usd_per_year"].values,
                interconnection_cost_usd_per_year=sp["interconnection_cost_usd_per_year"].values,
                net_locational_cost_usd_per_year=sp["net_locational_cost_usd_per_year"].values,
                generation_mwh_per_year=sp["generation_mwh_per_year"].values,
                cell_lmp=[float(LMP2D[r,c]) for r,c in zip(rr,cc)]))
            fleet.append(out)
            for tid in TECH_ORDER:
                m=(sp["tech_id"]==tid).values
                if m.any():
                    occ[tid][rr[m],cc[m]]=True
                    xy[tid].extend(list(zip(sp["xcoord"].values[m],sp["ycoord"].values[m])))
                    got[tid]=int(m.sum())
        log.append((Y,req[1],got[1],req[2],got[2]))
        print(f"  [{scen}] {Y0}->{Y}: wind req={req[1]} sited={got[1]}  solar req={req[2]} sited={got[2]}")
    fl=pd.concat(fleet,ignore_index=True)
    fl=reconcile_fleet_to_gcam(fl,piv)   # FLEET-FIX: cap/retire to EXACT GCAM-2050 per subregion+tech (unmet = policy-stranded)
    fl.to_csv(f"{OUT}/fleet_policy_{scen}{os.environ.get('OUTTAG','')}.csv",index=False)
    # stranded (wind capacity that GCAM wanted but policy left unsited)
    L=pd.DataFrame(log,columns=["year","wreq","wgot","sreq","sgot"])
    strand_wind=int((L.wreq-L.wgot).clip(lower=0).sum()); strand_solar=int((L.sreq-L.sgot).clip(lower=0).sum())
    print(f"  [{scen}] wrote fleet_policy_{scen}.csv n={len(fl)} "
          f"wind_stranded_sites={strand_wind} ({strand_wind*TECH[1]['unit_size_mw']/1000:.1f}GW) ({time.time()-t:.1f}s)")
    return fl,L,cf_alb,ssp,strand_wind,strand_solar

def summarize(scen,fl,cf_alb,ssp,strand_wind,strand_solar):
    g=pd.read_csv(GCAM_CSV); g=g[(g.scenario==scen)&(g.tech.isin(["wind","solar"]))]
    piv=g.pivot_table(index="tech",columns="year",values="capacity_GW",aggfunc="sum").fillna(0.0)
    rec={"scenario":scen}
    for kind in ("wind","solar"):
        f=fl[(fl.tech==kind)&(fl.retirement_year=="")]   # operating in 2050 (retired excluded)
        rec[f"{kind}_fleet2050_GW"]=round(f.capacity_mw.sum()/1000.0,1)
        rec[f"{kind}_seed2020_GW"]=round(f[f.sited_year==SEED_YEAR].capacity_mw.sum()/1000.0,1)
        rec[f"{kind}_sited_add_GW"]=round(f[f.sited_year>SEED_YEAR].capacity_mw.sum()/1000.0,1)
        rec[f"{kind}_GCAM2050_GW"]=round(float(piv.loc[kind,2050]),1)
    rec["wind_stranded_GW"]=round(strand_wind*TECH[1]["unit_size_mw"]/1000.0,1)
    rec["solar_stranded_GW"]=round(strand_solar*TECH[2]["unit_size_mw"]/1000.0,1)
    for kind in ("wind","solar"):
        comp=get_comp(ssp,2050,kind); base=np.isfinite(cf_alb[kind])&(cf_alb[kind]>0.02)&(region_raster>0)&(comp==0)
        f=fl[(fl.tech==kind)&(fl.sited_year>SEED_YEAR)&(fl.retirement_year=="")]
        upl=[]
        for name in f.subregion.unique():
            rid=[k for k,v in RID2NAME.items() if v==name]
            if not rid: continue
            m=base&(region_raster==rid[0]); sm=float(np.nanmean(cf_alb[kind][m])) if m.sum() else np.nan
            for cf in f[f.subregion==name].cell_cf_4km.values: upl.append((cf,sm))
        upl=np.array(upl)
        if len(upl):
            rec[f"{kind}_site_cf"]=round(float(np.nanmean(upl[:,0])),3)
    fig,ax=plt.subplots(1,2,figsize=(17,5.6))
    for a,kind,cmap in [(ax[0],"wind","viridis"),(ax[1],"solar","plasma")]:
        s=fl[(fl.tech==kind)&(fl.retirement_year=="")]
        sc=a.scatter(s.lon,s.lat,c=s.cell_cf_4km,cmap=cmap,s=5,alpha=0.6,linewidth=0)
        a.set_title(f"POLICY {scen} {kind} 2050 fleet n={len(s)} {s.capacity_mw.sum()/1000:.0f} GW")
        a.set_xlim(-125,-66); a.set_ylim(24,50); plt.colorbar(sc,ax=a,label="4km CF",shrink=.8)
    plt.tight_layout(); plt.savefig(f"{OUT}/map_policy_{scen}.png",dpi=100); plt.close()
    return rec

if __name__=="__main__":
    only=sys.argv[1] if len(sys.argv)>1 else None
    recs=[]
    for climate in CLIMATES:
        for ssp in SSPS:
            scen=f"{climate}_{ssp}"
            if only and only!=scen: continue
            fl,L,cf_alb,ssp_,sw,ss=run_scenario(climate,ssp)
            recs.append(summarize(scen,fl,cf_alb,ssp_,sw,ss))
    S=pd.DataFrame(recs)
    tag=os.environ.get("SUMTAG","")
    S.to_csv(f"{OUT}/SUMMARY_policy_fleets{tag}.csv",index=False)
    pd.set_option("display.width",260); pd.set_option("display.max_columns",50)
    print("\n============ 8-SCENARIO POLICY SUMMARY ============")
    print(S.to_string(index=False))
    print(f"\n[ALL DONE] {time.time()-T0:.1f}s")
