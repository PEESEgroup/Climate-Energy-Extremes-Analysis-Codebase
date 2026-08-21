#!/usr/bin/env python
"""Build per-subregion REAL hourly hydro product + dispatch shape for the 18 study subregions.
Sources (all on box): EIA-930 PUDL parquet (universal backbone, all BAs, 2018-07->2026),
plus deeper dedicated ISO feeds BPA/CAISO/NYISO/SPP/ERCOT that take priority where they exist.
BA -> subregion via controlled allocation using cched plant roster nameplate + subregion mask.
Outputs: /data/hydro/real/subregion_hydro_hourly_real.npz , /data/hydro/real/dispatch_shape.npz
"""
import numpy as np, pandas as pd
import pyarrow.dataset as ds

TAR = "/data/hydro/38277_cd39165337f52ba0f091ad0ef9efeca7.tar"
REAL = "/data/hydro/real"
GRID = "/data/datasets/grid"

SUBS = ['CAISO','ERCOT','FRCC','ISONE','MISO_North','MISO_Central','MISO_South','NYISO',
        'NorthernGrid_East','NorthernGrid_South','NorthernGrid_West','PJM_East','PJM_West',
        'SERTP','SPP_North','SPP_South','WestConnect_North','WestConnect_South']
SUBIDX = {s:i for i,s in enumerate(SUBS)}

# local timezone per subregion (for hour-of-day dispatch shape)
TZ = {'CAISO':'America/Los_Angeles','ERCOT':'America/Chicago','FRCC':'America/New_York',
      'ISONE':'America/New_York','MISO_North':'America/Chicago','MISO_Central':'America/Chicago',
      'MISO_South':'America/Chicago','NYISO':'America/New_York','NorthernGrid_East':'America/Denver',
      'NorthernGrid_South':'America/Denver','NorthernGrid_West':'America/Los_Angeles',
      'PJM_East':'America/New_York','PJM_West':'America/New_York','SERTP':'America/New_York',
      'SPP_North':'America/Chicago','SPP_South':'America/Chicago','WestConnect_North':'America/Denver',
      'WestConnect_South':'America/Phoenix'}

# ---------- BA -> subregion allocation framework ----------
SINGLE = {  # whole BA -> one subregion
  'AEC':'SERTP','AVRN':'NorthernGrid_West','AZPS':'WestConnect_South','BANC':'CAISO',
  'BHBA':'WestConnect_North','BPAT':'NorthernGrid_West','CISO':'CAISO','CPLE':'SERTP',
  'CPLW':'SERTP','DUK':'SERTP','ERCO':'ERCOT','FPC':'FRCC','IID':'CAISO','ISNE':'ISONE',
  'LDWP':'CAISO','LGEE':'SERTP','NYIS':'NYISO','PNM':'WestConnect_South','SC':'SERTP',
  'SCEG':'SERTP','SEPA':'SERTP','SOCO':'SERTP','TAL':'FRCC','TIDC':'CAISO','TVA':'SERTP','YAD':'SERTP',
  # FIX 2026-07-13: FRCC previously had ONLY FPC+TAL = 24% of Florida; add all FL BAs + other dropped BAs
  'FPL':'FRCC','TEC':'FRCC','FMPP':'FRCC','JEA':'FRCC','GVL':'FRCC','HST':'FRCC','SEC':'FRCC','NSB':'FRCC',
  'AECI':'SPP_South','TEPC':'WestConnect_South','EPE':'WestConnect_South','EEI':'MISO_Central',
  'GWA':'NorthernGrid_East','WWA':'NorthernGrid_East','DEAA':'WestConnect_South','HGMA':'WestConnect_South',
  'GRIF':'WestConnect_South','GRID':'SPP_North'}
SPLIT = {  # RTO -> split only within these subregions, by plant capacity share
  'PJM':['PJM_East','PJM_West'],
  'MISO':['MISO_North','MISO_Central','MISO_South'],
  'SWPP':['SPP_North','SPP_South']}
WEST = ['NorthernGrid_East','NorthernGrid_South','NorthernGrid_West','WestConnect_North','WestConnect_South']
WEST_BAS = ['AVA','CHPD','DOPD','GCPD','IPCO','NEVP','NWMT','PACE','PACW','PGE','PSCO','PSEI',
            'SCL','SRP','TPWR','WACM','WALC','WAUW']
# SPA handled by full-mask fallback (its AR/MO/OK plants land in SPP_South/MISO_South/SPP_North)

def build_alloc():
    roster = pd.read_csv(f"{TAR}/cched_hydro_plants_with_huc4.csv")
    m = np.load(f"{GRID}/subregion_mask.npz", allow_pickle=True)
    mask = m["subregion_mask"]
    id2sub = {int(r[0]): r[1] for r in m["id_to_subregion"]}
    c = np.load(f"{GRID}/coordinate.npz", allow_pickle=True)
    lat = c["lat"].astype(float); lon = c["lon"].astype(float)
    def to_sub(la, lo):
        if not np.isfinite(la) or not np.isfinite(lo): return None
        return id2sub.get(int(mask[int(np.abs(lat-la).argmin()), int(np.abs(lon-lo).argmin())]), None)
    roster["sub"] = [to_sub(a,o) for a,o in zip(roster.lat, roster.lon)]

    def shares_within(ba, allowed):
        r = roster[(roster.ba==ba) & roster["sub"].isin(allowed)]
        cap = r.groupby("sub").nameplate_capacity.sum()
        if cap.sum() > 0:
            return (cap/cap.sum()).to_dict()
        return {s: 1.0/len(allowed) for s in allowed}   # uniform fallback

    alloc = {}   # ba -> {sub: weight}
    for ba, sub in SINGLE.items():
        alloc[ba] = {sub: 1.0}
    for ba, allowed in SPLIT.items():
        alloc[ba] = shares_within(ba, allowed)
    for ba in WEST_BAS:
        alloc[ba] = shares_within(ba, WEST)
    # SPA: full mask over whatever subs its plants map to
    r = roster[(roster.ba=='SPA') & roster["sub"].notna()]
    cap = r.groupby("sub").nameplate_capacity.sum()
    alloc['SPA'] = (cap/cap.sum()).to_dict() if cap.sum()>0 else {'SPP_South':1.0}
    return alloc

# ---------- load sources into BA-level hourly series (tz-naive UTC) ----------
def load_eia930():
    dset = ds.dataset(f"{REAL}/eia930_hourly_netgen_by_source.parquet", format="parquet")
    t = dset.to_table(columns=["datetime_utc","balancing_authority_code_eia","generation_energy_source","net_generation_reported_mwh"],
                      filter=(ds.field("generation_energy_source").isin(["hydro","hydro_excluding_pumped_storage"])))
    df = t.to_pandas()
    df = df[df.net_generation_reported_mwh.notna()]
    df.loc[df.net_generation_reported_mwh.abs() > 1e5, "net_generation_reported_mwh"] = np.nan  # kill corrupt (2.58e9 BANC etc)
    df = df.dropna(subset=["net_generation_reported_mwh"])
    df["dt"] = pd.to_datetime(df.datetime_utc)
    # stitch: hydro (<2024-07-01) + hydro_excl_ps (>=), average any tiny seam overlap
    g = df.groupby(["balancing_authority_code_eia","dt"], as_index=False).net_generation_reported_mwh.mean()
    piv = g.pivot(index="dt", columns="balancing_authority_code_eia", values="net_generation_reported_mwh")
    return piv

def loc_to_utc(idx, tz):
    return (pd.DatetimeIndex(idx).tz_localize(tz, ambiguous='NaT', nonexistent='shift_forward')
              .tz_convert('UTC').tz_localize(None))

def load_dedicated():
    out = {}
    # BPA -> BPAT (UTC naive)
    b = pd.read_parquet(f"{REAL}/bpa/bpa_hydro_hourly_2007_2026_utc.parquet")
    out['BPAT'] = pd.Series(b.hydro_mw.values, index=pd.DatetimeIndex(b.dt_utc)).sort_index()
    # NYISO -> NYIS (UTC naive)
    n = pd.read_parquet(f"{REAL}/nyiso/nyiso_hydro_hourly.parquet")
    out['NYIS'] = pd.Series(n.hydro_mw.values, index=pd.DatetimeIndex(n.datetime_utc)).sort_index()
    # SPP -> SWPP (tz-aware UTC)
    s = pd.read_parquet(f"{REAL}/spp/SPP_hydro_hourly_2014_2025.parquet")
    out['SWPP'] = pd.Series(s.hydro_mw.values, index=pd.DatetimeIndex(s.datetime_utc).tz_convert('UTC').tz_localize(None)).sort_index()
    # CAISO -> CISO (Pacific local naive -> UTC)
    c = pd.read_parquet(f"{REAL}/caiso/caiso_hydro_hourly.parquet")
    ci = loc_to_utc(c.datetime, 'America/Los_Angeles')
    cc = pd.Series(c.hydro_MW.values, index=ci); cc = cc[cc.index.notna()].sort_index()
    out['CISO'] = cc[~cc.index.duplicated(keep='first')]
    # ERCOT -> ERCO (Central local naive -> UTC)
    e = pd.read_parquet(f"{REAL}/ercot/ercot_hydro_hourly_2007_2026.parquet")
    ei = loc_to_utc(e.datetime_central, 'America/Chicago')
    ee = pd.Series(e.hydro_mw.values, index=ei); ee = ee[ee.index.notna()].sort_index()
    out['ERCO'] = ee[~ee.index.duplicated(keep='first')]
    for k in out:
        out[k] = out[k].clip(lower=-1e5, upper=1e5)
    return out

def main():
    alloc = build_alloc()
    print("=== ALLOCATION (ba -> sub: weight) ===")
    for ba in sorted(alloc):
        print(f"  {ba:6s} -> " + ", ".join(f"{s}:{w:.2f}" for s,w in sorted(alloc[ba].items(), key=lambda x:-x[1])))

    eia = load_eia930()
    ded = load_dedicated()
    print("\n=== EIA930 BA span:", eia.index.min(), "->", eia.index.max(), "nBA", eia.shape[1])
    for k,v in ded.items():
        print(f"  dedicated {k}: {v.index.min()} -> {v.index.max()} n={len(v)}")

    # common hourly UTC grid = union of all
    tmin = min([eia.index.min()] + [v.index.min() for v in ded.values()])
    tmax = max([eia.index.max()] + [v.index.max() for v in ded.values()])
    grid = pd.date_range(tmin.floor('h'), tmax.ceil('h'), freq='h')
    print(f"\n=== UTC grid {grid[0]} -> {grid[-1]}  ({len(grid)} hours)")

    # BA-level combined table: dedicated priority, EIA930 fill
    all_bas = sorted(set(eia.columns) | set(ded.keys()))
    ba_tab = pd.DataFrame(index=grid, columns=all_bas, dtype=float)
    for ba in eia.columns:
        ba_tab[ba] = eia[ba].reindex(grid)
    for ba, s in ded.items():
        s = s.reindex(grid)
        base = ba_tab[ba] if ba in ba_tab else pd.Series(np.nan, index=grid)
        ba_tab[ba] = s.where(s.notna(), base)   # dedicated where present else EIA930

    # per-BA robust despike: isolated corrupt spikes/huge negatives (EIA-930 has records
    # up to 2.58e9 MWh). Legit BAs have max~1.1*p999; corrupt spikes are 10-700x p99.
    # cap = max(5*p99, 500) preserves legit peaks, kills order-of-magnitude outliers.
    n_clip = 0
    for ba in ba_tab.columns:
        x = ba_tab[ba]
        p99 = np.nanpercentile(x.values, 99) if x.notna().any() else 0.0
        cap = max(5*abs(p99), 500.0)
        bad = x.abs() > cap
        n_clip += int(bad.sum())
        ba_tab.loc[bad, ba] = np.nan
    print(f"despike: {n_clip} corrupt BA-hours set NaN")

    # allocate BA -> subregion
    sub_tab = pd.DataFrame(index=grid, columns=SUBS, dtype=float)
    contrib_frames = {s: [] for s in SUBS}
    for ba, wmap in alloc.items():
        if ba not in ba_tab: continue
        col = ba_tab[ba]
        for sub, w in wmap.items():
            contrib_frames[sub].append(col * w)
    for sub in SUBS:
        if contrib_frames[sub]:
            sub_tab[sub] = pd.concat(contrib_frames[sub], axis=1).sum(axis=1, min_count=1)

    hydro = sub_tab[SUBS].to_numpy(dtype=float).T   # (n_sub, n_hours)
    times = grid.values.astype('datetime64[s]')

    # source label per subregion
    src = {}
    for sub in SUBS:
        if sub=='CAISO': src[sub]='CAISO feed(2018-04)+EIA930(CISO/BANC/LDWP/IID/TIDC)'
        elif sub=='ERCOT': src[sub]='ERCOT feed(2007)'
        elif sub=='NYISO': src[sub]='NYISO feed(2015-12)'
        elif sub=='ISONE': src[sub]='EIA930(ISNE,2018-07)'
        elif sub=='NorthernGrid_West': src[sub]='BPA feed(2007)+EIA930(PACW/PGE/PSEI/SCL/TPWR/...)'
        elif sub in ('SPP_North','SPP_South'): src[sub]='SPP feed(2014)+EIA930(SWPP/SPA), capacity-split'
        elif sub in ('PJM_East','PJM_West'): src[sub]='EIA930(PJM,2018-07), capacity-split'
        elif sub.startswith('MISO'): src[sub]='EIA930(MISO,2018-07), capacity-split'
        elif sub=='FRCC': src[sub]='EIA930(FPC/TAL,2018-07) - tiny'
        elif sub=='SERTP': src[sub]='EIA930(SOCO/TVA/DUK/CPLE/SC/SCEG/LGEE/...,2018-07)'
        else: src[sub]='EIA930(western BAs,2018-07), capacity-split'

    np.savez(f"{REAL}/subregion_hydro_hourly_real.npz",
             hydro=hydro.astype(np.float32), times=times,
             subregions=np.array(SUBS), source_per_sub=np.array([src[s] for s in SUBS]))
    print(f"\nSAVED {REAL}/subregion_hydro_hourly_real.npz  hydro{hydro.shape}")

    # ---------- dispatch shape: normalized hour-of-day x month per subregion ----------
    prof = np.full((len(SUBS), 24, 12), np.nan, dtype=float)
    ndays = np.zeros((len(SUBS), 12), dtype=int)
    for si, sub in enumerate(SUBS):
        s = pd.Series(hydro[si], index=grid).dropna()
        if s.empty: continue
        loc = s.copy(); loc.index = pd.DatetimeIndex(s.index).tz_localize('UTC').tz_convert(TZ[sub])
        v = loc.clip(lower=0.0)
        df = pd.DataFrame({'v': v.values, 'hod': loc.index.hour, 'm': loc.index.month,
                           'day': loc.index.normalize()})
        # per-day fraction, require near-full day and positive total
        daytot = df.groupby('day').v.transform('sum')
        daycnt = df.groupby('day').v.transform('count')
        df = df[(daytot > 0) & (daycnt >= 20)]
        df['frac'] = df.v / df.groupby('day').v.transform('sum')
        gm = df.groupby(['m','hod']).frac.mean().unstack('hod')  # rows=month, cols=hod
        for m in range(1,13):
            if m in gm.index:
                row = gm.loc[m].reindex(range(24)).to_numpy()
                if np.nansum(row) > 0:
                    prof[si,:,m-1] = row/np.nansum(row)
                    ndays[si,m-1] = df[df.m==m].day.nunique()
    np.savez(f"{REAL}/dispatch_shape.npz",
             profile=prof.astype(np.float32), subregions=np.array(SUBS),
             hod=np.arange(24), month=np.arange(1,13), n_days=ndays)
    print(f"SAVED {REAL}/dispatch_shape.npz  profile{prof.shape}")

    # ---------- validation ----------
    print("\n================ VALIDATION ================")
    sdf = pd.DataFrame(hydro.T, index=grid, columns=SUBS)
    print("\nPer-subregion: first_real_hour  last  mean_MW  peak_MW  %hours-with-data")
    n = len(grid)
    for sub in SUBS:
        col = sdf[sub]
        nz = col.dropna()
        fr = nz.index.min() if len(nz) else None
        print(f"  {sub:20s} {str(fr):19s} {str(nz.index.max()):19s} "
              f"{col.mean():8.1f} {col.max():9.1f}  {100*col.notna().mean():5.1f}%  shape_days_min={ndays[SUBIDX[sub]].min()}")

    # fleet TWh for representative full years
    for yr in [2019, 2022, 2023]:
        yv = sdf[(sdf.index>=f'{yr}-01-01') & (sdf.index<f'{yr+1}-01-01')]
        twh = yv.sum().sum()/1e6   # MWh -> TWh (1h steps)
        cover = (yv.notna().mean()*100).round(0).astype(int)
        print(f"\nFLEET {yr}: {twh:6.1f} TWh/yr over 18 subregions ({len(yv)} hrs)  "
              f"per-sub coverage%: {dict(cover)}")
        pos = yv.clip(lower=0).sum()/1e6
        print(f"   per-sub TWh {yr}: " + ", ".join(f"{s}:{pos[s]:.1f}" for s in SUBS))

    shape_ok = int((ndays.min(axis=1) >= 20).sum())
    print(f"\nSubregions with usable dispatch shape (>=20 days every month): {shape_ok}/18")

if __name__ == "__main__":
    main()
