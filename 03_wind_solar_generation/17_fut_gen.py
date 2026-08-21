"""R4 FUTURE-GENERATION pass: apply the VALIDATED deployment pipeline
(3D-SRGAN hub wind -> PySAM Windpower w/ 15% turb loss + actual T2/PSFC density ;
 native-12km SWDOWN -> PVWatts) to FUTURE TGW weather sampled at the CERF-SITED
future fleets. Produces per-subregion hourly wind & solar generation for the 8
canonical scenarios (rcp{45,85}{cooler,hotter}_ssp{3,5}).

Weather is per-CLIMATE (4). Each climate serves 2 ssp fleets. So downscale each
climate ONCE (union of both ssp fleets' plant cells), then aggregate per fleet.

PHASES (env PHASE):
  wind_infer  /opt/pytorch (GPU)  : SRGAN 4km hub wind at union wind sites + native T2/PSFC
                                    -> fut_wind_wx_{CLIMATE}.npz  (per-site hub speed, T2, PSFC)
  wind_pysam  /data/genenv        : PySAM Windpower (shear=0, config turb_generic_loss=15,
                                    actual per-site T2/PSFC) -> fut_wind_cf_{CLIMATE}.npz
  solar_gen   /data/genenv        : native SWDOWN/T2/wspd/PSFC at union solar sites -> PVWatts
                                    (per-year 8760/8784 embed) -> fut_solar_cf_{CLIMATE}.npz
  aggregate   /data/genenv        : map fleets -> site CF, cap*CF -> per-subregion hourly gen
                                    -> /data/cerf_out/futuregen_{scenario}.npz  (x8)

ENV: PHASE(req), CLIMATE(rcp45cooler|...), YEARS(2030-2050), TSTRIDE(2 wind 6-hourly),
     BATCH(8), TAG(=CLIMATE), NPROC(10), SMOKE_YEAR(optional single year override).
Byte-for-byte identical LR build / regrid / z-score / G / hr-denorm as future_downscale.py.
Read-only on /data/tgw_3d_future + /data/ckpt + /data/datasets + /data/cerf_out/fleet_*.csv.
Writes: /data/gen_targets/srgan3d_val/futgen/*  and /data/cerf_out/futuregen_*.npz
"""
import os, sys, glob, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "02_downscale_wind"))

PHASE   = os.environ["PHASE"]
CLIMATE = os.environ.get("CLIMATE", "rcp45cooler")
YEARS   = os.environ.get("YEARS", "2030-2050")
TSTRIDE = int(os.environ.get("TSTRIDE", "2"))
BATCH   = int(os.environ.get("BATCH", "8"))
TAG     = os.environ.get("TAG", CLIMATE)
NPROC   = int(os.environ.get("NPROC", "10"))
NW      = int(os.environ.get("NW", "8"))       # wind_infer CPU loader/regrid workers (parallel, overlap GPU)
y0, y1  = [int(x) for x in YEARS.split("-")]

D3F   = os.environ.get("D3F", "/data/tgw_3d_future")
GRID  = "/data/datasets/grid"
TRN   = "/data/datasets/train"
GC    = "/data/datasets/gen/tgw-gen-historical"
CERF  = "/data/cerf_out"
OUT   = os.environ.get("OUT", "/data/gen_targets/srgan3d_val/futgen"); os.makedirs(OUT, exist_ok=True)
CKPT  = os.environ.get("CKPT", "/data/ckpt/g3d_full/G_deploy.pth")
# FOUR fleet variants sampled in ONE downscaling pass (union of sites). {scenario} -> rcp45cooler_ssp3.
# Each variant -> its own futuregen_{variant}_{scenario}.npz. Paths overridable via env.
VARIANTS = {
    "nopolicy": os.environ.get("FLEET_TMPL_NOPOLICY", f"{CERF}/fleet_{{scenario}}.csv"),
    "policy":   os.environ.get("FLEET_TMPL_POLICY",   f"{CERF}/fleet_policy_{{scenario}}.csv"),
    "obbba":    os.environ.get("FLEET_TMPL_OBBBA",    f"{CERF}/fleet_obbba_{{scenario}}.csv"),
    # 4th arm: county ordinances/bans/setbacks/prime-farmland ON, RPS/CES siting pull OFF
    # (07_cerf_policy.py run with RPS_ALPHA=0 OUTTAG=_ordonly). Same GCAM capacity as the policy arm;
    # only the siting differs -> isolates the local-restriction mechanism from the incentive pull.
    "ordonly":  os.environ.get("FLEET_TMPL_ORDONLY",  f"{CERF}/fleet_policy_{{scenario}}_ordonly.csv"),
}
HEIGHTS = np.array([10, 40, 80, 110, 140, 160, 200], np.float32)

# ---- OFFSHORE WIND hook (mandate-driven; guarded by if offshore_fleet_{climate}.csv exists) ----
# Offshore is per-CLIMATE (one file serves both ssp fleets + all 3 policy variants: state law is a
# binding floor). Offshore cells join the WIND union so SRGAN+PySAM give real hourly CF; offshore
# sites use a 15 MW IEA-class turbine (hub 150 m + offshore power curve) instead of nearest-EIA-onshore.
OFFSHORE_HUB = 150.0
def _iea15_curve():
    ws = [round(x * 0.5, 2) for x in range(0, 61)]          # 0..30 m/s @ 0.5
    rated, cin, cr, cout = 15000.0, 3.0, 10.59, 25.0        # IEA-15MW-RWT class: cut-in3/rated10.59/cut-out25
    pw = []
    for v in ws:
        if v < cin or v > cout:
            p = 0.0
        elif v < cr:
            p = rated * ((v ** 3 - cin ** 3) / (cr ** 3 - cin ** 3))   # cubic ramp to rated
        else:
            p = rated
        pw.append(round(p, 2))
    return ws, pw
OFFSHORE_WS, OFFSHORE_PW = _iea15_curve()

SCENS = {"rcp45cooler": ["rcp45cooler_ssp3", "rcp45cooler_ssp5"],
         "rcp45hotter": ["rcp45hotter_ssp3", "rcp45hotter_ssp5"],
         "rcp85cooler": ["rcp85cooler_ssp3", "rcp85cooler_ssp5"],
         "rcp85hotter": ["rcp85hotter_ssp3", "rcp85hotter_ssp5"]}

# ---------------- shared: 4km rectilinear grid + native curvilinear grid ----------------
co  = np.load(f"{GRID}/coordinate.npz"); LAT = co["lat"].astype("float64"); LON = co["lon"].astype("float64")
assert LAT.shape == (625,) and LON.shape == (1475,)
_g  = np.load(f"{D3F}/tgw3d_grid.npz"); XLAT = _g["XLAT"].astype("float64"); XLONG = _g["XLONG"].astype("float64")
XLATf = XLAT.ravel(); XLONGf = XLONG.ravel()

def cell_hr(lat, lon):                       # nearest 4km (625x1475) rectilinear cell -> flat idx
    ih = np.abs(LAT[None, :] - np.asarray(lat, float)[:, None]).argmin(1)
    iw = np.abs(LON[None, :] - np.asarray(lon, float)[:, None]).argmin(1)
    return ih * 1475 + iw

def native_cell(lat, lon):                   # nearest native 12km (299x424) curvilinear cell -> flat idx
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    out = np.empty(len(lat), np.int64)
    for k in range(len(lat)):
        out[k] = np.argmin((XLATf - lat[k]) ** 2 + (XLONGf - lon[k]) ** 2)
    return out

# ---------------- fleet -> union sites per tech (across ALL variants x both ssp fleets) -----
def active_variants(climate):
    """variants whose fleet CSVs exist for BOTH ssp scenarios of this climate."""
    out = {}
    for v, tmpl in VARIANTS.items():
        if all(os.path.exists(tmpl.format(scenario=sc)) for sc in SCENS[climate]):
            out[v] = tmpl
    return out

def load_fleet(scenario, tech, tmpl):
    d = pd.read_csv(tmpl.format(scenario=scenario))
    d = d[(d.sited_year <= 2050) & (d.tech == tech)].copy()          # fixed-2050 fleet
    d = d[(d.retirement_year.isna()) | (d.retirement_year > 2050)]   # operating in 2050
    d = d[np.isfinite(d.lon) & np.isfinite(d.lat)]
    d = d[(d.lat >= LAT.min()) & (d.lat <= LAT.max()) & (d.lon >= LON.min()) & (d.lon <= LON.max())]
    d["key"] = list(zip(d.lon.round(6), d.lat.round(6)))
    return d.reset_index(drop=True)

def union_sites(climate, tech):
    keys = {}
    av = active_variants(climate)
    for v, tmpl in av.items():
        for sc in SCENS[climate]:
            d = load_fleet(sc, tech, tmpl)
            for k, lo, la in zip(d.key, d.lon, d.lat):
                if k not in keys: keys[k] = (float(lo), float(la))
    ks = list(keys.keys()); lon = np.array([keys[k][0] for k in ks]); lat = np.array([keys[k][1] for k in ks])
    print(f"[union_sites] {climate} {tech}: {len(ks)} unique sites over variants {list(av)}", flush=True)
    return ks, lon, lat

def assign_nearest_config(site_lat, site_lon, cfg_df):
    """nearest EIA config plant_code_unique per site (lat/lon KDTree in degrees)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([cfg_df.lat.values, cfg_df.lon.values]))
    _, idx = tree.query(np.column_stack([site_lat, site_lon]))
    return cfg_df.plant_code_unique.values[idx], idx

def load_wind_cfg():
    c = pd.read_csv(f"{GC}/eia_wind_configs.csv", dtype={"plant_code_unique": str}).drop_duplicates("plant_code_unique")
    return c[np.isfinite(c.lat) & np.isfinite(c.lon)].reset_index(drop=True)

def load_solar_cfg():
    c = pd.read_csv(f"{GC}/eia_solar_configs.csv", dtype={"plant_code_unique": str}).drop_duplicates("plant_code_unique")
    return c[np.isfinite(c.lat) & np.isfinite(c.lon)].reset_index(drop=True)

# ---------------- offshore fleet (per-climate, mandate-driven) -----------------------------
def offshore_path(climate):
    return os.environ.get("OFFSHORE_TMPL", f"{CERF}/offshore_fleet_{{climate}}.csv").format(climate=climate)

def has_offshore(climate):
    return os.path.exists(offshore_path(climate))

def load_offshore(climate):
    """offshore_wind plants for this climate (same 14-col schema as onshore fleets)."""
    d = pd.read_csv(offshore_path(climate))
    d = d[(d.tech == "offshore_wind") & (d.sited_year <= 2050)].copy()
    d = d[(d.retirement_year.isna()) | (d.retirement_year > 2050)]
    d = d[np.isfinite(d.lon) & np.isfinite(d.lat)]
    d = d[(d.lat >= LAT.min()) & (d.lat <= LAT.max()) & (d.lon >= LON.min()) & (d.lon <= LON.max())]
    d["key"] = list(zip(d.lon.round(6), d.lat.round(6)))
    return d.reset_index(drop=True)

# ---------------- stamp selection (dedup across overlapping weekly files) -----------------
def sel_files(climate):
    fs = sorted(glob.glob(f"{D3F}/tgw_wrf_{climate}_three_hourly_*.npz"))
    def fy(fp): return int(os.path.basename(fp).split("_three_hourly_")[1][:4])
    return [f for f in fs if y0 <= fy(f) <= y1]

def select_stamps(climate, tstride):
    sset = set(); seen = []
    for fp in sel_files(climate):
        z = np.load(fp, allow_pickle=True); tms = [str(t) for t in z["times"]]; z.close()
        for s in tms:
            if y0 <= int(s[:4]) <= y1 and s not in sset: sset.add(s); seen.append(s)
    seen = sorted(seen); return seen[::tstride]

# ================================================================= WIND INFER (GPU)
if PHASE == "wind_infer":
    import torch, torch.nn.functional as F
    from model.srgan import SRGAN_g
    assert torch.cuda.is_available(); dev = "cuda"; torch.backends.cudnn.benchmark = True

    # area-average regrid TGW 299x424 -> LR 89x211 (verbatim future_downscale.py)
    def pool1d(a, n): return F.adaptive_avg_pool1d(torch.tensor(np.asarray(a, dtype='float64'))[None, None], n)[0, 0].numpy()
    def cell_edges(c): m = 0.5 * (c[1:] + c[:-1]); return np.concatenate([[2 * c[0] - m[0]], m, [2 * c[-1] - m[-1]]])
    LATLR = pool1d(co['lat'], 89); LONLR = pool1d(co['lon'], 211); LATE = cell_edges(LATLR); LONE = cell_edges(LONLR)
    NLAT, NLON = 89, 211
    _bil = np.digitize(XLATf, LATE) - 1; _bio = np.digitize(XLONGf, LONE) - 1
    VALID = (_bil >= 0) & (_bil < NLAT) & (_bio >= 0) & (_bio < NLON)
    CELL = (_bil * NLON + _bio)[VALID]; COUNTS = np.bincount(CELL, minlength=NLAT * NLON).astype('float64'); NZ = COUNTS > 0
    def regrid(arr):
        C = arr.shape[0]; out = np.full((C, NLAT * NLON), np.nan); af = arr.reshape(C, -1)[:, VALID]
        for ch in range(C):
            s = np.bincount(CELL, weights=af[ch], minlength=NLAT * NLON); out[ch, NZ] = s[NZ] / COUNTS[NZ]
        return out.reshape(C, NLAT, NLON)
    def _interp(arr, z, h):
        NLz = arr.shape[0]
        kk = np.clip((z <= h).sum(0) - 1, 0, NLz - 2)[None]
        z0 = np.take_along_axis(z, kk, 0)[0]; z1 = np.take_along_axis(z, kk + 1, 0)[0]
        a0 = np.take_along_axis(arr, kk, 0)[0]; a1 = np.take_along_axis(arr, kk + 1, 0)[0]
        w = (h - z0) / np.where(z1 > z0, z1 - z0, 1.0)
        out = a0 + w * (a1 - a0)
        out = np.where(h < z[0], arr[0], out); out = np.where(h >= z[-1], np.nan, out)
        return out.astype(np.float32)
    def build_lr(uv_t, zagl_t, surf_t, pwat_t):
        u = uv_t[:, 0]; v = uv_t[:, 1]; wind = []
        for h in HEIGHTS:
            wind.append(_interp(u, zagl_t, h)); wind.append(_interp(v, zagl_t, h))
        wind = np.stack(wind)
        ctx = np.stack([surf_t[0], surf_t[1], surf_t[2], surf_t[3], pwat_t / 1000.0, surf_t[6], surf_t[5]])
        return np.concatenate([wind, ctx], 0)

    _S = np.load(f"{GRID}/static_terrain_landmask.npz")
    hgt = _S["hgt"].astype("float32"); lmask = _S["landmask"].astype("float32"); lake = _S["lakemask"].astype("float32")
    island = ((lmask > 0.5) & (lake < 0.5)).astype("float32"); hgtz = (hgt - hgt.mean()) / (hgt.std() + 1e-6)
    static_hr = np.stack([hgtz, island * 2 - 1], 0)
    static_lr = F.adaptive_avg_pool2d(torch.from_numpy(static_hr)[None], (89, 211))[0].numpy()
    st = np.load(f"{TRN}/norm_stats_3d.npz")
    lr_mean = st["lr_mean"].astype(np.float32).reshape(21, 1, 1); lr_std = st["lr_std"].astype(np.float32).reshape(21, 1, 1)

    # union wind sites for this climate + nearest-EIA config (for hub height & pysam)
    if CLIMATE == "historical":
        # observed fixed fleet: the same 1,395 EIA plants the historical analysis has always used,
        # filtered to those present in the published historical CF so the two are directly comparable.
        _c = pd.read_csv(f"{GC}/eia_wind_configs.csv", dtype={"plant_code_unique": str}).drop_duplicates("plant_code_unique")
        _c = _c[np.isfinite(_c.lat) & np.isfinite(_c.lon)]
        _g = np.load("/data/gen_targets/wind_cf_1980_2019.npz", allow_pickle=True)
        _keep = set(_g["plants"].astype(str)); _g.close()
        _c = _c[_c.plant_code_unique.isin(_keep)].reset_index(drop=True)
        ks = list(_c.plant_code_unique.values); slon = _c.lon.values.astype(float); slat = _c.lat.values.astype(float)
        is_off = np.zeros(len(ks), bool)
        print(f"[wind_infer] historical fixed fleet: {len(ks)} EIA plants", flush=True)
    else:
        ks, slon, slat = union_sites(CLIMATE, "wind")
        is_off = np.zeros(len(ks), bool)
    if CLIMATE != "historical" and has_offshore(CLIMATE):                 # append offshore-wind cells to the wind union (guarded)
        od = load_offshore(CLIMATE); exist = set(ks); ak = []; alo = []; ala = []
        for k, lo, la in zip(od.key, od.lon.values, od.lat.values):
            if k not in exist:
                exist.add(k); ak.append(k); alo.append(float(lo)); ala.append(float(la))
        if ak:
            ks = ks + ak
            slon = np.concatenate([slon, np.array(alo)]); slat = np.concatenate([slat, np.array(ala)])
            is_off = np.concatenate([is_off, np.ones(len(ak), bool)])
        print(f"[wind_infer] +offshore: {len(ak)} new offshore cells appended -> {len(ks)} union sites", flush=True)
    wcfg = load_wind_cfg(); cfg_pcu, cfg_idx = assign_nearest_config(slat, slon, wcfg)
    hub = wcfg.wind_turbine_hub_ht.values[cfg_idx].astype(np.float32)
    hub[is_off] = OFFSHORE_HUB                # 15 MW offshore turbine hub height (interp 140<->160 m SRGAN levels)
    hr4 = cell_hr(slat, slon)                 # 4km flat cell per site
    natc = native_cell(slat, slon)            # native flat cell per site (for T2/PSFC)
    nsite = len(ks)
    # precompute hub interp bracket (lo,hi,w) along HEIGHTS
    hi = np.clip(np.searchsorted(HEIGHTS, hub), 1, len(HEIGHTS) - 1); lo = hi - 1
    wgt = (hub - HEIGHTS[lo]) / (HEIGHTS[hi] - HEIGHTS[lo])
    ar = np.arange(nsite)

    SEL = select_stamps(CLIMATE, TSTRIDE); NH = len(SEL); pos = {s: i for i, s in enumerate(SEL)}
    print(f"[wind_infer] CLIMATE={CLIMATE} YEARS={YEARS} TSTRIDE={TSTRIDE} -> {NH} frames  "
          f"{nsite} union wind sites  BATCH={BATCH} NW={NW}  ckpt={CKPT}", flush=True)
    wsphub = np.full((nsite, NH), np.nan, np.float32)
    T2 = np.full((nsite, NH), np.nan, np.float32); PS = np.full((nsite, NH), np.nan, np.float32)
    done = np.zeros(NH, bool)

    # ---- CPU worker (parallel across cores): load npz + build_lr + regrid + zscore + native T2/PSFC.
    #      Pure numpy (no torch/CUDA). Returns per-frame (j, zc[21,89,211], t2v[nsite], psv[nsite]). ----
    def process_file(fp):
        z = np.load(fp, allow_pickle=True); tms = [str(t) for t in z["times"]]
        need = [(ti, s) for ti, s in enumerate(tms) if s in pos]
        if not need: z.close(); return []
        uv = z["uv"].astype(np.float32); zagl = z["zagl"].astype(np.float32)
        surf = z["surf"].astype(np.float32); pwat = z["pwat"].astype(np.float32); z.close()
        res = []
        for ti, s in need:
            j = pos[s]; sfc = surf[ti]
            t2v = sfc[2].ravel()[natc].astype(np.float32); psv = sfc[3].ravel()[natc].astype(np.float32)  # T2(K), PSFC(hPa)
            lrr = regrid(build_lr(uv[ti], zagl[ti], surf[ti], pwat[ti]))
            zc = np.nan_to_num((lrr - lr_mean) / lr_std, nan=0.0).astype(np.float32)
            res.append((j, zc, t2v, psv))
        return res

    # ---- fork the CPU worker pool BEFORE any CUDA init (workers never call torch/CUDA) ----
    import multiprocessing as mp
    from collections import deque
    pool = mp.get_context("fork").Pool(NW)

    # ---- NOW init CUDA + model in the MAIN process only ----
    G = SRGAN_g(in_channels=21, out_channels=14, crop_size=(625, 1475),
                static_lr=static_lr, static_hr=static_hr, dyn_lr_ch=0, dyn_hr_ch=0)
    G.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=True); G.eval().to(dev)
    hr_mean_t = torch.from_numpy(st["hr_mean"].astype(np.float32).reshape(1, 14, 1, 1)).to(dev)
    hr_std_t  = torch.from_numpy(st["hr_std"].astype(np.float32).reshape(1, 14, 1, 1)).to(dev)
    hr4_t = torch.from_numpy(hr4).to(dev)
    batch = []; jbuf = []; t0 = time.time(); nproc = [0]; nextrep = [2000]; diag = [True]

    def flush():
        if not batch: return
        x = torch.from_numpy(np.stack(batch)).to(dev)
        with torch.no_grad():
            y = G(x) * hr_std_t + hr_mean_t                    # [B,14,625,1475]
            u = y[:, 0::2]; v = y[:, 1::2]                     # [B,7,625,1475]
            sp = torch.sqrt(u * u + v * v).reshape(len(batch), 7, -1)[:, :, hr4_t]  # [B,7,nsite]
            spc = sp.cpu().numpy()
        for bi, j in enumerate(jbuf):
            s7 = spc[bi]                                       # (7,nsite)
            wsphub[:, j] = s7[lo, ar] * (1 - wgt) + s7[hi, ar] * wgt
            nproc[0] += 1
        batch.clear(); jbuf.clear()
        if nproc[0] >= nextrep[0]:
            el = time.time() - t0
            print(f"  {nproc[0]}/{NH}  {el:.0f}s ({el/max(nproc[0],1):.3f}s/frame)", flush=True); nextrep[0] += 2000

    # ---- windowed producer/consumer: bounded in-flight files -> CPU workers overlap the GPU forward ----
    files = sel_files(CLIMATE); it = iter(files); inflight = deque()
    for _ in range(NW + 3):
        try: inflight.append(pool.apply_async(process_file, (next(it),)))
        except StopIteration: break
    while inflight:
        res = inflight.popleft().get()
        try: inflight.append(pool.apply_async(process_file, (next(it),)))
        except StopIteration: pass
        for (j, zc, t2v, psv) in res:
            if done[j]: continue                               # dedup overlapping weekly-file stamps
            done[j] = True; T2[:, j] = t2v; PS[:, j] = psv
            if diag[0]:
                print(f"[diag] frame0 t2 {t2v.mean():.1f}K psfc {psv.mean():.1f}hPa "
                      f"zc_finite {np.isfinite(zc).mean():.3f} zc|mean| {np.abs(zc).mean():.3f}", flush=True); diag[0] = False
            batch.append(zc); jbuf.append(j)
            if len(batch) >= BATCH: flush()
    flush()
    pool.close(); pool.join()
    np.savez(f"{OUT}/fut_wind_wx_{TAG}.npz",
             lon=slon, lat=slat, hub=hub, cfg_pcu=cfg_pcu.astype(str), hr4=hr4, natc=natc,
             stamps=np.array(SEL), wsphub=wsphub, t2=T2, psfc=PS, done=done, is_off=is_off,
             climate=CLIMATE, years=YEARS, tstride=TSTRIDE, ckpt=CKPT)
    print(f"[wind_infer] DONE {int(done.sum())}/{NH}  wsphub mean {np.nanmean(wsphub):.2f} m/s  "
          f"T2 mean {np.nanmean(T2):.1f}K  PSFC mean {np.nanmean(PS):.1f}hPa  {time.time()-t0:.0f}s "
          f"-> {OUT}/fut_wind_wx_{TAG}.npz", flush=True)

# ================================================================= WIND PYSAM (genenv)
elif PHASE == "wind_pysam":
    from gen_physics import wind_cf
    import multiprocessing as mp
    Z = np.load(f"{OUT}/fut_wind_wx_{TAG}.npz", allow_pickle=True)
    done = np.asarray(Z["done"], bool)
    stamps = Z["stamps"].astype(str)[done]; n = len(stamps)
    slon = Z["lon"]; slat = Z["lat"]; hub = Z["hub"].astype(float); cfg_pcu = Z["cfg_pcu"].astype(str)
    WH = Z["wsphub"][:, done]; T2C = (Z["t2"][:, done] - 273.15).astype(np.float32); PMB = Z["psfc"][:, done].astype(np.float32)
    nsite = len(slon)
    IS_OFF = np.asarray(Z["is_off"], bool) if "is_off" in Z.files else np.zeros(nsite, bool)
    cfg = pd.read_csv(f"{GC}/eia_wind_configs.csv", dtype={"plant_code_unique": str}).drop_duplicates("plant_code_unique").set_index("plant_code_unique")
    L = ((n + 8759) // 8760) * 8760
    FULLy = pd.date_range("2015-01-01", periods=L, freq="h")
    _G = {"wh": WH, "t2": T2C, "pmb": PMB}
    def one(pi):
        if IS_OFF[pi]:                        # 15 MW IEA-class offshore turbine (hub 150 m, offshore curve)
            ws, pw = OFFSHORE_WS, OFFSHORE_PW; tgl = 15.0
            base = dict(wind_turbine_hub_ht=OFFSHORE_HUB, wind_turbine_rotor_diameter=240.0,
                        system_capacity=15000.0, wind_turbine_powercurve_windspeeds=list(ws),
                        wind_turbine_powercurve_powerout=list(pw))
        else:
            c = cfg.loc[cfg_pcu[pi]]
            ws = eval(c.wind_turbine_powercurve_windspeeds) if isinstance(c.wind_turbine_powercurve_windspeeds, str) else c.wind_turbine_powercurve_windspeeds
            pw = eval(c.wind_turbine_powercurve_powerout) if isinstance(c.wind_turbine_powercurve_powerout, str) else c.wind_turbine_powercurve_powerout
            tgl = float(c.turb_generic_loss) if "turb_generic_loss" in c.index and pd.notna(c.turb_generic_loss) else 15.0
            base = dict(wind_turbine_hub_ht=float(c.wind_turbine_hub_ht), wind_turbine_rotor_diameter=float(c.wind_turbine_rotor_diameter),
                        system_capacity=float(max(pw)), wind_turbine_powercurve_windspeeds=list(ws), wind_turbine_powercurve_powerout=list(pw))
        whub = np.zeros(L, np.float32); whub[:n] = np.nan_to_num(_G["wh"][pi])
        t2c = np.full(L, 15.0, np.float32); t2c[:n] = np.nan_to_num(_G["t2"][pi], nan=10.0)
        pmb = np.full(L, 1013.0, np.float32); pmb[:n] = np.nan_to_num(_G["pmb"][pi], nan=1013.0)
        cf = wind_cf(FULLy, whub, t2c, pmb, float(slat[pi]), float(slon[pi]), {**base, "shear": 0.0, "turb_generic_loss": tgl})[:n]
        return pi, np.asarray(cf, np.float32)
    CF = np.full((nsite, n), np.nan, np.float32); t0 = time.time()
    with mp.get_context("fork").Pool(NPROC, maxtasksperchild=100) as pool:
        for k, (pi, cf) in enumerate(pool.imap_unordered(one, range(nsite), chunksize=16)):
            CF[pi] = cf
            if k % 1000 == 0: print(f"  [wind_pysam] {k}/{nsite} {time.time()-t0:.0f}s", flush=True)
    np.savez(f"{OUT}/fut_wind_cf_{TAG}.npz", lon=slon, lat=slat, hub=hub, cfg_pcu=cfg_pcu, stamps=stamps, cf=CF)
    print(f"[wind_pysam] DONE n={n} nsite={nsite}  NET CF mean {np.nanmean(CF):.3f}  {time.time()-t0:.0f}s "
          f"-> {OUT}/fut_wind_cf_{TAG}.npz", flush=True)

# ================================================================= SOLAR (genenv, native+PVWatts)
elif PHASE == "solar_gen":
    from gen_physics import solar_cf
    import multiprocessing as mp
    ks, slon, slat = union_sites(CLIMATE, "solar")
    scfg = load_solar_cfg(); cfg_pcu, cfg_idx = assign_nearest_config(slat, slon, scfg)
    natc = native_cell(slat, slon); nsite = len(ks)
    SEL = select_stamps(CLIMATE, 1); NH = len(SEL); pos = {s: i for i, s in enumerate(SEL)}  # solar = all 3-hourly
    print(f"[solar_gen] CLIMATE={CLIMATE} YEARS={YEARS} -> {NH} frames (3-hourly)  {nsite} union solar sites", flush=True)
    SW = np.full((nsite, NH), np.nan, np.float32); T2 = np.full((nsite, NH), np.nan, np.float32)
    WS = np.full((nsite, NH), np.nan, np.float32); PS = np.full((nsite, NH), np.nan, np.float32)
    done = np.zeros(NH, bool); t0 = time.time()
    for fp in sel_files(CLIMATE):
        z = np.load(fp, allow_pickle=True); tms = [str(t) for t in z["times"]]
        need = [(ti, s) for ti, s in enumerate(tms) if s in pos and not done[pos[s]]]
        if not need: z.close(); continue
        surf = z["surf"].astype(np.float32); z.close()
        for ti, s in need:
            j = pos[s]; sfc = surf[ti]
            SW[:, j] = sfc[6].ravel()[natc]; T2[:, j] = sfc[2].ravel()[natc]
            WS[:, j] = np.hypot(sfc[0].ravel()[natc], sfc[1].ravel()[natc]); PS[:, j] = sfc[3].ravel()[natc]
            done[j] = True
    print(f"[solar_gen] sampled {int(done.sum())}/{NH}  SWDOWN mean {np.nanmean(SW):.1f}  {time.time()-t0:.0f}s -> PVWatts", flush=True)
    stamps = np.array(SEL)
    dt = pd.to_datetime(stamps, format="%Y%m%d%H")
    years = np.array([int(s[:4]) for s in stamps])
    # per-year hourly embed (matches validated solar_srvsnative pysam). Always 8760h:
    # PVWatts v5 skips Feb-29 and returns exactly 8760 CF, so build the grid on the actual
    # year's dates MINUS Feb-29 -> length 8760 in every year (leap or not), day-of-year kept
    # correct for the actual (leap) year. Non-leap years have no Feb-29 -> grid unchanged (byte-for-byte).
    year_grids = {}
    for yr in range(y0, y1 + 1):
        FULL = pd.date_range(f"{yr}-01-01", f"{yr}-12-31 23:00", freq="h")
        FULL = FULL[~((FULL.month == 2) & (FULL.day == 29))]
        hkey = {t.strftime("%Y%m%d%H"): k for k, t in enumerate(FULL)}
        year_grids[yr] = (FULL, hkey)
    cfgpar = scfg.set_index("plant_code_unique")
    _G = {"sw": SW, "t2": T2, "ws": WS, "ps": PS}
    def one(pi):
        c = cfgpar.loc[cfg_pcu[pi]]
        gcfg = dict(system_capacity=float(c.system_capacity), losses=float(c.losses),
                    array_type=int(c.array_type), module_type=int(c.module_type),
                    azimuth=float(c.azimuth), tilt=float(c.tilt))
        out = np.full(NH, np.nan, np.float32)
        fails = []
        for yr in range(y0, y1 + 1):
            m = np.where(years == yr)[0]
            if len(m) == 0: continue
            FULL, hkey = year_grids[yr]; H = len(FULL)
            # map each sampled stamp -> hour-of-year slot; Feb-29 frames (~8/leap-yr) have no
            # slot on the 8760 grid -> dropped (left NaN), never zeroing the rest of the year.
            loc = np.array([hkey.get(stamps[k], -1) for k in m])
            keep = loc >= 0
            mk = m[keep]; hidx = loc[keep]
            if len(mk) == 0: continue
            ghi = np.zeros(H); ghi[hidx] = np.clip(np.nan_to_num(_G["sw"][pi, mk]), 0, 1400)
            t2c = np.full(H, 15.0); t2c[hidx] = np.nan_to_num(_G["t2"][pi, mk], nan=288.0) - 273.15
            wsp = np.full(H, 2.0); wsp[hidx] = np.nan_to_num(_G["ws"][pi, mk], nan=2.0)
            prs = np.full(H, 1013.0); prs[hidx] = np.nan_to_num(_G["ps"][pi, mk], nan=1013.0)
            try:
                cf = solar_cf(FULL, ghi, t2c, wsp, prs, float(slat[pi]), float(slon[pi]), gcfg)
                out[mk] = np.clip(np.asarray(cf, np.float32)[hidx], 0, 1.05)
            except Exception as e:
                fails.append((int(yr), repr(e)))
        return pi, out, fails
    CF = np.full((nsite, NH), np.nan, np.float32); t1 = time.time()
    nfail = 0; fail_ex = []
    # maxtasksperchild kept SMALL: PySAM leaks ~17 MB per (site) inside the worker, so at
    # maxtasksperchild=100 (800 sites/worker) each of 8 workers reached ~8.6 GB PRIVATE and the
    # box OOMed at the enlarged 4-variant union. Recycling every 4 chunks (32 sites) caps it at
    # a few hundred MB. Pure process management - identical numerics, negligible fork cost.
    with mp.get_context("fork").Pool(NPROC, maxtasksperchild=4) as pool:
        for k, (pi, cf, fails) in enumerate(pool.imap_unordered(one, range(nsite), chunksize=8)):
            CF[pi] = cf
            for yr, err in fails:
                nfail += 1
                if len(fail_ex) < 20: fail_ex.append((int(pi), yr, err))
            if k % 1000 == 0: print(f"  [pvwatts] {k}/{nsite} {time.time()-t1:.0f}s", flush=True)
    if nfail:
        print(f"[solar_gen] WARNING: {nfail} PVWatts (site,year) failures (NOT silently swallowed); first {len(fail_ex)}:", flush=True)
        for pi, yr, err in fail_ex:
            print(f"    site {pi} lon={float(slon[pi]):.3f} lat={float(slat[pi]):.3f} year {yr}: {err}", flush=True)
    else:
        print("[solar_gen] PVWatts failures: 0", flush=True)
    np.savez(f"{OUT}/fut_solar_cf_{TAG}.npz", lon=slon, lat=slat, cfg_pcu=cfg_pcu.astype(str), stamps=stamps, cf=CF)
    print(f"[solar_gen] DONE n={NH} nsite={nsite}  CF mean {np.nanmean(CF):.3f}  {time.time()-t1:.0f}s "
          f"-> {OUT}/fut_solar_cf_{TAG}.npz", flush=True)

# ================================================================= AGGREGATE (genenv)
elif PHASE == "aggregate":
    # ONE shared union CF -> aggregate per fleet VARIANT x ssp scenario -> futuregen_{variant}_{scenario}.npz
    WZ = np.load(f"{OUT}/fut_wind_cf_{TAG}.npz", allow_pickle=True)
    SZ = np.load(f"{OUT}/fut_solar_cf_{TAG}.npz", allow_pickle=True)
    wkey = {(round(float(lo), 6), round(float(la), 6)): i for i, (lo, la) in enumerate(zip(WZ["lon"], WZ["lat"]))}
    skey = {(round(float(lo), 6), round(float(la), 6)): i for i, (lo, la) in enumerate(zip(SZ["lon"], SZ["lat"]))}
    w_stamps = WZ["stamps"].astype(str); s_stamps = SZ["stamps"].astype(str)
    wcf = WZ["cf"]; scf = SZ["cf"]; nw = len(w_stamps); ns = len(s_stamps)
    av = active_variants(CLIMATE)
    # canonical subregion axis: union over all active variants x ssp x tech (stable across the 6 outputs)
    subset = set()
    for v, tmpl in av.items():
        for sc in SCENS[CLIMATE]:
            for tech in ("wind", "solar"):
                subset |= set(load_fleet(sc, tech, tmpl).subregion.dropna().astype(str))
    if has_offshore(CLIMATE):
        subset |= set(load_offshore(CLIMATE).subregion.dropna().astype(str))
    subs = sorted(subset); si = {s: i for i, s in enumerate(subs)}
    # offshore-wind generation: per-CLIMATE, IDENTICAL across all variants x ssp (state-law floor).
    # offshore cells were folded into the wind union -> reuse wind-union hourly CF (wcf) at each cell.
    off_gen = np.zeros((len(subs), nw), np.float64); off_cap = np.zeros(len(subs)); miss_o = 0
    if has_offshore(CLIMATE):
        do = load_offshore(CLIMATE)
        for cap, k, sub in zip(do.capacity_mw.values, do.key, do.subregion.astype(str).values):
            i = wkey.get(k)
            if i is None: miss_o += 1; continue
            if sub not in si: continue
            off_gen[si[sub]] += cap * np.nan_to_num(wcf[i]); off_cap[si[sub]] += cap
        ocf = (off_gen.sum(0).mean() / off_cap.sum()) if off_cap.sum() > 0 else float("nan")
        print(f"[agg] OFFSHORE {CLIMATE}: cap {off_cap.sum()/1e3:.1f}GW mean {off_gen.sum(0).mean()/1e3:.1f}GW "
              f"CF {ocf:.3f} (miss {miss_o})", flush=True)
    for variant, tmpl in av.items():
        for scenario in SCENS[CLIMATE]:
            wind_gen = np.zeros((len(subs), nw), np.float64); solar_gen = np.zeros((len(subs), ns), np.float64)
            wcap = np.zeros(len(subs)); scap = np.zeros(len(subs)); miss_w = miss_s = 0
            dw = load_fleet(scenario, "wind", tmpl)
            for cap, k, sub in zip(dw.capacity_mw.values, dw.key, dw.subregion.astype(str).values):
                i = wkey.get(k)
                if i is None: miss_w += 1; continue
                wind_gen[si[sub]] += cap * np.nan_to_num(wcf[i]); wcap[si[sub]] += cap
            ds = load_fleet(scenario, "solar", tmpl)
            for cap, k, sub in zip(ds.capacity_mw.values, ds.key, ds.subregion.astype(str).values):
                i = skey.get(k)
                if i is None: miss_s += 1; continue
                solar_gen[si[sub]] += cap * np.nan_to_num(scf[i]); scap[si[sub]] += cap
            wtot = wcap.sum() / 1e3; stot = scap.sum() / 1e3
            wgw = wind_gen.sum(0).mean() / 1e3; sgw = solar_gen.sum(0).mean() / 1e3
            wcf_nat = wgw / wtot if wtot > 0 else np.nan; scf_nat = sgw / stot if stot > 0 else np.nan
            np.savez(f"{CERF}/futuregen_{variant}_{scenario}.npz",
                     subregions=np.array(subs), wind_stamps=w_stamps, solar_stamps=s_stamps,
                     wind_gen=wind_gen.astype(np.float32), solar_gen=solar_gen.astype(np.float32),
                     wind_cap_mw=wcap, solar_cap_mw=scap,
                     offshore_gen=off_gen.astype(np.float32), offshore_cap_mw=off_cap,
                     climate=CLIMATE, scenario=scenario, variant=variant,
                     note="per-subregion hourly generation MW; wind 6-hourly(TSTRIDE2), solar 3-hourly; "
                          "wind=3D-SRGAN hub->PySAM Windpower(shear0,turb_generic_loss=15,actual T2/PSFC); "
                          "solar=native SWDOWN->PVWatts; nearest-EIA config templates; CF shared across variants; "
                          "offshore_gen=mandate-driven offshore_wind(15MW/hub150, shared across variants), "
                          "wind_stamps axis; offshore capacity is a state-law floor identical in all 3 policy variants")
            print(f"[agg] {variant}/{scenario}: wind cap {wtot:.1f}GW mean {wgw:.1f}GW CF {wcf_nat:.3f} (miss {miss_w}) | "
                  f"solar cap {stot:.1f}GW mean {sgw:.1f}GW CF {scf_nat:.3f} (miss {miss_s}) "
                  f"-> futuregen_{variant}_{scenario}.npz", flush=True)
else:
    raise SystemExit(f"unknown PHASE={PHASE}")
