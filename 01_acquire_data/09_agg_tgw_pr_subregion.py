"""Aggregate TGW 12km precip/snow (3ch [precip,swe,snowh], 299x424) -> 18-subregion DAILY series.
Route: 12km cell -> county (county_mask_tgw.npz) -> subregion (fips_to_subregion, code 1..18).
precip = daily SUM of subregion-mean hourly (mm/day); swe/snowh = daily MEAN of subregion-mean,
computed over land cells EXCLUDING permanent-ice cells (SWE runaway-accumulation artifact).
Boundary-split days accumulate across files (date buckets finalized at end). Env: /data/tellenv.
Usage: python 09_agg_tgw_pr_subregion.py --stream historical [--limit N] [--icefile ...]"""
import argparse, glob, os, numpy as np

ROOT = "/data/tgw_precip"
CMASK = "/data/loads_measured/county_mask_tgw.npz"
FMAP = "/tmp/fips_to_subregion_mapping.csv"
OUTDIR = f"{ROOT}/agg"
ICE_THRESH = 5000.0  # mm SWE; seasonal snow rarely > ~3000mm -> above this = permanent ice

ap = argparse.ArgumentParser()
ap.add_argument("--stream", required=True)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--icefile", default=f"{OUTDIR}/ice_mask.npz")
ap.add_argument("--build-ice", action="store_true", help="build ice mask from last file of this stream")
args = ap.parse_args()
os.makedirs(OUTDIR, exist_ok=True)

# ---- cell -> subregion code ----
cm = np.load(CMASK, allow_pickle=True)
fips = cm["fips"].astype(str)
H, W = int(cm["H"]), int(cm["W"])
pair_fips = cm["pair_fips"]; pair_cell = cm["pair_cell"].astype(np.int64)
fmap = {}
with open(FMAP) as f:
    next(f)
    for ln in f:
        p = ln.rstrip("\n").split(",")
        if len(p) < 3: continue
        fmap[p[0].zfill(5)] = int(p[2])
sub_of_pair = np.array([fmap.get(fips[i].zfill(5), 0) for i in pair_fips], dtype=np.int16)  # 0 = unmapped
valid = sub_of_pair > 0
lc = pair_cell[valid]              # flat 12km cell indices (land, mapped)
sub_lc = sub_of_pair[valid]       # subregion code 1..18
n_unmapped = int((~valid).sum())
print(f"[{args.stream}] land cells mapped={lc.size} unmapped_pairs={n_unmapped} H,W={H},{W}", flush=True)

files = sorted(glob.glob(f"{ROOT}/{args.stream}/tgw_pr_*.npz"))
if args.limit: files = files[:args.limit]
assert files, "no files"

def load(fp):
    z = np.load(fp, allow_pickle=True)
    return z["data"], z["times"].astype(str)

# ---- ice mask ----
if args.build_ice:
    d, _ = load(files[-1])                       # last file = max accumulation
    swe_max = d[:, 1].astype("float32").max(0).ravel()  # (H*W,)
    ice = swe_max > ICE_THRESH
    np.savez(args.icefile, ice=ice.reshape(H, W), thresh=ICE_THRESH, H=H, W=W)
    print(f"[ice] built from {os.path.basename(files[-1])}: {int(ice.sum())} permanent-ice cells "
          f"(of {lc.size} land)", flush=True)
ice = np.load(args.icefile)["ice"].ravel() if os.path.exists(args.icefile) else np.zeros(H*W, bool)

# precompute per-subregion cell index lists (precip=all land; swe/snowh=non-ice land)
noice = ~ice[lc]
IDX = {s: lc[sub_lc == s] for s in range(1, 19)}
IDXN = {s: lc[(sub_lc == s) & noice] for s in range(1, 19)}
cnt = {s: IDX[s].size for s in range(1, 19)}
cntn = {s: IDXN[s].size for s in range(1, 19)}

# date buckets
p_sum = {}; s_sum = {}; h_sum = {}; day_cnt = {}
for k, fp in enumerate(files):
    d, t = load(fp)
    F = d.shape[0]
    flat = d.astype("float32").reshape(F, 3, -1)
    # per-frame subregion means -> (F,19)
    pm = np.zeros((F, 19), "float32"); sm = np.zeros((F, 19), "float32"); hm = np.zeros((F, 19), "float32")
    for s in range(1, 19):
        if cnt[s]:  pm[:, s] = flat[:, 0][:, IDX[s]].mean(1)
        if cntn[s]:
            sm[:, s] = flat[:, 1][:, IDXN[s]].mean(1)
            hm[:, s] = flat[:, 2][:, IDXN[s]].mean(1)
    for f in range(F):
        dt = t[f][:8]
        if dt not in p_sum:
            p_sum[dt] = np.zeros(19, "float32"); s_sum[dt] = np.zeros(19, "float32")
            h_sum[dt] = np.zeros(19, "float32"); day_cnt[dt] = 0
        p_sum[dt] += pm[f]; s_sum[dt] += sm[f]; h_sum[dt] += hm[f]; day_cnt[dt] += 1
    if k % 200 == 0: print(f"  {k}/{len(files)} {os.path.basename(fp)} dates={len(p_sum)}", flush=True)

dates = sorted(p_sum)
D = len(dates)
precip = np.zeros((D, 18), "float32"); swe = np.zeros((D, 18), "float32"); snowh = np.zeros((D, 18), "float32")
for i, dt in enumerate(dates):
    c = day_cnt[dt]
    precip[i] = p_sum[dt][1:]              # daily SUM of hourly subregion-mean (mm/day)
    swe[i] = s_sum[dt][1:] / max(c, 1)     # daily MEAN
    snowh[i] = h_sum[dt][1:] / max(c, 1)
out = f"{OUTDIR}/{args.stream}_subregion_daily.npz"
np.savez(out, dates=np.array(dates), precip=precip, swe=swe, snowh=snowh,
         sub_codes=np.arange(1, 19), cells_per_sub=np.array([cnt[s] for s in range(1, 19)]),
         noice_cells_per_sub=np.array([cntn[s] for s in range(1, 19)]))
print(f"WROTE {out}  D={D} dates {dates[0]}..{dates[-1]}", flush=True)
print(f"  precip mm/day mean={precip.mean():.2f} p99={np.percentile(precip,99):.1f} max={precip.max():.1f}", flush=True)
print(f"  swe mm mean={swe.mean():.1f} max={swe.max():.1f} | snowh m mean={snowh.mean():.3f} max={snowh.max():.2f}", flush=True)
