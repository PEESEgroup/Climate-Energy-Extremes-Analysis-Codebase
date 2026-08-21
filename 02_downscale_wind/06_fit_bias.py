"""Fit TGW-hist(WRF 12km) -> CONUS404(4km) LR bias correction on 1997-2019, per cell (89x211) per month.
Methods (evidence-based, see TGW_DOWNSCALING_WORKFLOW.md Step 2):
  winds u10,v10 -> EQM/QDM on wind SPEED (+ circular direction offset)   [fixes 16% var + 11-13% tail bias]
  t2,psfc,long  -> additive offset  (variance already matched -> linear scaling, trend-preserving)
  q2            -> multiplicative ratio (positive-definite; q2 floored at 0)
  short (SWDOWN)-> clearness-index Kt = SW/(S0*cosSZA) ratio, daytime-only (correct atmosphere transmission,
                   NOT the deterministic solar geometry both models share)
Reference = CONUS404 LR (cache, physical = z*lr_std+lr_mean, ALL frames = best climatology).
Outputs /data/tgw_hist/bias_fit.npz (fit on ALL years) + prints held-out odd/even-year validation.
Run: /opt/pytorch/bin/python 06_fit_bias.py
"""
import numpy as np, datetime as _dt, sys, time
NLAT, NLON = 89, 211; S0 = 1361.0
VN = ['u10', 'v10', 'q2', 'psfc', 't2', 'long', 'short']
QLEV = np.array([0.5, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 70, 75, 80, 85, 90, 93, 95, 97, 98, 99, 99.5, 99.7, 99.9])

# ---------- load meta ----------
t0 = time.time()
M = np.load("/data/tgw_hist/full_lr_meta.npz", allow_pickle=True)
Ntot = int(M['N']); stampsT = M['stamps'].astype(str)
yrT = np.array([int(s[:4]) for s in stampsT]); moT = np.array([int(s[4:6]) for s in stampsT])
TGW = np.memmap("/data/tgw_hist/full_lr.dat", dtype='float32', mode='r', shape=(Ntot, 7, NLAT, NLON))

cm = np.load("/scratch/cache/cache_meta.npz", allow_pickle=True)
stampsC = cm['stamps'].astype(str); Nc = len(stampsC)
yrC = np.array([int(s[:4]) for s in stampsC]); moC = np.array([int(s[4:6]) for s in stampsC])
ns = np.load("/data/datasets/train/norm_stats.npz"); LRM = ns['lr_mean'].astype('float32'); LRS = ns['lr_std'].astype('float32')
CACHE = np.memmap("/scratch/cache/lr_z_fp16.dat", dtype=np.float16, mode='r', shape=(Nc, 7, NLAT, NLON))

# ---------- solar cos(zenith) on LR grid ----------
co = np.load("/data/datasets/grid/coordinate.npz")
import torch, torch.nn.functional as _Fn
_latlr = _Fn.adaptive_avg_pool1d(torch.tensor(co['lat'].astype('float64'))[None, None], 89)[0, 0].numpy()
_lonlr = _Fn.adaptive_avg_pool1d(torch.tensor(co['lon'].astype('float64'))[None, None], 211)[0, 0].numpy()
LATg = np.deg2rad(_latlr)[:, None] * np.ones((1, NLON)); LONg = np.ones((NLAT, 1)) * _lonlr[None, :]
def cosz_stack(stamps):
    out = np.empty((len(stamps), NLAT, NLON), np.float32)
    for i, s in enumerate(stamps):
        yr, mo, dy, hh = int(s[:4]), int(s[4:6]), int(s[6:8]), int(s[8:10])
        doy = _dt.date(yr, mo, dy).timetuple().tm_yday; g = 2 * np.pi / 365.0 * (doy - 1 + (hh - 12) / 24.0)
        eq = 229.18 * (7.5e-5 + 1.868e-3 * np.cos(g) - .032077 * np.sin(g) - .014615 * np.cos(2 * g) - .040849 * np.sin(2 * g))
        dec = .006918 - .399912 * np.cos(g) + .070257 * np.sin(g) - .006758 * np.cos(2 * g) + 9.07e-4 * np.sin(2 * g) - .002697 * np.cos(3 * g) + .00148 * np.sin(3 * g)
        ha = np.deg2rad((hh * 60 + eq + 4 * LONg) / 4.0 - 180.0)
        out[i] = np.clip(np.sin(LATg) * np.sin(dec) + np.cos(LATg) * np.cos(dec) * np.cos(ha), 0, 1)
    return out

def load_month(month, ymask_T, ymask_C):
    rT = np.nonzero((moT == month) & ymask_T)[0]
    rC = np.nonzero((moC == month) & ymask_C)[0]
    aT = np.asarray(TGW[rT]).astype('float32')                       # (nT,7,89,211) physical
    aC = np.asarray(CACHE[rC]).astype('float32') * LRS[None, :, None, None] + LRM[None, :, None, None]
    aT[:, 2] = np.maximum(aT[:, 2], 0.0); aC[:, 2] = np.maximum(aC[:, 2], 0.0)  # q2 floor >=0
    return aT, aC, stampsT[rT], stampsC[rC]

def fit(ymask_T, ymask_C):
    """Return correction dict fit on the given year masks."""
    R = {k: np.full((12, NLAT, NLON), np.nan, np.float32) for k in
         ['t2_off', 'psfc_off', 'long_off', 'q2_ratio', 'dir_off', 'sw_kt_ratio']}
    Tq = np.full((len(QLEV), 12, NLAT, NLON), np.nan, np.float32)  # TGW speed quantiles
    Cq = np.full((len(QLEV), 12, NLAT, NLON), np.nan, np.float32)  # CONUS speed quantiles
    for m in range(1, 13):
        aT, aC, sT, sC = load_month(m, ymask_T, ymask_C)
        j = m - 1
        # additive offsets (mean_C - mean_T)
        R['t2_off'][j] = np.nanmean(aC[:, 4], 0) - np.nanmean(aT[:, 4], 0)
        R['psfc_off'][j] = np.nanmean(aC[:, 3], 0) - np.nanmean(aT[:, 3], 0)
        R['long_off'][j] = np.nanmean(aC[:, 5], 0) - np.nanmean(aT[:, 5], 0)
        # q2 multiplicative
        mT = np.nanmean(aT[:, 2], 0); mC = np.nanmean(aC[:, 2], 0)
        R['q2_ratio'][j] = mC / np.where(mT > 1e-9, mT, np.nan)
        # winds: speed quantiles + direction circular offset
        spT = np.sqrt(aT[:, 0]**2 + aT[:, 1]**2); spC = np.sqrt(aC[:, 0]**2 + aC[:, 1]**2)
        Tq[:, j] = np.nanpercentile(spT, QLEV, axis=0); Cq[:, j] = np.nanpercentile(spC, QLEV, axis=0)
        dirT = np.arctan2(np.nanmean(aT[:, 1], 0), np.nanmean(aT[:, 0], 0))
        dirC = np.arctan2(np.nanmean(aC[:, 1], 0), np.nanmean(aC[:, 0], 0))
        R['dir_off'][j] = np.arctan2(np.sin(dirC - dirT), np.cos(dirC - dirT))
        # shortwave clearness index Kt (daytime cosz>0.15)
        czT = cosz_stack(sT); czC = cosz_stack(sC)
        dayT = czT > 0.15; dayC = czC > 0.15
        KtT = np.where(dayT, aT[:, 6] / (S0 * np.where(dayT, czT, 1)), np.nan)
        KtC = np.where(dayC, aC[:, 6] / (S0 * np.where(dayC, czC, 1)), np.nan)
        mktT = np.nanmean(KtT, 0); mktC = np.nanmean(KtC, 0)
        R['sw_kt_ratio'][j] = mktC / np.where(mktT > 1e-6, mktT, np.nan)
        print(f"    month {m:2d}: nT={len(sT)} nC={len(sC)}  ({time.time()-t0:.0f}s)", flush=True)
    R['wind_Tq'] = Tq; R['wind_Cq'] = Cq; R['qlev'] = QLEV
    return R

def apply_hist(aT_val, R, stamps_val):
    """Apply correction (hist mode: EQM for winds) to a validation TGW array (n,7,89,211) by its months."""
    out = aT_val.copy(); mo = np.array([int(s[4:6]) for s in stamps_val])
    for m in range(1, 13):
        idx = np.nonzero(mo == m)[0]
        if not len(idx): continue
        j = m - 1
        out[idx, 4] += R['t2_off'][j]; out[idx, 3] += R['psfc_off'][j]; out[idx, 5] += R['long_off'][j]
        out[idx, 2] = np.maximum(out[idx, 2], 0) * np.nan_to_num(R['q2_ratio'][j], nan=1.0)
        # winds EQM on speed, preserve (rotated) direction
        u = aT_val[idx, 0]; v = aT_val[idx, 1]; sp = np.sqrt(u**2 + v**2); ang = np.arctan2(v, u)
        Tq = R['wind_Tq'][:, j]; Cq = R['wind_Cq'][:, j]                       # (nq,89,211)
        spc = np.empty_like(sp)
        for a in range(NLAT):
            for b in range(NLON):
                xp = Tq[:, a, b]
                if not np.isfinite(xp).all(): spc[:, a, b] = sp[:, a, b]; continue
                spc[:, a, b] = np.interp(sp[:, a, b], xp, Cq[:, a, b])
        ang2 = ang + np.nan_to_num(R['dir_off'][j], nan=0.0)
        out[idx, 0] = spc * np.cos(ang2); out[idx, 1] = spc * np.sin(ang2)
        # shortwave: Kt ratio, daytime only
        cz = cosz_stack(stamps_val[idx]); day = cz > 0.15
        sw = aT_val[idx, 6]; Kt = np.where(day, sw / (S0 * np.where(day, cz, 1)), 0)
        Ktc = Kt * np.nan_to_num(R['sw_kt_ratio'][j], nan=1.0)
        out[idx, 6] = np.where(day, np.clip(Ktc, 0, 1.2) * S0 * cz, sw)
    return out

def bias_metrics(aT, aC, tag):
    """Distribution-bias metrics of TGW(aT) vs CONUS(aC) reference (same month pool). Uses climatological
    per-cell stats; winds also p99 & ρv³ (density~1.2)."""
    res = {}
    valid = np.isfinite(aT[0, 4]) & np.isfinite(aC[0, 4]) if len(aC) else np.isfinite(aT[0, 4])
    for ch, v in enumerate(VN):
        # RMS of per-cell mean difference (bias in the mean field)
        mT = np.nanmean(aT[:, ch], 0); mC = np.nanmean(aC[:, ch], 0)
        res[f'{v}_meanbias'] = float(np.sqrt(np.nanmean((mT - mC)[valid]**2)))
    # winds tail + power
    spT = np.sqrt(aT[:, 0]**2 + aT[:, 1]**2); spC = np.sqrt(aC[:, 0]**2 + aC[:, 1]**2)
    res['wspd_p99bias'] = float(np.sqrt(np.nanmean((np.nanpercentile(spT, 99, 0) - np.nanpercentile(spC, 99, 0))[valid]**2)))
    pT = np.nanmean(0.5 * 1.2 * spT**3, 0); pC = np.nanmean(0.5 * 1.2 * spC**3, 0)  # mean wind power density
    res['windpow_relbias'] = float(np.nanmean(((pT - pC) / np.where(pC > 1e-6, pC, np.nan))[valid]))
    return res

if __name__ == '__main__':
    allT = (yrT >= 1997) & (yrT <= 2019); allC = (yrC >= 1997) & (yrC <= 2019)
    # ---- held-out validation: fit odd years, test even (and vice versa) ----
    print("=== HELD-OUT VALIDATION (fit odd yrs -> test even, and vice versa) ===", flush=True)
    for fit_odd in (True, False):
        fmaskT = allT & ((yrT % 2 == 1) == fit_odd); fmaskC = allC & ((yrC % 2 == 1) == fit_odd)
        tmaskT = allT & ((yrT % 2 == 1) != fit_odd); tmaskC = allC & ((yrC % 2 == 1) != fit_odd)
        R = fit(fmaskT, fmaskC)
        # evaluate on a pooled sample of held-out test months (subsample for speed)
        rT = np.nonzero(tmaskT)[0]; rC = np.nonzero(tmaskC)[0]
        subT = rT[::7]; subC = rC[::5]                        # ~pooled, all months represented
        aTv = np.asarray(TGW[subT]).astype('float32'); aTv[:, 2] = np.maximum(aTv[:, 2], 0)
        aCv = np.asarray(CACHE[subC]).astype('float32') * LRS[None, :, None, None] + LRM[None, :, None, None]; aCv[:, 2] = np.maximum(aCv[:, 2], 0)
        before = bias_metrics(aTv, aCv, 'before')
        aTc = apply_hist(aTv, R, stampsT[subT])
        after = bias_metrics(aTc, aCv, 'after')
        lab = 'FIT-odd/TEST-even' if fit_odd else 'FIT-even/TEST-odd'
        print(f"\n--- {lab} (nTest={len(subT)}) ---")
        print(f"{'metric':18s}{'before':>12s}{'after':>12s}{'reduction':>11s}")
        for k in before:
            b, a = before[k], after[k]
            red = (1 - abs(a) / abs(b)) * 100 if abs(b) > 1e-12 else 0
            print(f"{k:18s}{b:12.4g}{a:12.4g}{red:10.0f}%")
    # ---- final fit on ALL years -> save ----
    print("\n=== FINAL FIT on ALL 1997-2019 ===", flush=True)
    Rall = fit(allT, allC)
    np.savez("/data/tgw_hist/bias_fit.npz", **Rall, vn=np.array(VN), S0=S0)
    print(f"saved /data/tgw_hist/bias_fit.npz  ({time.time()-t0:.0f}s total)", flush=True)
