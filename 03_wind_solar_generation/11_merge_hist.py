"""Merge the 1980-1996 extension with the 1997-2019 arm into one 1980-2019 product."""
import numpy as np
B = "/data/gen_targets/srgan3d_val/hist_v5"
for tag, early, late, out in [
    ("wind",  f"{B}/hist_cf_hourly_1980_1996.npz",   f"{B}/hist_cf_hourly.npz",   f"{B}/hist_cf_hourly_1980_2019.npz"),
    ("solar", f"{B}/hist_solar_cf1h_1980_1996.npz",  f"{B}/hist_solar_cf1h.npz",  f"{B}/hist_solar_cf1h_1980_2019.npz")]:
    A = np.load(early, allow_pickle=True); C = np.load(late, allow_pickle=True)
    pa, pc = A["plants"].astype(str), C["plants"].astype(str)
    assert len(pa) == len(pc) and (pa == pc).all(), f"{tag}: plant order differs"
    sa, sc = A["stamps"].astype(str), C["stamps"].astype(str)
    assert sa[-1] < sc[0], f"{tag}: windows overlap ({sa[-1]} vs {sc[0]})"
    cf = np.concatenate([np.asarray(A["cf"]), np.asarray(C["cf"])], axis=1)
    st = np.concatenate([sa, sc])
    d = {}
    for k in A.files:
        if k in ("cf", "stamps"):
            continue
        a = np.asarray(A[k])
        if a.ndim == 2 and a.shape[1] == len(sa):        # time-indexed, must be concatenated too,
            d[k] = np.concatenate([a, np.asarray(C[k])], axis=1)   # not carried over stale
            print("   concatenated time-indexed key %s -> %s" % (k, d[k].shape))
        else:
            d[k] = A[k]
    np.savez(out, cf=cf, stamps=st, **d)
    m = np.nanmean(cf)
    print("%-6s %s + %s = %s stamps  %s -> %s  plain-mean CF %.4f  %.1f GB"
          % (tag, len(sa), len(sc), len(st), st[0], st[-1], m, cf.nbytes / 1e9), flush=True)
