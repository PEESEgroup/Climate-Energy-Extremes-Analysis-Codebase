"""3-way time-based split for the 3D-wind SRGAN over the 1356 c404_3d frames (FORK of 03_generate_split.py).
 test = held-out whole years {2010,2015} that are PRESENT in the 3D set (2010 absent -> test = 2015 only).
        if NEITHER present, fall back to a seeded random 10% held-out.
 val  = day-of-year %10==0 of the remaining years (season-balanced, for early stopping).
 train= the rest.
Aligned to cache row order (sorted frames, grid file excluded). Writes split3d.npz."""
import glob, os, numpy as np, datetime as dt
C3="/data/c404_3d"; T="/data/datasets/train"; TEST_YEARS={2010,2015}
def frame_list(): return [f for f in sorted(glob.glob(f"{C3}/c404_3d_*.npz")) if "grid" not in os.path.basename(f)]
def stamp(f): return os.path.basename(f)[8:18]                     # c404_3d_YYYYMMDDHH.npz
frames=frame_list(); stamps=np.array([stamp(f) for f in frames]); N=len(frames)
years=np.array([int(s[:4]) for s in stamps])
present_test=sorted(y for y in TEST_YEARS if (years==y).any())
lab=np.empty(N,dtype='<U5')
for i,s in enumerate(stamps):
    y=int(s[:4]); doy=dt.date(y,int(s[4:6]),int(s[6:8])).timetuple().tm_yday
    lab[i]='val' if doy%10==0 else 'train'
if present_test:
    for i,s in enumerate(stamps):
        if int(s[:4]) in present_test: lab[i]='test'
    test_note=f"years{present_test}"
else:                                                              # neither test year present -> seeded random 10%
    rng=np.random.default_rng(0); k=max(1,int(round(0.10*N)))
    test_idx=rng.choice(N,size=k,replace=False); lab[test_idx]='test'
    test_note="random10pct_seed0"
np.savez(f"{T}/split3d.npz", split=lab, stamps=stamps, test_years=np.array(sorted(TEST_YEARS)),
         note=f"3D wind SRGAN 3-way: test={test_note}; val=doy%10==0 of rest; train=rest")
u,c=np.unique(lab,return_counts=True)
print("N=",N, dict(zip(u.tolist(),c.tolist())),
      "| pct:", {k:round(100*v/N,1) for k,v in zip(u.tolist(),c.tolist())}, "| test=",test_note, flush=True)
