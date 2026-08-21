"""One pass over the 1356 c404_3d frames -> fp16 land-ROI z-scored HR+LR cache on NVMe (FORK of 05_build_cache.py).
HR3D = 14 wind ch [u@10,v@10,...,u@200,v@200] fine 625x1475 (SR target, wind-only, GROUP {1:(0,14)}).
LR3D = 14 zoomed wind + 7 zoomed pred-context [u10,v10,t2,psfc,pwat,short,long] coarse 89x211 (21 ch).
Z-score uses norm_stats_3d.npz (land-ROI stats). Winds z ~ O(1) so fp16 is safe.
Writes /opt/dlami/nvme/cache3d/{hr3d_z_fp16.dat,lr3d_z_fp16.dat,cache_meta_3d.npz}. Env NPROC (<=6)."""
import os, glob, zipfile, numpy as np
from scipy.ndimage import zoom
G="/data/datasets/grid"; C3="/data/c404_3d"; T="/data/datasets/train"; CACHE="/data/cache3d"
os.makedirs(CACHE, exist_ok=True)
NPROC=int(os.environ.get("NPROC","6"))
nn=np.load(f"{G}/c404_to_fine_nn.npz")['idx']
PRED_CTX=[0,1,2,4,5,6,7]                              # pred_vars -> [u10,v10,t2,psfc,pwat,short,long]
st=np.load(f"{T}/norm_stats_3d.npz")
hrm=st['hr_mean'].astype(np.float32).reshape(14,1,1); hrs=st['hr_std'].astype(np.float32).reshape(14,1,1)
lrm=st['lr_mean'].astype(np.float32).reshape(21,1,1); lrs=st['lr_std'].astype(np.float32).reshape(21,1,1)
def frame_list(): return [f for f in sorted(glob.glob(f"{C3}/c404_3d_*.npz")) if "grid" not in os.path.basename(f)]
def stamp(f): return os.path.basename(f)[8:18]
def regrid(a2d): return a2d.ravel()[nn].reshape(625,1475).astype(np.float32)
frames=frame_list(); N=len(frames); stamps=np.array([stamp(f) for f in frames])
sp=np.load(f"{T}/split3d_full.npz"); assert (sp['stamps']==stamps).all(), "split3d/frame order mismatch"
splits=sp['split']
HRSH=(N,14,625,1475); LRSH=(N,21,89,211)
np.memmap(f"{CACHE}/hr3d_z_fp16.dat",dtype=np.float16,mode='w+',shape=HRSH).flush()
np.memmap(f"{CACHE}/lr3d_z_fp16.dat",dtype=np.float16,mode='w+',shape=LRSH).flush()
print(f"allocated cache N={N} HR={HRSH} LR={LRSH} on {CACHE}",flush=True)
def assemble(f):
    d=np.load(f)
    wind14=d['hub'].astype(np.float32).reshape(14,1015,1367)
    hr=np.stack([regrid(wind14[c]) for c in range(14)])                          # (14,625,1475)
    lrw=np.stack([zoom(hr[c],1/7,order=1) for c in range(14)])                   # (14,89,211)
    pred=d['pred'].astype(np.float32)
    ctx=np.stack([zoom(regrid(pred[k]),1/7,order=1) for k in PRED_CTX])          # (7,89,211)
    lr=np.concatenate([lrw,ctx],0)                                              # (21,89,211)
    return hr,lr
def work(rows):
    hr=np.memmap(f"{CACHE}/hr3d_z_fp16.dat",dtype=np.float16,mode='r+',shape=HRSH)
    lr=np.memmap(f"{CACHE}/lr3d_z_fp16.dat",dtype=np.float16,mode='r+',shape=LRSH)
    bad=[]
    for i in rows:
        try: h,l=assemble(frames[i])
        except (zipfile.BadZipFile,EOFError,ValueError,OSError): bad.append(i); hr[i]=0; lr[i]=0; continue
        hr[i]=((h-hrm)/hrs).astype(np.float16)
        lr[i]=((l-lrm)/lrs).astype(np.float16)
    hr.flush(); lr.flush()
    return bad
if __name__=="__main__":
    import multiprocessing as mp
    chunks=[list(range(i,N,NPROC*4)) for i in range(NPROC*4)]; chunks=[c for c in chunks if c]
    with mp.Pool(NPROC) as p: parts=p.map(work,chunks)
    bad=sorted(sum(parts,[]))
    valid=np.ones(N,bool)
    for b in bad: valid[b]=False
    np.savez(f"{CACHE}/cache_meta_3d.npz",stamps=stamps,splits=splits,valid=valid,N=N,
             hr_shape=np.array([14,625,1475]),lr_shape=np.array([21,89,211]),
             years=np.array([int(s[:4]) for s in stamps]),
             store="landroi_zscore_fp16 via norm_stats_3d.npz; HR=14 wind, LR=14 wind+7 pred-context")
    u,c=np.unique(splits[valid],return_counts=True)
    print(f"DONE N={N} bad={len(bad)} valid={int(valid.sum())} split={dict(zip(u.tolist(),c.tolist()))}",flush=True)
