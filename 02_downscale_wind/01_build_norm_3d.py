"""Per-channel land-ROI z-score stats for the 3D-wind SRGAN (FORK of 01_build_norm.py).
HR = 14 wind channels [u@10,v@10,u@40,v@40,...,u@200,v@200] on the fine 625x1475 grid (SR target).
LR = 14 zoomed wind + 7 zoomed pred-context [u10,v10,t2,psfc,pwat,short,long] on coarse 89x211.
Stats over the TRAIN split only (split3d.npz), over the dilated land-ROI pixels (land_roi_mask.npz).
Writes /data/datasets/train/norm_stats_3d.npz (hr_mean/std (14,), lr_mean/std (21,)).  Env: SAMPLE, NPROC."""
import os, glob, zipfile, numpy as np
from scipy.ndimage import zoom
G="/data/datasets/grid"; C3="/data/c404_3d"; OUT="/data/datasets/train"
SAMPLE=int(os.environ.get("SAMPLE","500"))          # train frames sampled for stats (0=all train)
NPROC=int(os.environ.get("NPROC","6"))
nn=np.load(f"{G}/c404_to_fine_nn.npz")['idx']
rm=np.load(f"{G}/land_roi_mask.npz"); M=rm['roi']; Mc=rm['roi_coarse']          # (625,1475),(89,211) bool
PRED_CTX=[0,1,2,4,5,6,7]                              # pred_vars=[u10,v10,t2,rh,psfc,pwat,short,long] -> drop rh
HR_KEYS_3D=[f"{c}@{h}" for h in [10,40,80,110,140,160,200] for c in ('u','v')]   # 14
LR_KEYS_3D=HR_KEYS_3D+['u10','v10','t2','psfc','pwat','short','long']            # 21
def frame_list(): return [f for f in sorted(glob.glob(f"{C3}/c404_3d_*.npz")) if "grid" not in os.path.basename(f)]
def stamp(f): return os.path.basename(f)[8:18]
def regrid(a2d): return a2d.ravel()[nn].reshape(625,1475).astype(np.float32)     # native (1015,1367) -> fine
def assemble(f):
    d=np.load(f)
    wind14=d['hub'].astype(np.float32).reshape(14,1015,1367)                     # [lvl,(u,v)] level-major
    hr=np.stack([regrid(wind14[c]) for c in range(14)])                          # (14,625,1475)
    lrw=np.stack([zoom(hr[c],1/7,order=1) for c in range(14)])                   # (14,89,211)
    pred=d['pred'].astype(np.float32)
    ctx=np.stack([zoom(regrid(pred[k]),1/7,order=1) for k in PRED_CTX])          # (7,89,211)
    lr=np.concatenate([lrw,ctx],0)                                              # (21,89,211)
    return hr,lr
def _acc(files):
    sH=np.zeros(14);s2H=np.zeros(14);cH=0; sL=np.zeros(21);s2L=np.zeros(21);cL=0; bad=0
    for f in files:
        try: hr,lr=assemble(f)
        except (zipfile.BadZipFile,EOFError,ValueError,OSError): bad+=1; continue
        vH=hr[:,M]; sH+=vH.sum(1); s2H+=(vH.astype(np.float64)**2).sum(1); cH+=vH.shape[1]
        vL=lr[:,Mc]; sL+=vL.sum(1); s2L+=(vL.astype(np.float64)**2).sum(1); cL+=vL.shape[1]
    return sH,s2H,cH,sL,s2L,cL,bad
if __name__=="__main__":
    frames=frame_list(); stamps=np.array([stamp(f) for f in frames])
    sp=np.load(f"{OUT}/split3d.npz"); assert (sp['stamps']==stamps).all(), "split3d/frame order mismatch"
    train=[f for f,l in zip(frames,sp['split']) if l=='train']
    samp = train if (SAMPLE==0 or len(train)<=SAMPLE) else [train[i] for i in np.linspace(0,len(train)-1,SAMPLE).astype(int)]
    print(f"train={len(train)} stats_sample={len(samp)} NPROC={NPROC}",flush=True)
    if NPROC>1:
        import multiprocessing as mp
        chunks=[samp[i::NPROC*4] for i in range(NPROC*4) if samp[i::NPROC*4]]
        with mp.Pool(NPROC) as p: parts=p.map(_acc,chunks)
    else: parts=[_acc(samp)]
    sH=sum(x[0] for x in parts);s2H=sum(x[1] for x in parts);cH=sum(x[2] for x in parts)
    sL=sum(x[3] for x in parts);s2L=sum(x[4] for x in parts);cL=sum(x[5] for x in parts)
    bad=sum(x[6] for x in parts)
    hr_mean=sH/cH; hr_std=np.sqrt(np.maximum(s2H/cH-hr_mean**2,1e-12))
    lr_mean=sL/cL; lr_std=np.sqrt(np.maximum(s2L/cL-lr_mean**2,1e-12))
    np.savez(f"{OUT}/norm_stats_3d.npz",
             hr_mean=hr_mean.astype(np.float32),hr_std=hr_std.astype(np.float32),
             lr_mean=lr_mean.astype(np.float32),lr_std=lr_std.astype(np.float32),
             hr_keys=np.array(HR_KEYS_3D),lr_keys=np.array(LR_KEYS_3D),
             n_sample=len(samp)-bad,roi="subregion_mask>0 dilate5")
    np.set_printoptions(precision=3,suppress=True)
    print(f"skipped {bad} corrupt | HR mean{hr_mean} std{hr_std}")
    print(f"LR mean{lr_mean}\nLR std {lr_std}")
    print(f"-> {OUT}/norm_stats_3d.npz  ({len(samp)-bad} frames)",flush=True)
