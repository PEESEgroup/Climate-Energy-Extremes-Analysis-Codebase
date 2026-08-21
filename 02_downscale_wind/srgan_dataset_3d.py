"""3D-wind SRGAN dataset from the fp16 NVMe cache (FORK of srgan_dataset.py -> C404SRGAN3D).
Cache (build_cache_3d.py) already stores LAND-ROI z-scored fp16 (norm_stats_3d.npz) so no per-step affine.
Single wind GROUP: HR = 14 wind ch (625,1475) target; LR = 14 wind + 7 pred-context (89,211) input.
Augmentation for the thin (~1.1k train frames) set:
  flip=True  : random horizontal (E-W) flip. u-components (even HR idx, even LR-wind idx, LR ctx u10@14)
               are NEGATED so the flipped field stays physically consistent. NOTE: incompatible with the
               generator's baked terrain buffers (they don't flip) -> use only with terrain conditioning OFF.
  crop=N     : random co-registered LR(N,N)/HR(7N,7N) tile (7x data ratio). Requires the generator built
               with crop_size=(7N,7N) and terrain OFF. crop=None (default) -> full frame.
Returns lr(21,89,211 or N,N), hr(14,625,1475 or 7N,7N)."""
import numpy as np, torch
from torch.utils.data import Dataset
import os as _os
T="/data/datasets/train"; CACHE=_os.environ.get("CACHE3D","/opt/dlami/nvme/cache3d"); _SPLIT=_os.environ.get("SPLIT3D","split3d")
HR_KEYS_3D=[f"{c}@{h}" for h in [10,40,80,110,140,160,200] for c in ('u','v')]     # 14
LR_KEYS_3D=HR_KEYS_3D+['u10','v10','t2','psfc','pwat','short','long']              # 21
GROUP={1:(0,14)}                                                                    # single wind group
U_HR=list(range(0,14,2))                        # HR u-channel indices 0,2,...,12
U_LR=list(range(0,14,2))+[14]                    # LR u-channels: winds 0,2,..,12 + ctx u10 at 14

class C404SRGAN3D(Dataset):
    def __init__(self, split='train', flip=False, crop=None):
        self.flip=flip; self.crop=crop
        m=np.load(f"{CACHE}/cache_meta_3d.npz", allow_pickle=True)
        valid=m['valid']; self.N=int(m['N'])
        self.hr_sh=tuple(int(x) for x in m['hr_shape']); self.lr_sh=tuple(int(x) for x in m['lr_shape'])
        sp=np.load(f"{T}/{_SPLIT}.npz", allow_pickle=True); labels=sp['split']
        assert len(labels)==self.N, f"split3d {len(labels)} != cache N {self.N}"
        assert (m['stamps']==sp['stamps']).all(), "split3d/cache stamp order mismatch"
        sel=(labels==split)&valid; self.idx=np.nonzero(sel)[0].astype(np.int64)
        self._hr=None; self._lr=None
    def __len__(self): return len(self.idx)
    def _open(self):
        if self._hr is None:
            self._hr=np.memmap(f"{CACHE}/hr3d_z_fp16.dat",dtype=np.float16,mode='r',shape=(self.N,)+self.hr_sh)
            self._lr=np.memmap(f"{CACHE}/lr3d_z_fp16.dat",dtype=np.float16,mode='r',shape=(self.N,)+self.lr_sh)
    def __getitem__(self,i):
        self._open(); r=int(self.idx[i])
        hr=np.asarray(self._hr[r],dtype=np.float32); lr=np.asarray(self._lr[r],dtype=np.float32)
        if self.crop is not None:
            N=self.crop; H,W=self.lr_sh[1],self.lr_sh[2]                          # LR grid 89,211
            Hh,Wh=self.hr_sh[1],self.hr_sh[2]                                     # HR grid 625,1475
            rmax=min(H, Hh//7)-N; cmax=min(W, Wh//7)-N                            # keep 7N tile inside HR
            r0=np.random.randint(0, rmax+1); c0=np.random.randint(0, cmax+1)
            lr=lr[:, r0:r0+N, c0:c0+N]
            hr=hr[:, 7*r0:7*r0+7*N, 7*c0:7*c0+7*N]                                # co-registered HR tile (7N,7N)
        if self.flip and np.random.rand()<0.5:
            hr=hr[:, :, ::-1].copy(); lr=lr[:, :, ::-1].copy()
            hr[U_HR]*=-1.0; lr[U_LR]*=-1.0                                        # E-W flip negates u-wind
        # belt-and-suspenders: sanitize any non-finite (corrupt frame) to finite z-space values before training
        lr_t=torch.nan_to_num(torch.from_numpy(np.ascontiguousarray(lr)), nan=0.0, posinf=10.0, neginf=-10.0)
        hr_t=torch.nan_to_num(torch.from_numpy(np.ascontiguousarray(hr)), nan=0.0, posinf=10.0, neginf=-10.0)
        return {'lr':lr_t, 'hr':hr_t, 'row':r}
