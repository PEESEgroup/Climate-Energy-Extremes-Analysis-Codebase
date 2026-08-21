#!/usr/bin/env bash
# Wait for the full /data cache3d build, sanity-check 1 real batch, then launch v2 3D-wind SRGAN training.
set -u
LOG=/data/logs/train3d.log; BLOG=/data/logs/build_cache3d_full.log
mkdir -p /data/logs /data/ckpt
echo "================ TRAIN3D LAUNCHER START $(date -u) ================" >> "$LOG"
echo "waiting for cache build DONE ..." >> "$LOG"
until grep -q "DONE N=" "$BLOG" 2>/dev/null; do sleep 60; done
echo "cache DONE: $(grep "DONE N=" "$BLOG" | tail -1)" >> "$LOG"
# sanity: load 1 train batch from the FULL cache via the dataset
CACHE3D=/data/cache3d SPLIT3D=split3d_full /opt/pytorch/bin/python - >> "$LOG" 2>&1 <<PY
import numpy as np, torch, sys
sys.path.insert(0,"/home/ubuntu/code")
from srgan_dataset_3d import C404SRGAN3D
from torch.utils.data import DataLoader
d=C404SRGAN3D("train"); print("dataset train N =", len(d), flush=True)
b=next(iter(DataLoader(d,batch_size=3,num_workers=2)))
lr,hr=b["lr"],b["hr"]
ok = tuple(lr.shape[1:])==(21,89,211) and tuple(hr.shape[1:])==(14,625,1475) and torch.isfinite(lr).all() and torch.isfinite(hr).all()
print("SANITY lr",tuple(lr.shape),"hr",tuple(hr.shape),"finite",bool(torch.isfinite(lr).all()),bool(torch.isfinite(hr).all()),"-> ",("OK" if ok else "FAIL"),flush=True)
sys.exit(0 if ok else 3)
PY
if [ $? -ne 0 ]; then echo "SANITY FAILED -> not launching" >> "$LOG"; exit 1; fi
echo "================ LAUNCH TRAINING $(date -u) ================" >> "$LOG"
cd /home/ubuntu/code
CACHE3D=/data/cache3d SPLIT3D=split3d_full /opt/pytorch/bin/python -u 04_train_c404_3d_v2.py \
  --speed_w 0.1 --psd_w 0.05 --vshear_w 1.0 --advloss hinge \
  --bs 3 --workers 6 --patience 5 --out /data/ckpt/g3d_full >> "$LOG" 2>&1
echo "================ TRAINING EXITED rc=$? $(date -u) ================" >> "$LOG"
