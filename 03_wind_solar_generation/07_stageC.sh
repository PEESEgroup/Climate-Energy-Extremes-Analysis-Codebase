#!/bin/bash
# STAGE C - historical wind, 1980-2019, through the SAME 3D SRGAN path as the future arm.
set -u
cd /data/gen_targets/srgan3d_val
export OUT=/data/gen_targets/srgan3d_val/hist_v5
mkdir -p "$OUT"
L=/data/logs/v5_stageC.log
E=/data/logs/v5_stageC_ERR.flag
echo "==== STAGE C start $(date) 1980-2019 BATCH=16 ====" | tee -a $L

FREE=$(df --output=avail -BG /data | tail -1 | tr -dc '0-9')
echo "free=${FREE}G" | tee -a $L
if [ "$FREE" -lt 400 ]; then echo "ABORT: only ${FREE}G free" | tee -a $L; date > $E; exit 1; fi

if [ ! -f "$OUT/fut_wind_wx_hist.npz" ]; then
  echo "---- WIND_INFER 1980-2019 $(date +%H:%M:%S) ----" | tee -a $L
  OUT=$OUT D3F=/data/tgw_3d CLIMATE=historical TAG=hist YEARS=1980-2019 TSTRIDE=1 \
  BATCH=16 NW=8 PHASE=wind_infer CUDA_VISIBLE_DEVICES=0 PYTHONWARNINGS=ignore \
    stdbuf -oL -eL /opt/pytorch/bin/python 17_fut_gen.py 2>&1 | grep -vE 'pynvml|FutureWarning' | tee -a $L
  if [ ! -f "$OUT/fut_wind_wx_hist.npz" ]; then echo "FAIL: wind_infer produced no output" | tee -a $L; date > $E; exit 1; fi
fi

if [ ! -f "$OUT/fut_wind_cf_hist.npz" ]; then
  echo "---- WIND_PYSAM $(date +%H:%M:%S) ----" | tee -a $L
  OUT=$OUT D3F=/data/tgw_3d CLIMATE=historical TAG=hist YEARS=1980-2019 TSTRIDE=1 NPROC=12 \
  PHASE=wind_pysam PYTHONWARNINGS=ignore \
    stdbuf -oL /data/genenv/bin/python 17_fut_gen.py 2>&1 | tee -a $L
  if [ ! -f "$OUT/fut_wind_cf_hist.npz" ]; then echo "FAIL: wind_pysam produced no output" | tee -a $L; date > $E; exit 1; fi
fi
echo "==== STAGE C DONE $(date) ====" | tee -a $L
date > /data/logs/v5_stageC_DONE.flag
