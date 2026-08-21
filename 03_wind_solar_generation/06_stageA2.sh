#!/bin/bash
# STAGE A (retry) - future wind at native 3-hourly, single stream, BATCH=16 (measured safe).
# Exit status is checked at every step so a failure cannot be masked as DONE.
set -u
cd /data/gen_targets/srgan3d_val
export OUT=/data/gen_targets/srgan3d_val/futgen_v5
mkdir -p "$OUT"
L=/data/logs/v5_stageA.log
echo "==== STAGE A retry start $(date) TSTRIDE=1 BATCH=16 single stream ====" | tee -a $L
for C in rcp45cooler rcp45hotter rcp85cooler rcp85hotter; do
  FREE=$(df --output=avail -BG /data | tail -1 | tr -dc '0-9')
  echo "---- $C free=${FREE}G $(date +%H:%M:%S) ----" | tee -a $L
  if [ "$FREE" -lt 400 ]; then echo "ABORT low disk" | tee -a $L; date > /data/logs/v5_stageA_ERR.flag; exit 1; fi
  if [ ! -f "$OUT/fut_wind_wx_${C}.npz" ]; then
    OUT=$OUT YEARS=2030-2050 TSTRIDE=1 BATCH=16 NW=8 PYTHONWARNINGS=ignore \
    CLIMATE=$C TAG=$C PHASE=wind_infer CUDA_VISIBLE_DEVICES=0 \
      stdbuf -oL -eL /opt/pytorch/bin/python 17_fut_gen.py 2>&1 \
      | stdbuf -oL grep --line-buffered -vE 'pynvml|FutureWarning' | tee -a $L
    if [ ! -f "$OUT/fut_wind_wx_${C}.npz" ]; then
      echo "FAIL wind_infer $C" | tee -a $L; date > /data/logs/v5_stageA_ERR.flag; exit 1; fi
  fi
  if [ ! -f "$OUT/fut_wind_cf_${C}.npz" ]; then
    OUT=$OUT YEARS=2030-2050 TSTRIDE=1 NPROC=12 PYTHONWARNINGS=ignore \
    CLIMATE=$C TAG=$C PHASE=wind_pysam \
      stdbuf -oL /data/genenv/bin/python 17_fut_gen.py 2>&1 | tee -a $L
    if [ ! -f "$OUT/fut_wind_cf_${C}.npz" ]; then
      echo "FAIL wind_pysam $C" | tee -a $L; date > /data/logs/v5_stageA_ERR.flag; exit 1; fi
  fi
  echo "[$C] OK $(date +%H:%M:%S)" | tee -a $L
done
echo "==== STAGE A DONE $(date) ====" | tee -a $L
date > /data/logs/v5_stageA_DONE.flag
