#!/bin/bash
# STAGE A - future wind at native 3-hourly (TSTRIDE=1), 4 climates, 2 concurrent GPU streams.
# Writes to futgen_v5/ only. Nothing existing is touched.
set -u
cd /data/gen_targets/srgan3d_val
export OUT=/data/gen_targets/srgan3d_val/futgen_v5
mkdir -p "$OUT"
L=/data/logs/v5_stageA.log
echo "==== STAGE A start $(date) TSTRIDE=1 BATCH=24 2-way concurrent ====" | tee -a $L

run_one () {
  local C=$1
  local LL=$OUT/stageA_${C}.log
  if [ ! -f "$OUT/fut_wind_wx_${C}.npz" ]; then
    echo "[$C] WIND_INFER $(date +%H:%M:%S)" | tee -a $L
    OUT=$OUT YEARS=2030-2050 TSTRIDE=1 BATCH=24 NW=6 PYTHONWARNINGS=ignore \
    CLIMATE=$C TAG=$C PHASE=wind_infer CUDA_VISIBLE_DEVICES=0 \
      stdbuf -oL -eL /opt/pytorch/bin/python 17_fut_gen.py > $LL 2>&1
  fi
  if [ ! -f "$OUT/fut_wind_cf_${C}.npz" ]; then
    echo "[$C] WIND_PYSAM $(date +%H:%M:%S)" | tee -a $L
    OUT=$OUT YEARS=2030-2050 TSTRIDE=1 NPROC=6 PYTHONWARNINGS=ignore \
    CLIMATE=$C TAG=$C PHASE=wind_pysam \
      stdbuf -oL /data/genenv/bin/python 17_fut_gen.py >> $LL 2>&1
  fi
  echo "[$C] DONE $(date +%H:%M:%S)" | tee -a $L
}

for PAIR in "rcp45cooler rcp45hotter" "rcp85cooler rcp85hotter"; do
  FREE=$(df --output=avail -BG /data | tail -1 | tr -dc '0-9')
  echo "---- batch [$PAIR] free=${FREE}G $(date +%H:%M:%S) ----" | tee -a $L
  if [ "$FREE" -lt 400 ]; then echo "ABORT: only ${FREE}G free" | tee -a $L; exit 1; fi
  for C in $PAIR; do run_one $C & done
  wait
done
echo "==== STAGE A DONE $(date) ====" | tee -a $L
date > /data/logs/v5_stageA_DONE.flag
