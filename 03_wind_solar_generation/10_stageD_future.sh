#!/bin/bash
# Temporal super-resolution of future hub wind, 3-hourly -> hourly, then the power curve.
# Y0/Y1 MUST be set: stageD defaults to the historical window and would select nothing.
# MEMORY GATE: the first attempt was OOM-killed at 40 GB resident while the solar job held 30 GB.
# stageD is now patched to ~18 GB peak, and each climate waits for real headroom before starting.
set -u
L=/data/logs/v5_stageD_future.log
O=/data/gen_targets/srgan3d_val/futgen_v5
NEED_GB=45
echo "==== STAGE D FUTURE start $(date) ====" | tee -a $L
for C in rcp45cooler rcp45hotter rcp85cooler rcp85hotter; do
  WX=$O/fut_wind_wx_${C}.npz
  OUT1=$O/fut_wind_cf1h_${C}.npz
  [ -f "$WX" ]  || { echo "[$C] wx missing, skip" | tee -a $L; continue; }
  [ -f "$OUT1" ] && { echo "[$C] exists, skip" | tee -a $L; continue; }
  while :; do
    AV=$(free -g | awk '/^Mem:/{print $7}')
    [ "$AV" -ge $NEED_GB ] && break
    echo "[$C] waiting for memory: ${AV}G available, need ${NEED_GB}G" | tee -a $L
    sleep 120
  done
  echo "---- $C $(date +%H:%M:%S) avail=${AV}G ----" | tee -a $L
  WX=$WX SRC=future CLIMATE=$C OUT=$OUT1 Y0=2030 Y1=2050 NPROC=6 \
    /data/genenv/bin/python /data/08_stageD.py 2>&1 | grep --line-buffered -E 'stamps|hourly|DONE|Error|Traceback' | tee -a $L
  if [ ! -f "$OUT1" ]; then echo "FAIL $C (check for an OOM kill: dmesg -T | grep -i oom)" | tee -a $L; date > /data/logs/v5_stageD_future_ERR.flag; exit 1; fi
done
echo "==== STAGE D FUTURE DONE $(date) ====" | tee -a $L
date > /data/logs/v5_stageD_future_DONE.flag
