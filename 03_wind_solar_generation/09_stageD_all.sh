#!/bin/bash
# Temporal super-resolution for BOTH arms on one code path and one grid: the 12 km native hourly
# surface archive. Historical runs first because it is the shorter job and gates the historical
# aggregation. Y0/Y1 are explicit for both; the default window would silently select nothing.
set -u
L=/data/logs/v5_stageD_all.log
H=/data/gen_targets/srgan3d_val/hist_v5
O=/data/gen_targets/srgan3d_val/futgen_v5
NEED_GB=45
gate () { while :; do AV=$(free -g | awk '/^Mem:/{print $7}'); [ "$AV" -ge $NEED_GB ] && break
  echo "  waiting for memory: ${AV}G available, need ${NEED_GB}G" | tee -a $L; sleep 120; done; }
echo "==== STAGE D ALL start $(date) ====" | tee -a $L
gate
echo "---- historical $(date +%H:%M:%S) avail=${AV}G ----" | tee -a $L
WX=$H/fut_wind_wx_hist.npz SRC=hist OUT=$H/hist_cf_hourly.npz Y0=1997 Y1=2019 NPROC=6 \
  /data/genenv/bin/python /data/08_stageD.py 2>&1 | grep --line-buffered -E 'mapping|stamps|hourly|DONE|Error|Traceback|Assertion' | tee -a $L
[ -f $H/hist_cf_hourly.npz ] || { echo 'FAIL historical' | tee -a $L; date > /data/logs/v5_stageD_all_ERR.flag; exit 1; }
for C in rcp45cooler rcp45hotter rcp85cooler rcp85hotter; do
  OUT1=$O/fut_wind_cf1h_${C}.npz
  [ -f "$OUT1" ] && { echo "[$C] exists, skip" | tee -a $L; continue; }
  gate
  echo "---- $C $(date +%H:%M:%S) avail=${AV}G ----" | tee -a $L
  WX=$O/fut_wind_wx_${C}.npz SRC=future CLIMATE=$C OUT=$OUT1 Y0=2030 Y1=2050 NPROC=6 \
    /data/genenv/bin/python /data/08_stageD.py 2>&1 | grep --line-buffered -E 'mapping|stamps|hourly|DONE|Error|Traceback|Assertion' | tee -a $L
  [ -f "$OUT1" ] || { echo "FAIL $C" | tee -a $L; date > /data/logs/v5_stageD_all_ERR.flag; exit 1; }
done
echo "==== STAGE D ALL DONE $(date) ====" | tee -a $L
date > /data/logs/v5_stageD_all_DONE.flag
