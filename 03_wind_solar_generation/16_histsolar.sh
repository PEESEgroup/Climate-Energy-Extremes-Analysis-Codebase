#!/bin/bash
# Historical hourly solar, 1997-2019, native 12 km, from the surface archive just downloaded.
# Same script and same PVWatts configuration as the future arm, with the corrected pressure handling.
# Memory gate for the same reason stage D needed one: peaks add up across concurrently running stages.
set -u
L=/data/logs/v5_histsolar.log
O=/data/gen_targets/srgan3d_val/hist_v5/hist_solar_cf1h.npz
NEED_GB=45
echo "==== HIST SOLAR start $(date) ====" | tee -a $L
if [ -f "$O" ]; then echo 'exists, skip' | tee -a $L; exit 0; fi
while :; do
  AV=$(free -g | awk '/^Mem:/{print $7}')
  [ "$AV" -ge $NEED_GB ] && break
  echo "waiting for memory: ${AV}G available, need ${NEED_GB}G" | tee -a $L
  sleep 120
done
echo "---- start $(date +%H:%M:%S) avail=${AV}G ----" | tee -a $L
CLIMATE=historical SFCDIR=/data/tgw_hist_sfc OUT=$O YEARS=1997-2019 NPROC=5 \
  /data/genenv/bin/python /data/14_futsolar.py 2>&1 | grep --line-buffered -E 'plants|stamps|read |pysam|DONE|failed|Error|Traceback' | tee -a $L
if [ ! -f "$O" ]; then echo 'FAIL (check dmesg -T | grep -i oom)' | tee -a $L; date > /data/logs/v5_histsolar_ERR.flag; exit 1; fi
date > /data/logs/v5_histsolar_DONE.flag
