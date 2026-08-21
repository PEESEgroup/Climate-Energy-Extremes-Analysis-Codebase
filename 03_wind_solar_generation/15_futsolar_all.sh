#!/bin/bash
# Future solar on the CERF policy fleets: ONE CLIMATE AT A TIME with a wide worker pool.
# Two at a time was tried and abandoned: the forked PySAM workers diverge from the parent page by
# page, so 14 workers added ~35 GB of private memory on top of the parent arrays and nearly caused
# an OOM. The driver arrays are now spilled to disk and read through the page cache, so the pool is
# cheap; the remaining peak is the read phase, which one climate at a time keeps comfortably safe.
set -u
L=/data/logs/v5_futsolar.log
O=/data/gen_targets/srgan3d_val/futgen_v5
echo "==== FUTURE SOLAR start $(date) ====" | tee -a $L
for C in rcp45cooler rcp85cooler rcp45hotter rcp85hotter; do
  if [ -f "$O/fut_solar_cf1h_${C}.npz" ]; then echo "[$C] exists, skip" | tee -a $L; continue; fi
  while :; do AV=$(free -g | awk '/^Mem:/{print $7}'); [ "$AV" -ge 70 ] && break
    echo "  [$C] waiting for memory: ${AV}G" | tee -a $L; sleep 120; done
  echo "---- $C start $(date +%H:%M:%S) avail=${AV}G ----" | tee -a $L
  CLIMATE=$C OUT=$O/fut_solar_cf1h_${C}.npz YEARS=2030-2050 NPROC=12 \
    /data/genenv/bin/python /data/14_futsolar.py > /data/logs/v5_futsolar_${C}.log 2>&1
  if [ -f "$O/fut_solar_cf1h_${C}.npz" ]; then
    echo "[$C] DONE $(date +%H:%M:%S)  $(grep -h '\[fs\] DONE' /data/logs/v5_futsolar_${C}.log | tail -1)" | tee -a $L
  else
    echo "[$C] FAIL $(date +%H:%M:%S) (see /data/logs/v5_futsolar_${C}.log)" | tee -a $L
    date > /data/logs/v5_futsolar_ERR.flag; exit 1
  fi
done
n=$(ls $O/fut_solar_cf1h_*.npz 2>/dev/null | wc -l)
echo "==== FUTURE SOLAR finished $(date), $n/4 ====" | tee -a $L
[ "$n" -eq 4 ] && date > /data/logs/v5_futsolar_DONE.flag
