#!/bin/bash
# Full-time-series future TGW-3D download: ALL 4 scenarios, 2030-2050, FULL weekly (kstride=1).
# There is no deadline on this arm, so concurrency is GENTLE (2 extract shards, modest batch).
# It trickles over 3 to 4 days WITHOUT hogging the EBS/CPU needed for validation experiments.
# Sequential (shared EBS). Starts only AFTER the historical controller (tmux tgw3dctl) finishes.
# Pairs with 02_tgw_3d_ctl.py + 03_process_tgw_3d.py (pwat-patched). Stored -> /data/tgw_3d_future/<scen>/.
set -u
CTL=/home/ubuntu/code/02_tgw_3d_ctl.py
PY=/data/tellenv/bin/python
LOG=/data/logs/tgw_3d_future_all.log
mkdir -p /data/logs /data/tgw_3d_future

echo "$(date -u) waiting for historical controller (tmux tgw3dctl) to finish..." | tee -a $LOG
while tmux has-session -t tgw3dctl 2>/dev/null; do sleep 120; done
echo "$(date -u) historical done -> starting FUTURE full-time-series (all 4 scenarios, gentle)" | tee -a $LOG

run(){
  local scen=$1
  echo "===== $(date -u) START future $scen (2030-2050 FULL) =====" | tee -a $LOG
  $PY -u $CTL --scenario "$scen" --src-path "/${scen}_2020_2059/three_hourly" \
     --dropdir /data/tgw_3d_drop --outdir "/data/tgw_3d_future/$scen" \
     --tmin 2030 --tmax 2050 --kstride 1 --shards 3 --batch 60 --max-pending 120 --min-free-gb 900 --poll 30 \
     >> $LOG 2>&1
  echo "===== $(date -u) END future $scen =====" | tee -a $LOG
}
run rcp85hotter
run rcp45cooler
run rcp85cooler
run rcp45hotter
echo "########## ALL FUTURE TGW-3D DONE $(date -u) ##########" | tee -a $LOG
