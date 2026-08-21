#!/bin/bash
# Sequential orchestrator for TGW precip/snow streams. Each stream = 14 parallel extract workers +
# disk-guarded batched Globus transfer. Sequential (not concurrent) because transfer-write and
# extract-read share one EBS volume -> 14 workers already saturate it; back-to-back avoids disk thrash
# and bounds peak raw to one drop dir (<=275 GB). Run under tmux; logs to /data/logs/tgw_pr_all.log.
set -u
mkdir -p /data/logs /data/tgw_raw /data/tgw_precip
CTL=/home/ubuntu/code/06_tgw_pr_ctl.py
PY=/data/tellenv/bin/python
SH=14
run(){
  local scen=$1 src=$2 y0=$3 y1=$4
  echo "===== $(date -u +%H:%M:%S) START $scen ($y0-$y1) ====="
  $PY $CTL --scenario "$scen" --src-path "$src" \
     --dropdir /data/tgw_raw/$scen --outdir /data/tgw_precip/$scen \
     --y0 $y0 --y1 $y1 --tmin $y0 --tmax $y1 --shards $SH --batch 60 \
     --min-free-gb 1500 --max-pending 250 --poll 20
  echo "===== $(date -u +%H:%M:%S) END $scen ====="
}
run historical  /historical_1980_2019/hourly  1980 2019
run rcp85hotter /rcp85hotter_2020_2059/hourly 2030 2050
run rcp85cooler /rcp85cooler_2020_2059/hourly 2030 2050
run rcp45hotter /rcp45hotter_2020_2059/hourly 2030 2050
run rcp45cooler /rcp45cooler_2020_2059/hourly 2030 2050
echo "########## ALL TGW PRECIP/SNOW STREAMS DONE $(date -u) ##########"
