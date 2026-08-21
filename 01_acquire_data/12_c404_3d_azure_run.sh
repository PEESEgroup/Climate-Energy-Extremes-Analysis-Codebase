#!/usr/bin/env bash
# Crash-restart wrapper for the Azure-zarr CONUS404-3D extractor. SAFETY: never pkills; touches only
# its own extractor. Idempotent (skips existing npz). STOP via touch /data/logs/c404_3d_azure.STOP.
set -u
PY=/data/kcenv/bin/python
EXE=/home/ubuntu/code/11_extract_c404_3d_azure.py
LOG=/data/logs/c404_3d_azure.log
STOP=/data/logs/c404_3d_azure.STOP
export N=${N:-14}; export ORDER=${ORDER:-asc}; export FLOOR_TB=${FLOOR_TB:-1.5}
mkdir -p /data/logs; rm -f "$STOP"
echo "================ AZURE WRAPPER START $(date -u) N=$N ORDER=$ORDER ================" >> "$LOG"
for pass in $(seq 1 500); do
  [ -f "$STOP" ] && { echo "STOPFLAG -> exit $(date -u)" >> "$LOG"; break; }
  echo "---------------- pass $pass START $(date -u) ----------------" >> "$LOG"
  nice -n 5 ionice -c3 "$PY" -u "$EXE" >> "$LOG" 2>&1
  rc=$?
  echo "---------------- pass $pass END rc=$rc $(date -u) ----------------" >> "$LOG"
  [ "$rc" -eq 0 ] && { echo "clean exit -> done $(date -u)" >> "$LOG"; break; }
  echo "rc=$rc -> restart after backoff" >> "$LOG"; sleep 15
done
echo "================ AZURE WRAPPER DONE $(date -u) ================" >> "$LOG"
