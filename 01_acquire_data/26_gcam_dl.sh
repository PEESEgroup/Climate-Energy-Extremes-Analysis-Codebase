#!/bin/bash
cd /data
mkdir -p /data/gcam_usa
export TERM=xterm
echo "START $(date)" > /data/gcam_dl.log
/data/msdenv/bin/msdlive download --dataset-id yb23g-44274 --output-dir /data/gcam_usa >> /data/gcam_dl.log 2>&1
echo "EXITCODE=$?" >> /data/gcam_dl.log
echo "END $(date)" >> /data/gcam_dl.log
du -sh /data/gcam_usa >> /data/gcam_dl.log 2>&1
