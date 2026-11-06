#!/bin/bash
set -e
export RAYON_NUM_THREADS=64
export PATH=$HOME/.cargo/bin:$PATH
for d in 20240205 20240329 20240813 20240930 20241008 20250407 20251127 20260311; do
  if [ -s out/$d.json ]; then echo "$d 已有，跳过"; continue; fi
  /usr/bin/time -f "$d elapsed %e s maxrss %M KB" ./target/release/second_census $d > out/$d.json 2> out/$d.log
  grep -E "valid|elapsed" out/$d.log | tr '\n' ' '; echo
done
