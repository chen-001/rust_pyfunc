#!/bin/bash
set -e
export RAYON_NUM_THREADS=50
for d in 20240205 20240329 20240813 20240930 20241008 20250407 20251127 20260311; do
  /usr/bin/time -f "$d elapsed %e s maxrss %M KB" ./target/release/ms_census $d 2> out/$d.log
  grep -E "valid|elapsed" out/$d.log | tr '\n' ' '; echo
done
