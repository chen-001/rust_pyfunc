#!/bin/bash
for d in 20240813 20240329 20240205 20250407 20240930 20241008; do
  echo "=== $d $(date +%H:%M:%S) ==="
  /usr/bin/time -f "elapsed %e s" ./target/release/trade_census $d > out/$d.json 2> out/$d.log
  tail -2 out/$d.log
  ls -lh out/$d.json | awk '{print $5}'
done
