#!/bin/bash
for d in 20240205 20240329 20240813 20240930 20241008 20250407 20251127 20260311; do
  /usr/bin/time -f "$d elapsed %e s" ./target/release/trade_census $d > out/$d.json 2> out/$d.log
  grep -E "valid|elapsed" out/$d.log | tr '\n' ' '; echo
done
