#!/bin/bash
cd /home/chenzongwei/rust_pyfunc/sandbox_align_mirror
B=./target/release/align_mirror_v3
run() {
  echo "== $4 start $(date +%T) load:$(cut -d' ' -f1-3 /proc/loadavg)"
  /usr/bin/time -f "wall=%e user=%U sys=%S maxrss_kb=%M exit=%x" $B $1 --engine $2 --out $3 2> results/v3full/$4.log
  tail -2 results/v3full/$4.log
  echo "== $4 end $(date +%T) load:$(cut -d' ' -f1-3 /proc/loadavg)"
}
run 20241231 fast results/v3full/fast3 lowload_fast_a
run 20241231 fast results/v3full/fast4 lowload_fast_b
run 20241231 base results/v3full/base2 lowload_base
cmp results/v3full/fast3/v3_20241231_fast.f32 results/v3full/fast1/v3_20241231_fast.f32 && cmp results/v3full/fast4/v3_20241231_fast.f32 results/v3full/fast1/v3_20241231_fast.f32 && cmp results/v3full/base2/v3_20241231_base.f32 results/v3full/fast1/v3_20241231_fast.f32 && echo ALL_FOUR_RUNS_IDENTICAL
echo ALL_DONE
