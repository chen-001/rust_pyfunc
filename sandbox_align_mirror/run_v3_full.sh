#!/bin/bash
# 依次跑全市场（同一时刻只有一个进程），每次记录开始与结束负载、墙钟、CPU、峰值内存
cd /home/chenzongwei/rust_pyfunc/sandbox_align_mirror
B=./target/release/align_mirror_v3
run() { # date engine outdir tag
  echo "== $4 start $(date +%T) load:$(cut -d' ' -f1-3 /proc/loadavg)"
  /usr/bin/time -f "wall=%e user=%U sys=%S maxrss_kb=%M exit=%x" $B $1 --engine $2 --out $3 2> results/v3full/$4.log
  tail -3 results/v3full/$4.log
  echo "== $4 end $(date +%T) load:$(cut -d' ' -f1-3 /proc/loadavg)"
}
run 20241231 base results/v3full/base base_20241231
run 20241231 fast results/v3full/fast1 fast1_20241231
run 20241231 fast results/v3full/fast2 fast2_20241231
run 20160630 fast results/v3full/fast1 fast1_20160630
run 20211105 fast results/v3full/fast1 fast1_20211105
run 20160630 base results/v3full/base base_20160630
run 20211105 base results/v3full/base base_20211105
echo ALL_DONE
