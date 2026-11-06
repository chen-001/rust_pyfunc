#!/bin/bash
# 拓展二沙箱评估：100 个单日 + 每段第 5/10/15/20/25 天做 5 日合并
O=/hdd/user_home_unsafe/chenzongwei/sandbox_lfsr_ext2_out
B=./target/release/lfsr_ext2_sandbox
mapfile -t D < $O/dates.txt
for d in "${D[@]}"; do [ -f $O/${d}_w1.bin ] || $B day $d $O keep_ext1.txt 2>>$O/log_day.txt; done
for i in "${!D[@]}"; do
  if [ $(( (i % 25 + 1) % 5 )) -eq 0 ]; then d=${D[$i]}; [ -f $O/${d}_w5.bin ] || $B pool $d 5 $O 2>>$O/log_pool.txt; fi
done
echo DONE
