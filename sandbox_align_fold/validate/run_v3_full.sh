#!/bin/bash
# 第三版全市场实测：依次运行（同一时刻只有一个全市场进程），每次记录前后负载
cd "$(dirname "$0")/.."
B=./target/release/align_fold_v3
run() { # 名称 日期 环境变量...
  local name=$1 d=$2; shift 2
  local lf=results/v3_${name}_${d}.log
  echo "开始负载: $(uptime)" > $lf
  env "$@" /usr/bin/time -v $B $d out/v3_$name >> $lf 2>&1
  echo "结束负载: $(uptime)" >> $lf
  echo "$name $d 完成: $(grep -E 'per_stock' $lf)"
}
for spec in "$@"; do
  IFS=: read -r name d envs <<< "$spec"
  run $name $d ${envs//,/ }
done
