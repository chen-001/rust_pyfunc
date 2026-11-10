#!/bin/bash
# 第三版白名单验收：(1) 不在股票表内的极端样本既不被读取也不影响结果；(2) 白名单缺失时报错退出。
set -u
cd "$(dirname "$0")"
BIN=./target/release/align_v3
ROOT=accept/root_v3
rm -rf "$ROOT"
for d in 20241230 20241231; do for s in transaction market_data; do mkdir -p $ROOT/$d/$s; done; done
for c in $(ls /nvme/level2/20241231/transaction | awk 'NR%100==1' | cut -c1-6); do
  for d in 20241230 20241231; do
    ln -sf /nvme/level2/$d/transaction/${c}_${d}_transaction.csv $ROOT/$d/transaction/
    [ -f /nvme/level2/$d/market_data/${c}_${d}_market_data.csv ] && ln -sf /nvme/level2/$d/market_data/${c}_${d}_market_data.csv $ROOT/$d/market_data/
  done
done
export RUST_PYFUNC_LEVEL2_ROOT=$ROOT RUST_PYFUNC_LEVEL2_PATH=$ROOT
$BIN 20241231 20241230 accept/v3_base > /dev/null 2>&1; echo "base exit=$?"
FUND=159915; SRC=$(ls $ROOT/20241231/transaction | head -1 | cut -c1-6)
grep -q "^$FUND," /ssd_data/data/basic_info/symbol_map.csv && { echo "错误：$FUND 在股票表内"; exit 1; }
for d in 20241230 20241231; do
  for s in transaction market_data; do
    awk -F, -v OFS=, -v f=$FUND 'NR==1{print;next}{$1=f; $9=$9*100000; print}' /nvme/level2/$d/$s/${SRC}_${d}_$s.csv > $ROOT/$d/$s/${FUND}_${d}_$s.csv
  done
done
$BIN 20241231 20241230 accept/v3_fund > /dev/null 2>&1; echo "with_fund exit=$?"
cmp accept/v3_base.f32 accept/v3_fund.f32 && cmp accept/v3_base.json accept/v3_fund.json && echo "PASS: 非股票样本不影响结果"
echo "基金代码出现次数: $(grep -o "\"$FUND\"" accept/v3_fund.json | wc -l)  股票数: $(python3 -c "import json;print(json.load(open('accept/v3_base.json'))['rows'])")"
RUST_PYFUNC_BASIC_INFO_DIR=accept/not_exist $BIN 20241231 20241230 accept/v3_err > /dev/null 2> accept/v3_err.txt; echo "missing_whitelist exit=$?"
grep -o "股票范围或原始目录检查失败[^,]*" accept/v3_err.txt | head -1
sha256sum /ssd_data/data/basic_info/symbol_map.csv
rm -rf "$ROOT" accept/v3_fund.* accept/v3_err.*
