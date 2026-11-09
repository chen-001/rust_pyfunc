#!/bin/bash
# 验收：(1) 不在股票表内的极端样本既不被读取也不影响结果；(2) 白名单缺失时报错退出。
# 用 30 只股票的小样本目录，避免重复跑全市场。
set -u
cd "$(dirname "$0")"
BIN=./target/release/align_day_vs_day
ROOT=accept/root
rm -rf "$ROOT"; mkdir -p accept
for d in 20241230 20241231; do for s in transaction market_data; do mkdir -p $ROOT/$d/$s; done; done
for c in $(ls /nvme/level2/20241231/transaction | head -30 | cut -c1-6); do
  for d in 20241230 20241231; do
    ln -sf /nvme/level2/$d/transaction/${c}_${d}_transaction.csv $ROOT/$d/transaction/
    ln -sf /nvme/level2/$d/market_data/${c}_${d}_market_data.csv $ROOT/$d/market_data/
  done
done
export RUST_PYFUNC_LEVEL2_ROOT=$ROOT RUST_PYFUNC_LEVEL2_PATH=$ROOT
$BIN 20241231 20241230 > accept/base.json 2>/dev/null; echo "base exit=$?"
# 放入一个不在股票表内的基金代码，成交量极端放大
FUND=159915; SRC=$(ls $ROOT/20241231/transaction | head -1 | cut -c1-6)
grep -q "^$FUND," /ssd_data/data/basic_info/symbol_map.csv && echo "错误：$FUND 在股票表内" && exit 1
for d in 20241230 20241231; do
  for s in transaction market_data; do
    sed "s/^$SRC,/$FUND,/" /nvme/level2/$d/$s/${SRC}_${d}_$s.csv | awk -F, -v OFS=, 'NR==1||1{ if(NR>1){$9=$9*100000} print }' > $ROOT/$d/$s/${FUND}_${d}_$s.csv
  done
done
$BIN 20241231 20241230 > accept/with_fund.json 2>/dev/null; echo "with_fund exit=$?"
cmp accept/base.json accept/with_fund.json && echo "PASS: 非股票样本不影响结果"
grep -c "\"$FUND\"" accept/with_fund.json | sed "s/^/基金代码出现次数: /"
RUST_PYFUNC_BASIC_INFO_DIR=accept/not_exist $BIN 20241231 20241230 > /dev/null 2> accept/err.txt; echo "missing_whitelist exit=$?"
head -c 200 accept/err.txt; echo
sha256sum /ssd_data/data/basic_info/symbol_map.csv
wc -l /ssd_data/data/basic_info/symbol_map.csv
