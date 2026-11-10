#!/bin/bash
# 第三版白名单验收：27 只股票 + 一个不在股票表、成交量放大 100 万倍的基金样本（今天和昨天都放）
cd /home/chenzongwei/rust_pyfunc/sandbox_align_mirror
T=results/v3wl; rm -rf $T; mkdir -p $T
BIN=./target/release/align_mirror_v3; D=20241231; P=20241230
for d in $D $P; do mkdir -p $T/root/$d/transaction $T/root/$d/market_data; done
for c in $(ls /ssd_data/stock/$D/transaction | head -2000 | awk 'NR%70==1{print substr($0,1,6)}'); do
  for d in $D $P; do
    ln -s /ssd_data/stock/$d/transaction/${c}_${d}_transaction.csv $T/root/$d/transaction/ 2>/dev/null
    ln -s /ssd_data/stock/$d/market_data/${c}_${d}_market_data.csv $T/root/$d/market_data/ 2>/dev/null
  done
done
export RUST_PYFUNC_LEVEL2_ROOT=$PWD/$T/root RUST_PYFUNC_LEVEL2_PATH=$PWD/$T/root
$BIN $D --out $T/a 2> $T/a.log; echo a_exit=$?; grep codes $T/a.log | cut -c1-120
FUND=010504
for d in $D $P; do
  awk -F, -v OFS=, 'NR==1{print;next}{$9=$9*1000000; print}' /ssd_data/stock/$d/transaction/${FUND}_${d}_transaction.csv > $T/root/$d/transaction/${FUND}_${d}_transaction.csv
done
$BIN $D --out $T/b 2> $T/b.log; echo b_exit=$?; grep codes $T/b.log | cut -c1-120
cmp $T/a/v3_${D}_fast.f32 $T/b/v3_${D}_fast.f32 && cmp $T/a/v3_${D}_fast.codes.json $T/b/v3_${D}_fast.codes.json && echo SAME_WITH_NONSTOCK_PRESENT
mkdir -p $T/basic_plus; cp /ssd_data/data/basic_info/symbol_map.csv $T/basic_plus/; echo "$FUND,99999" >> $T/basic_plus/symbol_map.csv
RUST_PYFUNC_BASIC_INFO_DIR=$PWD/$T/basic_plus $BIN $D --out $T/c 2> $T/c.log; echo c_exit=$?; grep codes $T/c.log | cut -c1-120
cmp -s $T/a/v3_${D}_fast.f32 $T/c/v3_${D}_fast.f32 && echo C_SAME || echo C_DIFFERS_AS_EXPECTED
RUST_PYFUNC_BASIC_INFO_DIR=$PWD/$T/nonexistent $BIN $D --out $T/d 2> $T/d.log; echo missing_exit=$?; cat $T/d.log
mkdir -p $T/basic_empty; printf 'symbol,pos\n' > $T/basic_empty/symbol_map.csv
RUST_PYFUNC_BASIC_INFO_DIR=$PWD/$T/basic_empty $BIN $D --out $T/e 2> $T/e.log; echo empty_exit=$?; cat $T/e.log
sha256sum /ssd_data/data/basic_info/symbol_map.csv
rm -rf $T/root
