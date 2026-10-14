#!/bin/bash
cd /home/chenzongwei/rust_pyfunc/perf_sandbox/hm32_bench
DAY=${1:-20250303}
STEP=${2:-50}
OUT=/home/chenzongwei/rust_pyfunc/perf_sandbox/hm32_bench/sample_${DAY}.txt
: > "$OUT"
for f in $(ls /ssd_data/stock/${DAY}/market_data/*.csv | awk -v s=$STEP 'NR%s==1'); do
  code=$(basename "$f" | cut -d_ -f1)
  line=$(taskset -c 300 ./hm32_bench $DAY $code both 2>&1)
  read_ms=$(echo "$line" | sed -n '1s/.*read_ms=\([0-9.]*\).*/\1/p')
  n_allday=$(echo "$line" | grep -o 'allday   \] n=[ ]*[0-9]*' | grep -o '[0-9]*$')
  tot=$(echo "$line" | sed -n 's/.*TOTAL naive: alloc=\([0-9.]*\) build=\([0-9.]*\) matvec=\([0-9.]*\) std=\([0-9.]*\).*/\1 \2 \3 \4/p')
  fast=$(echo "$line" | sed -n 's/.*TOTAL fast_ms=\([0-9.]*\)  maxdiff_all=\(.*\)/\1 \2/p')
  nan=$(echo "$line" | grep -c 'nan=true')
  echo "$code $n_allday $read_ms $tot $fast $nan" >> "$OUT"
done
echo DONE >> "$OUT"
wc -l "$OUT"
