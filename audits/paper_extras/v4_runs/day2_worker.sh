#!/usr/bin/env bash
# YEU_CAU v4, ngay 09/10 (nguoi dung: chay song song 2 viec, dung toan bo may). Moi worker lay viec theo THU TU danh sach rieng, nhan viec
# bang mkdir khoa (nguyen tu) trong v4_runs/claims/ -> 2 worker khong bao gio chay trung mot viec; viec da co JSON thi bo qua.
# Worker "sweep": quet thue phang con lai (bench-v1, tag bench-v1, khong sua; chuoi ARMS giong run_flat_sweep.sh) roi CHUYEN sang giup v2.1
# tu CUOI danh sach. Worker "v2": bench-v2.1 tu DAU danh sach; cuoi cung chay kiem ghep cap P0.3 (a) va (c) tren seed 42.
# Dung: bash day2_worker.sh sweep | bash day2_worker.sh v2
cd "$(dirname "$0")/../../.." || exit 1
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate gpt_eco
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
W=$1; OUT=audits/paper_extras/v4_runs; LOG=$OUT/day2_$W.log; mkdir -p $OUT/claims $OUT/checks
ARMS="free_market+rl_aux,rl_mean_fixed,flat_pit0.10_cit0,flat_pit0.20_cit0,flat_pit0.30_cit0,flat_pit0.50_cit0,flat_pit0_cit0.10,flat_pit0_cit0.20,flat_pit0_cit0.30,flat_pit0_cit0.50"
V2ARMS="free_market,us_federal,saez,free_market+rl_aux,us_federal+rl_aux,saez+rl_aux,rl_mean_fixed,flat_pit0.10_cit0,flat_pit0.20_cit0,flat_pit0.30_cit0,flat_pit0.50_cit0,flat_pit0_cit0.10,flat_pit0_cit0.20,flat_pit0_cit0.30,flat_pit0_cit0.50"
V2LIST="42:exp 202:det 202:exp 303:det 303:exp 404:det 404:exp 505:det 505:exp 606:det 606:exp 707:det 707:exp 808:det 808:exp 909:det 909:exp"
echo "== $W BAT DAU $(date '+%F %T') commit $(git rev-parse --short HEAD)" >> $LOG
claim() { mkdir "$OUT/claims/$1" 2>/dev/null; }
run_v2() {  # $1 seed $2 mode
  local S=$1 M=$2 J=$OUT/v2/bench2_seed${1}_${2}.json EX=""; [ "$M" = "exp" ] && EX="--explore"
  [ -f $J ] && return; claim v2_${S}_${M} || return
  echo "== v2 seed $S $M bat dau $(date '+%T')" >> $LOG
  python audits/paper_extras/benchmark_v2.py --config scenarios/em_baseline.yaml --checkpoint be/checkpoint/final_v037_seed$S/iter_100 \
    --episodes 10 --max-steps 240 --batched-inference $EX --arms "$V2ARMS" --json-out $J > $OUT/v2/bench2_seed${S}_${M}.log 2>&1
  echo "== v2 seed $S $M xong rc=$? $(grep -a -o 'sha256 truoc/sau trung: [A-Za-z]*' $OUT/v2/bench2_seed${S}_${M}.log) $(date '+%T')" >> $LOG
}
if [ "$W" = "sweep" ]; then
  for S in 404 505 606 707 808 909; do for M in det exp; do
    J=$OUT/v1/sweep_seed${S}_${M}.json; EX=""; [ "$M" = "exp" ] && EX="--explore"
    [ -f $J ] && continue; claim sweep_${S}_$M || continue
    echo "== quet seed $S $M bat dau $(date '+%T')" >> $LOG
    python -m be.benchmark --config scenarios/em_baseline.yaml --checkpoint be/checkpoint/final_v037_seed$S/iter_100 --episodes 10 \
      --max-steps 240 --batched-inference $EX --arms "$ARMS" --json-out $J > $OUT/v1/sweep_seed${S}_${M}.log 2>&1
    echo "== quet seed $S $M xong rc=$? nap=$(grep -c 'Da nap checkpoint' $OUT/v1/sweep_seed${S}_${M}.log) $(date '+%T')" >> $LOG
  done; done
  echo "== quet XONG $(date '+%T') -> giup v2.1 tu cuoi danh sach" >> $LOG
  for P in $(echo $V2LIST | tr ' ' '\n' | tac); do run_v2 ${P%%:*} ${P##*:}; done
else
  for P in $V2LIST; do run_v2 ${P%%:*} ${P##*:}; done
  if claim checks; then
    echo "== P0.3a ghep cap 240 buoc bat dau $(date '+%T')" >> $LOG
    python audits/paper_extras/pairing_coverage_v2.py --checkpoint be/checkpoint/final_v037_seed42/iter_100 --episodes 2 \
      --csv-out $OUT/checks/pairing_coverage_seed42.csv > $OUT/checks/pairing_coverage_seed42.log 2>&1
    echo "== P0.3a xong rc=$? $(date '+%T')" >> $LOG
    REV=$(echo $V2ARMS | tr ',' '\n' | tac | paste -sd, -)
    echo "== P0.3c thu tu dao nguoc bat dau $(date '+%T')" >> $LOG
    python audits/paper_extras/benchmark_v2.py --config scenarios/em_baseline.yaml --checkpoint be/checkpoint/final_v037_seed42/iter_100 \
      --episodes 10 --max-steps 240 --batched-inference --explore --arms "$REV" --json-out $OUT/checks/order_reversed_seed42_exp.json \
      > $OUT/checks/order_reversed_seed42_exp.log 2>&1
    echo "== P0.3c xong rc=$? $(date '+%T')" >> $LOG
  fi
fi
echo "== $W XONG $(date '+%F %T')" >> $LOG
