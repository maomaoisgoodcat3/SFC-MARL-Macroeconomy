#!/usr/bin/env bash
# YEU_CAU v4 ngay 09/10: CHAY LAI cac viec loi CHUONG TRINH (Ray khong khoi dong duoc khi 2 tien trinh danh gia cung chay — xung dot cong /
# dang ky worker voi raylet; KHONG phai loi mo hinh). Nguoi dung duyet chay lai 09/10. Doi CA HAI worker xong roi chay TUAN TU MOT tien trinh
# (khong co Ray thu 2 song song -> tranh xung dot). Chi chay viec CHUA co JSON. Cung lenh/tham so voi day2_worker.sh.
cd "$(dirname "$0")/../../.." || exit 1
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate gpt_eco
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
OUT=audits/paper_extras/v4_runs; LOG=$OUT/day2_retry.log
ARMS="free_market+rl_aux,rl_mean_fixed,flat_pit0.10_cit0,flat_pit0.20_cit0,flat_pit0.30_cit0,flat_pit0.50_cit0,flat_pit0_cit0.10,flat_pit0_cit0.20,flat_pit0_cit0.30,flat_pit0_cit0.50"
V2ARMS="free_market,us_federal,saez,free_market+rl_aux,us_federal+rl_aux,saez+rl_aux,rl_mean_fixed,flat_pit0.10_cit0,flat_pit0.20_cit0,flat_pit0.30_cit0,flat_pit0.50_cit0,flat_pit0_cit0.10,flat_pit0_cit0.20,flat_pit0_cit0.30,flat_pit0_cit0.50"
echo "== retry CHO 2 worker xong $(date '+%F %T')" >> $LOG
until grep -q "^== v2 XONG" $OUT/day2_v2.log && grep -q "^== sweep XONG" $OUT/day2_sweep.log; do sleep 60; done
echo "== retry BAT DAU $(date '+%F %T') commit $(git rev-parse --short HEAD)" >> $LOG
for S in 404 505 606 707 808 909; do for M in det exp; do
  J=$OUT/v1/sweep_seed${S}_${M}.json; [ -f $J ] && continue; EX=""; [ "$M" = "exp" ] && EX="--explore"
  mv $OUT/v1/sweep_seed${S}_${M}.log $OUT/v1/sweep_seed${S}_${M}.FAILED_$(date +%H%M).log 2>/dev/null
  echo "== CHAY LAI quet seed $S $M bat dau $(date '+%T')" >> $LOG
  python -m be.benchmark --config scenarios/em_baseline.yaml --checkpoint be/checkpoint/final_v037_seed$S/iter_100 --episodes 10 \
    --max-steps 240 --batched-inference $EX --arms "$ARMS" --json-out $J > $OUT/v1/sweep_seed${S}_${M}.log 2>&1
  echo "== CHAY LAI quet seed $S $M xong rc=$? nap=$(grep -c 'Da nap checkpoint' $OUT/v1/sweep_seed${S}_${M}.log) $(date '+%T')" >> $LOG
done; done
for S in 42 202 303 404 505 606 707 808 909; do for M in det exp; do
  J=$OUT/v2/bench2_seed${S}_${M}.json; [ -f $J ] && continue; EX=""; [ "$M" = "exp" ] && EX="--explore"
  mv $OUT/v2/bench2_seed${S}_${M}.log $OUT/v2/bench2_seed${S}_${M}.FAILED_$(date +%H%M).log 2>/dev/null
  echo "== CHAY LAI v2 seed $S $M bat dau $(date '+%T')" >> $LOG
  python audits/paper_extras/benchmark_v2.py --config scenarios/em_baseline.yaml --checkpoint be/checkpoint/final_v037_seed$S/iter_100 \
    --episodes 10 --max-steps 240 --batched-inference $EX --arms "$V2ARMS" --json-out $J > $OUT/v2/bench2_seed${S}_${M}.log 2>&1
  echo "== CHAY LAI v2 seed $S $M xong rc=$? $(grep -a -o 'sha256 truoc/sau trung: [A-Za-z]*' $OUT/v2/bench2_seed${S}_${M}.log) $(date '+%T')" >> $LOG
done; done
echo "== retry XONG $(date '+%F %T')" >> $LOG
