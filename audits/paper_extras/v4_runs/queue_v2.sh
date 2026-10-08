#!/usr/bin/env bash
# YEU_CAU v4 P0.3: hang doi B — bench-v2.1 (tag bench-v2.1; dang ky 6a95505 + 452fb00 + 649087d) cho CA 9 seed x 2 che do x 16 nhanh;
# sau do kiem ghep cap du 240 buoc moi nhanh (P0.3a) va doc lap thu tu nhanh (P0.3c, seed 42 lay mau, thu tu dao nguoc).
cd "$(dirname "$0")/../../.." || exit 1
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate gpt_eco
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
OUT=audits/paper_extras/v4_runs; LOG=$OUT/queue_v2.log; mkdir -p $OUT/v2 $OUT/checks
ARMS="free_market,us_federal,saez,free_market+rl_aux,us_federal+rl_aux,saez+rl_aux,rl_mean_fixed,flat_pit0.10_cit0,flat_pit0.20_cit0,flat_pit0.30_cit0,flat_pit0.50_cit0,flat_pit0_cit0.10,flat_pit0_cit0.20,flat_pit0_cit0.30,flat_pit0_cit0.50"
REV="flat_pit0_cit0.50,flat_pit0_cit0.30,flat_pit0_cit0.20,flat_pit0_cit0.10,flat_pit0.50_cit0,flat_pit0.30_cit0,flat_pit0.20_cit0,flat_pit0.10_cit0,rl_mean_fixed,saez+rl_aux,us_federal+rl_aux,free_market+rl_aux,saez,us_federal,free_market"
echo "== BAT DAU $(date '+%F %T') commit $(git rev-parse --short HEAD)" >> $LOG
for S in 42 202 303 404 505 606 707 808 909; do
  for M in det exp; do
    EX=""; [ "$M" = "exp" ] && EX="--explore"
    echo "== v2 seed $S $M bat dau $(date '+%T')" >> $LOG
    python audits/paper_extras/benchmark_v2.py --config scenarios/em_baseline.yaml --checkpoint be/checkpoint/final_v037_seed$S/iter_100 \
      --episodes 10 --max-steps 240 --batched-inference $EX --arms "$ARMS" --json-out $OUT/v2/bench2_seed${S}_${M}.json > $OUT/v2/bench2_seed${S}_${M}.log 2>&1
    echo "== v2 seed $S $M xong rc=$? $(grep -a -o 'sha256 truoc/sau trung: [A-Za-z]*' $OUT/v2/bench2_seed${S}_${M}.log) $(date '+%T')" >> $LOG
  done
done
echo "== P0.3a ghep cap 240 buoc bat dau $(date '+%T')" >> $LOG
python audits/paper_extras/pairing_coverage_v2.py --checkpoint be/checkpoint/final_v037_seed42/iter_100 --episodes 2 \
  --csv-out $OUT/checks/pairing_coverage_seed42.csv > $OUT/checks/pairing_coverage_seed42.log 2>&1
echo "== P0.3a xong rc=$? $(date '+%T')" >> $LOG
echo "== P0.3c thu tu dao nguoc bat dau $(date '+%T')" >> $LOG
python audits/paper_extras/benchmark_v2.py --config scenarios/em_baseline.yaml --checkpoint be/checkpoint/final_v037_seed42/iter_100 \
  --episodes 10 --max-steps 240 --batched-inference --explore --arms "$REV" --json-out $OUT/checks/order_reversed_seed42_exp.json > $OUT/checks/order_reversed_seed42_exp.log 2>&1
echo "== P0.3c xong rc=$? $(date '+%T')" >> $LOG
echo "== HANG DOI B XONG $(date '+%F %T')" >> $LOG
