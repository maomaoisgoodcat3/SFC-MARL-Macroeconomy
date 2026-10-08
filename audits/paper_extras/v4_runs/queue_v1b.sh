#!/usr/bin/env bash
# YEU_CAU v4 P0.1 (dem 08/10, nguoi dung doi quet thue phang sang 09/10): CHI lan chay 7 nhanh mac dinh bench-v1 (be/benchmark.py, tag
# bench-v1, KHONG sua; giong negwealth_7arms/run_one.sh) cho cac seed x che do con lai. Bo qua cai da co JSON.
cd "$(dirname "$0")/../../.." || exit 1
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate gpt_eco
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
OUT=audits/paper_extras/v4_runs; LOG=$OUT/queue_v1.log
echo "== BAT DAU queue_v1b $(date '+%F %T') commit $(git rev-parse --short HEAD)" >> $LOG
for S in 404 505 606 707 808 909; do
  for M in det exp; do
    J=$OUT/v1/bench7_seed${S}_${M}.json
    [ -f $J ] && continue
    EX=""; [ "$M" = "exp" ] && EX="--explore"
    echo "== v1 7nhanh seed $S $M bat dau $(date '+%T')" >> $LOG
    python -m be.benchmark --config scenarios/em_baseline.yaml --checkpoint be/checkpoint/final_v037_seed$S/iter_100 --episodes 10 \
      --max-steps 240 --batched-inference $EX --json-out $J > $OUT/v1/bench7_seed${S}_${M}.log 2>&1
    echo "== v1 7nhanh seed $S $M xong rc=$? $(grep -c 'Da nap checkpoint' $OUT/v1/bench7_seed${S}_${M}.log) $(date '+%T')" >> $LOG
  done
done
echo "== HANG DOI A XONG (chi 7 nhanh; quet doi sang 09/10) $(date '+%F %T')" >> $LOG
