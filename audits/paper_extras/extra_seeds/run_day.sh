#!/usr/bin/env bash
# Train cac seed theo PREREG_extra_seeds.yaml (commit 3719961): train lien 100 iter, KHONG dung khi cong truot;
# sau khi train XONG het seed trong ngay moi chay cong iter 40 + iter 100 (chi de bao cao, khong song song voi train).
# Dung: bash run_day.sh 404 505
cd "$(dirname "$0")/../../.." || exit 1
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate gpt_eco
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
OUT=audits/paper_extras/extra_seeds; LOG=$OUT/run.log
echo "== NGAY BAT DAU $(date '+%F %T') commit $(git rev-parse --short HEAD) seeds: $*" >> $LOG
for S in "$@"; do
  echo "== seed $S train bat dau $(date '+%F %T')" >> $LOG
  python -m be.main --mode train --config scenarios/em_baseline.yaml --train-iters 100 --num-workers 4 \
    --train-batch-size 4000 --minibatch-size 256 --checkpoint-freq 10 --checkpoint-dir be/checkpoint \
    --scenario-name final_v037_seed$S --seed $S > $OUT/train_seed$S.log 2>&1
  echo "== seed $S train xong rc=$? $(date '+%F %T')" >> $LOG
done
for S in "$@"; do
  for IT in 40 100; do
    if [ -d be/checkpoint/final_v037_seed$S/iter_$IT ]; then
      echo "== seed $S cong iter $IT bat dau $(date '+%F %T')" >> $LOG
      python audits/final_run/gate_eval.py be/checkpoint/final_v037_seed$S/iter_$IT --c3-mode split \
        --json-out $OUT/gate_seed${S}_iter$IT.json > $OUT/gate_seed${S}_iter$IT.log 2>&1
      echo "== seed $S cong iter $IT xong rc=$? $(grep -a -o 'GATE_RESULT: [A-Z]*' $OUT/gate_seed${S}_iter$IT.log) $(date '+%F %T')" >> $LOG
    else
      echo "== seed $S KHONG co iter_$IT" >> $LOG
    fi
  done
done
echo "== NGAY XONG $(date '+%F %T')" >> $LOG
