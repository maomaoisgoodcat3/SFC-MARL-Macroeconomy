#!/usr/bin/env bash
# Chay quet thue phang + rl_mean_fixed (dang ky truoc: audits/final_run/PREREG_flat_sweep.md, commit 4e0327d).
# 3 seed x 2 che do x 11 nhanh (rl_learned, free_market+rl_aux, rl_mean_fixed, 8 nhanh flat) x 10 episode ghep cap.
# Gioi han 3 gio may: kiem truoc moi lan chay; vuot -> dung, ghi ly do.
cd "$(dirname "$0")/../.." || exit 1
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate gpt_eco
OUT=audits/final_run/logs/flat_sweep
ARMS="free_market+rl_aux,rl_mean_fixed,flat_pit0.10_cit0,flat_pit0.20_cit0,flat_pit0.30_cit0,flat_pit0.50_cit0,flat_pit0_cit0.10,flat_pit0_cit0.20,flat_pit0_cit0.30,flat_pit0_cit0.50"
LIMIT=10800
T0=$(date +%s)
echo "BAT DAU $(date '+%F %T')"
for S in 42 202 303; do
  for MODE in det exp; do
    EL=$(( $(date +%s) - T0 ))
    if [ $EL -gt $LIMIT ]; then echo "DUNG: vuot 3 gio may ($EL s) truoc seed $S $MODE"; exit 3; fi
    EX=""; [ "$MODE" = "exp" ] && EX="--explore"
    echo "== seed $S $MODE bat dau $(date '+%T') (da chay $EL s)"
    PYTHONIOENCODING=utf-8 python -m be.benchmark --config scenarios/em_baseline.yaml \
      --checkpoint be/checkpoint/final_v037_seed$S/iter_100 --episodes 10 --max-steps 240 --batched-inference $EX \
      --arms "$ARMS" --json-out $OUT/sweep_seed${S}_${MODE}.json > $OUT/sweep_seed${S}_${MODE}.log 2>&1
    RC=$?
    echo "== seed $S $MODE xong rc=$RC $(date '+%T')"
    grep -q "Da nap checkpoint" $OUT/sweep_seed${S}_${MODE}.log || { echo "LOI: khong nap duoc checkpoint seed $S"; exit 4; }
    [ $RC -ne 0 ] && { echo "LOI chuong trinh seed $S $MODE"; exit 5; }
  done
done
echo "XONG $(date '+%F %T'), tong $(( $(date +%s) - T0 )) s"
