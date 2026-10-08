#!/usr/bin/env bash
# YEU_CAU v4 P0.1/P0.4: hang doi A — (1) cong lai seed 42 (chi danh gia, checkpoint dong bang); (2) bench-v1 (be/benchmark.py, tag bench-v1,
# KHONG sua) cho 6 seed lap lai: lan 7 nhanh mac dinh (giong negwealth_7arms/run_one.sh) + lan quet (chuoi ARMS giong run_flat_sweep.sh).
# Dang ky truoc: PREREG_A8_replication.yaml (649087d). Seed goc 42/202/303 dung file cu (negwealth_7arms, flat_sweep) — da tai lap trung.
cd "$(dirname "$0")/../../.." || exit 1
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate gpt_eco
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
OUT=audits/paper_extras/v4_runs; LOG=$OUT/queue_v1.log; mkdir -p $OUT/v1 $OUT/gate42
ARMS="free_market+rl_aux,rl_mean_fixed,flat_pit0.10_cit0,flat_pit0.20_cit0,flat_pit0.30_cit0,flat_pit0.50_cit0,flat_pit0_cit0.10,flat_pit0_cit0.20,flat_pit0_cit0.30,flat_pit0_cit0.50"
echo "== BAT DAU $(date '+%F %T') commit $(git rev-parse --short HEAD)" >> $LOG
for IT in 40 100; do
  echo "== cong seed42 iter $IT bat dau $(date '+%T')" >> $LOG
  python audits/final_run/gate_eval.py be/checkpoint/final_v037_seed42/iter_$IT --c3-mode split \
    --json-out $OUT/gate42/gate_seed42_iter$IT.json > $OUT/gate42/gate_seed42_iter$IT.log 2>&1
  echo "== cong seed42 iter $IT xong rc=$? $(grep -a -o 'GATE_RESULT: [A-Z]*' $OUT/gate42/gate_seed42_iter$IT.log) $(date '+%T')" >> $LOG
done
for S in 404 505 606 707 808 909; do
  for M in det exp; do
    EX=""; [ "$M" = "exp" ] && EX="--explore"
    CK=be/checkpoint/final_v037_seed$S/iter_100
    echo "== v1 7nhanh seed $S $M bat dau $(date '+%T')" >> $LOG
    python -m be.benchmark --config scenarios/em_baseline.yaml --checkpoint $CK --episodes 10 --max-steps 240 --batched-inference $EX \
      --json-out $OUT/v1/bench7_seed${S}_${M}.json > $OUT/v1/bench7_seed${S}_${M}.log 2>&1
    echo "== v1 7nhanh seed $S $M xong rc=$? $(date '+%T')" >> $LOG
    grep -q "Da nap checkpoint" $OUT/v1/bench7_seed${S}_${M}.log || echo "== LOI: khong nap checkpoint seed $S" >> $LOG
    echo "== v1 quet seed $S $M bat dau $(date '+%T')" >> $LOG
    python -m be.benchmark --config scenarios/em_baseline.yaml --checkpoint $CK --episodes 10 --max-steps 240 --batched-inference $EX \
      --arms "$ARMS" --json-out $OUT/v1/sweep_seed${S}_${M}.json > $OUT/v1/sweep_seed${S}_${M}.log 2>&1
    echo "== v1 quet seed $S $M xong rc=$? $(date '+%T')" >> $LOG
  done
done
echo "== HANG DOI A XONG $(date '+%F %T')" >> $LOG
