#!/usr/bin/env bash
# Do ty le buoc co nguoi am tai san cho 7 nhanh benchmark mac dinh (dung thu tu goc -> tai lap dung so cu).
# Dung: bash run_one.sh <seed> <det|exp>. Cong cu: be/benchmark.py (khong doi tu tag bench-v1).
cd "$(dirname "$0")/../../.." || exit 1
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate gpt_eco
S=$1; M=$2; OUT=audits/paper_extras/negwealth_7arms; EX=""; [ "$M" = "exp" ] && EX="--explore"
echo "== seed $S $M bat dau $(date '+%F %T') commit $(git rev-parse --short HEAD)" >> $OUT/run.log
PYTHONIOENCODING=utf-8 python -m be.benchmark --config scenarios/em_baseline.yaml --checkpoint be/checkpoint/final_v037_seed$S/iter_100 \
  --episodes 10 --max-steps 240 --batched-inference $EX --json-out $OUT/bench7_seed${S}_${M}.json > $OUT/bench7_seed${S}_${M}.log 2>&1
echo "== seed $S $M xong rc=$? $(date '+%F %T')" >> $OUT/run.log
