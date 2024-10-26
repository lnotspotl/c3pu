#!/bin/bash
#BSUB -n 1
#BSUB -J task03
#BSUB -W 24:00
#BSUB -q gpu
#BSUB -gpu "num=2:mode=shared:mps=no"
#BSUB -o out.%J
#BSUB -e err.%J

cd /share/ece592f24/team-ajak/task03/cache
module load cuda
source activate /share/ece592f24/team-ajak/task03/cache/cache_env

python3 -m task03.cache_model_main03 \
  --experiment_base_dir=./outputs/train \
  --experiment_name=run1 \
  --cache_configs=cache_replacement/policy_learning/cache/configs/default.json \
  --model_bindings="loss=[\"ndcg\", \"reuse_dist\"]" \
  --model_bindings="address_embedder.max_vocab_size=5000" \
  --train_memtrace=/share/ece592f24/team-ajak/traces/astar_313B_train.csv \
  --valid_memtrace=/share/ece592f24/team-ajak/traces/astar_313B_valid.csv

conda deactivate