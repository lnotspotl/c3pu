# Must be in cache/ directory.

python3 -m HW3FeatureEngineering.Big4.cache_model_main \
  --experiment_base_dir=../outputs/train \
  --experiment_name=astar_big4_dynamic \
  --cache_configs=cache_replacement/policy_learning/cache/configs/default.json \
  --model_bindings="loss=[\"ndcg\", \"reuse_dist\"]" \
  --model_configs=HW3FeatureEngineering/default-byte.json \
  --train_memtrace=/share/ece592f24/team-ajak/traces_features/astar_313B_train_features.csv \
  --valid_memtrace=/share/ece592f24/team-ajak/traces_features/astar_313B_valid_features.csv
