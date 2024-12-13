# Must be in cache/ directory

python3 -m HW3FeatureEngineering.Tags.cache_main \
  --experiment_base_dir=../outputs/eval \
  --experiment_name=astar_big4_dynamic_20000 \
  --cache_configs="cache_replacement/policy_learning/cache/configs/default.json" \
  --cache_configs="cache_replacement/policy_learning/cache/configs/eviction_policy/learned.json" \
  --memtrace_file="../traces/astar_313B_test_features.csv" \
  --config_bindings="associativity=16" \
  --config_bindings="capacity=2097152" \
  --config_bindings="eviction_policy.scorer.checkpoint=\"../outputs/train/astar_big4_dynamic/checkpoints/20000.ckpt\"" \
  --config_bindings="eviction_policy.scorer.config_path=\"../outputs/train/astar_big4_dynamic/model_config.json\"" \
  --warmup_period=0
