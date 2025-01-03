Replaces the LSTM in PARROT with an RNN and removes the use of the cell state.
Modifies files noted in changes.txt to implement architecutre.

Example training command:
# Current directory is cache (../task04)
python3 -m task04.cache_model_main \
  --experiment_base_dir=task04/outputs/train \
  --experiment_name=astar_313B \
  --cache_configs=cache_replacement/policy_learning/cache/configs/default.json \
  --model_configs=task04/default.json \
  --model_bindings="loss=[\"ndcg\", \"reuse_dist\"]" \
  --model_bindings="address_embedder.max_vocab_size=5000" \
  --train_memtrace=cache_replacement/policy_learning/cache/traces/astar_train.csv \
  --valid_memtrace=cache_replacement/policy_learning/cache/traces/astar_valid.csv

Example evaluation command:
# Current directory is cache (../task04)
python3 -m task04.cache_main \
  --experiment_base_dir=task04/outputs/eval \
  --experiment_name=astar_313B \
  --cache_configs="cache_replacement/policy_learning/cache/configs/default.json" \
  --cache_configs="cache_replacement/policy_learning/cache/configs/eviction_policy/learned.json>
  --memtrace_file="cache_replacement/policy_learning/cache/traces/astar_test.csv" \
  --config_bindings="associativity=16" \
  --config_bindings="capacity=2097152" \
  --config_bindings="eviction_policy.scorer.checkpoint=\"task04/outputs/train/astar_313B/checkpoints/20000.ckpt\"" \
  --config_bindings="eviction_policy.scorer.config_path=\"task04/outputs/train/astar_313B/model_config.json\"" \
  --warmup_period=0