python3 -m task02.cache_model__main \
  --experiment_base_dir=/./task02/outputs \
  --experiment_name=debug \
  --cache_configs=task02/cache_pc_embeddings.json \
  --model_bindings="loss=[\"ndcg\", \"reuse_dist\"]" \
  --model_bindings="address_embedder.max_vocab_size=5000" \
  --train_memtrace=cache_replacement/policy_learning/cache/traces/astar_train.csv \
  --valid_memtrace=cache_replacement/policy_learning/cache/traces/astar_valid.csv