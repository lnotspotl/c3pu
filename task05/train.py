#!/usr/bin/env python3

import os
import tqdm
import time
import argparse

# pytorch
import torch
import torch.nn as nn
import torch.optim as optim

# cache replacement imports
from cache_replacement.policy_learning.cache_model.model import EvictionPolicyModel
from cache_replacement.policy_learning.cache_model.eviction_policy import LearnedScorer
from cache_replacement.policy_learning.cache_model.schedules import LinearSchedule, ConstantSchedule
from cache_replacement.policy_learning.cache.eviction_policy import BeladyScorer
from cache_replacement.policy_learning.cache.cache import Cache
from cache_replacement.policy_learning.cache.evict_trace import EvictionEntry
from cache_replacement.policy_learning.cache_model.utils import as_batches


def measure_cache_hit_rate(
    memtrace_path,
    cache_config,
    eviction_model,
    model_prob_schedule,
    get_step,
    max_examples=None,
    use_oracle_scores=True,
    k=5,
    eval=False,
):
    if max_examples is None:
        max_examples = np.inf

    line_size = cache_config.get("cache_line_size")
    with MemoryTrace(memtrace_path, cache_line_size=line_size) as trace:

        def create_eviction_policy(model_prob):
            oracle_scorer = BeladyScorer(trace)
            learned_scorer = LearnedScorer(eviction_model)

            policies = [eviction_policy.GreedyEvictionPolicy(oracle_scorer), eviction_policy.GreedyEvictionPolicy(learned_scorer)]
            policy_probs = [1 - model_prob, model_prob]
            scoring_policy_index = 0 if use_oracle_scores else None

            return eviction_policy.MixturePolicy(
                policies, policy_probs, scoring_policy_index=scoring_policy_index
            )

        policy = create_eviction_policy(model_prob_schedule.value(get_step()))
        cache = Cache.from_config(cache_config, eviction_policy=policy, trace=trace)

        addresses = set()
        pcs = set()
        desc = "Collecting data from {} with mixture parameter: {}".format(
            memtrace_path, model_prob_schedule.value(get_step())
        )

        with tqdm.tqdm(desc=desc) as pbar:
            while not trace.done():
                data = []
                hit_rates = []
                model_prob = model_prob_schedule.value(get_step())
                cache.set_eviction_policy(create_eviction_policy(model_prob))
                # discard stats from previous iterations
                cache.hit_rate_statistic.reset()

                # callback to add trainig data to a data buffer
                def add_to_data(cache_access, eviction_decision):
                    entry = EvictionEntry(cache_access, eviction_decision)
                    data.append(entry)
                    addresses.add(cache_access.address)

                while len(data) < max_examples and not trace.done():
                    pc, address = trace.next()
                    pcs.add(pc)
                    cache.read(pc, address, [add_to_data])
                    hit_rates.append(cache.hit_rate_statistic.success_rate())
                    pbar.update(1)

                skip_len = len(hit_rates) // k
                hit_rates = hit_rates[skip_len : skip_len * (k - 1) + 1 : skip_len] + [hit_rates[-1]]
                yield data, hit_rates

def get_num_params(model):
    return sum(param.numel() for param in model.parameters())

def get_model_device(model):
    return next(model.parameters()).device

def evaluate_model(test_trace: str, model: nn.Module, cache_config: dict, warmup_period=int(2e3)):
    
    # Load test trace into memory
    memory_trace = MemoryTrace(test_trace, cache_line_size=cache_config["cache_line_size"])
    
    # Initialize greedy learned policy
    learned_scorer = LearnedScorer(eviction_model)
    greedy_policy = eviction_policy.GreedyEvictionPolicy(learned_scorer)

    # Initialize cache
    cache = Cache.from_config(cache_config, eviction_policy=greedy_policy)

    # Evaluate hit rate
    with memory_trace:

        # Warmup cache
        for i in range(warmup_steps):
            assert not memory_trace.done(), "memory trace depleted in warmup stage"
            pc, address = memory_trace.next()
            cache.read(pc, address)

        # Start tracking hit rate
        cache.hit_rate_statistic.reset()
        while not memory_trace.done():
            pc, address = memory_trace.next()
            cache.read(pc, address)

        # Calculate hit rate
        hit_rate = cache.hit_rate_statistic.success_rate()

    return hit_rate

def main(args: argparse.Namespace):

    # Paths to train, validation and test datasets    
    trace_folder = os.path.join(args.input_folder, args.trace)
    assert os.path.exists(trace_folder), f"Trace folder `{trace_folder}` does not exist."

    train_trace = os.path.join(trace_folder, "train.csv")
    valid_trace = os.path.join(trace_folder, "valid.csv")
    test_trace = os.path.join(trace_folder, "test.csv")

    # Path to output files
    experiment_folder = os.path.join(args.output_folder, args.experiment_name)
    checkpoint_folder = os.path.join(experiment_folder, "checkpoints")
    logs_file = os.path.join(experiment_folder, "logs.txt")
    os.makedirs(experiment_folder, exist_ok=args.override_outputs)


    # Cache config
    cache_config = {"cache_line_size": 64, "capacity": args.cache_capacity, "associativity": 16}

    # Model config
    model_config = {
        "address_embedder" : {"type": "dynamic-vocab", "embed_dim": 64, "max_vocab_size": 10000},
        "pc_embedder" : {"type": "dynamic-vocab", "embed_dim": 64, "max_vocab_size": 10000},
        "cache_line_embedder" : "address_embedder",
        "cache_pc_embedder" : "none",
        "positional_embedder" : {"type": "positional", "embed_dim": 128},
        "lstm_hidden_size" : 128,
        "max_attention_history" : 30,
        "sequence_length" : 80,
        "loss" : ["reuse_dist", "log_likelihood"],
        "lr" : 0.001,
        "epochs" : args.epochs
    }

    # Create DAgger scheduler
    schedule_config = {"type": "linear", "initial": 0.0, "final": 1.0, "num_steps": 200000, "update_freq": 10000}
    schedule = LinearSchedule(schedule_config["num_steps"], initial_p=schedule_config["initial"], final_p=schedule_config["final"])
    step = 0.0
    def get_step(): return step

    # Training device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device) # Allocate memory on this device by default

    # Create eviction model
    eviction_model = EvictionPolicyModel.from_config(model_config).to(device)
    print(f"Eviction model has {get_num_params(eviction_model) / 1e6:.3f} million parameters")

    # Create optimizer
    optimizer = optim.Adam(policy_model.parameters(), lr=model_config.get("lr"))

    # Start training
    for epoch in range(1, model_config["epochs"] + 1):

        # Book-keeping
        overall_loss = 0.0
        iterations = 0

        t_epoch_start = time.time()

        data_generator = measure_cache_hit_rate(
            filename, 
            cache_config,
            eviction_model,
            schedule,
            get_step
        )

        for train_data, hit_rates in data_generator:
            for batch_number, batch in enumerate(as_batches([train_data], batch_size, model_config["sequence_length"])):
                optimizer.zero_grad()
                losses = eviction_model.loss(batch, model_config["sequence_length"]//2)
                total_loss = sum(losses.values())
                total_loss.backward()
                optimizer.step()

                # Store checkpoint - yes, this stores the model at time 0
                if step % args.checkpoint_freq == 0:
                    hit_rate = f"{evaluate_model(test_trace, eviction_model, cache_config):.5f}"
                    model_name = "checkpoint_step={step}_hr={hit_rate}_torch.pt"
                    model_path = os.path.join(checkpoint_folder, model_name)
                    checkpoint_buffer = io.BytesIO()
                    torch.save(policy_model.state_dict(), checkpoint_buffer)
                    with open(model_path, "wb") as save_file:
                        save_file.write(checkpoint_buffer.get_value())

                overall_loss += total_loss.item()
                iterations += 1
                step += 1

                # DAgger break
                if batch_number == schedule_config["update_freq"]:
                    break

            description = f"Epoch: {epoch+1} | Average loss: {overall_loss/iterations}"
            print(description)

        t_epoch_end = time.time()
        t_epoch_s = t_epoch_end - t_epoch_start


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--trace", type=str)
    parser.add_argument("--cache_capacity", type=int, default=2**21)
    parser.add_argument("--input_folder", type=str, default="inputs")
    parser.add_argument("--output_folder", type=str, default="outputs")
    parser.add_argument("--override_outputs", type=bool, default=False)
    parser.add_argument("--checkpoint_freq", type=int, default=int(1e3))
    parser.add_argument("--epochs", type=int, default=int(1e7))
    args = parser.parse_args()

    main(args)
