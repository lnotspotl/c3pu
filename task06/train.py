#!/usr/bin/env python3

import argparse
import io
import json
import logging
import os
import shutil
from collections import defaultdict

import numpy as np

# pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

# Updated EvictionPolicyModel
from model import EvictionPolicyModel
from rich.console import Console
from rich.logging import RichHandler

# cache replacement related
from cache_replacement.policy_learning.cache.cache import Cache
from cache_replacement.policy_learning.cache.evict_trace import EvictionEntry
from cache_replacement.policy_learning.cache.eviction_policy import BeladyScorer, GreedyEvictionPolicy, MixturePolicy
from cache_replacement.policy_learning.cache.memtrace import MemoryTrace
from cache_replacement.policy_learning.cache_model.eviction_policy import LearnedScorer
from cache_replacement.policy_learning.cache_model.schedules import LinearSchedule
from cache_replacement.policy_learning.cache_model.utils import as_batches

# Custom EvictionPolicyModel
from model import EvictionPolicyModel


def get_logger(
    name: str, log_to_stdout: bool = False, log_to_file: bool = False, level: str = "INFO", log_file: str = "./logs.txt"
) -> logging.Logger:
    assert log_to_stdout or log_to_file, "At least one of log_to_stdout or log_to_file must be True"
    assert level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], f"Invalid log level: {level}"

    handlers = list()
    if log_to_file:
        console = Console(file=open(log_file, "w"), width=120)  # TODO: this file should be closed
        handlers.append(RichHandler(console=console))

    if log_to_stdout:
        handlers.append(RichHandler(rich_tracebacks=True, tracebacks_show_locals=True))

    FORMAT = "%(name)s: %(message)s"
    logging.basicConfig(
        level=logging.CRITICAL,
        format=FORMAT,
        datefmt="[%X]",
        handlers=handlers,
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


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

            policies = [GreedyEvictionPolicy(oracle_scorer), GreedyEvictionPolicy(learned_scorer)]
            policy_probs = [1 - model_prob, model_prob]
            scoring_policy_index = 0 if use_oracle_scores else None

            return MixturePolicy(policies, policy_probs, scoring_policy_index=scoring_policy_index)

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
    learned_scorer = LearnedScorer(model)
    greedy_policy = GreedyEvictionPolicy(learned_scorer)

    # Initialize cache
    cache = Cache.from_config(cache_config, eviction_policy=greedy_policy)

    # Evaluate hit rate
    with memory_trace:
        # Warmup cache
        for i in range(warmup_period):
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


def store_config(filename, config):
    assert isinstance(config, dict)
    with open(filename, "w") as f:
        f.write(json.dumps(config))


def get_embedder_config(embedding_type: str, embedding_size: int):
    if embedding_type == "dynamic-vocab":
        return {"type": "dynamic-vocab", "embed_dim": embedding_size, "max_vocab_size": 10000}

    if embedding_type == "byte":
        return {"type": "byte", "embed_dim": embedding_size, "bytes_per_entry": embedding_size // 8}

    raise ValueError(f"Unknown embedding type: {embedding_type}")


def main(args: argparse.Namespace):
    # Paths to train, validation and test datasets
    trace_folder = os.path.join(args.input_folder, args.trace)
    assert os.path.exists(trace_folder), f"Trace folder `{trace_folder}` does not exist."

    train_trace = os.path.join(trace_folder, "train.csv")
    valid_trace = os.path.join(trace_folder, "valid.csv")
    test_trace = os.path.join(trace_folder, "test.csv")

    # Path to output files
    checkpoint_folder = os.path.join(args.experiment_folder, "checkpoints")
    config_folder = os.path.join(args.experiment_folder, "configs")
    stats_file = os.path.join(args.experiment_folder, "stats.npz")
    log_file = os.path.join(args.experiment_folder, "train.log")
    if args.override_outputs and os.path.exists(args.experiment_folder):
        shutil.rmtree(args.experiment_folder, ignore_errors=True)
    os.makedirs(args.experiment_folder, exist_ok=True)
    os.makedirs(checkpoint_folder, exist_ok=True)

    # Set up logger
    logger = get_logger(args.trace, log_to_stdout=True, log_to_file=args.log_to_file, log_file=log_file)

    # Cache config
    cache_config = {"cache_line_size": 64, "capacity": 2**21, "associativity": 16}

    # Model config
    model_config = {
        "address_embedder": get_embedder_config(args.embedding_type, args.embedding_size),
        "pc_embedder": get_embedder_config(args.embedding_type, args.embedding_size),
        "cache_line_embedder": "address_embedder",
        "cache_pc_embedder": "none",
        "rnn_type": args.rnn_type,
        "rnn_cell_nonlinearity": args.rnn_cell_nonlinearity,
        "positional_embedder": {"type": "positional", "embed_dim": 128},
        "rnn_hidden_size": args.rnn_hidden_size,
        "max_attention_history": 30,
        "sequence_length": 80,
        "loss": ["reuse_dist", "log_likelihood"],
        "lr": 0.001,
        "total_training_steps": args.total_training_steps,
        "batch_size": args.batch_size,
    }

    # Create DAgger scheduler
    schedule_config = {"type": "linear", "initial": 0.0, "final": 1.0, "num_steps": 200000, "update_freq": 10000}
    schedule = LinearSchedule(
        schedule_config["num_steps"], initial_p=schedule_config["initial"], final_p=schedule_config["final"]
    )
    step = 0.0

    # Store all the configs above
    if args.store_configs:
        os.makedirs(config_folder, exist_ok=True)
        model_config_path = os.path.join(config_folder, "model_config.json")
        cache_config_path = os.path.join(config_folder, "cache_config.json")
        schedule_config_path = os.path.join(config_folder, "schedule_config.json")
        store_config(model_config_path, model_config)
        store_config(cache_config_path, cache_config)
        store_config(schedule_config_path, schedule_config)

    def get_step():
        return step

    # Training device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)  # Allocate memory on this device by default

    # Create eviction model
    model_logger = get_logger(
        "[EvictionModel]", log_to_stdout=True, log_to_file=args.log_to_file, log_file=log_file
    )  # Note: logging is thread safe :)
    eviction_model = EvictionPolicyModel.from_config(model_config, logger=model_logger).to(device)
    logger.info(f"Eviction model has {get_num_params(eviction_model) / 1e6:.3f} million parameters")

    # Create optimizer
    optimizer = optim.Adam(eviction_model.parameters(), lr=model_config.get("lr"))
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=model_config.get("total_training_steps"), eta_min=model_config.get("lr") / 10
    )

    stats = defaultdict(list)

    # Start training
    while True:
        # Book-keeping
        overall_loss = 0.0
        iterations = 0

        data_generator = measure_cache_hit_rate(train_trace, cache_config, eviction_model, schedule, get_step)

        for train_data, hit_rates in data_generator:
            for batch_number, batch in enumerate(
                as_batches([train_data], model_config["batch_size"], model_config["sequence_length"])
            ):
                # Perform a training step
                optimizer.zero_grad()
                losses = eviction_model.loss(batch, model_config["sequence_length"] // 2)
                total_loss = sum(losses.values())
                total_loss.backward()
                optimizer.step()

                # Update learning rate
                lr_scheduler.step()

                # Update stats
                stats["step"].append(step)
                stats["total_loss"].append(total_loss.item())

                # Store checkpoint - yes, this stores the model at time 0
                if step % args.checkpoint_freq == 0:
                    logger.info(f"Step: {int(step)} | Evaluating model on test trace")
                    hit_rate = f"{evaluate_model(test_trace, eviction_model, cache_config):.5f}"
                    model_name = f"checkpoint_step={step}_hr={hit_rate}_torch.pt"
                    model_path = os.path.join(checkpoint_folder, model_name)
                    checkpoint_buffer = io.BytesIO()
                    torch.save(eviction_model.state_dict(), checkpoint_buffer)
                    with open(model_path, "wb") as save_file:
                        save_file.write(checkpoint_buffer.getvalue())
                    logger.info(f"Step: {int(step)} | Model saved at {model_path}")
                    logger.info(f"Step: {int(step)} | Storing stats to {stats_file}")
                    if os.path.exists(stats_file):
                        os.remove(stats_file)
                    np.savez(stats_file, **stats)

                overall_loss += total_loss.item()
                iterations += 1
                step += 1

                description = f"Step: {int(step)} | Average loss: {overall_loss/iterations}"
                logger.info(description)

                if step == model_config["total_training_steps"]:
                    logger.info("Training finished!")
                    return

                # DAgger break
                if batch_number == schedule_config["update_freq"]:
                    break

            description = f"Step: {int(step)} | Average loss: {overall_loss/iterations}"
            logger.info(description)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Regular parameters
    parser.add_argument("--trace", type=str)
    parser.add_argument("--experiment_folder", type=str)
    parser.add_argument("--input_folder", type=str, default="inputs")
    parser.add_argument("--override_outputs", type=bool, default=False)
    parser.add_argument("--checkpoint_freq", type=int, default=int(1e3))
    parser.add_argument("--total_training_steps", type=int, default=int(1e6))
    parser.add_argument("--log_to_file", type=bool, default=False)
    parser.add_argument("--store_configs", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=32)

    # Hyper-parameters
    parser.add_argument("--rnn_type", type=str, default="lstm")
    parser.add_argument("--rnn_cell_nonlinearity", type=str, default="tanh")
    parser.add_argument("--rnn_hidden_size", type=int, default=128)
    parser.add_argument("--embedding_type", type=str, default="dynamic-vocab")
    parser.add_argument("--embedding_size", type=int, default=64)

    args = parser.parse_args()

    main(args)
