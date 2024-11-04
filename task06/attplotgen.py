#!/usr/bin/env python3

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch

from cache_replacement.environment.memtrace import MemoryTrace
from cache_replacement.policy_learning.cache.cache import Cache
from cache_replacement.policy_learning.cache.evict_trace import EvictionEntry
from cache_replacement.policy_learning.cache.eviction_policy import GreedyEvictionPolicy
from cache_replacement.policy_learning.cache_model.eviction_policy import LearnedScorer
from cache_replacement.policy_learning.cache_model.model import EvictionPolicyModel


class AttentionObserver:
    def __init__(self):
        self.eviction_entries = list()

    def __len__(self):
        return len(self.eviction_entries)

    def __call__(self, cache_access, eviction_decision):
        entry = EvictionEntry(cache_access, eviction_decision)
        self.eviction_entries.append(entry)


def load_config(filename: str):
    assert os.path.exists(filename), f"Provided config {filename} does not exist!"
    with open(filename, "r") as f:
        return json.loads(f.read())


def main(args: argparse.Namespace):
    # Load configs into memory
    model_config = load_config(args.model_config)
    cache_config = load_config(args.cache_config)

    # Load model - initialize weight randomly
    eviction_model = EvictionPolicyModel.from_config(model_config)
    learned_scorer = LearnedScorer(eviction_model)
    greedy_policy = GreedyEvictionPolicy(learned_scorer)

    # Load weights
    assert os.path.exists(args.checkpoint), f"Provided checkpoint path {args.checkpoint} does not exist!"
    checkpoint = torch.load(args.checkpoint, map_location=torch.device(args.model_device))
    eviction_model.load_state_dict(checkpoint)

    # Set default torch device
    torch.set_default_device(args.model_device)

    # Load memory trace
    cache_line_size = cache_config["cache_line_size"]
    memory_trace = MemoryTrace(args.memory_trace, max_look_ahead=int(1e5), cache_line_size=cache_line_size)

    # Prepare cache policy
    cache = Cache.from_config(cache_config, eviction_policy=greedy_policy)

    # Generate data for attention plotting
    attention_observer = AttentionObserver()
    with memory_trace:
        while not memory_trace.done():
            pc, address = memory_trace.next()
            cache.read(pc, address, observers=[attention_observer])

    # Check hit_rate if you want :)
    hit_rate = cache.hit_rate_statistic.success_rate()
    print(f"Hit rate: {hit_rate}")

    # Prepare for plot generation
    assert args.batch_size >= 1, f"Batch size has to be a positive number (>=1)! Given: {args.batch_size}"
    assert len(attention_observer) >= 0, "Attention observer is empty!"
    data = attention_observer.eviction_entries
    n_plots = len(data) // args.batch_size

    print(f"Generating {n_plots} plots.")
    subsequences = [data[i * n_plots : (i + 1) * n_plots] for i in range(args.batch_size)]

    hidden_state = None
    for fig_idx, batch in enumerate(zip(*subsequences)):
        probs, pred_reuse_distances, hidden_state, attention = eviction_model(
            [entry.cache_access for entry in batch], hidden_state, inference=True
        )

        # Warmup
        for _ in range(args.batch_size - 1):
            next(attention)

        # Finally, generate the much wanted plot :)
        entry, probs, attention, pred_reuse_distances = (
            batch[-1],
            probs[-1],
            list(next(attention)),
            pred_reuse_distances[-1],
        )
        cache_access = entry.cache_access
        num_cache_lines = len(cache_access.cache_lines)
        row_data = list()
        for i, (attention_weights, access) in enumerate(reversed(attention)):
            row_data.append([weight.item() for weight in attention_weights[:num_cache_lines]])
        scores = np.array(row_data)

        # Generate attention weights
        attention_probs = scipy.special.softmax(scores * args.scaling_factor, axis=1)

        # Start plotting
        figure_name = f"attention_{fig_idx}"
        figure_path = os.path.join(args.output_dir, figure_name + ".png")
        plt.imshow(attention_probs, cmap="gray", interpolation="nearest")
        plt.savefig(figure_path)
        print(f"Generated figure {figure_name} stored at {figure_path}")

    print("Done generating graphs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, help="Path to model config")
    parser.add_argument("--cache_config", type=str, help="Path to cache config")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--memory_trace", type=str, help="Path to memory trace")
    parser.add_argument("--output_dir", type=str, help="Directory where attention plots will be stored")
    parser.add_argument("--model_device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, help="Number of samples in a single batch")
    parser.add_argument("--scaling_factor", type=float, default=1.0, help="Attention score scaling factor")
    args = parser.parse_args()

    main(args)
