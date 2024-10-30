#!/usr/bin/env python3

import argparse
import os
from collections import namedtuple

import pandas as pd

from cache_replacement.policy_learning.cache.cache import Cache
from cache_replacement.policy_learning.cache.eviction_policy import BeladyScorer, GreedyEvictionPolicy
from cache_replacement.policy_learning.cache.memtrace import MemoryTrace


class MPKIObserver:
    def __init__(self):
        self.cache_accesses = 0
        self.cache_misses = 0

    def __call__(self, cache_access, eviction_decision):
        is_miss = eviction_decision.evict
        self.cache_misses += int(is_miss)
        self.cache_accesses += 1

    def compute_mpki(self):
        print(self.cache_misses, self.cache_accesses)
        return self.cache_misses / self.cache_accesses * 1000


def evaluate_mpki(trace_file: str, cache_config: dict) -> float:
    # Load memory trace into memory
    memtrace = MemoryTrace(trace_file, cache_line_size=cache_config["cache_line_size"])

    # Initialize Belady's optimal eviction policy
    belady_scorer = BeladyScorer(memtrace)
    belady_policy = GreedyEvictionPolicy(belady_scorer)

    # Initialize cache
    cache = Cache.from_config(cache_config, eviction_policy=belady_policy)

    # Initialize MPKI observer
    mpki_observer = MPKIObserver()

    # Calculate MPKI
    with memtrace:
        while not memtrace.done():
            pc, address = memtrace.next()
            cache.read(pc, address, observers=[mpki_observer])
    mpki = mpki_observer.compute_mpki()
    return mpki


def main(args: argparse.Namespace):
    # Find all available traces
    trace_folders = [
        item for item in os.listdir(args.input_folder) if os.path.isdir(os.path.join(args.input_folder, item))
    ]
    traces = list()
    Trace = namedtuple("Trace", ["name", "path"])
    for trace_folder in trace_folders:
        folder_path = os.path.join(args.input_folder, trace_folder)
        for trace in os.listdir(folder_path):
            if trace.endswith(".csv"):
                trace_path = os.path.join(folder_path, trace)
                traces.append(Trace(name=trace, path=trace_path))

    # Just one cache config - might want to evaluate MPKI for multiple cache configurations
    cache_config = {"cache_line_size": 64, "capacity": 2**21, "associativity": 16}

    # Evaluate MPKI for all traces
    mpki_values = dict()

    for trace in traces:
        print(f"Evaluating MPKI for {trace.name}")
        mpki = evaluate_mpki(trace.path, cache_config)
        mpki_values[trace.name] = mpki
        print(f"MPKI for {trace.name}: {mpki}")
        print("-" * 80)

    # Turn into a pandas frame and then to markdown
    df = pd.DataFrame(mpki_values.items(), columns=["Trace", "MPKI"])
    result_file = os.path.join(args.output_folder, "mpki.csv")
    df.to_csv(result_file, index=False)
    print(mpki_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default="inputs")
    parser.add_argument("--output_folder", type=str, default="outputs")
    args = parser.parse_args()

    main(args)
