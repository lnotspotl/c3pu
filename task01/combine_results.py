#!/usr/bin/env python3

import argparse
import os
from collections import namedtuple

import pandas as pd


def read_from_file(file_path: str):
    """Read MPKI and hit rate from a file"""

    with open(file_path, "r") as file:
        mpki = float(file.readline().strip().split(" ")[-1])
        hit_rate = float(file.readline().strip().split(" ")[-1])

    return mpki, hit_rate


def main(args: argparse.Namespace):
    stats = list()
    TraceStatistics = namedtuple("TraceStatistics", ["trace", "mpki", "hit_rate"])

    for trace_file in os.listdir(args.results_folder):
        if not trace_file.endswith(".txt"):
            continue
        trace_name = trace_file.split(".")[0]  # remove the .txt extension
        trace_path = os.path.join(args.results_folder, trace_file)
        mpki, hit_rate = read_from_file(trace_path)
        stats.append(TraceStatistics(trace_name, mpki, hit_rate))

    # Turn into a pandas frame and then to markdown
    df = pd.DataFrame(stats)
    df.to_csv(args.output_csv, index=False)
    print(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_folder", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="outputs.csv")
    args = parser.parse_args()

    main(args)
