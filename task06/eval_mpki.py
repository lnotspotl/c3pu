#!/usr/bin/env python3

import argparse
import json
import os

import torch
import tqdm
from model import EvictionPolicyModel

from cache_replacement.policy_learning.cache.cache import Cache
from cache_replacement.policy_learning.cache.eviction_policy import GreedyEvictionPolicy
from cache_replacement.policy_learning.cache.memtrace import MemoryTrace
from cache_replacement.policy_learning.cache_model.eviction_policy import LearnedScorer

JOB_TEMPLATE = """#!/usr/bin/bash --login
#BSUB -n {num_cpus}
#BSUB -W {job_time_minutes}
#BSUB -J {job_name}
#BSUB -o stdout.%J
#BSUB -e stderr.%J
{queue_options}

# Initialize conda environment
conda init >> /dev/null 2>&1
conda activate {conda_env_path} >> /dev/null 2>&1

# Go to task directory
cd {task_path}

# Start training
python3 eval_mpki.py \
    --eval_mpki=True \
    --trace_file={trace_file} \
    --cache_config_path={cache_config_path} \
    --model_config_path={model_config_path} \
    --checkpoint_path={checkpoint_path} \
    --results_file={results_file} \
    --num_cpus={num_cpus} \
"""

GPU_OPTIONS = """
#BSUB -q gpu
#BSUB -gpu "num=1:mode=shared:mps=no"
#BSUB -R "span[hosts=1]"
"""

CPU_OPTIONS = """"""


def submit_jobs(args: argparse.Namespace):
    # Find all possible trace folders
    trace_folders = os.listdir(args.input_folder)

    # Find all possible models
    models = os.listdir(args.output_folder)

    # Get conda env path
    CACHE_CONDA_ENV_PATH = os.environ.get("CACHE_CONDA_ENV_PATH")
    CACHE_TASK_PATH = os.environ.get("CACHE_TASK_PATH")

    assert CACHE_CONDA_ENV_PATH is not None, "Please set CACHE_CONDA_ENV_PATH environment variable."
    assert CACHE_TASK_PATH is not None, "Please set CACHE_TASK_PATH environment variable."

    for model in models:
        model_path = os.path.join(args.output_folder, model)
        if not os.path.isdir(model_path):
            print(f"Skipping model {model} as it does not exist")
            continue

        # Find what trace the model corresponds to
        trace_name = None
        for trace_folder in trace_folders:
            if model.startswith(trace_folder):
                trace_name = trace_folder
                break

        if trace_name is None:
            print(f"Skipping model {model} as it does not correspond to any trace")
            continue

        # Find config
        cache_config_path = os.path.join(model_path, "configs", "cache_config.json")
        model_config_path = os.path.join(model_path, "configs", "model_config.json")

        assert os.path.exists(cache_config_path), f"Cache config not found for model {model}"
        assert os.path.exists(model_config_path), f"Model config not found for model {model}"

        # Find the best checkpoint
        checkpoint_folder = os.path.join(model_path, "checkpoints")
        best_hr, best_checkpoint = None, None
        for checkpoint in os.listdir(checkpoint_folder):
            if not checkpoint.startswith("checkpoint"):
                continue

            hr = float(checkpoint.split("hr=")[-1].split("_")[0])
            if best_hr is None or hr > best_hr:
                best_hr = hr
                best_checkpoint = os.path.join(checkpoint_folder, checkpoint)

        assert best_checkpoint is not None, f"Best checkpoint not found for model {model}"

        # Submit job
        job_name = f"{model}_eval_mpki"
        job_script = JOB_TEMPLATE.format(
            job_name=job_name,
            num_cpus=args.num_cpus,
            queue_options=GPU_OPTIONS if args.use_gpu else CPU_OPTIONS,
            conda_env_path=CACHE_CONDA_ENV_PATH,
            task_path=CACHE_TASK_PATH,
            trace_file=os.path.join(args.input_folder, trace_name, "llc_access_trace.csv"),
            cache_config_path=cache_config_path,
            model_config_path=model_config_path,
            checkpoint_path=best_checkpoint,
            results_file=os.path.join(model_path, "eval_mpki_results2.csv"),
            job_time_minutes=args.job_time_minutes,
        )

        with open(f"{job_name}.sh", "w") as f:
            f.write(job_script)

        os.system(f"bsub < {job_name}.sh")
        os.remove(f"{job_name}.sh")
        print(f"Submitted job {job_name}")


class CacheObserver:
    def __init__(self, multiplier=1):
        self.reset()
        self.num_instructions = 1_000_000_000  # champsim simulates this number of instructions and than crashes
        self.multiplier = multiplier

    def update(self, hit):
        self.cache_hits += int(hit)
        self.cache_accesses += 1

    def reset(self):
        self.cache_accesses = 0
        self.cache_hits = 0

    @property
    def cache_misses(self):
        return self.cache_accesses - self.cache_hits

    def compute_mpki(self):
        return (self.cache_misses * self.multiplier / self.num_instructions) * 1000

    def compute_hit_rate(self):
        return self.cache_hits / self.cache_accesses


def evaluate_trace(trace_file: str, cache_config: dict, model_config: dict, checkpoint_path: str) -> float:
    # Determine number of cache accesses
    num_cache_accesses = sum(1 for _ in open(trace_file))
    multiplier = num_cache_accesses / int(1e7)
    num_cache_accesses = min(num_cache_accesses, int(1e7))
    if num_cache_accesses == int(1e7):
        multiplier = 1.0

    # Load memory trace into memory
    # Make sure the trace_file is the 'llc_access_trace.csv' file as generated by champsim
    memtrace = MemoryTrace(
        trace_file, cache_line_size=cache_config["cache_line_size"], max_look_ahead=num_cache_accesses
    )

    # Initialize Belady's optimal eviction policy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    learned_model = EvictionPolicyModel.from_config(model_config).to(device)
    learned_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    learned_scorer = LearnedScorer(learned_model)
    learned_policy = GreedyEvictionPolicy(learned_scorer)

    # Initialize cache
    cache = Cache.from_config(cache_config, eviction_policy=learned_policy)

    # Initialize observer
    cache_observer = CacheObserver(multiplier)

    # Calculate MPKI and hit rate
    with memtrace:
        for read_idx in tqdm.tqdm(range(num_cache_accesses), desc=f"trace: {trace_file}"):
            pc, address = memtrace.next()
            cache_observer.update(cache.read(pc, address))
    mpki = cache_observer.compute_mpki()
    hit_rate = cache_observer.compute_hit_rate()
    return mpki, hit_rate


def eval_mpki(args: argparse.Namespace):
    cache_config = json.load(open(args.cache_config_path))
    model_config = json.load(open(args.model_config_path))

    mpki, hit_rate = evaluate_trace(args.trace_file, cache_config, model_config, args.checkpoint_path)
    print(f"MPKI: {mpki:.2f}, Hit rate: {hit_rate:.2f}")

    with open(args.results_file, "w") as f:
        f.write("model,mpki,hit_rate\n")
        f.write(f"{args.trace_file},{mpki:.2f},{hit_rate:.2f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Submit script arguments
    parser.add_argument("--input_folder", type=str, default="inputs")
    parser.add_argument("--output_folder", type=str, default="outputs")
    parser.add_argument("--submit_script", type=bool, default=False)

    # Evaluation arguments
    parser.add_argument("--eval_mpki", type=bool, default=False)
    parser.add_argument("--trace_file", type=str)
    parser.add_argument("--cache_config_path", type=str)
    parser.add_argument("--model_config_path", type=str)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--results_file", type=str)
    parser.add_argument("--num_cpus", type=int, default=1)
    parser.add_argument("--use_gpu", type=bool, default=False)
    parser.add_argument("--job_time_minutes", type=int, default=500)

    args = parser.parse_args()

    if args.eval_mpki:
        assert args.submit_script is False, "Cannot submit script when evaluating MPKI"
        eval_mpki(args)

    if args.submit_script:
        assert args.eval_mpki is False, "Cannot evaluate MPKI when submitting script"
        submit_jobs(args)
