#!/usr/bin/env python3

import argparse
import itertools
import os
from collections import namedtuple

JOB_TEMPLATE = """#!/usr/bin/bash --login
#BSUB -n {num_cpus}
#BSUB -W {job_time_minutes}
#BSUB -J {job_name}
#BSUB -o stdout.%J
#BSUB -e stderr.%J
#BSUB -q gpu
#BSUB -gpu "num=1:mode=shared:mps=no"
#BSUB -R "span[hosts=1]"

# Initialize conda environment
conda init >> /dev/null 2>&1
conda activate {conda_env_path} >> /dev/null 2>&1

# Go to task directory
cd {task_path}

# Start training
python3 train.py \
    --experiment_name={experiment_name} \
    --trace={trace_name} \
    --override_outputs={override_outputs} \
    --cache_capacity={cache_capacity}
"""


def main(args: argparse.Namespace):
    # Find all possible trace in inputs if it is a folder
    traces = list()
    Trace = namedtuple("Trace", ["name", "path"])
    print(f"Finding traces in input folder: {args.input_folder}")
    for name in os.listdir(args.input_folder):
        path = os.path.join(args.input_folder, name)
        if os.path.isdir(path):
            print("Found trace:", name)
            traces.append(Trace(name=name, path=path))
    print("-" * 80)

    # Create a folder for job scripts
    job_folder = os.path.join(args.output_folder, "jobs")
    os.makedirs(job_folder, exist_ok=True)

    # Get conda env path
    CACHE_CONDA_ENV_PATH = os.environ.get("CACHE_CONDA_ENV_PATH")
    CACHE_TASK_PATH = os.environ.get("CACHE_TASK_PATH")

    # assert CACHE_CONDA_ENV_PATH is not None, "Please set CACHE_CONDA_ENV_PATH environment variable."
    # assert CACHE_TASK_PATH is not None, "Please set CACHE_TASK_PATH environment variable."

    # For each trace, submit a job
    for trace, capacity in itertools.product(traces, args.cache_capacities):
        script_path = os.path.join(job_folder, f"submit_{trace.name}_capacity={capacity}.sh")
        with open(script_path, "w") as f:
            experiment_name = f"{trace.name}_capacity={capacity}"
            f.write(
                JOB_TEMPLATE.format(
                    num_cpus=args.num_cpus,
                    job_time_minutes=str(args.job_time_minutes),
                    job_name=f"train_{trace.name}_capacity={capacity}",
                    conda_env_path=CACHE_CONDA_ENV_PATH,
                    task_path=CACHE_TASK_PATH,
                    experiment_name=experiment_name,
                    trace_name=trace.name,
                    override_outputs=args.override_outputs,
                    cache_capacity=int(capacity),
                )
            )
        print("Job script written to:", script_path)
        print(f"Submitting job for trace {trace.name} with cache capacity {capacity}")
        os.system(f"bsub < {script_path}")
        print("Job submitted!")
        print("-" * 80)

    print("All jobs submitted!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default="inputs")
    parser.add_argument("--output_folder", type=str, default="outputs")
    parser.add_argument("--override_outputs", type=bool, default=False)
    parser.add_argument("--job_time_minutes", type=int, default=100)
    parser.add_argument("--num_cpus", type=int, default=2)
    parser.add_argument("--cache_capacities", type=int, nargs="+", default=[2**21])
    args = parser.parse_args()

    main(args)
