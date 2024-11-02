#!/usr/bin/env python3

import argparse
import os
import random
from collections import namedtuple

JOB_TEMPLATE = """#!/usr/bin/bash --login
#BSUB -n 1
#BSUB -W {job_time_minutes}
#BSUB -J {job_name}
#BSUB -o stdout.%J
#BSUB -e stderr.%J

# Initialize conda environment
conda init >> /dev/null 2>&1
conda activate {conda_env_path} >> /dev/null 2>&1

# Go to task directory
cd {task_path}

# Start evaluation
python3 eval_trace.py --trace_file={trace_file} --output_file={output_file}
"""


def main(args: argparse.Namespace):
    # Find all available traces
    trace_folders = [
        item for item in os.listdir(args.input_folder) if os.path.isdir(os.path.join(args.input_folder, item))
    ]
    traces = list()
    Trace = namedtuple("Trace", ["name", "path"])
    for trace_folder in trace_folders:
        folder_path = os.path.join(args.input_folder, trace_folder)
        trace = "llc_access_trace.csv"
        trace_path = os.path.join(folder_path, trace)
        assert os.path.isfile(trace_path), f"llc_access_trace.csv does not exist in {folder_path}"
        traces.append(Trace(name=f"{trace_folder}/{trace}", path=trace_path))

    CACHE_CONDA_ENV_PATH = os.environ.get("CACHE_CONDA_ENV_PATH")
    CACHE_TASK_PATH = os.environ.get("CACHE_TASK_PATH")
    assert CACHE_CONDA_ENV_PATH is not None, "Please set the CACHE_CONDA_ENV_PATH environment variable"
    assert CACHE_TASK_PATH is not None, "Please set the CACHE_TASK_PATH environment variable"
    assert os.path.exists(args.output_folder), f"{args.output_folder} does not exist"

    for trace in traces:
        job_name = f"eval_{trace.name}"
        output_file = os.path.join(args.output_folder, f"{trace.name}.txt")
        job_content = JOB_TEMPLATE.format(
            job_time_minutes=args.job_time_minutes,
            job_name=job_name,
            conda_env_path=CACHE_CONDA_ENV_PATH,
            task_path=CACHE_TASK_PATH,
            trace_file=trace.path,
            output_file=output_file,
        )
        script_path = f"tmp_{random.random()}.bash"
        with open(script_path, "w") as file:
            file.write(job_content)
        os.system(f"bsub < {script_path}")
        os.remove(script_path)
        print(f"Submitted job {job_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default="inputs")
    parser.add_argument("--output_folder", type=str, default="outputs")
    parser.add_argument("--job_time_minutes", type=int, default=60)
    args = parser.parse_args()

    main(args)
