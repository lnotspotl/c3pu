#!/usr/bin/env python3

import argparse
import os
import shutil
from collections import namedtuple

JOB_TEMPLATE = """#!/usr/bin/bash --login
#BSUB -n {num_cpus}
#BSUB -W {job_time_minutes}
#BSUB -J {job_name}
#BSUB -o stdout.%J
#BSUB -e stderr.%J
{queue_options}
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
    --experiment_folder={experiment_folder} \
    --trace={trace_name} \
    --override_outputs={override_outputs} \
    --log_to_file={log_to_file} \
    --store_configs={store_configs} \
    --rnn_type={rnn_type} \
    --rnn_cell_nonlinearity={rnn_cell_nonlinearity} \
    --rnn_hidden_size={rnn_hidden_size} \
    --embedding_type={embedding_type} \
    --embedding_size={embedding_size} \
    --num_cpus={num_cpus}
"""

GPU_OPTIONS = """
#BSUB -q gpu
#BSUB -gpu "num=1:mode=shared:mps=no"
#BSUB -R "span[hosts=1]"
"""

CPU_OPTIONS = """"""


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

    assert CACHE_CONDA_ENV_PATH is not None, "Please set CACHE_CONDA_ENV_PATH environment variable."
    assert CACHE_TASK_PATH is not None, "Please set CACHE_TASK_PATH environment variable."

    for trace in traces:
        for hyperparam in args.hyperparameters:
            rnn_type, rnn_cell_nonlinearity, rnn_hidden_size, embedding_type, embedding_size = hyperparam
            experiment_name = trace.name + "_" + str(hyperparam).replace(" ", "").replace("(", "_").replace(")", "")
            script_path = os.path.join(job_folder, "submit_" + experiment_name + ".sh")
            with open(script_path, "w") as f:
                experiment_folder = os.path.join(args.output_folder, experiment_name)
                if args.override_outputs and os.path.exists(experiment_folder):
                    shutil.rmtree(experiment_folder, ignore_errors=True)
                os.makedirs(experiment_folder, exist_ok=True)
                f.write(
                    JOB_TEMPLATE.format(
                        num_cpus=args.num_cpus,
                        log_to_file=args.log_to_file,
                        job_time_minutes=str(args.job_time_minutes),
                        job_name="task06_" + experiment_name,
                        conda_env_path=CACHE_CONDA_ENV_PATH,
                        task_path=CACHE_TASK_PATH,
                        experiment_folder=experiment_folder,
                        trace_name=trace.name,
                        override_outputs=args.override_outputs,
                        store_configs=args.store_configs,
                        rnn_type=rnn_type,
                        rnn_cell_nonlinearity=rnn_cell_nonlinearity,
                        rnn_hidden_size=rnn_hidden_size,
                        embedding_type=embedding_type,
                        embedding_size=embedding_size,
                        queue_options=GPU_OPTIONS if args.queue == "gpu" else CPU_OPTIONS,
                    )
                )
            print("Job script written to:", script_path)
            print(f"Submitting job for trace {trace.name} with experiment name: {experiment_name}")
            os.system(f"bsub < {script_path}")
            print("Job submitted!")
            print("-" * 80)

    print("All jobs submitted!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Regular parameters
    parser.add_argument("--input_folder", type=str, default="inputs")
    parser.add_argument("--output_folder", type=str, default="outputs")
    parser.add_argument("--override_outputs", type=bool, default=False)
    parser.add_argument("--job_time_minutes", type=int, default=100)
    parser.add_argument("--num_cpus", type=int, default=2)
    parser.add_argument("--log_to_file", type=bool, default=True)
    parser.add_argument("--store_configs", type=bool, default=True)
    parser.add_argument("--queue", type=str, default="gpu")
    args = parser.parse_args()

    # Define hyper-parameter configurations
    Hyperparameters = namedtuple(
        "Hyperparameters", ["rnn_type", "rnn_cell_nonlinearity", "rnn_hidden_size", "embedding_type", "embedding_size"]
    )
    hyperparameters = [
        Hyperparameters("lstm", "tanh", 128, "dynamic-vocab", 64),
        Hyperparameters("gru", "tanh", 128, "dynamic-vocab", 64),
        Hyperparameters("rnn", "tanh", 128, "dynamic-vocab", 64),
        Hyperparameters("rnn", "relu", 128, "dynamic-vocab", 64),
        Hyperparameters("lstm", "tanh", 256, "dynamic-vocab", 64),
        Hyperparameters("lstm", "tanh", 128, "byte", 64),
        Hyperparameters("lstm", "tanh", 128, "dynamic-vocab", 128),
        Hyperparameters("lstm", "tanh", 128, "dynamic-vocab", 32),
    ]

    args.hyperparameters = hyperparameters

    main(args)
