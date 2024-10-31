#!/usr/bin/env python3

import argparse
import os

import cache_replacement

JOB_TEMPLATE = """#!/usr/bin/bash --login
#BSUB -n {num_cpus}
#BSUB -W {job_time_minutes}
#BSUB -J {job_name}
#BSUB -o stdout.%J
#BSUB -e stderr.%J
#BSUB -q cpu
#BSUB -R "span[hosts=1]"

# Initialize conda environment
conda init >> /dev/null 2>&1
conda activate {conda_env_path} >> /dev/null 2>&1

cd {task_directory}

# Copy over champsim
cp {champsim_tar} . && tar -xf {champsim_tar} && cd ChampSim

git checkout 8798bed8117b2873e34c47733d2ba4f79b6014d4
git apply {champsim_patch}
chmod +x build_champsim.sh && ./build_champsim.sh bimodal no no no no create_llc_trace 1
cp {trace_path} . 
chmod +x run_champsim.sh && ./run_champsim.sh bimodal-no-no-no-no-create_llc_trace-1core 0 {num_instr_in_millions} {trace}

# Once finished, split 'llm_access_trace.csv` into train, validation and test splits
python3 {split_script} ./llc_access_trace.csv

# Move stuff to parent folder
mv train.csv ..
mv valid.csv ..
mv test.csv ..
mv llc_access_trace.csv ..
mv results_* ..

# Clean up
cd .. && rm -rf ChampSim && rm champsim.tar.gz
"""

DEFAULT_TRACES = [
    "astar_23B.trace.xz",
    "bwaves_98B.trace.xz",
    "bzip2_183B.trace.xz",
    "gcc_13B.trace.xz",
    "gamess_196B.trace.xz",
    "leslie3d_94B.trace.xz",
    "mcf_46B.trace.xz",
    "milc_360B.trace.xz",
    "namd_400B.trace.xz",
    "omnetpp_4B.trace.xz",
    "perlbench_53B.trace.xz",
    "povray_250B.trace.xz",
    "sjeng_358B.trace.xz",
    "sphinx3_883B.trace.xz",
]


def main(args: argparse.Namespace):
    CACHE_CONDA_ENV_PATH = os.environ.get("CACHE_CONDA_ENV_PATH")
    CACHE_TASK_PATH = os.environ.get("CACHE_TASK_PATH")

    assert CACHE_CONDA_ENV_PATH is not None, "Please set CACHE_CONDA_ENV_PATH environment variable."
    assert CACHE_TASK_PATH is not None, "Please set CACHE_TASK_PATH environment variable."

    for trace in args.traces:
        job_name = "generate_" + trace.replace(".xz", "")
        job_folder = os.path.join(args.output_folder, job_name)
        job_script = os.path.join(job_folder, "job.sh")
        champsim_tar = os.path.join(CACHE_TASK_PATH, "champsim.tar.gz")
        cache_replacement_directory = cache_replacement.__path__[0]
        champsim_path = os.path.join(
            cache_replacement_directory, "policy_learning", "cache", "traces", "champsim.patch"
        )
        num_instr_in_millions = str(trace.split("_")[-1].replace("B.trace.xz", "")) + "000"
        split_script = os.path.join(
            cache_replacement_directory, "policy_learning", "cache", "traces", "train_test_split.py"
        )
        trace_path = os.path.join(args.folder_with_traces, trace)

        os.makedirs(job_folder, exist_ok=True)
        with open(job_script, "w") as f:
            f.write(
                JOB_TEMPLATE.format(
                    num_cpus=args.num_cpus,
                    job_time_minutes=args.job_time_minutes,
                    job_name=job_name,
                    conda_env_path=CACHE_CONDA_ENV_PATH,
                    task_directory=job_folder,
                    champsim_tar=champsim_tar,
                    champsim_patch=champsim_path,
                    num_instr_in_millions=num_instr_in_millions,
                    trace=trace,
                    trace_path=trace_path,
                    split_script=split_script,
                )
            )

        print(f"Submitting job for trace: {trace}")
        os.system(f"bsub < {job_script}")
        print("Job submitted!")
        print("-" * 80)
        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_with_traces", type=str, required=True, help="Folder with all trace.xz files")
    parser.add_argument("--traces", type=str, nargs="+", default=DEFAULT_TRACES)
    parser.add_argument("--output_folder", type=str, default="outputs")
    parser.add_argument("--num_cpus", type=int, default=1)
    parser.add_argument("--job_time_minutes", type=int, default=3 * 60)
    args = parser.parse_args()

    main(args)
