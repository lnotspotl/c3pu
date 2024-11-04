## Task 08

### Submit jobs

```bash
export CACHE_CONDA_ENV_PATH=<path-to-conda-env>
export CACHE_TASK_PATH=<path-to-task> # path to task08 folder
python3 submit_jobs.py 
```

### Evaluate MPKI

```bash
export CACHE_CONDA_ENV_PATH=<path-to-conda-env>
export CACHE_TASK_PATH=<path-to-task> # path to task05 folder
python3 eval_mpki.py
```

The script above will generate a CSV file with the MPKI and hit rate for each model.
