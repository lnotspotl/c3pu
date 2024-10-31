## Generating traces

### Submit jobs

```bash
export CACHE_CONDA_ENV_PATH=<path-to-conda-env>
export CACHE_TASK_PATH=<path-to-task> # path to task00 folder
export TRACE_FOLDER=<path-to-folder-with-traces>  #.trace.xz
python3 submit_jobs.py --folder_with_traces=$TRACE_FOLDER
```
