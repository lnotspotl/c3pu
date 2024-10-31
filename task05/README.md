## Task 05

### Parameters

We'll be varying the *capacity* parameter:

- **Small Size cache**: 1048576 (2**20)
- **Normal Size cache**: 2097152 (2**21)
- **Larger Size cache**: 4194304 (2**22)

The rest of the parameters (*cache_line_size = 64* and *associativity = 16*) stay the same.

### Submit jobs

```bash
export CACHE_CONDA_ENV_PATH=<path-to-conda-env>
export CACHE_TASK_PATH=<path-to-task> # path to task05 folder
python3 submit_jobs.py --cache_capacities 1048576 2097152 4194304
```
