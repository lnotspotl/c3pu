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


### Evaluate MPKI

```bash
export CACHE_CONDA_ENV_PATH=<path-to-conda-env>
export CACHE_TASK_PATH=<path-to-task> # path to task05 folder
python3 eval_mpki.py
```

The script above will generate a CSV file with the MPKI and hit rate for each model.

### Generate attention plots

```bash

# Creates a bunch of attention plots over time
python3 attplotgen.py --model_config <path-to-model-config> --cache_config <path-to-cache-config> --checkpoint <path-to-checkpoint> --memory_trace <path-to-memory-trace> --output_dir <path-to-output-dir>

# Combines attention plots into a video
python3 attcombine.py --input_dir <path-to-input-dir> --output_video <path-to-output-video>
```

## Results

| Trace | Cache Capacity (bytes) | Hit Rate (%) | MPKI  | Replacement Policy |
|-------|----------|--------------|-------|-----------|
| `astar_313B`  | `2**21`     | `38.26`       | `27.90` | `Bellady` |
| `astar_313B`  | `2**20`     | `31.02`       | `31.18` | `Parrot` |
| `astar_313B`  | `2**21`     | `32.52`       | `30.49` | `Parrot` |
| `astar_313B`  | `2**22`     | `34.29`       | `29.69` | `Parrot` |


| Trace | Cache Capacity (bytes) | Hit Rate (%) | MPKI  | Replacement Policy |
|-------|----------|--------------|-------|-----------|
| `mcf_250B`  | `2**21`     | `46.71`       | `65.96` | `Bellady` |
| `mcf_250B`  | `2**20`     | `44.41`       | `68.82` | `Parrot` |
| `mcf_250B`  | `2**21`     | `45.52`       | `67.45` | `Parrot` |
| `mcf_250B`  | `2**22`     | `45.99`       | `66.87` | `Parrot` |


| Trace | Cache Capacity (bytes) | Hit Rate (%) | MPKI  | Replacement Policy |
|-------|----------|--------------|-------|-----------|
| `milc_409B`  | `2**21`     | `6.54`       | `25.19` | `Bellady` |
| `milc_409B`  | `2**20`     | `2.76`       | `26.21` | `Parrot` |
| `milc_409B`  | `2**21`     | `2.78`       | `26.20` | `Parrot` |
| `milc_409B`  | `2**22`     | `3.57`       | `25.99` | `Parrot` |

