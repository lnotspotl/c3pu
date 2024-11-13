## Task 06

#### Parameters:
We will be varying the used RNN cell (LSTM, GRU, classical RNN).
Moreover, in case of the RNN cell, we will change the internal nonlinearity(*tanh* and *relu*)
Finally, to satisfy the requirement for changing the number of neurons, we will be varying the *lstm_hidden_size* parameter specifying the size of the densely connected layer within the RNN cell.

When it comes to the embedding layer, we'll be varying the embedding layer type(dynamic-vocab, byte), embedding size (16, 32, 64, 128).

### Submit jobs

```bash
export CACHE_CONDA_ENV_PATH=<path-to-conda-env>
export CACHE_TASK_PATH=<path-to-task> # path to task06 folder
python3 submit_jobs.py 
```


### Evaluate MPKI

```bash
export CACHE_CONDA_ENV_PATH=<path-to-conda-env>
export CACHE_TASK_PATH=<path-to-task> # path to task06 folder
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
| Trace       | RNN Type | RNN Cell Nonlinearity | RNN Hidden Size | Embedding Type | Embedding Size | Hit Rate (%) | MPKI   | Policy  |
|-------------|----------|------------------------|-----------------|----------------|----------------|--------------|--------|---------|
| `astar_313B`  | `N/A`      | `N/A`                  | `N/A`             | `N/A`  | `N/A`             | `38.26`          | `27.90`    | `Bellady` |
| `astar_313B`  | `GRU`      | `tanh`                  | `128`             | `dynamic-vocab`  | `64`             | `31.96`          | `30.75`    | `Parrot1`  |
| `astar_313B`  | `LSTM`     | `tanh`                  | `128`             | `byte`          | `64`             | `28.89`          | `32.14`    | `Parrot2`  |
| `astar_313B`  | `LSTM`     | `tanh`                  | `128`             | `dynamic-vocab`  | `128`            | `32.72`          | `30.41`    | `Parrot3`  |
| `astar_313B`  | `LSTM`     | `tanh`                  | `128`             | `dynamic-vocab`  | `64`             | `32.45`          | `30.53`    | `Parrot4`  |
| `astar_313B`  | `LSTM`     | `tanh`                  | `128`             | `dynamic-vocab`  | `32`             | `32.42`          | `30.54`    | `Parrot5`  |
| `astar_313B`  | `LSTM`     | `tanh`                  | `256`             | `dynamic-vocab`  | `64`             | `32.39`          | `30.56`    | `Parrot6`  |
| `astar_313B`  | `RNN`      | `relu`                  | `128`             | `dynamic-vocab`  | `64`             | `31.89`          | `30.78`    | `Parrot7` |
| `astar_313B`  | `RNN`      | `tanh`                  | `128`             | `dynamic-vocab`  | `64`             | `32.06`          | `30.71`    | `Parrot8`  |

| Trace       | RNN Type | RNN Cell Nonlinearity | RNN Hidden Size | Embedding Type | Embedding Size | Hit Rate (%) | MPKI   | Policy  |
|-------------|----------|------------------------|-----------------|----------------|----------------|--------------|--------|---------|
| `mcf_250B`  | `N/A`      | `N/A`                  | `N/A`             | `N/A`  | `N/A`             | `46.71`          | `65.96`    | `Bellady` |
| `mcf_250B`  | `GRU`      | `tanh`                  | `128`             | `dynamic-vocab`  | `64`             | `43.27`          | `70.24`    | `Parrot1`  |
| `mcf_250B`  | `LSTM`     | `tanh`                  | `128`             | `byte`          | `64`             | `39.52`          | `74.88`    | `Parrot2`  |
| `mcf_250B`  | `LSTM`     | `tanh`                  | `128`             | `dynamic-vocab`  | `128`            | `46.10`          | `66.73`    | `Parrot3`  |
| `mcf_250B`  | `LSTM`     | `tanh`                  | `128`             | `dynamic-vocab`  | `64`             | `46.02`          | `66.84`    | `Parrot4`  |
| `mcf_250B`  | `LSTM`     | `tanh`                  | `128`             | `dynamic-vocab`  | `32`             | `45.89`          | `66.99`    | `Parrot5`  |
| `mcf_250B`  | `LSTM`     | `tanh`                  | `256`             | `dynamic-vocab`  | `64`             | `45.90`          | `66.98`    | `Parrot6`  |
| `mcf_250B`  | `RNN`      | `relu`                  | `128`             | `dynamic-vocab`  | `64`             | `43.26`          | `70.25`    | `Parrot7` |
| `mcf_250B`  | `RNN`      | `tanh`                  | `128`             | `dynamic-vocab`  | `64`             | `44.32`          | `68.94`    | `Parrot8`  |


| Trace       | RNN Type | RNN Cell Nonlinearity | RNN Hidden Size | Embedding Type | Embedding Size | Hit Rate (%) | MPKI   | Policy  |
|-------------|----------|------------------------|-----------------|----------------|----------------|--------------|--------|---------|
| `milc_409B` | `N/A`      | `N/A`                  | `N/A`             | `N/A`  | `N/A`             | `6.54`          | `25.19`    | `Bellady` |
| `milc_409B` | `GRU`      | `tanh`                  | `128`             | `dynamic-vocab`  | `64`             | `2.65`          | `26.24`    | `Parrot1`  |
| `milc_409B` | `LSTM`     | `tanh`                  | `128`             | `byte`          | `64`             | `1.98`          | `26.42`    | `Parrot2`  |
| `milc_409B` | `LSTM`     | `tanh`                  | `128`             | `dynamic-vocab`  | `128`            | `2.78`          | `26.21`    | `Parrot3`  |
| `milc_409B` | `LSTM`     | `tanh`                  | `128`             | `dynamic-vocab`  | `64`             | `2.75`          | `26.22`    | `Parrot4`  |
| `milc_409B` | `LSTM`     | `tanh`                  | `128`             | `dynamic-vocab`  | `32`             | `2.54`          | `26.27`    | `Parrot5`  |
| `milc_409B` | `LSTM`     | `tanh`                  | `256`             | `dynamic-vocab`  | `64`             | `2.58`          | `26.26`    | `Parrot6`  |
| `milc_409B` | `RNN`      | `relu`                  | `128`             | `dynamic-vocab`  | `64`             | `1.92`          | `26.44`    | `Parrot7` |
| `milc_409B` | `RNN`      | `tanh`                  | `128`             | `dynamic-vocab`  | `64`             | `2.00`          | `26.42`    | `Parrot8`  |


## Plots

![astar_plot8](https://github.com/user-attachments/assets/9e88fe53-089a-4aec-8d75-c4ccf402f467)
![mcf_plot8](https://github.com/user-attachments/assets/e31d4d62-4c6c-478e-bcb0-81ce80d111e1)
![milc_plot8](https://github.com/user-attachments/assets/01621426-65ed-41bb-ab22-949b3bf3037c)

