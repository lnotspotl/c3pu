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
export CACHE_TASK_PATH=<path-to-task> # path to task05 folder
python3 eval_mpki.py
```

The script above will generate a CSV file with the MPKI and hit rate for each model.
