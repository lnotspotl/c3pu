## Task 05

### Task definition

- Objective: Explore the impact of cache size on the model performance and behavior.
- Tasks:
    - Change the size of the Last Level Cache (LLC) for each model configuration:
        - **Very Small**
        - **Normal Size**
        - **Larger Size**
    - Retrain PARROT for each cache size configuration to see how it adapts to the changes
    - Plot the attention layers for each cache configuration and explain how the model’s attention changes with varying cache sizes.
    - Compare your attention layer results with those presented in the Glider paper (Section 5.5). Discuss the differences and try to relate them to the workload characteristics and even changes in cache size

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