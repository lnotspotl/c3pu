## Task 06

### Task definition

#### Part A: Experiment with Different Embeddings
- Objective: Evaluate the impact of different embedding representations on model performance.
- Tasks:
    - Experiment with at least three different types of embeddings (e.g., word embeddings, positional embeddings, and learned embeddings).
    - Report the MPKI values for each embedding on the top three workloads.
    - Discuss how each embedding influences the model’s perception of workload characteristics and its subsequent cache replacement decisions.

#### Part B: Model Space Exploration
- Objective: Explore different configurations to find the optimal architecture for cache replacement.
- Tasks:
    -  Experiment with the following configurations for each model:
        - Activation Functions: ReLU, Sigmoid, Tanh, Softmax, etc.
        - Architecture Variations: Test different RNN architectures such as LSTMs and GRUs (see Chapter 10).
        - Width and Depth: Vary the number of neurons and layers.
    - Document your experiments and present the MPKI values for all configurations using bar charts.
    - Analyze which configurations performed best for each workload and why certain configurations might be more suited for specific types of cache behaviors.

#### Part C: Model Space Exploration
- Plot the attention layers for each embedding and explain how the model’s attention changes with varying embedding.

#### Parameters:
We will be varying the used RNN cell (LSTM, GRU, classical RNN).
Moreover, in case of the RNN cell, we will change the internal nonlinearity(*tanh* and *relu*)
Finally, to satisfy the requirement for changing the number of neurons, we will be varying the *lstm_hidden_size* parameter specifying the size of the densely connected layer within the RNN cell.

### Submit jobs

```bash
python3 submit_jobs.py
```