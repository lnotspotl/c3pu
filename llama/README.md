## Results

### LLM RAG system

```
# Tested on raggen_1000.txt - Correct
question = "When the PC was 0x413a5b, what did LRU evict? Is it optimal according to Belady?"

answer = """
According to the provided context, when the PC was 0x413a5b, LRU evicted 0xb0b2ed1db0. 
 However, Belady Evicted: 0x160a69b9db0 was also recorded for the same PC. 
 So, according to Belady, it is not optimal, as Belady was evicted instead of LRU.
"""
```

```
# Tested on arch.pdf - Correct
question = "What is PC?"
answer = """Based on the provided context, "PC" likely refers to "Program Counter."""
```


```
# Tested on arch.pdf - Correct
question = "What is a cache hit?"
answer = """According to the provided context, a cache hit occurs when the processor finds a requested data item in the cache."""
```

```
# Tested on astar_313B_test_features.csv - Somewhat correct, says it does not know initially, but proceeds by giving the correct answer
question = "Give me an example of data/parameters correspoding to PC 0x4090c3"
answer = """
I don't have any information about the data or parameters corresponding to PC 0x4090c3 from my previous answers. My previous answers were based on the context from the provided documents, but I couldn't find any relevant information about PC 0x4090c3.

However, based on the provided documents, I can try to find some information about PC 0x4090c3.

According to the provided documents, PC 0x4090c3 is a memory address (0x309dcdc90f90) with a tag (0x184ee6e4) and a block offset (0x10). The cache set is 1086, and the cache way is 1.

Unfortunately, I couldn't find any specific information about the data or parameters corresponding to PC 0x4090c3. If you could provide more context or information about what you are looking for, I'll try to help you further.
"""
```

```
# Tested on astar_313B_test_features.csv - Incorrect
question = "Count how many times the PC 0x4090c3 occurs in the data."
answer = """I cannot find any information about the PC 0x4090c3 in the provided data."""
```

### LLM as a data scientist
```
# Tested on astar_313B_test_features.csv - Correct
question = "Count how many times the PC 0x4090c3 occurs in the data."
answer= 12454
```

```
# Tested on astar_313B_test_features.csv - Cannot produce code - Failure
question = "Which PC is encoutered/seen the most?""
```

```
# Tested on astar_313B_test_features.csv - Correct
question = "What are the 5 most frequent memory_address values?"
answer = """
   memory_address  count
0   0xc1b5f0bbda0     27
1  0x8ad3301ad780     26
2  0xed3fb8bd5240     26
3  0x24656f19a4d0     26
4  0xfc47cbba5a80     26
"""
```

```
# Tested on astar_313B_test_features.csv - Produces correct plot
question = "Can you plot a histogram for recency?"
## Produced plot below
```
![image](https://github.com/user-attachments/assets/364fa534-578b-40d8-8e92-828fbb4fb64d)

### Finetuned LLM

We finetuned our `unsloth/Llama-3.2-3B-Instruct`, trying to make it to become good at evicting cache lines by imitating the Belady’s algorithm.
Subsequently, we started asking the fine-tuned model questions as to what better features could be used for it to make better informed decisions.

We generated multiple checkpoints (`0`, `1000`, `2000`, `3000`, `4000`) during the fine-tuning process, allowing us to compare how the model's 
answer evolves over time, as the fine-tuning process proceeds.

The following are the answers to our prompt for the unfine-tuned model (`0`) and fully fine-tuned model (`4000`).


##### `0`
```
# Testing plain model (no fine-tuning)
Prompt(
    system="You are a helpful assistant helping with CPU cache line eviction.",
    user="""
        This is our current prompt to an eviction policy.
        What can we changes should be made to help the model better identify the cache line to evict and ultimately make the eviction policy more efficient?
        
        Prompt: 
        Current PC is <pc>
        Current address: <list-of-addresses>
        Cache lines are: <list-of-cache-lines>
        Eviction:
        """
    ),
    
Answer:

To improve the eviction policy's efficiency, here are some potential changes that can be made:

1. **Use a more sophisticated cache line selection algorithm**: Instead of simply selecting the least recently used (LRU) cache line, consider using algorithms like the Least Frequently Used (LFU) or the Most Recently Used (MRU) cache line. This can help reduce the number of cache lines that need to be evicted.

2. **Consider the cache line's contents**: In addition to the cache line's address, consider the contents of the cache line when selecting the line to evict. For example, if the cache line contains a frequently accessed value, it may be more beneficial to evict a cache line with a less frequently accessed value.

3. **Use a hybrid eviction policy**: Combine the LRU and LFU algorithms to create a hybrid eviction policy. For example, evict the LRU cache line if there are multiple cache lines with the same frequency of access.

4. **Consider the cache line's size**: If the cache line is larger than the smallest cache line, consider evicting the larger cache line to make room for the smaller cache line.

5. **Use a more efficient data structure**: Consider using a data structure like a binary search tree (BST) or a heap to store the cache lines and their access frequencies. This can help reduce the time complexity of the eviction algorithm.

6. **Consider the cache line's age**: In addition to the access frequency, consider the age of the cache line when selecting the line to evict. For example, if a cache line has been in the cache for a long time without being accessed, it may be more beneficial to evict it.

7. **Use a more efficient eviction algorithm**: Consider using an algorithm like the "First-In-First-Out" (FIFO) or the "Last-In-First-Out" (LIFO) algorithm to select the cache line to evict.

8. **Consider the cache line's priority**: If there are multiple cache lines with the same access frequency, consider assigning a priority to each cache line based on its contents or other factors.

Here's an updated version of the prompt:


Current PC is <pc>
Current address: <list-of-addresses>
Cache lines are: <list-of-cache-lines>
Eviction:
  - Use LRU or LFU algorithm to select cache line
  - Consider cache line contents and access frequency
  - Use a hybrid eviction policy
  - Consider cache line size
  - Use a data structure like BST or heap to store cache lines
  - Consider cache line age
  - Use an efficient eviction algorithm like FIFO or LIFO
  - Assign priority to cache lines based on contents or other factors

Note that the specific changes will depend on the requirements and constraints of the system.

Summarized by GPT4:

The model that’s not fine-tuned at all, gives us a general advice to use features that are fed into the LRU algorithms, or use a hybrid eviction algorithm.
```

##### `4000`

```
# Testing on fine-tuned model
Prompt(
    system="You are a helpful assistant helping with CPU cache line eviction.",
    user="""
        This is our current prompt to an eviction policy.
        What can we changes should be made to help the model better identify the cache line to evict and ultimately make the eviction policy more efficient?
        
        Prompt: 
        Current PC is <pc>
        Current address: <list-of-addresses>
        Cache lines are: <list-of-cache-lines>
        Eviction:
        """
    ),
    
Answer:

To improve the eviction policy, here are some potential changes that can be made:

1. **Use a more sophisticated cache line selection algorithm**: Instead of simply selecting the oldest cache line, consider using an algorithm that takes into account factors such as:
	* The frequency of access to each cache line
	* The distance between the current address and the cache line
	* The age of the cache line (but with a decay factor to prevent too much emphasis on the oldest line)
2. **Consider the cache line's contents**: If the cache line contains data that is likely to be accessed again soon (e.g., a frequently accessed instruction or a recently loaded data), it may be beneficial to keep it in the cache for a longer period.
3. **Use a more efficient eviction strategy**: Instead of simply removing the oldest cache line, consider using a strategy that:
	* Removes the least frequently accessed cache line
	* Removes the cache line that is most likely to be accessed again soon
	* Removes the cache line that is closest to the current address
4. **Consider the cache line's size**: If the cache line is larger than the current address, it may be beneficial to remove the smaller cache line to make room for the larger one.
5. **Use a more advanced cache replacement policy**: Consider using a policy that takes into account the entire cache, such as:
	* The Least Recently Used (LRU) policy
	* The Least Frequently Used (LFU) policy
	* The Optimal Cache Replacement (OCR) policy
6. **Consider the cache's capacity**: If the cache is full, consider removing the least frequently accessed cache line to make room for new data.
7. **Use a more efficient cache line representation**: Consider using a more efficient representation of the cache lines, such as a binary search tree or a hash table, to improve the eviction algorithm's performance.

Here's an updated prompt that incorporates some of these changes:

Current PC: <pc>
Current address: <list-of-addresses>
Cache lines are: <list-of-cache-lines>
Eviction:
  - Use LRU algorithm with decay factor
  - Consider cache line contents
  - Remove least frequently accessed cache line
  - Remove cache line closest to current address
  - Consider cache line size
  - Use Optimal Cache Replacement (OCR) policy
  - Consider cache capacity
  - Use binary search tree representation for cache lines

Note that the specific changes will depend on the requirements and constraints of the system.

Summarized by GPT4:
The fully fine-tuned model started talking about binary search tree representations for cache lines,
 incorporating cache capacity and cache line sizes into into the vector of features.
 It also told us to focus on frequently occuring access patterns in the traces.
 
 
Overall, a much better response that of the unfine-tuned model.
```

We evaluated the answers for all the above-mentioned checkpoints, getting five answers overall.
We then proceeded by asking `GPT4` to compare the individual answers, giving us an answer ordering:

```
The following is a prompt we fed into two LLM systems:

<prompt>

The first model answered the following:

<answer_model_1>

The second model answered the following:

<answer_model_2>

Based on these two answers, which of the two models performed better at giving advice
as to how to improve the prompt? Say just model one or two. Then, justify your decision.
```

The following model ordering came to being:
```
0 < 1000 < 2000 < 3000 < 4000
```

This shows that the longer-fine-tuned models do perform better at giving advice as to how to improve the features going into the neural cache eviction model.
We do agree with that ordering ourselves, as checkpoint `0` gives a general advice while the response produced by `4000` contains more specific techniques and suggestions.


