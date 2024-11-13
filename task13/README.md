## Task13

### Using LLM as a replacement policy

```bash
python3 gpt_evictor.py --memory_trace=<path-to-memory-trace> --rag_file=<path-to-rag-file>
```

Generally, the LLM replacement policy does not work great and exhibits little to no performance.
There is still a lot of work to be done on this front.

### Zero-shot prompting

```bash
python3 prompter_zero_shot.py --rag_file=<path-to-rag-file> --query=<question>
```

Check out [example prompts and answers](https://github.com/lnotspotl/raggen/tree/main/task12)


### One-shot prompting


```bash
python3 prompter_one_shot.py --rag_file=<path-to-rag-file> --query=<question> --example=<one-example>
```

### Self-consistency


```bash
python3 prompter_self_consistency.py --rag_file=<path-to-rag-file> --query=<question> --num_queries=3
```

An example of where zero-shot prompting fails and self-consistency saves the day is for the following query:
**Could you please look at the memory trace of astar benchmark from accesses 1 to 200 and find an example of eviction where policy matches belady?**

For this query, zero-shot-prompted LLM responds with
```
I don't have the specific memory trace for accesses 1 to 200 of the astar benchmark to provide you with the exact example of an eviction where the policy matches Belady.
```

On the other hand, self-consistent LLM responds with (and succeeds)
```
Looking at the memory trace of the astar benchmark from accesses 1 to 200, we can see the following eviction where policy matches Belady:
- Workload: astar | PC: 0x409270 | Belady Evicted: 0x31232a40e93 | Policy04 Evicted: 0x31232a40e93                                                                            
                                                                                                                                                                                                  
In this case, both the Belady eviction and the Policy04 eviction have the same memory address, which is 0x31232a40e93.
```
