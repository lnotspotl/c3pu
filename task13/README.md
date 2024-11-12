## Task13

### Using LLM as a replacement policy

```bash
python3 llm_evictor.py --memory_trace=<path-to-memory-trace> --rag_file=<path-to-rag-file>
```

### Zero-shot prompting

```bash
python3 prompter_zero_shot.py --rag_file=<path-to-rag-file> --query=<question>
```

### One-shot prompting


```bash
python3 prompter_one_shot.py --rag_file=<path-to-rag-file> --query=<question> --example=<one-example>
```

### Self-consistency


```bash
python3 prompter_self_consistency.py --rag_file=<path-to-rag-file> --query=<question> --num_queries=3
```
