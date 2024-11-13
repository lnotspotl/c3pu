#!/usr/bin/env python3

import argparse
import os

import torch
from cache import Cache
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, ProgressCallback, Trainer, TrainingArguments

from cache_replacement.policy_learning.cache.eviction_policy import BeladyScorer, GreedyEvictionPolicy
from cache_replacement.policy_learning.cache.memtrace import MemoryTrace


def get_prompt(pc, address, cache_lines):
    prompt = ""
    prompt += f"Current PC is {hex(pc)}\n"
    prompt += f"Current address: {hex(address)}\n"
    prompt += f"Cache lines are: {cache_lines}\n"
    prompt += "Eviction: "
    return prompt


def get_formatted_prompt(input, target):
    return f"### Input: {input}\n### Response: {target}"


def generate_dataset(args):
    cache_config = {"cache_line_size": 64, "capacity": 64 * 16 * 64, "associativity": 16}
    memtrace_path = "./sample_trace.csv"
    memtrace = MemoryTrace(memtrace_path, cache_line_size=cache_config["cache_line_size"], max_look_ahead=int(1e5))

    # Initialize Belady's optimal eviction policy
    belady_scorer = BeladyScorer(memtrace)
    belady_policy = GreedyEvictionPolicy(belady_scorer)

    cache = Cache.from_config(cache_config, eviction_policy=belady_policy)

    class BeladyObserver:
        def __init__(self):
            self.last_cache_lines = list()

        def __call__(self, cache_access, eviction_decision):
            cache_lines = [hex(line) for (line, _) in cache_access.cache_lines]
            self.last_cache_lines = cache_lines

    belady_observer = BeladyObserver()

    dataset = list()

    with memtrace:
        while not memtrace.done():
            pc, address = memtrace.next()
            cache.read(pc, address, observers=[belady_observer])

            belady_eviction = cache.last_evicted_cache_line
            if belady_eviction is not None:
                prompt = get_prompt(pc, address, belady_observer.last_cache_lines)
                assert str(hex(belady_eviction)) in belady_observer.last_cache_lines
                datapoint = {"input": prompt, "target": str(hex(belady_eviction))}
                dataset.append(datapoint)
    hit_rate = cache.hit_rate_statistic.success_rate()
    print(hit_rate)

    return dataset


def prepare_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "up_proj", "o_proj", "k_proj", "down_proj", "gate_proj", "v_proj"],
        lora_dropout=0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()

    return model, tokenizer


def main(args):
    train_data = generate_dataset(args)

    model, tokenizer = prepare_model(args.model_name)

    # Format the example
    formatted_text = [get_formatted_prompt(item["input"], item["target"]) for item in train_data]

    # Tokenize with labels for training
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors=None,
        )
        # Create labels (same as input_ids for causal LM)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    # Create dataset
    dataset = Dataset.from_dict({"text": formatted_text})
    tokenized_dataset = dataset.map(tokenize_function, remove_columns=dataset.column_names, batched=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=args.logging_steps,
        save_strategy="no",
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        report_to="none",
        disable_tqdm=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        callbacks=[ProgressCallback()],
    )

    trainer.train()


if __name__ == "__main__":
    # Make sure the HF token is set
    assert os.environ.get("HF_TOKEN") is not None, "HF_TOKEN is not set"

    parser = argparse.ArgumentParser()
    parser.add_argument("--memory_trace_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    args = parser.parse_args()

    main(args)
