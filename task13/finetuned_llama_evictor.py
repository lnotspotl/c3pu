import argparse

import numpy as np
import torch
from cache import Cache
from finetune_llama_evictor import get_formatted_prompt, get_prompt
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from cache_replacement.policy_learning.cache.eviction_policy import EvictionPolicy
from cache_replacement.policy_learning.cache.memtrace import MemoryTrace


class LLamaPolicy(EvictionPolicy):
    def __init__(self, model_name, model_dir):
        seed = 42
        self._random = np.random.RandomState(seed)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self.model = PeftModel.from_pretrained(self.base_model, model_dir)

    def __call__(self, cache_access, access_times):
        # All scores the same.
        scores = {line: 0 for (line, _) in cache_access.cache_lines}
        if not scores:
            return None, scores

        # print("Prompting")
        del access_times
        pc, address = cache_access.pc, cache_access.address
        cache_lines = [hex(line) for (line, _) in cache_access.cache_lines]
        prompt = get_prompt(pc, address, cache_lines)

        formatted_prompt = get_formatted_prompt(prompt, "")
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            return_attention_mask=True,
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=512,
            temperature=0.1,
            top_p=0.95,
            repetition_penalty=1.0,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        resp = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        try:
            selected = int(resp.split("### Response:")[-1].strip(), 16)
        except Exception as e:
            print(f"Error parsing: {e}")
            selected = cache_access.cache_lines[self._random.randint(len(scores))][0]

        if selected not in [line for (line, _) in cache_access.cache_lines]:
            print("LLM picked non-existent cache line")
            selected = cache_access.cache_lines[self._random.randint(len(scores))][0]
        scores = {line: 0 for (line, _) in cache_access.cache_lines}
        return selected, scores


def main(args):
    cache_config = {"cache_line_size": 64, "capacity": 64 * 16 * 64, "associativity": 16}
    memtrace = MemoryTrace(
        args.memory_trace_file, cache_line_size=cache_config["cache_line_size"], max_look_ahead=int(1e5)
    )

    cache = Cache.from_config(cache_config, eviction_policy=LLamaPolicy())

    iter = 0
    with memtrace:
        while not memtrace.done():
            pc, address = memtrace.next()
            cache.read(pc, address)
            print(f"Iter: {iter} | Hit rate: {cache.hit_rate_statistic.success_rate()}")
            iter += 1
    hit_rate = cache.hit_rate_statistic.success_rate()
    print(hit_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory_trace_file", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B")
    args = parser.parse_args()

    main(args)
