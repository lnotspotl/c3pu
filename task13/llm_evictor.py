#!/usr/bin/env python3

import argparse
import json
import logging
import os
import random
import warnings

import pydantic
import tqdm
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from rich.logging import RichHandler

from cache_replacement.policy_learning.cache.cache import Cache
from cache_replacement.policy_learning.cache.eviction_policy import EvictionPolicy, RandomPolicy
from cache_replacement.policy_learning.cache.memtrace import MemoryTrace

RAG_SUMMARY = """

This is a summary of the RAG file above.

We have a cache with 64 * 16 * 64 capacity in bytes, cache line size of 64 bytes and associativity of 16.

We have 5 different eviction policies being tested: Belady, LRU, Policy02, Policy03, Policy04.

The traces used are: omnetpp, astar, lbm

The optimal eviction policy is Belady.

The MPKI (misses per 1000 instructions) values for the different policies and traces are as follows:

Policy02: omnetpp: 0.12334, astar: 0.93688, lbm: 0.75931
Policy03: omnetpp: 0.1233, astar: 0.93786, lbm: 0.76127
Policy04: omnetpp: 0.15811, astar: 0.94474, lbm: 0.95424

Policy02 is a single layer model MLP.
Policy03 is two-layer model MLP.
Policy04 is original Parror model with RNN used instead of an LSTM.

"""


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    assert level.lower() in ["debug", "info", "warning", "error", "critical"], f"Invalid log level: {level}"
    warnings.filterwarnings("ignore")

    handlers = list()
    handlers.append(RichHandler(rich_tracebacks=True, tracebacks_show_locals=True))

    FORMAT = "%(name)s: %(message)s"
    logging.basicConfig(
        level=logging.CRITICAL,
        format=FORMAT,
        datefmt="[%X]",
        handlers=handlers,
    )

    logger = logging.getLogger(name)
    logger.setLevel(level.upper())
    return logger


class AddressResponse(pydantic.BaseModel):
    address: int


class LLMEvictionPolicy(EvictionPolicy):
    def __init__(self, rag_chain: RetrievalQA, workload: str, logger: logging.Logger):
        self.rag_chain = rag_chain
        self.logger = logger
        self.workload = workload

    def __call__(self, cache_access, access_times):
        self.logger.debug(f"Cache access: {cache_access}")
        self.logger.debug(f"Access times: {access_times}")

        if not access_times:
            self.logger.warning("No access times. Returning None")
            return None, {}

        # Define query
        query = f"What line would you evict if the PC was {cache_access.pc:x} and the workload was {self.workload}?"
        query += "\n\n Respond only in the format {'address': 'str'}. For example: {'address': '0x12345678'}"
        query += f"\n\n Pick one of the following addresses in the cache: {[hex(line) for (line, _) in cache_access.cache_lines]}"

        self.logger.debug(f"Query: {query}\nRetrieving answer from LLM")
        response = self.rag_chain.invoke({"query": query})

        evicted_address = None

        try:
            evicted_address = int(json.loads(response["result"].replace("'", '"'))["address"], 16)
        except Exception:
            self.logger.warning("Invalid response from LLM. Returning random address")
            evicted_address = random.choice([line for (line, _) in cache_access.cache_lines])

        if evicted_address not in access_times:
            prev = evicted_address
            evicted_address = random.choice([line for (line, _) in cache_access.cache_lines])
            self.logger.warning(f"Invalid address: {prev}. Returning random address: {evicted_address}")

        self.logger.info(f"Evicted address: {hex(evicted_address)}")
        self.logger.info(f"Possible addresses: {[hex(line) for (line, _) in cache_access.cache_lines]}")
        scores = {address: 0 for address in access_times.keys()}
        scores[evicted_address] = 1
        return evicted_address, scores

    @classmethod
    def from_rag_file(cls, rag_file: str, workload, logger: logging.Logger, **kwargs) -> "LLMEvictionPolicy":
        docs = TextLoader(rag_file).load()
        # Add summary to all docs
        for doc in docs:
            doc.page_content = doc.page_content + "\n\n" + RAG_SUMMARY

        text_splitter = CharacterTextSplitter(separator="\n")
        docs = text_splitter.split_documents(docs)

        embedding = OpenAIEmbeddings()

        persist_directory = "./outputs/chroma_db"
        if os.path.exists(persist_directory) and args.use_cached_vector_db:
            logger.info("Using cached vector database")
            vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding)
        else:
            logger.info("Creating new vector database")
            vectorstore = Chroma.from_documents(
                documents=docs, embedding=embedding, persist_directory=persist_directory
            )

        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            return_source_documents=True,
            verbose=True,
        )

        return cls(rag_chain=rag_chain, workload=workload, logger=logger, **kwargs)


class CacheObserver:
    def __init__(self, multiplier=1):
        self.reset()
        self.num_instructions = 1_000_000_000  # champsim simulates this number of instructions and than crashes
        self.multiplier = multiplier

    def update(self, hit):
        self.cache_hits += int(hit)
        self.cache_accesses += 1

    def reset(self):
        self.cache_accesses = 0
        self.cache_hits = 0

    @property
    def cache_misses(self):
        return self.cache_accesses - self.cache_hits

    def compute_mpki(self):
        return (self.cache_misses * self.multiplier / self.num_instructions) * 1000

    def compute_hit_rate(self):
        return self.cache_hits / self.cache_accesses


def main(args: argparse.Namespace):
    load_dotenv()
    assert os.getenv("OPENAI_API_KEY") is not None, "OPENAI_API_KEY is not set. Please set it in .env file"

    logger = get_logger("[runqa]", level=args.log_level)
    logger.info(f"RAG file: {args.rag_file}")

    cache_config = {"cache_line_size": 64, "capacity": 64 * 16 * 64, "associativity": 16}
    memtrace = MemoryTrace(args.memory_trace, cache_line_size=cache_config["cache_line_size"], max_look_ahead=int(1e5))

    llm_evictor = LLMEvictionPolicy.from_rag_file(args.rag_file, "astar", logger)
    random_evictor = RandomPolicy()
    cache_observer_llm = CacheObserver(multiplier=1)
    cache_observer_random = CacheObserver(multiplier=1)

    cache_llm = Cache.from_config(cache_config, eviction_policy=llm_evictor, trace=memtrace)
    cache_random = Cache.from_config(cache_config, eviction_policy=random_evictor, trace=memtrace)

    print("Starting")

    num_cache_accesses = sum(1 for _ in open(args.memory_trace))
    with memtrace:
        for read_idx in tqdm.tqdm(range(num_cache_accesses), desc=f"trace: {args.memory_trace}"):
            assert not memtrace.done()
            pc, address = memtrace.next()
            hit = cache_llm.read(pc, address)
            cache_observer_llm.update(hit)

            hit = cache_random.read(pc, address)
            cache_observer_random.update(hit)

            if read_idx == 200:
                break

    print(
        f"LLM Hit rate: {cache_observer_llm.compute_hit_rate()} | Random Hit rate: {cache_observer_random.compute_hit_rate()}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory_trace", type=str, required=True)
    parser.add_argument("--rag_file", type=str, required=True)
    parser.add_argument("--use_cached_vector_db", type=bool, default=True)
    parser.add_argument("--log_level", type=str, default="info")
    args = parser.parse_args()

    main(args)
