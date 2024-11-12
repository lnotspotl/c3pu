#!/usr/bin/env python3

import argparse
import logging
import os
import warnings

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from rich.logging import RichHandler

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
Policy04 is original Parror model with RNN used instead of LSTM

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


def main(args: argparse.Namespace):
    load_dotenv()
    assert os.getenv("OPENAI_API_KEY") is not None, "OPENAI_API_KEY is not set. Please set it in .env file"

    logger = get_logger("[runqa]", level=args.log_level)
    logger.info(f"RAG file: {args.rag_file}")

    docs = TextLoader(args.rag_file).load()

    # Add summary to all docs
    for doc in docs:
        doc.page_content = doc.page_content + "\n\n" + RAG_SUMMARY

    text_splitter = CharacterTextSplitter(separator="\n")
    docs = text_splitter.split_documents(docs)

    llm = ChatOpenAI(model="gpt-3.5-turbo")
    embedding = OpenAIEmbeddings()

    persist_directory = "./outputs/chroma_db"
    if os.path.exists(persist_directory) and args.use_cached_vector_db:
        logger.info("Using cached vector database")
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    else:
        logger.info("Creating new vector database")
        vectorstore = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory=persist_directory)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever(), return_source_documents=True, verbose=True
    )

    logger.info(f"Query: {args.query}")
    response = rag_chain.invoke({"query": args.query})

    logger.debug("Retrieved Documents:")
    for doc in response["source_documents"]:
        logger.debug(f"Document: {doc.page_content}")
        logger.debug(f"Source: {doc.metadata}")

    logger.info(f"Response: {response['result']}")

    print(f"Q: {args.query}")
    print(f"A: {response['result']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag_file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--use_cached_vector_db", type=bool, default=True)
    parser.add_argument("--log_level", type=str, default="info")
    args = parser.parse_args()

    main(args)
