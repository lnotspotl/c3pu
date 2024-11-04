#!/usr/bin/env python3

import argparse
import os

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter


def main(args: argparse.Namespace):
    load_dotenv()
    assert os.getenv("OPENAI_API_KEY") is not None, "OPENAI_API_KEY is not set. Please set it in .env file"
    print(f"RAG file: {args.rag_file}")

    docs = TextLoader(args.rag_file).load()
    text_splitter = CharacterTextSplitter(separator="\n")
    docs = text_splitter.split_documents(docs)

    llm = ChatOpenAI(model="gpt-3.5-turbo")
    embedding = OpenAIEmbeddings()

    persist_directory = "./outputs/chroma_db"
    if os.path.exists(persist_directory):
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    else:
        vectorstore = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory=persist_directory)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever(), return_source_documents=True, verbose=True
    )

    response = rag_chain.invoke({"query": args.query})
    print(response["result"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag_file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    args = parser.parse_args()

    main(args)
