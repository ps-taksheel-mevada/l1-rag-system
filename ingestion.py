import os
import json
import shutil
from uuid import uuid4
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import get_embedding_model, get_collection_name, get_database_location


def get_embeddings():
    return OllamaEmbeddings(model=get_embedding_model())


def get_or_create_vector_store(embeddings: OllamaEmbeddings | None = None):
    if embeddings is None:
        embeddings = get_embeddings()
    return Chroma(
        collection_name=get_collection_name(),
        embedding_function=embeddings,
        persist_directory=get_database_location(),
    )


def get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )


def reset_vector_store():
    db_path = get_database_location()
    if os.path.exists(db_path):
        shutil.rmtree(db_path, ignore_errors=True)


def ingest_jsonl_content(file_content: str, source_name: str = "uploaded"):
    """Ingest JSON-lines content into the vector store."""
    embeddings = get_embeddings()
    splitter = get_text_splitter()
    vector_store = get_or_create_vector_store(embeddings)

    total_chunks = 0
    for line in file_content.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        raw_text = obj.get("raw_text", "")
        url = obj.get("url", source_name)
        title = obj.get("title", "")

        if not raw_text:
            continue

        docs = splitter.create_documents(
            [raw_text],
            metadatas=[{"source": url, "title": title}],
        )
        uuids = [str(uuid4()) for _ in range(len(docs))]
        vector_store.add_documents(documents=docs, ids=uuids)
        total_chunks += len(docs)

    return total_chunks


def ingest_text(text: str, source_name: str = "uploaded"):
    """Ingest plain text with optional metadata."""
    embeddings = get_embeddings()
    splitter = get_text_splitter()
    vector_store = get_or_create_vector_store(embeddings)

    docs = splitter.create_documents(
        [text],
        metadatas=[{"source": source_name, "title": source_name}],
    )
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(documents=docs, ids=uuids)

    return len(docs)


def load_vector_store():
    embeddings = get_embeddings()
    return Chroma(
        collection_name=get_collection_name(),
        embedding_function=embeddings,
        persist_directory=get_database_location(),
    )
