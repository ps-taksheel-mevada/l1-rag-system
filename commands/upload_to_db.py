from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import shutil
import os
from uuid import uuid4
from stqdm import stqdm
from commands.init_db import vector_store

class UploadDocs:

    def __init__(self, database_location, collection_name, embedding_model):
        self.vector_store = vector_store
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )

    def __call__(self, data):
        for doc in stqdm(data):
            texts = self.text_splitter.create_documents([doc['content']], metadatas=[{"source":f"https://en.wikipedia.org/wiki/{doc['key']}", "title":doc["key"]}])
            uuids = [str(uuid4()) for _ in texts]
            self.vector_store.add_documents(documents=texts, ids=uuids)

# embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")

# if os.path.exists(os.getenv("DATABASE_LOCATION")):
#     shutil.rmtree(os.getenv("DATABASE_LOCATION"))

# vector_store = Chroma(
#     collection_name=os.getenv("COLLECTION_NAME"),
#     embedding_function=embeddings,
#     persist_directory=os.getenv("DATABASE_LOCATION")
# )

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200,
#     length_function=len,
#     is_separator_regex=False,
# )

# def upload(data):
#     for doc in data:
#         texts = text_splitter.create_documents([doc['content']],metadatas=[{"source":"https://en.wikipedia.org/wiki/{doc["key"]}", "title":doc["key"]}])
#         uuids = [str(uuid4()) for _ in texts]

#         vector_store.add_documents(documents=texts, ids=uuids)