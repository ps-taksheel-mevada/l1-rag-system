from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="mxbai-embed-large:latest",
)

vector_store = Chroma(
    collection_name="rag_data",
    embedding_function=embeddings,
    persist_directory="chroma_db", 
)