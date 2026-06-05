import os
from dotenv import load_dotenv

load_dotenv()


def get_embedding_model() -> str:
    return os.getenv("EMBEDDING_MODEL", "mxbai-embed-large:latest")


def get_chat_model() -> str:
    return os.getenv("CHAT_MODEL", "llama3.2:latest")


def get_model_provider() -> str:
    return os.getenv("MODEL_PROVIDER", "ollama")


def get_database_location() -> str:
    return os.getenv("DATABASE_LOCATION", "chroma_db")


def get_collection_name() -> str:
    return os.getenv("COLLECTION_NAME", "rag_data")


def get_checkpoint_db() -> str:
    return os.getenv("CHECKPOINT_DB", "checkpoints.db")


def get_dataset_folder() -> str:
    return os.getenv("DATASET_STORAGE_FOLDER", "datasets/")
