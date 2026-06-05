# RAG Agent CLI

A local Retrieval-Augmented Generation (RAG) agent powered by LangGraph and Ollama.

## Prerequisites

- Python 3.13+
- [Ollama](https://ollama.com) installed and running

Pull the required models:

```
ollama pull llama3.2:latest
ollama pull mxbai-embed-large:latest
```

## Installation

```
git clone https://github.com/ps-taksheel-mevada/l1-rag-system
cd l1-rag-system
```

### Using uv (recommended)

```
uv sync
```

### Using pip

```
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and configure if needed:

```
cp .env.example .env
```

## CLI Usage

> If using uv, prefix commands with `uv run` (e.g. `uv run python cli.py`).

### Interactive Mode

```
python cli.py
```

### Commands

| Command                  | Description                                    |
|--------------------------|------------------------------------------------|
| `/help`                  | Show all available commands                    |
| `/quit`                  | Exit the chat                                  |
| `/clear`                 | Start a new conversation                       |
| `/ingest PATH`           | Ingest a `.txt` or `.jsonl` file               |
| `/wiki KEYWORD [N]`      | Fetch & ingest N Wikipedia pages (default 3)   |
| `/reset-db`              | Delete and recreate the vector store           |
| `/model NAME`            | Switch chat model (e.g. `/model gemma3:4b`)    |

### Non-interactive Mode

Ingest a file before starting:

```
python cli.py --ingest data.txt
```

Run a single query and exit:

```
python cli.py --query "What is retrieval augmented generation?"
```

Fetch Wikipedia pages and query:

```
python cli.py --wiki "machine learning" --wiki-pages 5
```

Clear vector store, fetch Wikipedia, and query:

```
python cli.py --wiki "deep learning" --wiki-clear --query "Explain deep learning"
```

Disable streaming output:

```
python cli.py --query "Hello" --no-stream
```

Use a specific model:

```
python cli.py --model gemma3:4b
```

### CLI Arguments

| Argument         | Description                                    |
|------------------|------------------------------------------------|
| `--ingest PATH`  | Ingest a `.txt` or `.jsonl` file               |
| `--reset-db`     | Clear the vector store before starting         |
| `--model NAME`   | Chat model to use (default: `llama3.2:latest`) |
| `--query TEXT`   | Run a single query and exit (non-interactive)  |
| `--no-stream`    | Disable streaming output                       |
| `--wiki KEYWORD` | Fetch Wikipedia pages and ingest               |
| `--wiki-pages N` | Number of Wikipedia pages to fetch (default: 3)|
| `--wiki-clear`   | Clear vector store before Wikipedia fetch      |

## Environment Variables

| Variable              | Default                    | Description            |
|-----------------------|----------------------------|------------------------|
| `EMBEDDING_MODEL`     | `mxbai-embed-large:latest` | Ollama embedding model |
| `CHAT_MODEL`          | `llama3.2:latest`          | Ollama chat model      |
| `DATABASE_LOCATION`   | `chroma_db`                | Vector store directory |
| `COLLECTION_NAME`     | `rag_data`                 | ChromaDB collection    |
| `CHECKPOINT_DB`       | `checkpoints.db`           | SQLite checkpoint DB   |
| `DATASET_STORAGE_FOLDER` | `datasets/`             | Dataset storage folder |
