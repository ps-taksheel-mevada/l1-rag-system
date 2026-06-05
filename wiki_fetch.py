import os
import warnings
import ssl
import functools

os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["CURL_CA_BUNDLE"] = ""
ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings("ignore", category=UserWarning, message=".*parser.*")

import requests as _requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

WIKI_USER_AGENT = "Local-RAG-Agent/1.0"

_orig_send = _requests.Session.send
_orig_get = _requests.get
_orig_post = _requests.post
_orig_head = _requests.head

def _patched_send(self, request, **kwargs):
    kwargs.setdefault("verify", False)
    kwargs.setdefault("timeout", 15)
    request.headers.setdefault("User-Agent", WIKI_USER_AGENT)
    return _orig_send(self, request, **kwargs)

def _patched_get(url, **kwargs):
    kwargs.setdefault("verify", False)
    kwargs.setdefault("timeout", 15)
    kwargs.setdefault("headers", {})
    kwargs["headers"].setdefault("User-Agent", WIKI_USER_AGENT)
    return _orig_get(url, **kwargs)

def _patched_post(url, **kwargs):
    kwargs.setdefault("verify", False)
    kwargs.setdefault("timeout", 15)
    kwargs.setdefault("headers", {})
    kwargs["headers"].setdefault("User-Agent", WIKI_USER_AGENT)
    return _orig_post(url, **kwargs)

def _patched_head(url, **kwargs):
    kwargs.setdefault("verify", False)
    kwargs.setdefault("timeout", 15)
    kwargs.setdefault("headers", {})
    kwargs["headers"].setdefault("User-Agent", WIKI_USER_AGENT)
    return _orig_head(url, **kwargs)

_requests.Session.send = _patched_send
_requests.get = _patched_get
_requests.post = _patched_post
_requests.head = _patched_head

try:
    import wikipedia
except ImportError:
    wikipedia = None

if wikipedia is not None:
    wikipedia.set_user_agent("Local-RAG-Agent/1.0 (https://github.com)")

from ingestion import get_text_splitter, get_embeddings, get_or_create_vector_store, reset_vector_store
from uuid import uuid4


def fetch_and_ingest(
    keywords: list[str],
    pages_per_keyword: int = 3,
    clear_existing: bool = False,
):
    if wikipedia is None:
        raise ImportError("Install wikipedia: uv pip install wikipedia")

    if clear_existing:
        reset_vector_store()

    splitter = get_text_splitter()
    embeddings = get_embeddings()
    vector_store = get_or_create_vector_store(embeddings)
    total_chunks = 0
    fetched_pages = []

    for keyword in keywords:
        print(f"Searching Wikipedia for: {keyword}")
        try:
            titles = wikipedia.search(keyword, results=pages_per_keyword)
        except Exception as e:
            print(f"  Search failed: {e}")
            continue

        for title in titles:
            page = None
            for attempt in range(3):
                try:
                    page = wikipedia.page(title, auto_suggest=False)
                    break
                except wikipedia.DisambiguationError as e:
                    if e.options:
                        try:
                            page = wikipedia.page(e.options[0], auto_suggest=False)
                            print(f"  Disambiguated '{title}' -> '{e.options[0]}'")
                            break
                        except Exception:
                            break
                    break
                except wikipedia.PageError as e:
                    print(f"  Skipping '{title}': {e}")
                    break
                except (ValueError, KeyError, _requests.JSONDecodeError):
                    if attempt == 2:
                        print(f"  Skipping '{title}' (bad response)")
                        break
                    continue
                except _requests.ConnectionError as e:
                    if "CERTIFICATE" in str(e):
                        print(f"  SSL error for '{title}', skipping")
                        break
                    if attempt == 2:
                        print(f"  Failed to fetch '{title}': {e}")
                        break
                    continue
                except Exception:
                    if attempt == 2:
                        print(f"  Skipping '{title}'")
                        break
                    continue

            if page is None:
                if title:
                    print(f"  Skipping '{title}'")
                continue

            content = page.content
            if not content:
                continue

            docs = splitter.create_documents(
                [content],
                metadatas=[{"source": page.url, "title": page.title}],
            )
            uuids = [str(uuid4()) for _ in range(len(docs))]
            vector_store.add_documents(documents=docs, ids=uuids)
            total_chunks += len(docs)
            fetched_pages.append({"title": page.title, "url": page.url, "chunks": len(docs)})
            print(f"  Ingested: {page.title} ({len(docs)} chunks)")

    return total_chunks, fetched_pages
