#!/usr/bin/env python
import argparse
import sys
import os
import json

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, ToolMessage
from agent.graph import build_graph
from ingestion import (
    load_vector_store,
    ingest_text,
    ingest_jsonl_content,
    reset_vector_store,
)
from config import (
    get_chat_model,
    get_embedding_model,
    get_checkpoint_db,
)
from wiki_fetch import fetch_and_ingest


CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def print_banner():
    print(f"{CYAN}{BOLD}+======================================+{RESET}")
    print(f"{CYAN}{BOLD}|      LangGraph RAG Agent (CLI)      |{RESET}")
    print(f"{CYAN}{BOLD}+======================================+{RESET}")
    print(f"{DIM}Chat model: {get_chat_model()} | Embedding: {get_embedding_model()}{RESET}")
    print(f"{DIM}Type /help for commands, /quit to exit{RESET}\n")


def print_help():
    print(f"\n{YELLOW}Commands:{RESET}")
    print(f"  {BOLD}/help{RESET}              Show this help")
    print(f"  {BOLD}/quit{RESET}              Exit the chat")
    print(f"  {BOLD}/clear{RESET}             Start a new conversation")
    print(f"  {BOLD}/ingest PATH{RESET}        Ingest a .txt or .jsonl file")
    print(f"  {BOLD}/wiki KEYWORD [N]{RESET}   Fetch & ingest N Wikipedia pages (default 3)")
    print(f"  {BOLD}/reset-db{RESET}          Delete and recreate the vector store")
    print(f"  {BOLD}/model NAME{RESET}        Switch chat model (e.g. /model gemma3:4b)")
    print()


def ingest_file(filepath: str):
    if not os.path.isfile(filepath):
        print(f"{RED}File not found: {filepath}{RESET}")
        return

    print(f"{YELLOW}Ingesting {filepath}...{RESET}")
    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    if filepath.endswith(".jsonl"):
        chunks = ingest_jsonl_content(content, source_name=os.path.basename(filepath))
    else:
        chunks = ingest_text(content, source_name=os.path.basename(filepath))

    print(f"{GREEN}Done. Ingested {chunks} chunks.{RESET}")


DARK_GRAY = "\033[90m"
TOOL_PREFIX = f"{DARK_GRAY}"


def stream_response(graph, messages, thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    full_response = ""
    shown_tool_calls = set()
    current_node = None
    step_count = 0
    agent_phase = "start"

    for message, metadata in graph.stream(
        {"messages": messages[-1:]},
        config,
        stream_mode="messages",
    ):
        node = metadata.get("langgraph_node", "")
        step = metadata.get("langgraph_step", 0)

        if step != step_count:
            step_count = step
            if node == "agent" and agent_phase == "start":
                print(f"\n{TOOL_PREFIX}[AGENT] Analyzing query...{RESET}")
                agent_phase = "thinking"

        if node != current_node:
            current_node = node
            if node == "agent" and agent_phase == "tools_done":
                print(f"\n{TOOL_PREFIX}[AGENT] Composing response...{RESET}")
                agent_phase = "composing"
            elif node == "tools":
                print(f"\n{TOOL_PREFIX}[TOOLS] Executing...{RESET}")
                agent_phase = "tools"

        if isinstance(message, (AIMessageChunk, AIMessage)):
            if message.tool_calls:
                for tc in message.tool_calls:
                    tc_id = tc.get("id", "")
                    tc_name = tc.get("name")
                    if tc_id and tc_name and tc_id not in shown_tool_calls:
                        shown_tool_calls.add(tc_id)
                        tc_args = tc.get("args", {})
                        args_str = json.dumps(tc_args, ensure_ascii=False)
                        print(f"\n{TOOL_PREFIX}[AGENT] Decided to call: {tc_name}({args_str}){RESET}")
                        agent_phase = "calling_tool"

            content = message.content
            if content:
                if isinstance(content, str):
                    full_response += content
                    sys.stdout.write(content)
                elif isinstance(content, list):
                    new_text = "".join(
                        item.get("text", "") if isinstance(item, dict) else str(item)
                        for item in content
                    )
                    full_response += new_text
                    sys.stdout.write(new_text)
                sys.stdout.flush()

        elif isinstance(message, ToolMessage):
            doc_count = message.content.count("[") if hasattr(message, "content") and isinstance(message.content, str) else 0
            if doc_count > 0:
                print(f"\n{TOOL_PREFIX}[TOOLS] Retrieved {doc_count} document(s) from knowledge base{RESET}")
            else:
                print(f"\n{TOOL_PREFIX}[TOOLS] Completed (no matches found){RESET}")
            agent_phase = "tools_done"

    if full_response:
        print()
    return full_response


def chat_loop():
    print_banner()

    llm = ChatOllama(model=get_chat_model(), temperature=0, streaming=True)
    vector_store = load_vector_store()
    graph, _tools = build_graph(llm, vector_store, get_checkpoint_db())
    thread_id = "cli-session"

    messages = []

    while True:
        try:
            user_input = input(f"\n{BOLD}{GREEN}You:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{YELLOW}Goodbye!{RESET}")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd, *args = user_input[1:].split(maxsplit=1)
            arg = args[0] if args else ""

            if cmd == "quit":
                print(f"{YELLOW}Goodbye!{RESET}")
                break
            elif cmd == "help":
                print_help()
            elif cmd == "clear":
                messages = []
                thread_id = f"cli-{os.urandom(4).hex()}"
                graph, _tools = build_graph(llm, vector_store, get_checkpoint_db())
                print(f"{GREEN}Conversation reset.{RESET}")
            elif cmd == "ingest":
                if arg:
                    ingest_file(arg)
                    vector_store = load_vector_store()
                    graph, _tools = build_graph(llm, vector_store, get_checkpoint_db())
                else:
                    print(f"{RED}Usage: /ingest <filepath>{RESET}")
            elif cmd == "reset-db":
                reset_vector_store()
                vector_store = load_vector_store()
                graph, _tools = build_graph(llm, vector_store, get_checkpoint_db())
                print(f"{YELLOW}Vector store cleared.{RESET}")
            elif cmd == "wiki":
                if arg:
                    parts = arg.split(maxsplit=1)
                    keyword = parts[0]
                    pages = int(parts[1]) if len(parts) > 1 else 3
                    print(f"{YELLOW}Fetching Wikipedia: '{keyword}' ({pages} pages)...{RESET}")
                    total, fetched = fetch_and_ingest([keyword], pages_per_keyword=pages)
                    vector_store = load_vector_store()
                    graph, _tools = build_graph(llm, vector_store, get_checkpoint_db())
                    print(f"{GREEN}Done. {total} chunks from {len(fetched)} pages ingested.{RESET}")
                    for p in fetched:
                        print(f"  - {p['title']} ({p['url']})")
                else:
                    print(f"{RED}Usage: /wiki <keyword> [pages]{RESET}")
            elif cmd == "model":
                if arg:
                    llm = ChatOllama(model=arg, temperature=0, streaming=True)
                    graph, _tools = build_graph(llm, vector_store, get_checkpoint_db())
                    print(f"{GREEN}Switched to model: {arg}{RESET}")
                else:
                    print(f"{RED}Usage: /model <name>{RESET}")
            else:
                print(f"{RED}Unknown command: /{cmd}. Type /help for commands.{RESET}")
            continue

        messages.append(HumanMessage(content=user_input))

        print(f"{BOLD}{CYAN}Agent:{RESET} ", end="", flush=True)
        response = stream_response(graph, messages, thread_id)
        messages.append(AIMessage(content=response))


def main():
    parser = argparse.ArgumentParser(description="LangGraph RAG Agent CLI")
    parser.add_argument(
        "--ingest",
        metavar="PATH",
        help="Ingest a file into the vector store (txt or jsonl)",
    )
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="Clear the vector store before starting",
    )
    parser.add_argument(
        "--model",
        metavar="NAME",
        default=get_chat_model(),
        help=f"Chat model to use (default: {get_chat_model()})",
    )
    parser.add_argument(
        "--query",
        metavar="TEXT",
        help="Run a single query and exit (non-interactive)",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output",
    )
    parser.add_argument(
        "--wiki",
        metavar="KEYWORD",
        help="Fetch Wikipedia pages for a keyword and ingest into vector store",
    )
    parser.add_argument(
        "--wiki-pages",
        metavar="N",
        type=int,
        default=3,
        help="Number of Wikipedia pages to fetch (default: 3)",
    )
    parser.add_argument(
        "--wiki-clear",
        action="store_true",
        help="Clear vector store before Wikipedia fetch",
    )
    args = parser.parse_args()

    if args.reset_db:
        reset_vector_store()
        print(f"{YELLOW}Vector store cleared.{RESET}")

    if args.ingest:
        ingest_file(args.ingest)

    if args.wiki:
        total, fetched = fetch_and_ingest(
            [args.wiki],
            pages_per_keyword=args.wiki_pages,
            clear_existing=args.wiki_clear,
        )
        print(f"{GREEN}Done. {total} chunks from {len(fetched)} pages ingested.{RESET}")
        for p in fetched:
            print(f"  - {p['title']} ({p['url']})")

    if args.query:
        llm = ChatOllama(model=args.model, temperature=0, streaming=True)
        vector_store = load_vector_store()
        graph, _tools = build_graph(llm, vector_store, get_checkpoint_db())
        messages = [HumanMessage(content=args.query)]
        thread_id = f"query-{os.urandom(4).hex()}"

        if args.no_stream:
            config = {"configurable": {"thread_id": thread_id}}
            result = graph.invoke({"messages": messages}, config)
            response = result["messages"][-1].content
            if isinstance(response, list):
                response = "".join(
                    c.get("text", "") if isinstance(c, dict) else str(c)
                    for c in response
                )
            print(response)
        else:
            print(f"{BOLD}{CYAN}Agent:{RESET} ", end="", flush=True)
            response = stream_response(graph, messages, thread_id)
        return

    chat_loop()


if __name__ == "__main__":
    main()
