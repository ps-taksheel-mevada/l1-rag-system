import uuid
import json
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, ToolMessage
from agent.graph import build_graph
from ingestion import load_vector_store, ingest_text, ingest_jsonl_content, reset_vector_store
from config import get_embedding_model, get_chat_model, get_checkpoint_db
from wiki_fetch import fetch_and_ingest

st.set_page_config(page_title="LangGraph RAG Agent", page_icon="🤖", layout="wide")
st.title("🤖 LangGraph RAG Agent")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_resource
def get_vector_store():
    return load_vector_store()


@st.cache_resource
def get_agent_graph(_llm, _vector_store, checkpoint_db):
    return build_graph(_llm, _vector_store, checkpoint_db)


with st.sidebar:
    st.header("⚙️ Settings")

    chat_model_name = st.text_input(
        "Chat Model",
        value=get_chat_model(),
        placeholder="eg. gemma3:4b, llama3.2:3b",
        help="Ollama model name (pull it first with `ollama pull <name>`)"
    )
    embedding_model = st.text_input(
        "Embedding Model",
        value=get_embedding_model(),
        placeholder="eg. nomic-embed-text",
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)

    st.divider()
    st.header("📁 Documents")

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["txt", "jsonl"],
        help="TXT for plain text. JSONL with keys: raw_text, url, title"
    )

    if uploaded_file:
        file_content = uploaded_file.read().decode("utf-8")
        file_name = uploaded_file.name

        if st.button("Ingest Document", type="primary"):
            with st.spinner("Chunking and embedding..."):
                if file_name.endswith(".jsonl"):
                    chunks = ingest_jsonl_content(file_content, source_name=file_name)
                else:
                    chunks = ingest_text(file_content, source_name=file_name)
            st.success(f"Ingested {chunks} chunks from {file_name}")
            st.cache_resource.clear()

    st.divider()
    st.header("🌐 Wikipedia Fetch")

    wiki_keyword = st.text_input("Keyword", placeholder="e.g. artificial intelligence")
    wiki_pages = st.number_input("Pages to fetch", min_value=1, max_value=10, value=3)

    if st.button("Fetch & Ingest Wikipedia", type="primary"):
        if wiki_keyword:
            with st.spinner(f"Fetching Wikipedia pages for '{wiki_keyword}'..."):
                total, fetched = fetch_and_ingest([wiki_keyword], pages_per_keyword=wiki_pages)
            st.success(f"Ingested {total} chunks from {len(fetched)} pages")
            for p in fetched:
                st.caption(f"- [{p['title']}]({p['url']}) ({p['chunks']} chunks)")
            st.cache_resource.clear()
        else:
            st.warning("Enter a keyword first")

    st.divider()

    if st.button("🔄 Reset Chat"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    if st.button("🗑️ Clear Vector Store", type="secondary"):
        reset_vector_store()
        st.cache_resource.clear()
        st.warning("Vector store cleared. Re-ingest documents to use RAG.")
        st.rerun()

    st.divider()
    st.caption(f"Thread: {st.session_state.thread_id[:8]}...")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_input = st.chat_input("Ask me anything...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            llm = ChatOllama(
                model=chat_model_name,
                temperature=temperature,
                streaming=True,
            )

            vector_store = get_vector_store()
            graph, _tools = get_agent_graph(llm, vector_store, get_checkpoint_db())

            langchain_messages = []
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    langchain_messages.append(AIMessage(content=msg["content"]))

            config = {"configurable": {"thread_id": st.session_state.thread_id}}

            response_placeholder = st.empty()
            full_response = ""
            shown_tool_calls = set()
            current_node = None
            step_count = 0
            agent_phase = "start"

            for message, metadata in graph.stream(
                {"messages": langchain_messages[-1:]},
                config,
                stream_mode="messages",
            ):
                node = metadata.get("langgraph_node", "")
                step = metadata.get("langgraph_step", 0)

                if step != step_count:
                    step_count = step
                    if node == "agent" and agent_phase == "start":
                        st.caption("[AGENT] Analyzing query...")
                        agent_phase = "thinking"

                if node != current_node:
                    current_node = node
                    if node == "agent" and agent_phase == "tools_done":
                        st.caption("[AGENT] Composing response...")
                        agent_phase = "composing"
                    elif node == "tools":
                        st.caption("[TOOLS] Executing...")
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
                                st.caption(f"[AGENT] Decided to call: {tc_name}({args_str})")
                                agent_phase = "calling_tool"

                    content = message.content
                    if content:
                        if isinstance(content, str):
                            full_response += content
                        elif isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict):
                                    full_response += item.get("text", "")
                                else:
                                    full_response += str(item)
                        response_placeholder.markdown(full_response + "▌")

                elif isinstance(message, ToolMessage):
                    doc_count = message.content.count("[") if hasattr(message, "content") and isinstance(message.content, str) else 0
                    if doc_count > 0:
                        st.caption(f"[TOOLS] Retrieved {doc_count} document(s) from knowledge base")
                    else:
                        st.caption(f"[TOOLS] Completed (no matches found)")
                    agent_phase = "tools_done"

            response_placeholder.markdown(full_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )

        except Exception as e:
            st.error(f"Error: {str(e)}")
